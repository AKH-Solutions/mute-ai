import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
from collections import deque
import datetime
import os
import argparse
from mss import mss
import pygetwindow as gw
import sys
import platform
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
import subprocess

# Configuration
FRAME_INTERVAL = 1  # Seconds between frame processing
VOTING_WINDOW = 10  # Number of frames to vote on
VOTING_THRESHOLD = 7  # Minimum "commercial" frames to trigger mute
FALLBACK_REGION = {"top": 100, "left": 100, "width": 1280, "height": 720}  # Hardcoded fallback

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("mps")  # Use MPS for M1 Mac Mini
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def get_chrome_window_region(title_keyword="Home - YouTube TV"):
    """Dynamically find Chrome window and get its region."""
    if platform.system() != 'Darwin':
        # Fallback for non-macOS
        try:
            chrome_windows = [w for w in gw.getAllWindows() if "Google Chrome" in w.title and title_keyword.lower() in w.title.lower()]
            if chrome_windows:
                window = chrome_windows[0]
                left, top, width, height = window.left, window.top, window.width, window.height
                print(f"Found Chrome window: '{window.title}' at ({left}, {top}), size {width}x{height}")
                return {"top": top, "left": left, "width": width, "height": height}
        except Exception as e:
            print(f"PyGetWindow error: {e}")
        return FALLBACK_REGION

    # macOS: Priority 1 - PyGetWindow (Accessibility API)
    try:
        print("Debug: Checking PyGetWindow for Chrome windows...")
        all_windows = gw.getAllWindows()
        chrome_windows = [w for w in all_windows if "Google Chrome" in w.title and title_keyword.lower() in w.title.lower()]
        if chrome_windows:
            window = chrome_windows[0]
            left, top, width, height = window.left, window.top, window.width, window.height
            print(f"Found Chrome window via PyGetWindow: '{window.title}' at ({left}, {top}), size {width}x{height}")
            return {"top": top, "left": left, "width": width, "height": height}
        # Debug: List all Chrome titles
        chrome_titles = [w.title for w in all_windows if "Google Chrome" in w.title]
        print(f"Debug PyGetWindow: Chrome titles: {chrome_titles}")
    except Exception as e:
        print(f"PyGetWindow error: {e}")

    # Priority 2: Fixed AppleScript using System Events (avoids Chrome coercion errors)
    try:
        applescript = '''
tell application "System Events"
    tell process "Google Chrome"
        set windowList to {}
        repeat with w in windows
            set end of windowList to {name of w, position of w, size of w}
        end repeat
        return windowList
    end tell
end tell
'''
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            # Parse AppleScript output (list of {name, {x,y}, {w,h}})
            # AppleScript returns as string like {{ "title1", {x,y}, {w,h} }, ...}, so use eval with care (controlled output)
            windows_info = eval(result.stdout.strip())
            for window_info in windows_info:
                title, pos, sz = window_info
                if title_keyword.lower() in str(title).lower():
                    left, top = pos
                    width, height = sz
                    print(f"Found Chrome window via AppleScript: '{title}' at ({left}, {top}), size {width}x{height}")
                    return {"top": top, "left": left, "width": width, "height": height}
            print(f"Debug AppleScript: No matching title '{title_keyword}', available titles: {[str(info[0]) for info in windows_info]}")
        else:
            print(f"AppleScript failed with return code {result.returncode}: {result.stderr}")
    except Exception as e:
        print(f"AppleScript error: {e}")

    # Priority 3: Quartz (use first Chrome window, since titles are often empty)
    try:
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        print("Debug: Listing Chrome windows (Quartz fallback):")
        chrome_windows = []
        for window in windows:
            owner_name = window.get('kCGWindowOwnerName', '')
            if owner_name == 'Google Chrome':
                window_name = window.get('kCGWindowName', '')
                chrome_windows.append(window)
                print(f"  Chrome window: Title '{window_name}' at ({window['kCGWindowBounds']['X']}, {window['kCGWindowBounds']['Y']}), size {window['kCGWindowBounds']['Width']}x{window['kCGWindowBounds']['Height']}")
        if chrome_windows:
            # Use first Chrome window (titles empty common in Quartz for tabs)
            bounds = chrome_windows[0]['kCGWindowBounds']
            print(f"Using first Chrome window via Quartz: at ({bounds['X']}, {bounds['Y']}), size {bounds['Width']}x{bounds['Height']}")
            return {"top": bounds['Y'], "left": bounds['X'], "width": bounds['Width'], "height": bounds['Height']}
        print("Debug: No Chrome windows found via Quartz.")
    except Exception as e:
        print(f"Quartz error: {e}")

    print(f"No Chrome window with '{title_keyword}' found. Using fallback region: {FALLBACK_REGION}")
    return FALLBACK_REGION

def classify_image(image):
    """Classify a single image as 'game' or 'commercial'."""
    image = preprocess(image).unsqueeze(0).to("mps")
    text = tokenizer([
        "live football game with players on the field",  # Active gameplay
        "football coaches and announcers discussing the game",  # Coaches/announcers
        "football game sidelines and crowd in stadium",  # Sidelines and stadium
        "NFL game broadcast with field and players",  # General game broadcast
        "TV commercial advertisement or product ad"  # Commercials
    ]).to("mps")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        probs = (image_features @ text_features.T).softmax(dim=-1)
    # Check if the top match is a game-related text (indices 0-3)
    if probs[0].argmax() in [0, 1, 2, 3]:
        return "game"
    else:
        return "commercial"

def process_frame(frame, history):
    """Process a frame and update voting history."""
    # Resize frame for faster processing
    frame = cv2.resize(frame, (224, 224))
    # Convert BGR (OpenCV/mss) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Classify frame
    result = classify_image(pil_image)
    history.append(result)
    
    # Keep only the last VOTING_WINDOW results
    if len(history) > VOTING_WINDOW:
        history.popleft()
    
    # Decide based on majority voting
    commercial_count = history.count("commercial")
    if commercial_count >= VOTING_THRESHOLD:
        return "commercial"
    return "game"

def main():
    parser = argparse.ArgumentParser(description='Mute AI screen capture processor')
    parser.add_argument('--show', action='store_true', help='Show screen capture view')
    args = parser.parse_args()
    show_screen = args.show
    
    # Dynamically get the Chrome window region
    screen_region = get_chrome_window_region("YouTube TV")  # Broader keyword to match "YouTube TV - Watch & DVR..."
    
    # Initialize screen capture with mss
    with mss() as sct:
        # Initialize voting history
        history = deque(maxlen=VOTING_WINDOW)
        last_state = "game"
        
        try:
            frame_count = 0
            while True:
                # Re-detect window every 10 frames to handle movement/resizing
                if frame_count % 10 == 0:
                    screen_region = get_chrome_window_region("YouTube TV")
                
                # Capture screen region
                screenshot = sct.grab(screen_region)
                # Convert to NumPy array (BGR format)
                frame = np.array(screenshot)
                
                # Debug: Print frame result and vote count
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                result = process_frame(frame, history)
                commercial_count = history.count("commercial")
                print(f"{timestamp} - Frame classified as: {result} (Votes: {commercial_count}/{len(history)})")
                
                # Simulate mute/unmute
                if result == "commercial" and last_state != "commercial":
                    print(f"{timestamp} - Mute TV (Votes: {commercial_count}/{len(history)})")
                    os.system('osascript -e "set volume output muted true"')
                    last_state = "commercial"
                elif result == "game" and last_state != "game":
                    print(f"{timestamp} - Unmute TV (Votes: {commercial_count}/{len(history)})")
                    os.system('osascript -e "set volume output muted false"')
                    last_state = "game"
                
                # Display frame for testing
                if show_screen:
                    cv2.imshow("Screen Feed", frame)
                if cv2.waitKey(FRAME_INTERVAL * 1000) & 0xFF == ord("q"):
                    break
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("Stopped by user.")
        
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()