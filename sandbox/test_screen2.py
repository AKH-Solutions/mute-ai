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
REDETECT_INTERVAL = 3  # Re-detect window every 3 frames

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("mps")  # Use MPS for M1 Mac Mini
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def get_chrome_window_region(title_keyword="YouTube TV"):
    """Dynamically find Chrome window and get its region."""
    if platform.system() != 'Darwin':
        # Fallback for non-macOS
        try:
            chrome_windows = [w for w in gw.getAllWindows() if "Google Chrome" in w.title and title_keyword.lower() in w.title.lower()]
            if chrome_windows:
                window = chrome_windows[0]
                left, top, width, height = window.left, window.top, window.width, window.height
                print(f"Found Chrome window via PyGetWindow: '{window.title}' at ({left}, {top}), size {width}x{height}")
                return {"top": top, "left": left, "width": width, "height": height}
        except Exception as e:
            print(f"PyGetWindow error: {e}")
        return FALLBACK_REGION

    # macOS: Priority 1 - AppleScript using System Events (skip PyGetWindow as it's unreliable on macOS)
    try:
        applescript = '''
tell application "System Events"
    tell process "Google Chrome"
        set windowList to ""
        repeat with w in windows
            set {x, y} to position of w
            set {wid, hgt} to size of w
            set windowList to windowList & (name of w) & "," & x & "," & y & "," & wid & "," & hgt & "\\n"
        end repeat
        return windowList
    end tell
end tell
'''
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) == 5:
                        title, left_str, top_str, width_str, height_str = parts
                        try:
                            left, top, width, height = int(float(left_str)), int(float(top_str)), int(float(width_str)), int(float(height_str))
                            if title_keyword.lower() in title.lower():
                                print(f"Found Chrome window via AppleScript: '{title}' at ({left}, {top}), size {width}x{height}")
                                return {"top": top, "left": left, "width": width, "height": height}
                        except ValueError:
                            continue
            print(f"Debug AppleScript: No matching title '{title_keyword}', available titles: {[line.split(',')[0] for line in lines if line.strip()]}")
        else:
            print(f"AppleScript failed with return code {result.returncode}: {result.stderr}")
    except Exception as e:
        print(f"AppleScript error: {e}")

    # Priority 3: Quartz (use first Chrome window matching title)
    try:
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        print("Debug: Listing Chrome windows (Quartz fallback):")
        matching_windows = []
        for window in windows:
            owner_name = window.get('kCGWindowOwnerName', '')
            if owner_name == 'Google Chrome':
                window_name = window.get('kCGWindowName', '')
                print(f"  Chrome window: Title '{window_name}' at ({window['kCGWindowBounds']['X']}, {window['kCGWindowBounds']['Y']}), size {window['kCGWindowBounds']['Width']}x{window['kCGWindowBounds']['Height']}")
                if title_keyword.lower() in window_name.lower():
                    matching_windows.append(window)
        if matching_windows:
            bounds = matching_windows[0]['kCGWindowBounds']
            print(f"Using first matching Chrome window via Quartz: '{matching_windows[0].get('kCGWindowName', '')}' at ({bounds['X']}, {bounds['Y']}), size {bounds['Width']}x{bounds['Height']}")
            return {"top": bounds['Y'], "left": bounds['X'], "width": bounds['Width'], "height": bounds['Height']}
        print("Debug: No Chrome windows with matching title found via Quartz.")
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
    screen_region = get_chrome_window_region("YouTube TV")
    
    # Initialize screen capture with mss
    with mss() as sct:
        # Initialize voting history
        history = deque(maxlen=VOTING_WINDOW)
        last_state = "game"
        
        try:
            frame_count = 0
            while True:
                # Re-detect window every 3 frames to handle movement/resizing
                if frame_count % REDETECT_INTERVAL == 0:
                    screen_region = get_chrome_window_region("YouTube TV")
                
                # Capture screen region
                screenshot = sct.grab(screen_region)
                # Convert to NumPy array (BGR format)
                frame = np.array(screenshot)
                
                # Debug: Check for black frame (DRM issue)
                if np.mean(frame) < 10:  # Low mean indicates black frame
                    print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - Warning: Black frame detected! (Mean pixel value: {np.mean(frame):.2f})")
                
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