import os
os.environ['HF_HOME'] = '/Volumes/Hawkive/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/Volumes/Hawkive/.cache/huggingface/hub'

import cv2
from PIL import Image
import numpy as np
from collections import deque
from mlx_vlm import load, generate
import datetime
import time

# Configuration
FRAME_INTERVAL = 2  # Seconds between frame processing
VOTING_WINDOW = 5  # Number of frames to vote on
VOTING_THRESHOLD = 3  # Minimum "commercial" frames to trigger mute
CAMERA_SOURCE = 0  # USB webcam

# Load Qwen2-VL-7B-Instruct model (quantized for M1 efficiency)
model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-4bit")

def classify_image(image):
    """Classify a single image as 'game' or 'commercial' using Qwen2-VL."""
    # Custom prompt for nuanced classification
    prompt = "Do not repeat the question. Classify this TV broadcast frame as 'game' if it's football-related (field, players, sidelines, stadium, coaches, announcers, replays) or 'commercial' if it's an ad (products, logos, non-game content). Respond only with 'game' or 'commercial'."
    
    # Generate response (mlx-vlm handles image and prompt formatting)
    response = generate(
        model,
        processor,
        prompt,
        image,
        max_tokens=5,  # Short response
        temp=0.0,       # Deterministic output
        verbose=False   # Suppress extra output
    )
    
    # Parse response
    print(f"Model response: '{response.text}'")
    result = response.text.strip().lower()
    return "game" if "game" in result else "commercial"

def process_frame(frame, history):
    """Process a frame and update voting history."""
    # Resize frame for faster processing
    frame = cv2.resize(frame, (224, 224))
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
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
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_SOURCE}. Try index 1 or check camera connection.")
        return
    
    # Wait a bit for camera to initialize
    time.sleep(2)
    
    # Debug: Print webcam properties
    print(f"Webcam opened at index {CAMERA_SOURCE}")
    print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Initialize voting history
    history = deque(maxlen=VOTING_WINDOW)
    last_state = "game"  # Track last mute state
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from webcam (frame {frame_count}). Check if webcam is in use or disconnected.")
                break
            
            print(f"Processing frame {frame_count}")
            frame_count += 1
            
            # Process frame
            result = process_frame(frame, history)
            commercial_count = history.count("commercial")
            
            # Debug: Print frame result and vote count
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} - Frame classified as: {result} (Votes: {commercial_count}/{len(history)})")
            
            # Mute/unmute based on decision
            if result == "commercial" and last_state != "commercial":
                print(f"{timestamp} - Mute TV (Votes: {commercial_count}/{len(history)})")
                os.system('osascript -e "set volume output muted true"')
                last_state = "commercial"
            elif result == "game" and last_state != "game":
                print(f"{timestamp} - Unmute TV (Votes: {commercial_count}/{len(history)})")
                os.system('osascript -e "set volume output muted false"')
                last_state = "game"
            
            # Display frame for testing
            cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(FRAME_INTERVAL * 1000) & 0xFF == ord("q"):
                break
    
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()