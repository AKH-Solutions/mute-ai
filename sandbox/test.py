import torch
import open_clip
import cv2
from PIL import Image
import numpy as np
from collections import deque
import datetime
import os
import argparse

# Configuration
FRAME_INTERVAL = 1  # Seconds between frame processing
VOTING_WINDOW = 10  # Number of frames to vote on
VOTING_THRESHOLD = 7  # Minimum "commercial" frames to trigger mute
CAMERA_SOURCE = 1  # USB webcam (confirmed as index 0)

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to("mps")  # Use MPS for M1 Mac Mini
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

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
    parser = argparse.ArgumentParser(description='Mute AI webcam processor')
    parser.add_argument('--show', action='store_true', help='Show webcam view')
    args = parser.parse_args()
    show_webcam = args.show
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_SOURCE}. Try index 1 or check camera connection.")
        return
    
    # Initialize voting history
    history = deque(maxlen=VOTING_WINDOW)
    last_state = "game"  # Track last mute state
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            # Flip frame vertically to correct upside-down orientation
            frame = cv2.flip(frame, 0)
            
            # Process frame
            result = process_frame(frame, history)
            commercial_count = history.count("commercial")
            
            # Debug: Print frame result and vote count
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} - Frame classified as: {result} (Votes: {commercial_count}/{len(history)})")
            
            # Simulate mute/unmute
            if result == "commercial" and last_state != "commercial":
                print(f"{timestamp} - Mute TV (Votes: {commercial_count}/{len(history)})")  # Logging
                os.system('osascript -e "set volume output muted true"')
                last_state = "commercial"
            elif result == "game" and last_state != "game":
                print(f"{timestamp} - Unmute TV (Votes: {commercial_count}/{len(history)})")
                os.system('osascript -e "set volume output muted false"')
                last_state = "game"
            
            # Display frame for testing
            if show_webcam:
                cv2.imshow("Webcam Feed", frame)
            if cv2.waitKey(FRAME_INTERVAL * 1000) & 0xFF == ord("q"):
                break
    
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  # Run live webcam feed only