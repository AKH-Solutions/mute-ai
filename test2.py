import cv2
from PIL import Image
import numpy as np
from collections import deque
from mlx_vlm import load, generate

# Configuration
FRAME_INTERVAL = 3  # Seconds between frame processing (adjusted for VLM inference time)
VOTING_WINDOW = 5  # Number of frames to vote on
VOTING_THRESHOLD = 3  # Minimum "commercial" frames to trigger mute
CAMERA_SOURCE = 0  # USB webcam (confirmed as index 0)

# Load Phi-3.5-vision model (quantized for M1 efficiency)
model, processor = load("mlx-community/Phi-3.5-vision-instruct-4bit", trust_remote_code=True)

def classify_image(image, processor):
    """Classify a single image as 'game' or 'commercial' using Phi-3.5-vision."""
    # Custom prompt with alternative image tag
    prompt = "<image>Is this a frame from a football game broadcast, which may include sidelines, announcers, replays, or the field, or a TV commercial? Respond with 'game' or 'commercial' only."
    
    # Ensure image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Preprocess image and prompt
    try:
        inputs = processor(prompt, images=[image], return_tensors="np")
    except Exception as e:
        print(f"Processor error: {e}")
        return "error"
    
    # Generate response
    try:
        response = generate(
            model=model,
            processor=processor,
            prompt=prompt,
            inputs=inputs,
            max_tokens=10,
            temp=0.0,
            verbose=False
        )
    except Exception as e:
        print(f"Generate error: {e}")
        return "error"
    
    # Parse response
    result = response.strip().lower()
    return "game" if "game" in result else "commercial"

def process_frame(frame, history, processor):
    """Process a frame and update voting history."""
    # Resize frame for faster processing
    frame = cv2.resize(frame, (224, 224))
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Classify frame
    result = classify_image(pil_image, processor)
    if result == "error":
        print("Skipping frame due to processing error")
        return None
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
    
    # Debug: Print webcam properties
    print(f"Webcam opened at index {CAMERA_SOURCE}")
    print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Initialize voting history
    history = deque(maxlen=VOTING_WINDOW)
    last_state = "game"
    
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
            result = process_frame(frame, history, processor)
            if result is None:
                continue  # Skip frame if processing failed
            
            # Simulate mute/unmute
            if result == "commercial" and last_state != "commercial":
                print("Mute TV")
                last_state = "commercial"
            elif result == "game" and last_state != "game":
                print("Unmute TV")
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