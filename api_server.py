"""
Refactored Mute AI from screen capture to FastAPI server for ESP32-CAM input.

Major changes:
- Removed camera capture, argparse, screen display, and macOS window detection.
- Added FastAPI with POST /process_frame endpoint to receive JPEG images.
- Integrated sqlite3 for persistent voting history.
- Replaced osascript with pyatv for Apple TV muting/unmuting.
- Added GET /health endpoint.
- Model loaded globally on startup.
- Async support for pyatv.

To run: uvicorn api_server:app --host 0.0.0.0 --port 8000
Test: curl -F "image=@test.jpg" http://localhost:8000/process_frame
"""

import asyncio
import sqlite3
import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
from collections import deque
import datetime
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pyatv
import logging

# Configuration
FRAME_INTERVAL = 1  # Not used in server, kept for reference
VOTING_WINDOW = 10
VOTING_THRESHOLD = 7

# Global variables
app = FastAPI()
model = None
preprocess = None
tokenizer = None
atv_config = None  # pyatv config for Apple TV
last_states = {}  # client_id -> last_state

# Database setup
DB_PATH = "mute_ai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            classification TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

class VoteTracker:
    def __init__(self, client_id):
        self.client_id = client_id

    def add_classification(self, classification):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute('INSERT INTO history (client_id, timestamp, classification) VALUES (?, ?, ?)',
                       (self.client_id, timestamp, classification))
        # Keep only last VOTING_WINDOW
        cursor.execute('''
            DELETE FROM history WHERE id IN (
                SELECT id FROM history WHERE client_id = ? ORDER BY timestamp DESC LIMIT -1 OFFSET ?
            )
        ''', (self.client_id, VOTING_WINDOW))
        conn.commit()
        conn.close()

    def get_vote_counts(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT classification FROM history WHERE client_id = ? ORDER BY timestamp DESC LIMIT ?',
                       (self.client_id, VOTING_WINDOW))
        results = cursor.fetchall()
        conn.close()
        classifications = [row[0] for row in results]
        commercial_count = classifications.count("commercial")
        total = len(classifications)
        return commercial_count, total

    def get_decision(self):
        commercial_count, total = self.get_vote_counts()
        if commercial_count >= VOTING_THRESHOLD:
            return "commercial"
        return "game"

def classify_image(image):
    """Classify a single image as 'game' or 'commercial'."""
    image = preprocess(image).unsqueeze(0).to("mps")
    text = tokenizer([
        "live football game with players on the field",
        "football coaches and announcers discussing the game",
        "football game sidelines and crowd in stadium",
        "NFL or College Football game broadcast with field and players",
        "TV commercial advertisement or product ad"
    ]).to("mps")
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        probs = (image_features @ text_features.T).softmax(dim=-1)
    if probs[0].argmax() in [0, 1, 2, 3]:
        return "game"
    else:
        return "commercial"

@app.on_event("startup")
async def startup_event():
    global model, preprocess, tokenizer, atv_config
    # Load OpenCLIP model
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to("mps")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    # Init DB
    init_db()

    # Discover Apple TV
    loop = asyncio.get_event_loop()
    devices = await pyatv.scan(loop=loop, timeout=5)
    if devices:
        atv_config = devices[0]  # Assume first device
        print(f"Found Apple TV: {atv_config.name}")
    else:
        print("No Apple TV found. Muting will not work.")

@app.post("/process_frame")
async def process_frame(image: UploadFile = File(...), client_id: str = Form("default")):
    try:
        # Read image bytes
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Check if image is valid (not black)
        if np.mean(frame) < 10:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        # Process frame
        frame = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        result = classify_image(pil_image)

        # Update tracker
        tracker = VoteTracker(client_id)
        tracker.add_classification(result)
        decision = tracker.get_decision()
        commercial_count, total = tracker.get_vote_counts()

        # Log
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp} - Client {client_id}: {result} -> {decision} (Votes: {commercial_count}/{total})")

        # Mute/unmute
        last_state = last_states.get(client_id, "game")
        muted = False
        if decision == "commercial" and last_state != "commercial":
            await mute_tv()
            muted = True
            print(f"{timestamp} - Mute TV")
        elif decision == "game" and last_state != "game":
            await unmute_tv()
            muted = False
            print(f"{timestamp} - Unmute TV")
        elif decision == "commercial":
            muted = True

        last_states[client_id] = decision

        return {
            "decision": decision,
            "votes": {"commercial": commercial_count, "total": total},
            "muted": muted
        }

    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def mute_tv():
    if atv_config:
        try:
            async with pyatv.connect(atv_config, loop=asyncio.get_event_loop()) as atv:
                await atv.audio.mute()
        except Exception as e:
            print(f"Failed to mute: {e}")

async def unmute_tv():
    if atv_config:
        try:
            async with pyatv.connect(atv_config, loop=asyncio.get_event_loop()) as atv:
                await atv.audio.unmute()
        except Exception as e:
            print(f"Failed to unmute: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}