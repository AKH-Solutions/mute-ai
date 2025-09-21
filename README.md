# Mute AI FastAPI Server

This is a refactored version of the Mute AI script, now running as a FastAPI web server. It receives JPEG images from an ESP32-CAM device, classifies them as "game" or "commercial" during a football broadcast, maintains voting history, and controls Apple TV volume muting/unmuting using pyatv.

## Setup

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Requirements

Install the required packages using the provided requirements.txt:

```bash
pip install -r requirements.txt
```

Note: pyatv requires pairing with your Apple TV beforehand. Follow the [pyatv documentation](https://pyatv.dev/) for pairing instructions.

## Running the Server

Start the server with:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The server will load the OpenCLIP model on startup and discover the Apple TV.

## Endpoints

### POST /process_frame

Receives a JPEG image and processes it for classification and voting.

- **Input**: Multipart form data with:
  - `image`: Required, JPEG file bytes.
  - `client_id`: Optional, string for multi-device support (default: "default").

- **Output**: JSON with:
  - `decision`: "game" or "commercial"
  - `votes`: {"commercial": count, "total": VOTING_WINDOW}
  - `muted`: true/false

Example test with curl:

```bash
curl -F "image=@test.jpg" http://localhost:8000/process_frame
```

### GET /health

Simple health check endpoint.

- **Output**: {"status": "ok"}

## Testing with REST Client

For easier testing in VS Code, use the REST Client extension.

1. Install the "REST Client" extension by Huachao Mao in VS Code.
2. Open the `api_tests/test_calls.http` file.
3. Click "Send Request" above each request block to test the endpoints.

The file includes tests for health check and processing frames with the images in `test_images/`.

## Configuration

- `VOTING_WINDOW`: 10 (frames to vote on)
- `VOTING_THRESHOLD`: 7 (min commercial votes to mute)
- Database: `mute_ai.db` for persistent history

## Changes from Original

- Converted to FastAPI server instead of screen capture loop.
- Input via HTTP POST instead of camera/screen capture.
- Used sqlite3 for persistent voting history.
- Replaced osascript with pyatv for Apple TV control.
- Added async support and error handling.
- Removed UI and macOS-specific code.