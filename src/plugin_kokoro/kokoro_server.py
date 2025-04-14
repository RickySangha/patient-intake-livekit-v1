import asyncio
import json
import os
import sys
import uuid
from typing import Dict, Any, List
from collections import deque

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

# Determine the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the 'models' directory within the script's directory
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.fp16-gpu.onnx")
DEFAULT_VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")
DEFAULT_SAMPLE_RATE = int(os.getenv("KOKORO_SAMPLE_RATE", "24000"))

# Global model instance
KOKORO_MODEL = None
ACTIVE_CONNECTIONS = set()
MAX_CONNECTIONS = int(os.getenv("KOKORO_MAX_CONNECTIONS", "10"))
# Maximum concurrent tasks per connection
MAX_CONCURRENT_TASKS = int(os.getenv("KOKORO_MAX_CONCURRENT_TASKS", "5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize model
    global KOKORO_MODEL
    try:
        from kokoro_onnx import Kokoro

        logger.info(f"Initializing Kokoro TTS model from {DEFAULT_MODEL_PATH}...")
        KOKORO_MODEL = Kokoro(DEFAULT_MODEL_PATH, DEFAULT_VOICES_PATH)
        logger.info("Kokoro TTS model initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize Kokoro TTS model: {e}")

    yield

    # Shutdown: cleanup if needed
    pass


# Create FastAPI app with lifespan
app = FastAPI(title="Kokoro TTS Microservice", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    """Request model for TTS API calls."""

    text: str
    voice_id: str = "af_nicole"
    speed: float = 1.0
    language: str = "en-us"
    sample_rate: int = DEFAULT_SAMPLE_RATE


class WebSocketManager:
    """Enhanced WebSocket connection manager that supports multiple concurrent tasks."""

    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        self.connection_locks: Dict[WebSocket, asyncio.Semaphore] = {}

    async def connect(self, websocket: WebSocket) -> bool:
        """Connect and register a new WebSocket connection."""
        # Check connection limit
        if len(self.active_connections) >= MAX_CONNECTIONS:
            logger.warning(f"Connection limit reached: {MAX_CONNECTIONS}")
            return False

        # Accept connection
        await websocket.accept()

        # Register connection with default settings
        self.active_connections[websocket] = {
            "settings": {
                "voice_id": "af_heart",
                "speed": 1.2,
                "language": "en-us",
                "sample_rate": DEFAULT_SAMPLE_RATE,
            },
            "active_tasks": {},  # Changed from active_request to active_tasks (dictionary)
            "task_queue": deque(),  # Queue to maintain order of tasks
        }
        self.connection_locks[websocket] = asyncio.Semaphore(
            1
        )  # Allow only 1 concurrent task
        logger.info(
            f"New connection accepted. Total active: {len(self.active_connections)}"
        )
        return True

    async def disconnect(self, websocket: WebSocket):
        """Disconnect and unregister a WebSocket connection."""
        if websocket in self.active_connections:
            # Cancel all active tasks for this connection
            for task in self.active_connections[websocket]["active_tasks"].values():
                if not task.done():
                    task.cancel()

            del self.active_connections[websocket]
            if websocket in self.connection_locks:
                del self.connection_locks[websocket]
            logger.info(
                f"Connection closed. Total active: {len(self.active_connections)}"
            )

    def get_settings(self, websocket: WebSocket) -> Dict[str, Any]:
        """Get the settings for a connection."""
        if websocket in self.active_connections:
            return self.active_connections[websocket]["settings"]
        return {}

    def update_settings(
        self, websocket: WebSocket, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update the settings for a connection."""
        if websocket in self.active_connections:
            for key, value in settings.items():
                if key in self.active_connections[websocket]["settings"]:
                    self.active_connections[websocket]["settings"][key] = value
            return self.active_connections[websocket]["settings"]
        return {}

    def add_task(self, websocket: WebSocket, request_id: str, task) -> bool:
        """Add a new task for processing a TTS request."""
        if websocket in self.active_connections:
            # Check if we're at the task limit
            active_tasks = self.active_connections[websocket]["active_tasks"]

            # Remove any completed tasks
            completed_tasks = [rid for rid, task in active_tasks.items() if task.done()]
            for rid in completed_tasks:
                del active_tasks[rid]

            # Check if we have room for a new task
            if len(active_tasks) >= MAX_CONCURRENT_TASKS:
                logger.warning(
                    f"Maximum concurrent tasks reached for connection: {MAX_CONCURRENT_TASKS}"
                )
                return False

            # Add the new task
            active_tasks[request_id] = task

            # Add to queue to maintain order
            self.active_connections[websocket]["task_queue"].append(request_id)

            return True
        return False

    def get_task(self, websocket: WebSocket, request_id: str):
        """Get a specific task for a connection."""
        if websocket in self.active_connections:
            return self.active_connections[websocket]["active_tasks"].get(request_id)
        return None

    def remove_task(self, websocket: WebSocket, request_id: str) -> bool:
        """Remove a task when it's completed."""
        if websocket in self.active_connections:
            if request_id in self.active_connections[websocket]["active_tasks"]:
                del self.active_connections[websocket]["active_tasks"][request_id]

                # Also remove from the task queue if present
                task_queue = self.active_connections[websocket]["task_queue"]
                if request_id in task_queue:
                    task_queue.remove(request_id)

                return True
        return False

    def get_all_tasks(self, websocket: WebSocket) -> Dict[str, asyncio.Task]:
        """Get all active tasks for a connection."""
        if websocket in self.active_connections:
            return self.active_connections[websocket]["active_tasks"]
        return {}

    def get_task_order(self, websocket: WebSocket) -> List[str]:
        """Get the order of tasks in the queue."""
        if websocket in self.active_connections:
            return list(self.active_connections[websocket]["task_queue"])
        return []


# Create a WebSocket manager
manager = WebSocketManager()


@app.websocket("/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS audio."""
    # Connect and register this WebSocket
    if not await manager.connect(websocket):
        # Connection limit reached
        await websocket.close(1013, "Maximum connections reached")
        return

    # Check if model is available
    if KOKORO_MODEL is None:
        await websocket.send_text(json.dumps({"error": "TTS model not initialized"}))
        await websocket.close(1011, "TTS model not initialized")
        await manager.disconnect(websocket)
        return

    async def process_text(text: str, request_id: str) -> None:
        """Process text and stream audio chunks to the client."""
        if not text.strip():
            return

        # Acquire the lock to ensure sequential processing
        async with manager.connection_locks[websocket]:
            try:
                # Get settings for this connection
                settings = manager.get_settings(websocket)

                # Send a start event
                await websocket.send_text(
                    json.dumps({"event": "tts_started", "request_id": request_id})
                )

                # Clean the input text
                text = text.strip()
                logger.info(f"Processing text: '{text}'")

                # Create a stream for this text
                stream = KOKORO_MODEL.create_stream(
                    text,
                    voice=settings["voice_id"],
                    speed=settings["speed"],
                    lang=settings["language"],
                )

                # Process the stream
                async for samples, original_sample_rate in stream:
                    # Properly handle the audio data
                    try:
                        import numpy as np
                        from scipy import signal

                        # Ensure samples is a numpy array
                        if not isinstance(samples, np.ndarray):
                            samples = np.array(samples)

                        # Resample if needed
                        target_sample_rate = settings["sample_rate"]
                        if original_sample_rate != target_sample_rate:
                            num_samples = int(
                                len(samples) * target_sample_rate / original_sample_rate
                            )
                            resampled = signal.resample(samples, num_samples)
                            audio_data = resampled
                        else:
                            audio_data = samples

                        # Convert to int16 for proper audio playback
                        if audio_data.dtype != np.int16:
                            if np.issubdtype(audio_data.dtype, np.floating):
                                if np.max(np.abs(audio_data)) <= 1.0:
                                    audio_data = (audio_data * 32767).astype(np.int16)
                                else:
                                    audio_data = np.clip(
                                        audio_data, -32768, 32767
                                    ).astype(np.int16)
                            else:
                                audio_data = np.clip(audio_data, -32768, 32767).astype(
                                    np.int16
                                )
                        else:
                            audio_data = samples
                    except ImportError:
                        logger.warning(
                            "Could not import scipy for resampling, using original audio"
                        )
                        audio_data = samples
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        audio_data = samples

                    # Convert to bytes
                    audio_bytes = audio_data.tobytes()

                    # Break into smaller chunks to ensure smooth streaming
                    chunk_size = 4096  # A reasonable chunk size
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        await websocket.send_bytes(chunk)

                # Send an end event
                await websocket.send_text(
                    json.dumps({"event": "tts_stopped", "request_id": request_id})
                )

            except Exception as e:
                logger.error(f"Error processing TTS request {request_id[:8]}: {e}")
                try:
                    await websocket.send_text(
                        json.dumps({"error": str(e), "request_id": request_id})
                    )
                except Exception:
                    # Ignore errors when sending to a closed websocket
                    pass
            finally:
                # Remove this task from active tasks
                manager.remove_task(websocket, request_id)

    try:
        # Main message loop
        while True:
            try:
                # Receive and parse the message
                message = await websocket.receive()

                # Check if text or bytes
                if "text" in message:
                    try:
                        data = json.loads(message["text"])

                        # Handle text request
                        if "text" in data:
                            text = data["text"]
                            request_id = data.get("request_id", str(uuid.uuid4()))
                            logger.info(
                                f"Processing TTS request: {request_id[:8]}... ({len(text)} chars)"
                            )

                            # Create a new task for this request
                            # Important: We no longer cancel previous tasks!
                            new_task = asyncio.create_task(
                                process_text(text, request_id)
                            )

                            # Add to our task tracking
                            if manager.add_task(websocket, request_id, new_task):
                                logger.info(
                                    f"Added task {request_id[:8]} to queue. Active tasks: {len(manager.get_all_tasks(websocket))}"
                                )
                            else:
                                # Too many concurrent tasks, cancel this one
                                new_task.cancel()
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "error": "Too many concurrent TTS requests",
                                            "request_id": request_id,
                                        }
                                    )
                                )

                        # Handle settings update
                        if "settings" in data and isinstance(data["settings"], dict):
                            updated_settings = manager.update_settings(
                                websocket, data["settings"]
                            )
                            # Confirm settings update
                            await websocket.send_text(
                                json.dumps({"settings_updated": updated_settings})
                            )
                            logger.info(f"Updated TTS settings: {updated_settings}")

                        # Handle interruption
                        if "event" in data and data["event"] == "interrupt":
                            logger.info("Received interruption request")

                            # Cancel all active tasks
                            active_tasks = manager.get_all_tasks(websocket)
                            for task_id, task in active_tasks.items():
                                if not task.done():
                                    task.cancel()

                            # Clear the active tasks tracking
                            for task_id in list(active_tasks.keys()):
                                manager.remove_task(websocket, task_id)

                            # Confirm interruption
                            await websocket.send_text(
                                json.dumps({"event": "interrupted"})
                            )

                    except json.JSONDecodeError:
                        # Invalid JSON
                        await websocket.send_text(
                            json.dumps({"error": "Invalid JSON message"})
                        )

            except WebSocketDisconnect:
                # Handle client disconnect
                logger.info("WebSocket disconnected")
                await manager.disconnect(websocket)
                break

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        await manager.disconnect(websocket)


# Rest of the API endpoints remain unchanged
@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    """REST API endpoint for TTS."""
    if KOKORO_MODEL is None:
        return {"error": "TTS model not initialized"}

    try:
        # Generate audio
        audio = KOKORO_MODEL.tts(
            request.text,
            voice=request.voice_id,
            speed=request.speed,
            lang=request.language,
        )

        # Return audio as base64
        import base64

        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio": audio_b64,
            "sample_rate": request.sample_rate,
            "format": "int16",
        }

    except Exception as e:
        logger.error(f"Error processing TTS request: {e}")
        return {"error": str(e)}


@app.get("/voices")
async def list_voices():
    """List available voices."""
    if KOKORO_MODEL is None:
        return {"error": "TTS model not initialized"}

    try:
        voices = KOKORO_MODEL.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": KOKORO_MODEL is not None}


if __name__ == "__main__":
    uvicorn.run("kokoro_server:app", host="0.0.0.0", port=8000, log_level="info")
