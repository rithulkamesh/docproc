"""Shared utilities for Ollama API integration."""

import io
import logging
import os
import time
import base64
import threading
import queue
from typing import Tuple, Dict, Any, Optional, Callable
import functools
import requests
from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Ollama API configuration
OLLAMA_API_HOST = os.environ.get("OLLAMA_API_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite3.2-vision")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))  # Increased from 15 to 60
OLLAMA_MAX_RETRIES = int(
    os.environ.get("OLLAMA_MAX_RETRIES", "3")
)  # Increased from 2 to 3
OLLAMA_CHECK_INTERVAL = int(os.environ.get("OLLAMA_CHECK_INTERVAL", "60"))  # seconds
OLLAMA_FALLBACK_TEXT = os.environ.get(
    "OLLAMA_FALLBACK_TEXT", "[Unable to process image]"
)

# Cache to prevent repeated server checks
_server_status_cache = {
    "status": None,
    "last_check": 0,
}

# Cache for recently processed images
_image_results_cache = {}

# Queue for async processing
_async_queue = queue.Queue()
_async_results = {}
_async_lock = threading.Lock()
_async_thread = None
_async_running = False


def optimize_image(
    image: Image.Image, quality: int = 80, max_dim: int = 512
) -> Image.Image:
    """Further optimize an image to reduce size and processing time.

    Args:
        image: PIL Image to optimize
        quality: JPEG quality for compression (1-100)
        max_dim: Maximum dimension size

    Returns:
        Optimized PIL Image
    """
    # Resize if needed
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Compress by converting to JPEG and back
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)


def preprocess_image(image) -> Image.Image:
    """Preprocess an image to make it suitable for the model.

    Args:
        image: PIL Image or numpy array

    Returns:
        PIL Image: Processed image ready for model
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            image = image[:, :, :3]

        image = Image.fromarray(image)

    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply minimal preprocessing - normalize size
    w, h = image.size
    max_dim = 768  # Limit max dimension
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Further optimize the image to reduce processing time
    image = optimize_image(image)

    return image


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 encoded string.

    Args:
        image: PIL Image

    Returns:
        str: Base64 encoded image string
    """
    buffered = io.BytesIO()
    image.save(
        buffered, format="JPEG", optimize=True, quality=85
    )  # Changed to JPEG for smaller size
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def check_ollama_server() -> bool:
    """Check if Ollama server is running and the required model is available.
    Uses caching to prevent frequent checks.

    Returns:
        bool: True if server is available and model exists
    """
    # Check cache first
    current_time = time.time()
    if (
        current_time - _server_status_cache["last_check"] < OLLAMA_CHECK_INTERVAL
        and _server_status_cache["status"] is not None
    ):
        return _server_status_cache["status"]

    # If cache expired or empty, check server
    try:
        response = requests.get(f"{OLLAMA_API_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Check for model with or without :latest suffix
            model_available = any(
                (
                    model.get("name", "").startswith(OLLAMA_MODEL)
                    or model.get("name", "").startswith(f"{OLLAMA_MODEL}:")
                )
                for model in models
            )

            # Update cache
            _server_status_cache["status"] = model_available
            _server_status_cache["last_check"] = current_time

            if not model_available:
                logger.warning(f"{OLLAMA_MODEL} model not found in Ollama server")
            else:
                logger.debug(f"Found {OLLAMA_MODEL} in Ollama server")

            return model_available
        else:
            logger.error(f"Error checking Ollama server: {response.status_code}")
            _server_status_cache["status"] = False
            _server_status_cache["last_check"] = current_time
            return False
    except requests.RequestException as e:
        logger.error(f"Failed to connect to Ollama server: {str(e)}")
        _server_status_cache["status"] = False
        _server_status_cache["last_check"] = current_time
        return False


def get_image_hash(image) -> str:
    """Generate a simple hash for an image to use as a cache key.

    Args:
        image: PIL Image or numpy array

    Returns:
        str: Hash string
    """
    if isinstance(image, np.ndarray):
        # For numpy arrays, create a hash from a downsampled version
        small = cv2.resize(image, (16, 16))  # Smaller for faster hashing
        return str(hash(small.tobytes()))
    elif isinstance(image, Image.Image):
        # For PIL images, create a hash from a thumbnail
        img_copy = image.copy()
        img_copy.thumbnail((16, 16))  # Smaller for faster hashing
        return str(hash(img_copy.tobytes()))
    return str(hash(image))


def _start_async_worker():
    """Start the async worker thread if not already running."""
    global _async_thread, _async_running

    if _async_running and _async_thread and _async_thread.is_alive():
        return  # Thread already running

    _async_running = True
    _async_thread = threading.Thread(
        target=_async_worker_loop, daemon=True, name="ollama-async-worker"
    )
    _async_thread.start()
    logger.debug("Started async Ollama worker thread")


def _async_worker_loop():
    """Worker thread that processes async Ollama requests."""
    global _async_running

    while _async_running:
        try:
            # Get task with timeout to check _async_running flag periodically
            try:
                task = _async_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task_id, function, args, kwargs, callback = task

            try:
                result = function(*args, **kwargs)

                # Store result
                with _async_lock:
                    _async_results[task_id] = result

                # Call callback if provided
                if callback:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in callback for task {task_id}: {e}")

            except Exception as e:
                logger.error(f"Error processing async task {task_id}: {e}")
                with _async_lock:
                    _async_results[task_id] = (f"[ERROR: {str(e)}]", 0.0)
            finally:
                _async_queue.task_done()

        except Exception as e:
            logger.error(f"Error in async worker loop: {e}")
            time.sleep(1.0)  # Avoid tight loop on error


def process_image_async(
    image,
    prompt: str = "Describe what you see in this image.",
    callback: Optional[Callable] = None,
    task_id: Optional[str] = None,
) -> str:
    """Process an image asynchronously with Ollama.

    Args:
        image: PIL Image or numpy array
        prompt: Text prompt for the model
        callback: Optional callback function to call when done
        task_id: Optional custom task ID

    Returns:
        str: Task ID for retrieving the result later
    """
    if task_id is None:
        task_id = f"task_{time.time()}_{get_image_hash(image)}"

    # Make sure worker is running
    _start_async_worker()

    # Add task to queue
    _async_queue.put(
        (
            task_id,
            process_image_with_ollama,
            (image, prompt),
            {"timeout": OLLAMA_TIMEOUT * 2},  # Extended timeout for async
            callback,
        )
    )

    return task_id


def get_async_result(task_id: str, remove: bool = True) -> Optional[Tuple[str, float]]:
    """Get result of an async processing task.

    Args:
        task_id: ID of the task
        remove: Whether to remove the result from cache after retrieving

    Returns:
        Optional result tuple or None if not ready
    """
    with _async_lock:
        if task_id in _async_results:
            result = _async_results[task_id]
            if remove:
                del _async_results[task_id]
            return result
    return None


def process_image_with_ollama(
    image,
    prompt: str = "Describe what you see in this image.",
    use_cache: bool = True,
    timeout: int = None,
) -> Tuple[str, float]:
    """Process image with Ollama API with retries and caching.

    Args:
        image: PIL Image or numpy array
        prompt: Text prompt for the model
        use_cache: Whether to use cached results
        timeout: Custom timeout in seconds

    Returns:
        Tuple[str, float]: Generated text and confidence score
    """
    if timeout is None:
        timeout = OLLAMA_TIMEOUT

    # Generate cache key from image and prompt
    if use_cache:
        img_hash = get_image_hash(image)
        cache_key = f"{img_hash}_{hash(prompt)}"

        # Check if we have a cached result
        if cache_key in _image_results_cache:
            return _image_results_cache[cache_key]

    # Check if Ollama server is running
    if not check_ollama_server():
        return (
            f"[ERROR: Ollama server not running or {OLLAMA_MODEL} model not available]",
            0.0,
        )

    # Preprocess image once outside the retry loop
    try:
        processed_image = preprocess_image(image)
        base64_image = image_to_base64(processed_image)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return f"[ERROR: Failed to preprocess image: {str(e)}]", 0.0

    # Retry logic with exponential backoff
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            # Calculate backoff time for retries
            backoff_time = min(2**attempt, 10) if attempt > 0 else 0
            if attempt > 0:
                logger.info(
                    f"Retrying after {backoff_time}s delay (attempt {attempt+1}/{OLLAMA_MAX_RETRIES+1})"
                )
                time.sleep(backoff_time)

            # Prepare the payload for Ollama API
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "images": [base64_image],
                # Additional options to help with performance
                "options": {
                    "num_predict": 256,  # Limit response length
                    "temperature": 0.5,  # Add some temperature to avoid getting stuck
                    "top_p": 0.95,  # Use top_p sampling
                },
            }

            # Call Ollama API with timeout
            logger.debug(f"Sending request to Ollama API (timeout: {timeout}s)")
            response = requests.post(
                f"{OLLAMA_API_HOST}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )

            if response.status_code == 200:
                result = response.json().get("response", "")
                confidence = 0.85  # Basic confidence estimation

                # Cache the result
                if use_cache and result:  # Only cache non-empty results
                    _image_results_cache[cache_key] = (result.strip(), confidence)

                    # Limit cache size
                    if len(_image_results_cache) > 100:
                        # Remove oldest entries
                        for _ in range(10):
                            if _image_results_cache:
                                _image_results_cache.pop(
                                    next(iter(_image_results_cache))
                                )

                return result.strip(), confidence
            else:
                logger.error(
                    f"Ollama API error (attempt {attempt+1}/{OLLAMA_MAX_RETRIES+1}): "
                    f"{response.status_code}, {response.text}"
                )
                if attempt < OLLAMA_MAX_RETRIES:
                    continue
                return (
                    f"[ERROR: Ollama API returned status {response.status_code}]",
                    0.0,
                )

        except requests.Timeout:
            logger.error(
                f"Ollama API timeout (attempt {attempt+1}/{OLLAMA_MAX_RETRIES+1})"
            )
            if attempt < OLLAMA_MAX_RETRIES:
                continue
            return "[ERROR: Ollama API request timed out]", 0.0

        except requests.ConnectionError as e:
            logger.error(
                f"Connection error (attempt {attempt+1}/{OLLAMA_MAX_RETRIES+1}): {str(e)}"
            )
            if attempt < OLLAMA_MAX_RETRIES:
                continue
            return f"[ERROR: Connection error: {str(e)}]", 0.0

        except Exception as e:
            logger.error(f"Error processing image with Ollama: {str(e)}")
            if attempt < OLLAMA_MAX_RETRIES:
                continue
            return f"[ERROR: {str(e)}]", 0.0

    # If we reach here, all retries failed
    return OLLAMA_FALLBACK_TEXT, 0.0


# Initialize the async worker on module import
_start_async_worker()
