import re
import sys
from typing import Set, Dict, List, Optional, Tuple, Any
import fitz
from PIL import Image
import io
import numpy as np
import logging
import cv2
import os
import queue
import threading
import uuid
import tempfile
import time

from docproc.doc.regions import Region
from docproc.doc.ollama_utils import (
    process_image_with_ollama,
    process_image_async,
    get_async_result,
    optimize_image,
)

logger = logging.getLogger(__name__)

# Constants for async processing
MAX_WAIT_TIME = 30.0  # Maximum time to wait for async result


class ImageProcessingQueue:
    """A queue-based manager for processing images through ML models.

    This class implements a producer-consumer pattern to control the flow of images
    to GPU resources, preventing out-of-memory errors and providing better resource management.
    """

    def __init__(self, max_queue_size: int = 10, num_workers: int = 2):
        """Initialize the image processing queue.

        Args:
            max_queue_size (int): Maximum number of items in the queue
            num_workers (int): Number of worker threads for processing
        """
        self.queue = queue.Queue(max_queue_size)
        self.num_workers = num_workers
        self.results = {}
        self._workers = []
        self._stop_event = threading.Event()
        self._processing_lock = threading.Lock()
        self._is_running = False

    def start(self, process_func):
        """Start the worker threads.

        Args:
            process_func: Function to use for processing queue items
        """
        if self._is_running:
            return

        self._is_running = True
        self._stop_event.clear()

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(process_func,),
                daemon=True,
                name=f"img-proc-worker-{i}",
            )
            self._workers.append(worker)
            worker.start()

        logger.debug(f"Started {self.num_workers} image processing workers")

    def stop(self):
        """Stop all worker threads gracefully."""
        if not self._is_running:
            return

        logger.debug("Stopping image processing queue...")
        self._stop_event.set()

        # Wait for workers to finish
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=2.0)

        self._workers = []
        self._is_running = False

        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

        logger.debug("Image processing queue stopped")

    def add_task(self, task_id: str, image_data: Any, **kwargs) -> bool:
        """Add an image processing task to the queue.

        Args:
            task_id (str): Unique identifier for this task
            image_data (Any): Image data to process
            **kwargs: Additional arguments for the processing function

        Returns:
            bool: True if task was added, False if queue is full
        """
        try:
            self.queue.put((task_id, image_data, kwargs), timeout=0.5)
            return True
        except queue.Full:
            logger.warning(f"Queue is full, couldn't add task {task_id}")
            return False

    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get the result of a processed task.

        Args:
            task_id (str): Task identifier
            timeout (float): How long to wait for the result

        Returns:
            Optional[Any]: Result if available, None otherwise
        """
        start_time = time.time() if timeout else None

        while timeout is None or (time.time() - start_time) < timeout:
            with self._processing_lock:
                if task_id in self.results:
                    result = self.results.pop(task_id)
                    return result

            # Wait a bit before checking again
            time.sleep(0.1)

        return None

    def _worker_loop(self, process_func):
        """Worker thread loop to process queue items.

        Args:
            process_func: Function to process each queue item
        """
        while not self._stop_event.is_set():
            try:
                # Get task with timeout to allow checking stop_event
                task = self.queue.get(timeout=0.5)
                if task is None:
                    self.queue.task_done()
                    continue

                task_id, image_data, kwargs = task

                try:
                    # Check if we're using a file path
                    use_file_path = kwargs.pop("use_file_path", False)

                    if use_file_path:
                        # If it's a file path, open it first
                        if isinstance(image_data, str) and os.path.exists(image_data):
                            from PIL import Image

                            img = Image.open(image_data)
                            # Process the image with the loaded PIL Image
                            result = process_func(img, **kwargs)
                            # Clean up temp file if needed
                            if kwargs.get("cleanup_file", False):
                                try:
                                    os.remove(image_data)
                                except Exception as e:
                                    logger.debug(f"Failed to remove temp file: {e}")
                        else:
                            result = f"[ERROR: Invalid image path: {image_data}]"
                    else:
                        # Process with the provided image data
                        result = process_func(image_data, **kwargs)

                    # Store the result
                    with self._processing_lock:
                        self.results[task_id] = result

                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {str(e)}")
                    with self._processing_lock:
                        self.results[task_id] = f"[ERROR: {str(e)}]"

                finally:
                    self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")


class EquationParser:
    """Takes an equation instance and uses Ollama with vision model to convert into LaTeX format"""

    def __init__(self):
        """Initialize the equation parser."""
        self._cache = {}
        # Create a temp dir for image extraction
        self._temp_dir = tempfile.mkdtemp(prefix="docproc_")
        # Initialize the processing queue
        self._processing_queue = ImageProcessingQueue(max_queue_size=5, num_workers=1)
        self._processing_queue.start(self._process_image_task)
        # Track pending async tasks
        self._pending_tasks = {}
        self._task_lock = threading.Lock()
        logger.debug(
            f"Created temporary directory for image extraction: {self._temp_dir}"
        )

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _process_image_task(self, img, prompt="") -> str:
        """Process image through Ollama's vision model.

        Args:
            img: Image to process (PIL Image or numpy array)
            prompt: Optional prompt for the model

        Returns:
            str: Processed result (LaTeX)
        """
        if not prompt:
            prompt = "This image contains a mathematical equation. Convert it to LaTeX format."

        # Optimize the image before processing
        if isinstance(img, Image.Image):
            img = optimize_image(img, quality=85, max_dim=512)

        # Use sync processing with a reasonable timeout
        result, _ = process_image_with_ollama(img, prompt, timeout=45)
        return self._clean_latex_output(result)

    def _save_image_to_temp(self, image) -> str:
        """Save image to a temporary file for processing.

        Args:
            image: PIL image or numpy array

        Returns:
            str: Path to saved file
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Create temp filename
        file_path = os.path.join(self._temp_dir, f"eq_{uuid.uuid4().hex}.png")

        # Save image
        image.save(file_path)
        return file_path

    def parse_equation(self, region: Region, page: fitz.Page) -> str:
        """Parse an equation region using Ollama's vision model.

        Args:
            region (Region): Region containing the equation image
            page (fitz.Page): PDF page containing the region

        Returns:
            str: LaTeX expression of the equation
        """
        cache_key = f"{page.number}_{region.bbox}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Extract image from PDF
            bbox = region.bbox
            x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            pix = page.get_pixmap(clip=(x1, y1, x2, y2), matrix=fitz.Matrix(1, 1))

            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )

            # Handle very small images - upscale to avoid processing issues
            h, w = img_array.shape[:2]
            min_size = 32
            if h < min_size or w < min_size:
                scale = max(min_size / h, min_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_array = cv2.resize(
                    img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
                )
                logger.info(
                    f"Scaled up small equation image from {h}x{w} to {new_h}x{new_w}"
                )

            # Create task ID
            task_id = f"eq_{page.number}_{x1}_{y1}_{x2}_{y2}"

            # First, check if we have a pending async task for this equation
            with self._task_lock:
                if task_id in self._pending_tasks:
                    async_id = self._pending_tasks[task_id]
                    # Check if async result is ready
                    result = get_async_result(async_id, remove=True)
                    if result:
                        # Got result, clean up and return
                        latex_text, _ = result
                        cleaned_result = self._clean_latex_output(latex_text)
                        self._cache[cache_key] = cleaned_result
                        del self._pending_tasks[task_id]
                        return cleaned_result

            # Try to use the processing queue first
            success = self._processing_queue.add_task(
                task_id=task_id,
                image_data=img_array,
                prompt="This image contains a mathematical equation. Convert it to LaTeX format.",
            )

            if success:
                # Wait for result with a short timeout
                queue_result = self._processing_queue.get_result(task_id, timeout=10.0)
                if queue_result:
                    self._cache[cache_key] = queue_result
                    return queue_result

                # If queue didn't return in time, fall back to async processing
                logger.debug(
                    f"Queue processing timeout for task {task_id}, using async processing"
                )

            # Fall back to async processing
            async_id = process_image_async(
                img_array,
                prompt="This image contains a mathematical equation. Convert it to LaTeX format.",
            )

            # Register pending task
            with self._task_lock:
                self._pending_tasks[task_id] = async_id

            # Try to get result with a reasonable timeout
            start_time = time.time()
            while time.time() - start_time < MAX_WAIT_TIME:
                result = get_async_result(async_id, remove=False)
                if result:
                    latex_text, _ = result
                    cleaned_result = self._clean_latex_output(latex_text)
                    self._cache[cache_key] = cleaned_result

                    # Clean up task reference
                    with self._task_lock:
                        if task_id in self._pending_tasks:
                            del self._pending_tasks[task_id]

                    return cleaned_result
                time.sleep(0.5)

            # If we're here, we couldn't get a result in time
            # Return a placeholder and keep the async task registered
            logger.warning(
                f"Async processing timeout for equation at {page.number}:{x1},{y1},{x2},{y2}"
            )
            return "[Processing equation...]"

        except Exception as e:
            error_msg = f"Error parsing equation: {str(e)}"
            logger.error(error_msg)
            return f"[OCR ERROR: {str(e)}]"

    def get_pending_results(self) -> Dict[str, str]:
        """Check for any pending equation results and update cache.

        Returns:
            Dictionary of region ids and their results
        """
        results = {}
        with self._task_lock:
            pending_tasks = list(self._pending_tasks.items())

        for task_id, async_id in pending_tasks:
            result = get_async_result(async_id)
            if result:
                latex_text, _ = result
                cleaned_result = self._clean_latex_output(latex_text)

                # Extract page and region info from task_id
                # Format: eq_{page_number}_{x1}_{y1}_{x2}_{y2}
                parts = task_id.split("_")
                if len(parts) >= 6:
                    page_num = parts[1]
                    bbox = "_".join(parts[2:])
                    cache_key = f"{page_num}_{bbox}"
                    self._cache[cache_key] = cleaned_result
                    results[task_id] = cleaned_result

                # Clean up task reference
                with self._task_lock:
                    if task_id in self._pending_tasks:
                        del self._pending_tasks[task_id]

        return results

    def cleanup(self):
        """Clean up resources when done."""
        if self._processing_queue:
            self._processing_queue.stop()

        # Clean up temp directory
        try:
            import shutil

            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Removed temporary directory: {self._temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory: {e}")

    def _clean_latex_output(self, text: str) -> str:
        """Clean the model output to extract proper LaTeX.

        Args:
            text (str): Raw model output

        Returns:
            str: Cleaned LaTeX expression
        """
        # Extract content between $ or $$ if present
        dollar_match = re.search(r"\$(.*?)\$", text)
        if dollar_match:
            return dollar_match.group(1)

        double_dollar_match = re.search(r"\$\$(.*?)\$\$", text)
        if double_dollar_match:
            return double_dollar_match.group(1)

        # If the model outputs "The LaTeX representation is..." or similar
        latex_intro_patterns = [
            r"the latex (?:representation|expression|formula) (?:is|would be)[:\s]+(.+)",
            r"in latex[:\s]+(.+)",
            r"latex[:\s]+(.+)",
        ]

        for pattern in latex_intro_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()

        # If none of the patterns match, return the original text
        return text.strip()


class UnicodeMathDetector:
    """Detector for mathematical content based on Unicode character patterns."""

    # Constant definitions
    MATH_RANGES = [
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
        (0x2100, 0x214F),  # Letterlike Symbols
        (0x2150, 0x218F),  # Number Forms
        (0x2190, 0x21FF),  # Arrows
        (0x0391, 0x03C9),  # Greek Letters
        (0x2080, 0x209C),  # Subscripts
        (0x2070, 0x207F),  # Superscripts
        (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    ]

    COMMON_MATH_CHARS = {
        "=",
        "+",
        "-",
        "×",
        "÷",
        "±",
        "∓",
        "∑",
        "∏",
        "∫",
        "∬",
        "∭",
        "∮",
        "∯",
        "∰",
        "∇",
        "∆",
        "√",
        "∛",
        "∜",
        "∝",
        "∞",
        "∟",
        "∠",
        "∡",
        "∢",
        "∥",
        "∦",
        "∧",
        "∨",
        "∩",
        "∪",
        "∈",
        "∉",
        "∊",
        "∋",
        "∌",
        "∍",
        "∎",
        "∏",
        "∐",
        "∑",
        "∓",
        "∔",
        "∕",
        "∖",
        "∗",
        "∘",
        "∙",
        "√",
        "∛",
        "∜",
        "∝",
        "∞",
        "∟",
        "∠",
        "∡",
        "∢",
        "∂",
        "∃",
        "∄",
        "∅",
        "∆",
        "∇",
        "∈",
        "∉",
        "∊",
        "∋",
        "∌",
        "∍",
        "∎",
        "∏",
        "≠",
        "≈",
        "≡",
        "≤",
        "≥",
        "⊂",
        "⊃",
        "⊄",
        "⊅",
        "⊆",
        "⊇",
        "⊈",
        "⊉",
        "⊊",
        "⊋",
        "⊌",
        "⊍",
        "⊎",
        "⊏",
        "⊐",
        "⊑",
        "⊒",
        "⊓",
        "⊔",
        "⊕",
        "⊖",
        "⊗",
        "⊘",
        "⊙",
        "⊚",
        "⊛",
        "⊜",
        "⊝",
        "⊞",
        "⊟",
        "⊠",
        "⊡",
        "⊢",
        "⊣",
        "⊤",
        "⊥",
        "⊦",
        "⊧",
        "⊨",
        "⊩",
        "⊪",
        "⊫",
        "⊬",
        "⊭",
        "⊮",
        "⊯",
        "⊰",
        "⊱",
        "⊲",
        "⊳",
        "⊴",
        "⊵",
        "⊶",
        "⊷",
        "⊸",
        "⊹",
        "⊺",
        "⊻",
        "⊼",
        "⊽",
        "⊾",
        "⊿",
        "⋀",
        "⋁",
        "⋂",
        "⋃",
        "⋄",
        "⋅",
        "⋆",
        "⋇",
        "⋈",
        "⋉",
        "⋊",
        "⋋",
        "⋌",
        "⋍",
        "⋎",
        "⋏",
        "⋐",
        "⋑",
        "⋒",
        "⋓",
        "⋔",
        "⋕",
        "⋖",
        "⋗",
        "⋘",
        "⋙",
        "⋚",
        "⋛",
        "⋜",
        "⋝",
        "⋞",
        "⋟",
        "⋠",
        "⋡",
        "⋢",
        "⋣",
        "⋤",
        "⋥",
        "⋦",
        "⋧",
        "⋨",
        "⋩",
        "⋪",
        "⋫",
        "⋬",
        "⋭",
        "⋮",
        "⋯",
        "⋰",
        "⋱",
        "⋲",
        "⋳",
        "⋴",
        "⋵",
        "⋶",
        "⋷",
        "⋸",
        "⋹",
        "⋺",
        "⋻",
        "⋼",
        "⋽",
        "⋾",
        "⋿",
    }

    FRACTIONS = {
        "½",
        "⅓",
        "⅔",
        "¼",
        "¾",
        "⅕",
        "⅖",
        "⅗",
        "⅘",
        "⅙",
        "⅚",
        "⅛",
        "⅜",
        "⅝",
        "⅞",
        "⅟",
        "↉",
    }

    def __init__(self):
        """Initialize the detector with precomputed character sets and precompiled regex patterns."""
        # Precompute math_chars only once since the constants never change
        self.math_chars = self._get_all_math_chars()
        # Precompile regex patterns
        math_chars_str = "".join(self.math_chars)
        self.math_sequence = re.compile(f"[{re.escape(math_chars_str)}]{{2,}}")
        self.superscript_pattern = re.compile(
            r"[\u2070\u00b9\u00b2\u00b3\u2074-\u207f]+"
        )
        self.subscript_pattern = re.compile(r"[\u2080-\u208e]+")

    def _get_all_math_chars(self) -> Set[str]:
        """Get a set of all mathematical Unicode characters."""
        chars = set()
        # Add characters from Unicode ranges
        for start, end in self.MATH_RANGES:
            for code_point in range(start, end + 1):
                try:
                    chars.add(chr(code_point))
                except ValueError:
                    continue
        # Add common math characters and fractions
        chars.update(self.COMMON_MATH_CHARS)
        chars.update(self.FRACTIONS)
        return chars

    def calculate_math_density(self, text: str) -> float:
        """Calculate the density of mathematical symbols in the text.

        Args:
            text (str): Text to analyze

        Returns:
            float: Ratio of mathematical symbols to total characters
        """
        if not text:
            return 0.0

        math_char_count = sum(1 for char in text if char in self.math_chars)
        return math_char_count / len(text)

    def has_math_pattern(self, text: str) -> bool:
        """Check for common mathematical patterns in the text.

        Looks for:
        - Fraction characters
        - Sequences of math symbols
        - Superscript/subscript numbers

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if mathematical patterns are found
        """
        # Check for fractions
        if any(frac in text for frac in self.FRACTIONS):
            return True
        # Check for sequences of math symbols using precompiled regex
        if self.math_sequence.search(text):
            return True
        # Check for superscript/subscript numbers via regex
        if self.superscript_pattern.search(text) or self.subscript_pattern.search(text):
            return True
        return False
