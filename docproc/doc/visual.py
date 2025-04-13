import io
import logging
import os
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoProcessor, AutoModelForVision2Seq

from docproc.doc.regions import Region, RegionType, BoundingBox
from docproc.doc.equations import (
    EquationParser,
)  # Import EquationParser to reuse GPU check

logger = logging.getLogger(__name__)


class ContentDetector:
    """Base class for detecting visual content in documents."""

    def __init__(self, threshold: float = 0.65):
        """
        Initialize the ContentDetector.

        Args:
            threshold (float): Score threshold for content detection confidence.
        """
        self.threshold = threshold

    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better detection.

        Args:
            image_array (np.ndarray): Image array (RGB or grayscale).

        Returns:
            np.ndarray: Preprocessed image.
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

        return denoised

    def is_diagram(self, image_array: np.ndarray) -> bool:
        """
        Detect if an image is a diagram rather than text.

        Args:
            image_array (np.ndarray): Image array.

        Returns:
            bool: True if the image is likely a diagram, False otherwise.
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # Threshold the image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        # Calculate image properties
        h, w = binary.shape
        total_area = h * w

        # Check contour properties
        large_contours = 0
        text_like_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area / total_area > 0.05:  # Large contour
                large_contours += 1
            if 0.0001 < area / total_area < 0.01:  # Text-like size
                text_like_contours += 1

        # Determine if it's a diagram
        return (large_contours > 3 and text_like_contours < 20) or large_contours > 5


class VisualContentProcessor:
    """Process PDF documents to detect visual content like equations and diagrams."""

    # Class variable for singleton instance
    _instance = None

    @classmethod
    def get_instance(cls, max_workers: int = 2):
        """Get or create the singleton instance of VisualContentProcessor."""
        if cls._instance is None:
            cls._instance = cls(max_workers)
        return cls._instance

    def __init__(self, max_workers: int = 2):
        """
        Initialize the VisualContentProcessor.

        Args:
            max_workers (int): Number of threads for concurrent processing
        """
        self.detector = ContentDetector()
        self.max_workers = max_workers
        self.granite_model = None
        self.granite_processor = None
        self._device = None
        self._loading_in_progress = False

    def _check_gpu_safely(self):
        """Safely check if GPU is available without crashing.
        Implements direct GPU detection instead of relying on EquationParser."""
        try:
            import torch
            import os

            # Check for environment variable to force CPU usage
            if os.environ.get("DOCPROC_FORCE_CPU", "").lower() in ("1", "true", "yes"):
                logger.info("GPU usage disabled by environment variable")
                return False

            # First, check if CUDA is available according to PyTorch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                logger.info("CUDA reported as not available by PyTorch")
                return False

            # Get device count and name
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Found GPU: {device_name}")

                # Try to initialize tensor on GPU to verify it works
                try:
                    test_tensor = torch.tensor([1.0], device="cuda:0")
                    del test_tensor
                    return True
                except Exception as e:
                    logger.warning(f"CUDA device found but failed initialization: {e}")
                    return False
            else:
                logger.info("No CUDA devices found by PyTorch")
                return False

        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False

    def _lazy_load_model(self):
        """Lazy load the Granite model when first needed."""
        if self.granite_model is None and not self._loading_in_progress:
            try:
                self._loading_in_progress = True
                logger.info(
                    "Loading IBM Granite 3.2 model for visual content recognition..."
                )

                # Import torch only when needed
                import torch

                # Safely check for GPU availability
                use_gpu = self._check_gpu_safely()

                if use_gpu:
                    self._device = "cuda:0"  # Use explicit cuda:0 device
                    # Set CUDA device properties to optimize performance
                    torch.backends.cudnn.benchmark = True
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

                    # Set memory efficient settings
                    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                            "expandable_segments:True,max_split_size_mb:128"
                        )
                        logger.info(
                            "Set PYTORCH_CUDA_ALLOC_CONF for better memory management"
                        )
                else:
                    self._device = "cpu"
                    logger.info(
                        "Using CPU for visual content processing (no GPU available or disabled)"
                    )

                model_name = "ibm-granite/granite-vision-3.2-2b"

                # Use fast processor explicitly
                self.granite_processor = AutoProcessor.from_pretrained(
                    model_name, use_fast=True
                )

                # Load model to GPU with optimized settings
                if self._device == "cuda:0":
                    try:
                        self.granite_model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            device_map="cuda:0",  # Use explicit device instead of "auto"
                            torch_dtype=torch.float16,  # Use half precision to save GPU memory
                            trust_remote_code=True,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load model on GPU, falling back to CPU: {e}"
                        )
                        self._device = "cpu"
                        self.granite_model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                        )
                else:
                    self.granite_model = AutoModelForVision2Seq.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )

                # Ensure model is in evaluation mode
                self.granite_model.eval()

                # Clear CUDA cache to free up memory
                if self._device == "cuda:0":
                    torch.cuda.empty_cache()

                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self.granite_model = None
                self.granite_processor = None
                raise
            finally:
                self._loading_in_progress = False

    def recognize_content(
        self, image_array: np.ndarray, prompt: str = ""
    ) -> Tuple[str, float]:
        """
        Recognize content in an image using IBM Granite 3.2 vision model.

        Args:
            image_array (np.ndarray): Image as numpy array
            prompt (str): Optional prompt to guide the model

        Returns:
            Tuple[str, float]: Recognized text and confidence score
        """
        self._lazy_load_model()

        if self.granite_model is None or self.granite_processor is None:
            return "Error: Model not loaded", 0.0

        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image_array, np.ndarray):
                if len(image_array.shape) == 2:
                    # Convert grayscale to RGB
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:
                    # Convert RGBA to RGB
                    image_array = image_array[:, :, :3]

                # Resize large images to reduce memory usage
                h, w = image_array.shape[:2]
                max_dim = 1024  # Set maximum dimension to limit memory usage
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image_array = cv2.resize(image_array, (new_w, new_h))
                    logger.info(
                        f"Resized image from {h}x{w} to {new_h}x{new_w} to save memory"
                    )

                image = Image.fromarray(image_array)
            else:
                image = image_array

            # Construct the prompt based on the analysis needed
            if not prompt:
                prompt = "Describe what you see in this image."

            import torch

            with torch.no_grad():  # Reduce memory usage during inference
                # Process on CPU first then move to proper device
                inputs = self.granite_processor(
                    text=prompt, images=image, return_tensors="pt"
                )

                # Move all tensor inputs to the same device
                inputs = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # Generate with beam search for better quality but limit tokens for memory
                generated_ids = self.granite_model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced from 512 to save memory
                    num_beams=2,  # Reduced from 4 to save memory
                    length_penalty=1.0,
                )

                # Always move to CPU for processing
                generated_ids = generated_ids.cpu()
                generated_text = self.granite_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            # Basic confidence estimation
            confidence = 0.85

            # Periodically clear CUDA cache to prevent memory buildup
            if self._device == "cuda:0":
                torch.cuda.empty_cache()

            return generated_text.strip(), confidence

        except Exception as e:
            logger.error(f"Error recognizing content: {e}")
            return f"Error: {str(e)}", 0.0

    def recognize_equation(self, image_array: np.ndarray) -> Tuple[str, float]:
        """
        Recognize mathematical equations in an image using IBM Granite 3.2.

        Args:
            image_array (np.ndarray): Image containing equation

        Returns:
            Tuple[str, float]: LaTeX representation and confidence score
        """
        prompt = "This image contains a mathematical equation. Transcribe it into LaTeX format."
        return self.recognize_content(image_array, prompt)

    def process_pdf(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Process the PDF document to detect and recognize visual content.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict: Mapping of page numbers to visual content detection results
        """
        # Open the PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            return {}

        logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")
        results = {}

        try:
            # First pass: Quick scan to find pages with potential visual content
            visual_pages = []
            for page_num in range(len(doc)):
                try:
                    if self._quick_check_page(doc[page_num]):
                        visual_pages.append(page_num)
                except Exception as e:
                    logger.error(f"Error in quick scan of page {page_num}: {e}")

            logger.info(
                f"Found {len(visual_pages)} pages with potential visual content"
            )

            # Process pages with visual content
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_page = {
                    executor.submit(self._process_page, doc, page_num): page_num
                    for page_num in visual_pages
                }

                for future in future_to_page:
                    page_num = future_to_page[future]
                    try:
                        has_visual_content, regions = future.result()
                        results[page_num] = {
                            "has_visual_content": has_visual_content,
                            "visual_regions": regions,
                        }
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")
                        results[page_num] = {
                            "has_visual_content": False,
                            "visual_regions": [],
                        }

            # Add empty results for pages without visual content
            for page_num in range(len(doc)):
                if page_num not in results:
                    results[page_num] = {
                        "has_visual_content": False,
                        "visual_regions": [],
                    }

            return results

        finally:
            # Close the document
            doc.close()

    def _quick_check_page(self, page) -> bool:
        """
        Check if a page might contain visual content.

        Args:
            page: A page from the PDF document

        Returns:
            bool: True if the page might contain visual content
        """
        # Check for images
        try:
            image_list = page.get_images(full=False)
            return len(image_list) > 0
        except Exception:
            return False

    def _process_page(self, doc, page_num: int) -> Tuple[bool, List[Dict]]:
        """
        Process a page to identify visual content.

        Args:
            doc: The PDF document
            page_num: Page number to process

        Returns:
            Tuple[bool, List[Dict]]: (has_visual_content, list of visual regions)
        """
        page = doc[page_num]
        image_list = page.get_images(full=True)

        visual_content_found = False
        regions = []

        for img_info in image_list:
            xref = img_info[0]

            try:
                # Extract image
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue

                # Convert to numpy array
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)

                # Skip very small images
                if image_array.shape[0] < 50 or image_array.shape[1] < 50:
                    continue

                # Get bbox
                bbox = page.get_image_bbox(img_info)

                # Check if it's a diagram
                is_diagram = self.detector.is_diagram(image_array)

                # Determine region type based on content
                if is_diagram:
                    region_type = RegionType.FIGURE
                    # Generate diagram description
                    content, confidence = self.recognize_content(
                        image_array, "Describe this diagram or figure in detail."
                    )
                else:
                    # Check if it might be an equation
                    # For potential equations, we'll use equation recognition
                    content, confidence = self.recognize_equation(image_array)

                    # Simple heuristic to detect if it's an equation
                    if "$" in content or "\\" in content:
                        region_type = RegionType.EQUATION
                    else:
                        region_type = RegionType.IMAGE

                visual_content_found = True

                # Create metadata
                metadata = {
                    "xref": xref,
                    "confidence": confidence,
                    "page_num": page_num,
                }

                # Add region
                regions.append(
                    {
                        "xref": xref,
                        "bbox": bbox,
                        "content": content,
                        "region_type": region_type,
                        "metadata": metadata,
                    }
                )

            except Exception as e:
                logger.warning(f"Error processing image {xref} on page {page_num}: {e}")

        return visual_content_found, regions

    def extract_regions_from_image(self, image_path: str) -> List[Region]:
        """
        Extract regions from a standalone image file.

        Args:
            image_path (str): Path to image file

        Returns:
            List[Region]: Detected regions in the image
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Check if it's a diagram
            is_diagram = self.detector.is_diagram(image_rgb)

            # Determine region type and extract content
            if is_diagram:
                content, confidence = self.recognize_content(
                    image_rgb, "Describe this diagram or figure in detail."
                )
                region_type = RegionType.FIGURE
            else:
                content, confidence = self.recognize_equation(image_rgb)
                if "$" in content or "\\" in content:
                    region_type = RegionType.EQUATION
                else:
                    region_type = RegionType.IMAGE

            # Create region
            h, w = image.shape[:2]
            region = Region(
                region_type=region_type,
                bbox=BoundingBox(x1=0, y1=0, x2=w, y2=h),
                confidence=confidence,
                content=content,
                metadata={"source_file": image_path},
            )

            return [region]

        except Exception as e:
            logger.error(f"Error processing image file {image_path}: {e}")
            return []
