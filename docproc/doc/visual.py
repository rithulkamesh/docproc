import io
import logging
import os
from typing import Dict, List, Tuple, Optional, Iterator
import re
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from docproc.doc.regions import Region, RegionType, BoundingBox
from docproc.doc.ollama_utils import process_image_with_ollama
from docproc.writer import FileWriter, JSONWriter, CSVWriter, SQLiteWriter

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
        self._result_cache = {}

    def recognize_content(
        self, image_array: np.ndarray, prompt: str = ""
    ) -> Tuple[str, float]:
        """
        Recognize content in an image using Ollama with granite3.2-vision model.

        Args:
            image_array (np.ndarray): Image as numpy array
            prompt (str): Optional prompt to guide the model

        Returns:
            Tuple[str, float]: Recognized text and confidence score
        """
        if not prompt:
            prompt = "Describe what you see in this image."

        return process_image_with_ollama(image_array, prompt)

    def recognize_equation(self, image_array: np.ndarray) -> Tuple[str, float]:
        """
        Recognize mathematical equations in an image using Ollama.

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
        # Check if we have cached results for this PDF
        if pdf_path in self._result_cache:
            return self._result_cache[pdf_path]

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

            # Cache results
            self._result_cache[pdf_path] = results

            # Limit cache size
            if len(self._result_cache) > 10:
                # Remove oldest entry
                self._result_cache.pop(next(iter(self._result_cache)))

            return results

        finally:
            # Close the document
            doc.close()

    def process_pdf_with_writer(
        self, pdf_path: str, writer: FileWriter, include_metadata: bool = True
    ) -> None:
        """
        Process the PDF document and write results incrementally to a file writer.

        Args:
            pdf_path (str): Path to the PDF file
            writer: FileWriter instance for output
            include_metadata (bool): Whether to include document metadata

        This method processes each page and writes results immediately to save memory.
        """
        # Initialize the writer
        writer.init_tables()

        # Open the PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            return

        logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")

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

            # Process each page with visual content and write results immediately
            for page_num in visual_pages:
                try:
                    has_visual_content, regions = self._process_page(doc, page_num)
                    if has_visual_content and regions:
                        # Convert regions to a format suitable for writing
                        records = self._regions_to_records(regions, pdf_path, page_num)
                        writer.write_data(iter(records))
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")

        finally:
            # Close the document
            doc.close()

    def _regions_to_records(
        self, regions: List[Dict], pdf_path: str, page_num: int
    ) -> List[Dict]:
        """
        Convert visual regions to records for writing.

        Args:
            regions: List of region dictionaries
            pdf_path: Path to the source PDF
            page_num: Page number

        Returns:
            List of record dictionaries ready for writing
        """
        records = []
        for i, region in enumerate(regions):
            record = {
                "document_path": pdf_path,
                "page_num": page_num,
                "region_id": f"{page_num}_{i}",
                "region_type": str(region["region_type"]),
                "content": region["content"],
                "bbox_x1": region["bbox"][0],
                "bbox_y1": region["bbox"][1],
                "bbox_x2": region["bbox"][2],
                "bbox_y2": region["bbox"][3],
                "confidence": region.get("metadata", {}).get("confidence", 0.0),
                "record_type": "visual_content",
            }
            records.append(record)
        return records

    def export_results(
        self, results: Dict[int, Dict], output_path: str, format_type: str = "json"
    ) -> None:
        """
        Export processing results to a file.

        Args:
            results: Dictionary of processing results
            output_path: Path to write output
            format_type: Format type ("json", "csv", or "sqlite")
        """
        # Create appropriate writer based on format_type
        if format_type.lower() == "json":
            writer = JSONWriter(output_path)
        elif format_type.lower() == "csv":
            writer = CSVWriter(output_path)
        elif format_type.lower() == "sqlite":
            writer = SQLiteWriter(output_path, "visual_content")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        try:
            writer.init_tables()

            # Convert results to records
            records = []
            for page_num, page_data in results.items():
                if page_data["has_visual_content"]:
                    for i, region in enumerate(page_data["visual_regions"]):
                        record = {
                            "page_num": page_num,
                            "region_id": f"{page_num}_{i}",
                            "region_type": str(region["region_type"]),
                            "content": region["content"],
                            "bbox_x1": region["bbox"][0],
                            "bbox_y1": region["bbox"][1],
                            "bbox_x2": region["bbox"][2],
                            "bbox_y2": region["bbox"][3],
                            "confidence": region.get("metadata", {}).get(
                                "confidence", 0.0
                            ),
                            "record_type": "visual_content",
                        }
                        records.append(record)

            # Write all records
            writer.write_data(iter(records))
        finally:
            writer.close()

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
