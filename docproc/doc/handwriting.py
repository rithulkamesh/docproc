import io
import logging
from typing import Dict

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HandwritingDetector:
    def __init__(self, threshold: float = 0.65):
        """
        Initialize the HandwritingDetector.

        Args:
            threshold (float): Score threshold to decide if an image contains handwriting.
        """
        self.threshold = threshold

    def is_handwritten(self, image_array: np.ndarray) -> bool:
        """
        Determine if the provided image likely contains handwriting.

        This method calculates a combination of stroke width variation and contour irregularity,
        which tend to be higher in handwritten content.

        Args:
            image_array (np.ndarray): Image array (RGB or grayscale).

        Returns:
            bool: True if the image is likely handwritten, False otherwise.
        """
        # Convert to grayscale if needed.
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # Calculate gradient magnitude for stroke width variation.
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        valid_mask = grad_magnitude > 0
        if np.count_nonzero(valid_mask) == 0:
            return False

        stroke_variation = np.std(grad_magnitude[valid_mask])

        # Calculate contour irregularity.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        irregularity_total = 0.0
        valid_contours = 0

        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # Irregularity ratio.
            irregularity_total += (len(approx) / perimeter) if perimeter > 0 else 0
            valid_contours += 1

        irregularity_score = (
            irregularity_total / valid_contours if valid_contours > 0 else 0
        )

        # Combine features to produce a handwriting score.
        handwriting_score = 0.7 * stroke_variation + 0.3 * irregularity_score

        return handwriting_score > self.threshold


class PDFHandwritingProcessor:
    def __init__(self, max_workers: int = 4):
        """
        Initialize the PDFHandwritingProcessor.

        Args:
            max_workers (int): Number of threads for concurrent page processing.
        """
        self.detector = HandwritingDetector()
        self.max_workers = max_workers

    def process_pdf(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Process the entire PDF and detect handwriting in images on each page.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            Dict[int, Dict]: Mapping from page numbers to handwriting detection results.
                             Each result includes a flag and a list of detected regions.
        """
        doc = fitz.open(pdf_path)
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_page = {
                executor.submit(self._process_page, doc, page_num): page_num
                for page_num in range(len(doc))
            }
            for future in future_to_page:
                page_num = future_to_page[future]
                try:
                    has_handwriting, regions = future.result()
                    results[page_num] = {
                        "has_handwriting": has_handwriting,
                        "handwriting_regions": regions,
                    }
                except Exception as exc:
                    logger.error(f"Error processing page {page_num}: {exc}")
                    results[page_num] = {
                        "has_handwriting": False,
                        "handwriting_regions": [],
                    }
        return results

    def _process_page(self, doc, page_num: int):
        """
        Process a single page of the PDF to detect handwriting regions.

        Args:
            doc: Opened PDF document.
            page_num (int): Page number to process.

        Returns:
            Tuple[bool, List[Dict]]: A flag indicating if handwriting was found and a list of regions.
        """
        page = doc[page_num]
        image_list = page.get_images(full=True)

        scale = 0.5  # Downscale factor for performance.
        handwriting_found = False
        regions = []

        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image.get("image")
            if not image_bytes:
                continue

            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)

            # Downscale image for quicker processing.
            h, w = image_array.shape[:2]
            small_img = cv2.resize(image_array, (int(w * scale), int(h * scale)))

            if self.detector.is_handwritten(small_img):
                handwriting_found = True
                bbox = page.get_image_bbox(img_info)
                regions.append({"xref": xref, "bbox": bbox})

        return handwriting_found, regions
