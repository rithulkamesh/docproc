import io
import logging
import string
import os
import tempfile
import re
from typing import Dict, List, Tuple, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HandwritingDetector:
    """Detect if an image contains handwritten content."""

    def __init__(self, threshold: float = 0.65):
        """
        Initialize the HandwritingDetector.

        Args:
            threshold (float): Score threshold to decide if an image contains handwriting.
        """
        self.threshold = threshold

    def is_handwritten(self, image_array: np.ndarray) -> bool:
        """
        Determine if the provided image likely contains handwritten content.

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


class MathHandwritingOCR:
    """Specialized OCR for handwritten mathematical content."""

    def __init__(self, debug_mode=False, output_dir=None):
        """
        Initialize the OCR engine for handwritten math content.

        Args:
            debug_mode (bool): Whether to save debug images
            output_dir (str): Directory to save debug images, if None uses current directory
        """
        self.debug_mode = debug_mode
        self.output_dir = output_dir or os.getcwd()

        # Make sure debug directory exists
        if self.debug_mode and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Tesseract configuration for different content types
        self.configs = {
            "normal": "--psm 6 -l eng --oem 1",  # Uniform text block
            "math": "--psm 7 -l eng --oem 1",  # Single line with math focus
            "sparse": "--psm 11 -l eng --oem 1",  # Sparse text with OSD
            "digit": "--psm 10 -l eng --oem 1 -c tessedit_char_whitelist=0123456789+-*/=()[]{}.",  # Digits only
        }

        # Valid characters to keep in OCR output
        self.valid_chars = set(
            string.ascii_letters
            + string.digits
            + "+-*/=()[]{}.,;:!?\"'$%&@#<>^_|\\~` \n\t"
        )

        # Special math symbols to preserve
        self.math_symbols = set("∫∑∏√∂∆πθλμαβγδεζηικνξρστφχψωΓΔΘΛΞΠΣΥΦΨΩ")
        self.valid_chars.update(self.math_symbols)

        # Common OCR substitution patterns for math symbols
        self.math_substitutions = {
            r"([0-9])([a-zA-Z])": r"\1 \2",  # Separate digits from letters
            r"([a-zA-Z])([0-9])": r"\1 \2",  # Separate letters from digits
            r"[\u2018\u2019]": "'",  # Normalize quotes
            r"[\u201C\u201D]": '"',  # Normalize double quotes
            r"[\'\"`]([a-zA-Z])": r"\1",  # Remove quotes around single variables
            r"\\": "/",  # Fix backslashes in fractions
            r"≈": "=",  # Approximate symbol
            r"∼": "~",  # Tilde
            r"([0-9])\s+([0-9])": r"\1\2",  # Join split numbers
            r"([a-z])\s+([a-z])": r"\1\2",  # Join split words
            r"\s*=\s*": " = ",  # Normalize spacing around equals
            r"\s*\+\s*": " + ",  # Normalize spacing around plus
            r"\s*\-\s*": " - ",  # Normalize spacing around minus
            r"\s*\*\s*": " * ",  # Normalize spacing around multiply
            r"\s*\/\s*": " / ",  # Normalize spacing around divide
            r"\(\s+": "(",  # Fix spacing in parentheses
            r"\s+\)": ")",  # Fix spacing in parentheses
            r"\[\s+": "[",  # Fix spacing in brackets
            r"\s+\]": "]",  # Fix spacing in brackets
            r"\{\s+": "{",  # Fix spacing in braces
            r"\s+\}": "}",  # Fix spacing in braces
            r"(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])": r"\1",  # Isolated single letters likely variables
            r"([0-9]+)\.([0-9]+)\.([0-9]+)": r"\1.\2\3",  # Fix double decimals
        }

        self.debug_counter = 0

    def _enhance_image_for_ocr(self, image_array: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple enhanced versions of the image for OCR.

        Args:
            image_array: Original image

        Returns:
            List of enhanced images for OCR attempts
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array.copy()

        # List to store enhanced versions
        enhanced_images = []

        # 1. Basic preprocessing
        # Resize if too small
        h, w = gray.shape[:2]
        min_dim = 800
        if h < min_dim or w < min_dim:
            scale = max(min_dim / h, min_dim / w)
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        # 2. Apply adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_images.append(adaptive)

        # 3. Apply Otsu's thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images.append(otsu)

        # 4. Denoise and sharpen
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        _, sharp_thresh = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        enhanced_images.append(sharp_thresh)

        # 5. Morphological operations to close gaps in text
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        enhanced_images.append(morph)

        # Save debug images if requested
        if self.debug_mode:
            self.debug_counter += 1
            cv2.imwrite(f"{self.output_dir}/debug_{self.debug_counter}_orig.png", gray)
            cv2.imwrite(
                f"{self.output_dir}/debug_{self.debug_counter}_adaptive.png", adaptive
            )
            cv2.imwrite(f"{self.output_dir}/debug_{self.debug_counter}_otsu.png", otsu)
            cv2.imwrite(
                f"{self.output_dir}/debug_{self.debug_counter}_sharp.png", sharp_thresh
            )
            cv2.imwrite(
                f"{self.output_dir}/debug_{self.debug_counter}_morph.png", morph
            )

        return enhanced_images

    def _is_mostly_diagram(self, image_array: np.ndarray) -> bool:
        """
        Determine if an image is mostly a diagram rather than text/equations.

        Args:
            image_array: Image to analyze

        Returns:
            True if the image appears to be a diagram
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

        # Threshold the image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Get image dimensions
        h, w = binary.shape

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False

        # Check if there are many small contours (text) or few large contours (diagram)
        total_area = h * w
        large_contours = 0
        text_like_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area / total_area > 0.05:  # Large contour
                large_contours += 1
            if 0.0001 < area / total_area < 0.01:  # Text-like size
                text_like_contours += 1

        # If we have few large contours and not many text-like contours, it's probably a diagram
        return (large_contours > 3 and text_like_contours < 20) or large_contours > 5

    def _clean_math_text(self, text: str) -> str:
        """
        Clean OCR output for mathematical content.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned mathematical text
        """
        # Remove invalid characters
        cleaned = "".join(c for c in text if c in self.valid_chars)

        # Apply math-specific substitutions
        for pattern, replacement in self.math_substitutions.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        # Remove duplicate spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

        # Handle common equation patterns
        if "=" in cleaned:
            # Try to identify left and right sides of equation
            parts = cleaned.split("=", 1)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                cleaned = f"{left} = {right}"

        return cleaned

    def is_equation(self, text: str) -> bool:
        """
        Determine if the text likely represents an equation.

        Args:
            text: OCR text to analyze

        Returns:
            True if the text appears to be an equation
        """
        # Common equation operators and symbols
        equation_markers = set("+-*/=^()[]{}∫∑∏√∂∆")

        # Quick exit for short texts
        if not text or len(text) < 3:
            return False

        # Calculate density of equation markers
        total_chars = max(len(text), 1)
        equation_marker_count = sum(1 for c in text if c in equation_markers)
        marker_density = equation_marker_count / total_chars

        # Look for specific patterns
        has_equality = "=" in text
        has_fractions = "/" in text and not text.startswith("http")
        has_multiple_operators = sum(1 for c in text if c in "+-*/") > 1

        # Check for digit and operator combinations
        contains_digit_with_operator = False
        for i in range(len(text) - 1):
            if (text[i].isdigit() and i + 1 < len(text) and text[i + 1] in "+-*/=") or (
                text[i] in "+-*/=" and i + 1 < len(text) and text[i + 1].isdigit()
            ):
                contains_digit_with_operator = True
                break

        return (
            marker_density > 0.08
            or (has_equality and has_fractions)
            or (has_equality and has_multiple_operators)
            or contains_digit_with_operator
        )

    def ocr_handwriting(self, image_array: np.ndarray) -> Tuple[str, bool, float]:
        """
        Perform OCR on handwritten content with mathematical awareness.

        Args:
            image_array: Image containing handwritten content

        Returns:
            Tuple[str, bool, float]: (OCR text, is_equation, confidence)
        """
        try:
            # First, check if this is mostly a diagram
            if self._is_mostly_diagram(image_array):
                return "[DIAGRAM OR FIGURE]", False, 0.8

            # Generate enhanced versions of the image
            enhanced_images = self._enhance_image_for_ocr(image_array)

            # Try OCR with different configurations and images
            best_text = ""
            best_confidence = 0.0

            # First try equation-specific OCR if the image looks like it contains equations
            # (uses different preprocessing techniques that work well for equations)
            for img in enhanced_images:
                # Try with math-specific config
                text = pytesseract.image_to_string(img, config=self.configs["math"])
                if text.strip():
                    cleaned = self._clean_math_text(text)
                    if cleaned and self.is_equation(cleaned):
                        # Looks like an equation, use this result
                        return cleaned, True, 0.7

            # Try with standard text config for all images
            all_results = []
            for img in enhanced_images:
                for config_name, config in self.configs.items():
                    try:
                        text = pytesseract.image_to_string(img, config=config)
                        if text.strip():
                            cleaned = self._clean_math_text(text)
                            # Simple heuristic: longer text with more valid chars is better
                            valid_chars = sum(
                                1
                                for c in cleaned
                                if c.isalnum() or c in "+-*/=()[]{}.,;:!?"
                            )
                            confidence = min(
                                0.9, valid_chars / max(len(cleaned), 1) * 0.9
                            )

                            if cleaned:
                                all_results.append((cleaned, confidence))
                    except Exception as e:
                        logger.debug(f"OCR error with {config_name}: {e}")

            # Sort by confidence
            all_results.sort(key=lambda x: x[1], reverse=True)

            if all_results:
                best_text, best_confidence = all_results[0]

            # Determine if this is an equation
            is_equation = self.is_equation(best_text)

            # If the result seems poor quality or empty, return appropriate message
            if not best_text or best_confidence < 0.2:
                return "[UNRECOGNIZED HANDWRITTEN CONTENT]", is_equation, 0.1

            return best_text, is_equation, best_confidence

        except Exception as e:
            logger.error(f"Handwriting OCR error: {e}")
            return f"[OCR ERROR: {str(e)}]", False, 0.0


class PDFHandwritingProcessor:
    """Process PDF documents to detect and OCR handwritten content."""

    def __init__(
        self,
        max_workers: int = 2,
        debug_mode: bool = False,
        debug_dir: Optional[str] = None,
    ):
        """
        Initialize the PDFHandwritingProcessor.

        Args:
            max_workers (int): Number of threads for concurrent processing
            debug_mode (bool): Whether to save debug images
            debug_dir (str): Directory to save debug images
        """
        self.detector = HandwritingDetector()
        self.max_workers = max_workers
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir

        # Create debug directory if needed
        if debug_mode and debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        # Initialize OCR engine
        self.ocr_engine = MathHandwritingOCR(
            debug_mode=debug_mode, output_dir=debug_dir
        )

    def process_pdf(self, pdf_path: str) -> Dict[int, Dict]:
        """
        Process the PDF document to detect and OCR handwritten content.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict mapping page numbers to handwriting detection results
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
            # First pass: Quick scan to find pages with potential handwriting
            handwriting_pages = []
            for page_num in range(len(doc)):
                try:
                    if self._quick_check_page(doc[page_num]):
                        handwriting_pages.append(page_num)
                except Exception as e:
                    logger.error(f"Error in quick scan of page {page_num}: {e}")

            logger.info(
                f"Found {len(handwriting_pages)} pages with potential handwriting"
            )

            # Second pass: Detailed processing of pages with handwriting
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_page = {
                    executor.submit(self._process_page, doc, page_num): page_num
                    for page_num in handwriting_pages
                }

                for future in future_to_page:
                    page_num = future_to_page[future]
                    try:
                        has_handwriting, regions = future.result()
                        results[page_num] = {
                            "has_handwriting": has_handwriting,
                            "handwriting_regions": regions,
                        }
                        logger.info(
                            f"Processed page {page_num}: found {len(regions)} handwriting regions"
                        )
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")
                        results[page_num] = {
                            "has_handwriting": False,
                            "handwriting_regions": [],
                        }

            # Add empty results for pages without handwriting
            for page_num in range(len(doc)):
                if page_num not in results:
                    results[page_num] = {
                        "has_handwriting": False,
                        "handwriting_regions": [],
                    }

            return results

        finally:
            # Close the document
            doc.close()

    def _quick_check_page(self, page) -> bool:
        """
        Quickly check if a page might contain handwriting.

        Args:
            page: A page from the PDF document

        Returns:
            True if the page might contain handwriting
        """
        # Check for images
        try:
            image_list = page.get_images(full=False)
            return len(image_list) > 0
        except Exception:
            return False

    def _process_page(self, doc, page_num: int) -> Tuple[bool, List[Dict]]:
        """
        Process a single page to find handwritten content.

        Args:
            doc: The PDF document
            page_num: Page number to process

        Returns:
            Tuple of (has_handwriting, list of handwriting regions)
        """
        logger.debug(f"Processing page {page_num}")
        page = doc[page_num]
        image_list = page.get_images(full=True)

        handwriting_found = False
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

                # Create a smaller version for detection
                h, w = image_array.shape[:2]
                small_img = cv2.resize(image_array, (int(w * 0.5), int(h * 0.5)))

                # Check if it contains handwriting
                if self.detector.is_handwritten(small_img):
                    handwriting_found = True
                    bbox = page.get_image_bbox(img_info)

                    # Save debug image if requested
                    if self.debug_mode and self.debug_dir:
                        try:
                            debug_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(
                                f"{self.debug_dir}/hw_page{page_num}_img{xref}.png",
                                debug_img,
                            )
                        except Exception as e:
                            logger.debug(f"Failed to save debug image: {e}")

                    # OCR the handwriting
                    ocr_text, is_equation, confidence = self.ocr_engine.ocr_handwriting(
                        image_array
                    )

                    # Create metadata
                    metadata = {
                        "xref": xref,
                        "possible_equation": is_equation,
                        "confidence": confidence,
                        "page_num": page_num,
                    }

                    # Add region
                    regions.append(
                        {
                            "xref": xref,
                            "bbox": bbox,
                            "ocr_text": ocr_text,
                            "metadata": metadata,
                        }
                    )

            except Exception as e:
                logger.warning(f"Error processing image {xref} on page {page_num}: {e}")

        return handwriting_found, regions
