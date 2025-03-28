import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Iterator, Dict
import io
from PIL import Image
import numpy as np
import pytesseract  # Add this import
import cv2  # Add this import for image preprocessing

from docproc.doc.equations import EquationParser, UnicodeMathDetector
from docproc.doc.regions import BoundingBox, Region, RegionType
from docproc.writer import FileWriter
from docproc.doc.handwriting import PDFHandwritingProcessor

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Process and extract structured information from document files.

    This class handles document parsing, region detection, content extraction, and export
    functionality for supported document formats (currently PDF). It provides methods to:
    - Load and parse document files
    - Detect and classify different types of regions (text, images, etc.)
    - Extract content from detected regions
    - Export processed results to various formats

    Args:
        filepath (str): Path to the input document file
        writer (type[FileWriter]): FileWriter class to use for exporting results

    Attributes:
        filepath (Path): Pathlib Path object for the input file
        file: File handle for the input document
        writer_class (type[FileWriter]): Class used for writing output
        regions (List[Region]): List of detected regions in the document

    Example:
        ```python
        with DocumentAnalyzer("document.pdf", CSVWriter) as analyzer:
            regions = analyzer.detect_regions()
            analyzer.export_regions()
        ```
    """

    def __init__(
        self,
        filepath: str,
        writer: type[FileWriter],
        output_path: str,
        region_types: Optional[List[RegionType]] = None,
        exclude_fields: Optional[List[str]] = None,
        enable_handwriting_detection: bool = False,
        convert_handwriting_to_latex: bool = False,
    ):
        """Initialize DocumentAnalyzer with input file and writer.

        Args:
            filepath (str): Path to the input document file
            writer (type[FileWriter]): FileWriter class to use for exporting results
            output_path (str): Path where output should be written
            region_types (Optional[List[RegionType]]): List of region types to scan for.
                                                      If None, scans for all types.
                                                      Example: [RegionType.TEXT, RegionType.EQUATION]
            exclude_fields (Optional[List[str]]): Fields to exclude from output
            enable_handwriting_detection (bool): Whether to enable handwriting detection
            convert_handwriting_to_latex (bool): Whether to convert handwriting to LaTeX
        """
        self.filepath = Path(filepath)
        self.file = open(filepath, "rb")
        self.writer_class = writer
        self.output_path = output_path
        self.region_types = region_types or list(RegionType)
        self.doc = None
        self._load_document()
        self.eqparser = EquationParser()
        self.detector = UnicodeMathDetector()
        self.exclude_fields = exclude_fields

        # Handwriting detection configuration
        self.enable_handwriting_detection = enable_handwriting_detection
        self.convert_handwriting_to_latex = convert_handwriting_to_latex

        if enable_handwriting_detection and RegionType.HANDWRITING in self.region_types:
            self.handwriting_processor = PDFHandwritingProcessor(max_workers=2)
            # Process handwriting early to avoid file handle issues
            self.handwriting_results = self._process_handwriting()
        else:
            self.handwriting_results = {}

        # Add OCR config
        self.ocr_config = "--psm 6"  # Assume a single block of text

    def _debug_image_extraction(self):
        """Debug method to check image extraction."""
        logger.info(f"Found {len(self.raw_images)} raw images in document")
        for i, (xref, bbox, page_num) in enumerate(self.raw_images[:5]):  # Show first 5
            logger.info(f"  Image {i}: xref={xref}, page={page_num}, bbox={bbox}")

    def _load_pdf(self) -> None:
        """Load and process a PDF document.

        Extracts text blocks and images from each page of the PDF document
        without classifying them into specific region types.
        """
        self.doc = fitz.open(self.file)
        self.raw_blocks = []
        self.raw_images = []

        try:
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]

                # Extract text blocks
                text_blocks = page.get_text("blocks")
                self.raw_blocks.extend((block, page_num) for block in text_blocks)

                # Extract images
                try:
                    page.get_pixmap()
                    images = page.get_images(full=True)
                    for img in images:
                        if img:
                            xref = img[0]
                            try:
                                bbox = page.get_image_bbox(img)
                                if bbox:
                                    self.raw_images.append((xref, bbox, page_num))
                            except Exception as e:
                                logger.warning(f"Failed to process image: {e}")
                except Exception as e:
                    logger.warning(
                        f"Failed to extract images from page {page_num}: {e}"
                    )

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def _load_document(self) -> None:
        """Load and process the input document based on its file type."""
        ext = self.filepath.suffix.lower()
        if ext == ".pdf":
            self._load_pdf()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_handwriting(self) -> Dict:
        """Process the document to find handwriting regions.

        Returns:
            Dict: Dictionary mapping page numbers to handwriting detection results
        """
        # Close the current file handle and get a fresh path
        filepath = str(self.filepath)
        self.file.close()

        # Process PDF for handwriting
        try:
            handwriting_results = self.handwriting_processor.process_pdf(filepath)
        except Exception as e:
            logger.error(f"Error processing handwriting: {e}")
            handwriting_results = {}

        # Reopen the file and reload document
        self.file = open(filepath, "rb")
        self._load_document()

        return handwriting_results

    def _preprocess_image_for_ocr(self, image_array):
        """Preprocess image for better OCR results on handwritten content.

        Args:
            image_array (numpy.ndarray): Image as numpy array

        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        # Invert back for OCR
        processed = cv2.bitwise_not(denoised)

        return processed

    def _ocr_handwriting(self, image_array):
        """Perform OCR on handwritten content using Tesseract.

        Args:
            image_array (numpy.ndarray): Image containing handwritten text

        Returns:
            str: Extracted text from handwriting
        """
        try:
            # Preprocess image for better OCR results
            processed_img = self._preprocess_image_for_ocr(image_array)

            # OCR with Tesseract
            text = pytesseract.image_to_string(processed_img, config=self.ocr_config)

            return text.strip() or "No text detected"
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return f"OCR failed: {e}"

    def _might_be_equation(self, text):
        """Simple heuristic to check if text contains mathematical equations.

        Args:
            text (str): Text to check

        Returns:
            bool: True if text likely contains equations
        """
        math_symbols = set("+-*/=^()[]{}∫∑∏√∂∆πθλμ")
        symbol_count = sum(1 for char in text if char in math_symbols)
        symbol_density = symbol_count / max(len(text), 1)

        # Check for common equation patterns
        has_fractions = "/" in text and not text.startswith("http")
        has_equality = "=" in text
        has_numbers = any(c.isdigit() for c in text)

        return (symbol_density > 0.1) or (
            has_fractions and has_equality and has_numbers
        )

    def get_page_regions(self, page: fitz.Page) -> List[Region]:
        """
        Extract candidate regions from a PDF page using text blocks.

        Args:
            page (fitz.Page): A page from the PDF document.

        Returns:
            List[Region]: List of candidate Region objects.
        """
        regions = []
        blocks = page.get_text("blocks")
        # Each block is a tuple: (x0, y0, x1, y1, text, block_no, ...)
        for block in blocks:
            text = block[4]
            if text.strip():
                bbox = BoundingBox(x1=block[0], y1=block[1], x2=block[2], y2=block[3])
                region = Region(
                    region_type=RegionType.TEXT,
                    bbox=bbox,
                    confidence=1.0,
                    content=text,
                    metadata={
                        "page_num": page.number if hasattr(page, "number") else 0
                    },
                )
                regions.append(region)
        return regions

    def detect_regions(self) -> Iterator[Region]:
        """Detect and yield regions from the document.

        This method integrates text region detection with handwriting detection
        to provide a unified stream of detected regions.

        Yields:
            Iterator[Region]: Stream of detected regions
        """
        # First, yield text and equation regions from each page
        for page in self.doc:
            page_num = page.number

            # Yield text regions if text is in requested types
            if (
                RegionType.TEXT in self.region_types
                or RegionType.EQUATION in self.region_types
            ):
                for candidate in self.get_page_regions(page):
                    region = self._classify_text_region(candidate, page)
                    if region.region_type in self.region_types:
                        yield region

            # Yield handwriting regions for this page if enabled
            if (
                self.enable_handwriting_detection
                and RegionType.HANDWRITING in self.region_types
                and page_num in self.handwriting_results
            ):
                result = self.handwriting_results[page_num]

                if result["has_handwriting"]:
                    for region_info in result["handwriting_regions"]:
                        bbox = region_info["bbox"]

                        # Extract image for OCR
                        xref = region_info["xref"]
                        try:
                            # Extract handwriting image
                            base_image = self.doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image = Image.open(io.BytesIO(image_bytes))
                            image_array = np.array(image)

                            # Perform OCR on the handwriting
                            ocr_text = self._ocr_handwriting(image_array)

                            # Create region with OCR text
                            region_type = RegionType.HANDWRITING

                            # Check if this might be a handwritten equation
                            if self._might_be_equation(ocr_text):
                                metadata = {
                                    "page_num": page_num,
                                    "xref": region_info["xref"],
                                    "possible_equation": True,
                                }

                                # If we should convert handwriting to LaTeX and it looks like an equation
                                if self.convert_handwriting_to_latex:
                                    region_type = RegionType.EQUATION
                                    # Process as equation
                                    region = Region(
                                        region_type=region_type,
                                        bbox=BoundingBox(
                                            x1=bbox.x0,
                                            y1=bbox.y0,
                                            x2=bbox.x1,
                                            y2=bbox.y1,
                                        ),
                                        content=ocr_text,  # Use OCR text as content
                                        metadata={**metadata, "source": "handwritten"},
                                    )
                                    # Try to parse as equation if possible
                                    try:
                                        equation_content = self.eqparser.parse_equation(
                                            region, page
                                        )
                                        region.content = equation_content
                                    except Exception as e:
                                        logger.warning(
                                            f"Could not parse handwritten equation: {e}"
                                        )
                            else:
                                # Regular handwriting
                                metadata = {
                                    "page_num": page_num,
                                    "xref": region_info["xref"],
                                }

                                region = Region(
                                    region_type=region_type,
                                    bbox=BoundingBox(
                                        x1=bbox.x0, y1=bbox.y0, x2=bbox.x1, y2=bbox.y1
                                    ),
                                    content=ocr_text,  # Use OCR text as content
                                    metadata=metadata,
                                )

                        except Exception as e:
                            logger.warning(f"Failed to OCR handwriting: {e}")
                            # Fallback to original behavior if OCR fails
                            region = Region(
                                region_type=RegionType.HANDWRITING,
                                bbox=BoundingBox(
                                    x1=bbox.x0, y1=bbox.y0, x2=bbox.x1, y2=bbox.y1
                                ),
                                content=f"Handwritten content (xref: {region_info['xref']}) - OCR failed",
                                metadata={
                                    "page_num": page_num,
                                    "xref": region_info["xref"],
                                    "ocr_failed": True,
                                },
                            )

                        yield region

    def _classify_text_region(self, region: Region, page: fitz.Page) -> Region:
        """Enhanced classification of text regions with Unicode math detection.

        Args:
            region (Region): The region to classify.
            page (fitz.Page): The page where the region was detected.

        Returns:
            Region: The classified region with updated content if necessary.
        """
        # Use the detector initialized in __init__
        math_density = self.detector.calculate_math_density(region.content)
        has_patterns = self.detector.has_math_pattern(region.content)

        if math_density > 0.15 or has_patterns:
            region.region_type = RegionType.EQUATION
            region.content = self.eqparser.parse_equation(region, page)
        else:
            region.region_type = RegionType.TEXT

        return region

    def get_regions_by_type(self, region_type: RegionType) -> List[Region]:
        """Filter regions by their type.

        Args:
            region_type (RegionType): Type of regions to filter for

        Returns:
            List[Region]: List of regions matching the specified type
        """
        return [r for r in self.regions if r.region_type == region_type]

    def merge_adjacent_equations(self) -> None:
        """Merge consecutive equation regions on the same page into one.
        If more than one equation is merged, perform OCR (via EquationParser)
        on the merged region content.
        """
        merged_regions = []
        i = 0
        while i < len(self.regions):
            region = self.regions[i]
            if region.region_type == RegionType.EQUATION:
                merged_region = region
                merged_count = 1
                j = i + 1
                while j < len(self.regions):
                    next_region = self.regions[j]
                    # Merge only consecutive equations on the same page.
                    if (
                        next_region.region_type == RegionType.EQUATION
                        and merged_region.metadata.get("page_num")
                        == next_region.metadata.get("page_num")
                    ):
                        # Update bounding box to enclose both regions.
                        merged_region.bbox = BoundingBox(
                            min(merged_region.bbox.x1, next_region.bbox.x1),
                            min(merged_region.bbox.y1, next_region.bbox.y1),
                            max(merged_region.bbox.x2, next_region.bbox.x2),
                            max(
                                merged_region.bbox.y2,
                                next_region.bbox.y2,
                            ),
                        )
                        # Concatenate contents with a space after trimming.
                        merged_region.content = (
                            merged_region.content.rstrip()
                            + " "
                            + next_region.content.lstrip()
                        )
                        merged_count += 1
                        j += 1
                    else:
                        break

                # If more than one equation was merged, process merged content via OCR.
                if merged_count > 1:
                    page_num = merged_region.metadata.get("page_num")
                    page_obj = self.doc[page_num]
                    merged_region.content = self.eqparser.parse_equation(
                        merged_region, page_obj
                    )
                merged_regions.append(merged_region)
                i = j
            else:
                merged_regions.append(region)
                i += 1
        self.regions = merged_regions

    def export_regions(self, regions: Iterator[Region] = None) -> None:
        """
        Processes regions from an iterator and exports them using the provided writer.
        If no iterator is provided, it fetches regions by calling detect_regions().
        """
        if regions is None:
            regions = self.detect_regions()

        with self.writer_class(self.output_path) as writer:
            writer.init_tables()
            # Use a generator to convert each Region into an exportable dict
            writer.write_data(
                (
                    region.to_json(exclude_fields=self.exclude_fields)
                    for region in regions
                )
            )

    def __enter__(self):
        """Context manager entry method.

        Returns:
            DocumentAnalyzer: The analyzer instance
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit method.

        Ensures proper cleanup by closing the input file.

        Args:
            exc_type: Type of exception that occurred, if any
            exc_value: Exception instance that occurred, if any
            traceback: Traceback information, if an exception occurred
        """
        self.file.close()
        if exc_type is not None:
            logger.error(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        return False  # Propagate exceptions

    def _process_handwriting(self) -> Dict:
        """
        Process the document to detect handwriting regions.

        Returns:
            Dict: A mapping of page numbers to handwriting detection results.
        """
        filepath_str = str(self.filepath)
        self.file.close()
        logger.debug(f"Starting handwriting detection for {filepath_str}")

        try:
            handwriting_results = self.handwriting_processor.process_pdf(filepath_str)
            total_pages = len(handwriting_results)
            pages_with_handwriting = sum(
                1
                for result in handwriting_results.values()
                if result["has_handwriting"]
            )
            logger.debug(
                f"Handwriting detection complete: {pages_with_handwriting}/{total_pages} pages contain handwriting"
            )
        except Exception as e:
            logger.error("Error processing handwriting", exc_info=True)
            handwriting_results = {}

        self.file = open(filepath_str, "rb")
        self._load_document()
        return handwriting_results
