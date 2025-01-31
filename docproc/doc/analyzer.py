from dataclasses import asdict, dataclass
from enum import Enum, auto
import json
from typing import Optional, Dict, List
from pathlib import Path
import fitz
import logging

from docproc.doc.equations import UnicodeMathDetector, EquationParser
from docproc.writer import FileWriter

logger = logging.getLogger(__name__)


class RegionType(Enum):
    """Enumeration of supported document region types.

    Defines the different types of regions that can be detected within a document:
    - TEXT: Regions containing textual content
    - EQUATION: Regions containing mathematical equations
    - IMAGE: Regions containing images or graphics
    - HANDWRITING: Regions containing handwritten content
    """

    TEXT = auto()
    EQUATION = auto()
    IMAGE = auto()
    HANDWRITING = auto()

    def to_json(self):
        return self.name


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box in a document.

    Stores the coordinates of a rectangular region defined by its top-left (x1, y1)
    and bottom-right (x2, y2) corners.

    Attributes:
        x1 (float): X-coordinate of the top-left corner
        y1 (float): Y-coordinate of the top-left corner
        x2 (float): X-coordinate of the bottom-right corner
        y2 (float): Y-coordinate of the bottom-right corner
    """

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Region:
    """Represents a detected region within a document.

    Stores information about a specific region including its type, location,
    detection confidence, content, and additional metadata.

    Attributes:
        region_type (RegionType): Type of the region (text, equation, image, etc.)
        bbox (BoundingBox): Bounding box coordinates of the region
        confidence (float): Confidence score of the region detection (0.0 to 1.0)
        content (Optional[str]): Extracted content from the region, if applicable
        metadata (Dict[str, any]): Additional metadata associated with the region
    """

    region_type: RegionType
    bbox: BoundingBox
    confidence: float
    content: Optional[str] = None
    metadata: Dict[str, any] = None

    def __post_init__(self):
        """Initialize empty metadata dictionary if none provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_json(self):
        """Convert region to SQLite-compatible dictionary."""
        return {
            "region_type": self.region_type.to_json(),
            "bbox": json.dumps(asdict(self.bbox)),  # Serialize bbox to JSON string
            "confidence": self.confidence,
            "content": self.content,
            "metadata": (
                json.dumps(self.metadata) if self.metadata else None
            ),  # Serialize metadata
        }


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
    ):
        """Initialize DocumentAnalyzer with input file and writer.

        Args:
            filepath (str): Path to the input document file
            writer (type[FileWriter]): FileWriter class to use for exporting results
            output_path (str): Path where output should be written
            region_types (Optional[List[RegionType]]): List of region types to scan for.
                                                      If None, scans for all types.
        """
        self.filepath = Path(filepath)
        self.file = open(filepath, "rb")
        self.writer_class = writer
        self.regions: List[Region] = []
        self.output_path = output_path
        self.region_types = region_types or list(RegionType)
        self.doc = None
        self._load_document()
        self.eqparser = EquationParser()

    def _load_pdf(self) -> None:
        """Load and process a PDF document.

        Extracts text blocks and images from each page of the PDF document
        without classifying them into specific region types.

        Raises:
            fitz.FileDataError: If the PDF file is corrupted or invalid
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
        """Load and process the input document based on its file type.

        Determines the appropriate loading method based on the file extension
        and delegates to the corresponding loader method.

        Raises:
            ValueError: If the file type is not supported
        """
        ext = self.filepath.suffix.lower()
        if ext == ".pdf":
            self._load_pdf()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def detect_regions(self) -> List[Region]:
        """Detect and classify regions in the document.

        Processes the document to identify and classify different types of regions
        such as text blocks, images, equations, and handwritten content.

        Returns:
            List[Region]: List of detected and classified regions
        """
        self.regions = []

        # Process text blocks
        for block, page_num in self.raw_blocks:
            x1, y1, x2, y2, text, *_ = block
            (region_type, content) = self._classify_text_region(text)
            self.regions.append(
                Region(
                    region_type=region_type,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=1.0,
                    content=content,
                    metadata={"page_num": page_num},
                )
            )

        # Process images
        for xref, bbox, page_num in self.raw_images:
            self.regions.append(
                Region(
                    region_type=RegionType.IMAGE,
                    bbox=BoundingBox(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    confidence=1.0,
                    metadata={"xref": xref, "page_num": page_num},
                )
            )

        return self.regions

    def _classify_text_region(self, text: str) -> (RegionType, str):
        """Enhanced classification of text regions with Unicode math detection.

        Args:
            text (str): The text content of the region

        Returns:
            RegionType: The classified region type
        """
        detector = UnicodeMathDetector()

        # Use multiple heuristics to detect mathematical content
        math_density = detector.calculate_math_density(text)
        has_patterns = detector.has_math_pattern(text)

        # Classify as equation if either:
        # 1. High density of mathematical symbols (>15%)
        # 2. Clear mathematical patterns are present
        if math_density > 0.15 or has_patterns:
            return RegionType.EQUATION, self.eqparser.parse_equation(text)

        return (RegionType.TEXT, text)

    def get_regions_by_type(self, region_type: RegionType) -> List[Region]:
        """Filter regions by their type.

        Args:
            region_type (RegionType): Type of regions to filter for

        Returns:
            List[Region]: List of regions matching the specified type
        """
        return [r for r in self.regions if r.region_type == region_type]

    def export_regions(self) -> None:
        """Export processed regions using the configured writer.

        Writes all detected regions to the output file using the configured
        FileWriter class. Provides progress updates during export.
        """

        def progress(count: int):
            """Callback function to report export progress.

            Args:
                count (int): Number of regions processed so far
            """
            logger.info(f"Processed {count} regions...")

        with self.writer_class(self.output_path, progress) as writer:
            writer.init_tables()
            writer.write_data(region.to_json() for region in self.regions)

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
