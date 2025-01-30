from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Optional, Dict, List
from pathlib import Path
import fitz

from docproc.writer import FileWriter


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

    def __init__(self, filepath: str, writer: type[FileWriter], output_path: str):
        """Initialize DocumentAnalyzer with input file and writer.

        Args:
            filepath (str): Path to the input document file
            writer (type[FileWriter]): FileWriter class to use for exporting results
        """
        self.filepath = Path(filepath)
        self.file = open(filepath, "rb")
        self.writer_class = writer
        self.regions: List[Region] = []
        self.output_path = output_path
        self._load_document()

    def _load_pdf(self) -> None:
        """Load and process a PDF document.

        Extracts text blocks and images from each page of the PDF document
        and creates corresponding Region objects. Text is extracted using PyMuPDF's
        block detection, while images are extracted with their bounding boxes.

        Raises:
            fitz.FileDataError: If the PDF file is corrupted or invalid
        """
        doc = fitz.open(self.file)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text blocks
                text_blocks = page.get_text("blocks")
                for block in text_blocks:
                    x1, y1, x2, y2, text, *_ = block
                    self.regions.append(
                        Region(
                            region_type=RegionType.TEXT,
                            bbox=BoundingBox(x1, y1, x2, y2),
                            confidence=1.0,
                            content=text.strip(),
                        )
                    )

                # Extract images with proper initialization
                try:
                    # Get page images with proper list initialization
                    page.get_pixmap()  # Initialize page image list
                    images = page.get_images(full=True)

                    for img in images:
                        if img:
                            xref = img[0]
                            try:
                                bbox = page.get_image_bbox(img)
                                if bbox:
                                    self.regions.append(
                                        Region(
                                            region_type=RegionType.IMAGE,
                                            bbox=BoundingBox(
                                                bbox.x0, bbox.y0, bbox.x1, bbox.y1
                                            ),
                                            confidence=1.0,
                                            metadata={"xref": xref},
                                        )
                                    )
                            except Exception as e:
                                print(f"Warning: Failed to process image: {e}")
                                continue

                except Exception as e:
                    print(
                        f"Warning: Failed to extract images from page {page_num}: {e}"
                    )
                    continue

        finally:
            doc.close()

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

        return self.regions

    def get_regions_by_type(self, region_type: RegionType) -> List[Region]:
        """Filter regions by their type.

        Args:
            region_type (RegionType): Type of regions to filter for

        Returns:
            List[Region]: List of regions matching the specified type
        """
        return [r for r in self.regions if r.region_type == region_type]

    def extract_region_content(self, region: Region) -> None:
        """Extract content from a specific region based on its type.

        Processes the region to extract its content according to the region type.
        For example, performs OCR on image regions or equation parsing for
        equation regions.

        Args:
            region (Region): Region to extract content from

        Note:
            This is a placeholder method - implementation needed for different region types
        """
        # Implement extraction logic per region type
        pass

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
            print(f"Processed {count} regions...")

        with self.writer_class(self.output_path, progress) as writer:
            writer.init_tables()
            writer.write_data(asdict(region) for region in self.regions)

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
