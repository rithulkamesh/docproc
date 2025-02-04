from typing import Optional, List
from pathlib import Path
import fitz
import logging
import concurrent

from docproc.doc.regions import Region, RegionType, BoundingBox
from docproc.doc.equations import UnicodeMathDetector, EquationParser
from docproc.writer import FileWriter

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
        # Process text blocks sequentially
        for block, page_num in self.raw_blocks:
            logger.info(f"Processing page {page_num}")
            x1, y1, x2, y2, text, *_ = block
            rgn = Region(
                region_type=RegionType.UNCLASSIFIED,
                bbox=BoundingBox(
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)
                ),
                confidence=1.0,
                content=text,
                metadata={"page_num": page_num},
            )
            # Classify and add region directly
            classified_region = self._classify_text_region(rgn, self.doc[page_num])
            self.regions.append(classified_region)

        # Process images
        for xref, bbox, page_num in self.raw_images:
            self.regions.append(
                Region(
                    region_type=RegionType.IMAGE,
                    bbox=BoundingBox(
                        round(bbox.x0, 2),
                        round(bbox.y0, 2),
                        round(bbox.x1, 2),
                        round(bbox.y1, 2),
                    ),
                    confidence=1.0,
                    metadata={"xref": xref, "page_num": page_num},
                )
            )

        return self.regions

    def _classify_text_region(self, region: Region, page: fitz.Page) -> Region:
        """Enhanced classification of text regions with Unicode math detection.

        Args:
            text (str): The text content of the region

        Returns:
            RegionType: The classified region type
        """
        detector = UnicodeMathDetector()

        # Use multiple heuristics to detect mathematical content
        math_density = detector.calculate_math_density(region.content)
        has_patterns = detector.has_math_pattern(region.content)

        # Classify as equation if either:
        # 1. High density of mathematical symbols (>15%)
        # 2. Clear mathematical patterns are present
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
