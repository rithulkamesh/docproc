import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Iterator, Dict
import cv2
from tqdm import tqdm  # For progress bars
from docproc.doc.equations import EquationParser, UnicodeMathDetector
from docproc.doc.regions import BoundingBox, Region, RegionType
from docproc.writer import FileWriter
from docproc.doc.visual import VisualContentProcessor


# Suppress PIL logs - add this at the top
logging.getLogger("PIL").setLevel(logging.WARNING)
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
        enable_visual_detection: bool = False,
        max_batch_size: int = 5,
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
            enable_visual_detection (bool): Whether to enable advanced visual content detection
            max_batch_size (int): Maximum number of images to process in a batch
        """
        # Suppress PIL logs immediately to prevent initial flood
        logging.getLogger("PIL").setLevel(logging.WARNING)

        # Visual detection configuration
        self.enable_visual_detection = enable_visual_detection
        self.max_batch_size = max_batch_size

        self.filepath = Path(filepath)
        self.file = open(filepath, "rb")
        self.writer_class = writer
        self.output_path = output_path
        self.region_types = region_types or list(RegionType)
        self.doc = None

        # Keep track of active tasks
        self._active_tasks = set()

        logger.info(f"Loading document: {filepath}")
        self._load_document()

        # Use singleton instances to avoid reloading models
        self.eqparser = EquationParser.get_instance()
        self.detector = UnicodeMathDetector()
        self.exclude_fields = exclude_fields

        if enable_visual_detection:
            logger.info("Initializing visual content detection...")
            # Use singleton instance of VisualContentProcessor
            self.visual_processor = VisualContentProcessor.get_instance(max_workers=2)
            # Process visual content early to avoid file handle issues
            self.visual_results = self._process_visual_content()
        else:
            self.visual_results = {}

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

        num_pages = len(self.doc)
        logger.info(f"Processing {num_pages} pages...")

        # Determine if we need to process images at all
        need_images = self.enable_visual_detection and (
            RegionType.EQUATION in self.region_types
            or RegionType.FIGURE in self.region_types
            or RegionType.IMAGE in self.region_types
        )

        try:
            # Add progress bar for document loading
            for page_num in tqdm(
                range(num_pages), desc="Loading document", unit="page"
            ):
                page = self.doc[page_num]

                # Extract text blocks (always needed)
                text_blocks = page.get_text("blocks")
                self.raw_blocks.extend((block, page_num) for block in text_blocks)

                # Only extract images if needed for visual detection
                if need_images:
                    try:
                        images = page.get_images(full=True)
                        # Process images in smaller batches to manage memory
                        current_batch = []
                        for img in images:
                            if img:
                                xref = img[0]
                                try:
                                    bbox = page.get_image_bbox(img)
                                    if bbox:
                                        current_batch.append((xref, bbox, page_num))

                                        # When batch is full, process it
                                        if len(current_batch) >= self.max_batch_size:
                                            self.raw_images.extend(current_batch)
                                            current_batch = []

                                except Exception as e:
                                    logger.debug(f"Failed to process image: {e}")

                        # Add any remaining images in the batch
                        if current_batch:
                            self.raw_images.extend(current_batch)

                    except Exception as e:
                        logger.debug(
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

    def _process_visual_content(self) -> Dict:
        """Process the document to find visual content like equations and diagrams.

        Returns:
            Dict: Dictionary mapping page numbers to visual content detection results
        """
        # Close the current file handle and get a fresh path
        filepath = str(self.filepath)
        self.file.close()

        # Process PDF for visual content
        try:
            logger.info("Processing document for visual content...")
            visual_results = self.visual_processor.process_pdf(filepath)
            total_pages = len(visual_results)
            pages_with_content = sum(
                1 for result in visual_results.values() if result["has_visual_content"]
            )
            logger.info(
                f"Visual content detection complete: {pages_with_content}/{total_pages} pages contain visual elements"
            )
        except Exception as e:
            logger.error(f"Error processing visual content: {e}")
            visual_results = {}

        # Reopen the file and reload document
        logger.info("Reloading document after visual content detection")
        self.file = open(filepath, "rb")
        self._load_document()

        return visual_results

    def extract_all_data(self) -> List[Dict]:
        """
        Extract all data from the document, including text, equations, and visual content.

        Returns:
            List[Dict]: A list of dictionaries, each representing a detected region with its content and metadata.
        """
        logger.info("Extracting all data from the document...")
        extracted_data = []

        # Detect regions in the document
        regions = list(self.detect_regions())

        for region in regions:
            # Convert each region to a dictionary with its content and metadata
            region_data = {
                "type": region.region_type.name,
                "content": region.content,
                "bbox": {
                    "x1": region.bbox.x1,
                    "y1": region.bbox.y1,
                    "x2": region.bbox.x2,
                    "y2": region.bbox.y2,
                },
                "metadata": region.metadata,
            }
            extracted_data.append(region_data)

        logger.info(f"Extraction complete. Total regions: {len(extracted_data)}")
        return extracted_data

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

        This method integrates text region detection with visual content detection
        to provide a unified stream of detected regions.

        Yields:
            Iterator[Region]: Stream of detected regions
        """
        # Show progress when processing pages for regions
        page_count = len(self.doc)
        with tqdm(total=page_count, desc="Detecting regions", unit="page") as pbar:
            for page in self.doc:
                page_num = page.number
                pbar.update(1)

                # Yield text regions if text is in requested types
                if (
                    RegionType.TEXT in self.region_types
                    or RegionType.EQUATION in self.region_types
                ):
                    for candidate in self.get_page_regions(page):
                        region = self._classify_text_region(candidate, page)
                        if region.region_type in self.region_types:
                            yield region

                # Yield visual content regions for this page if enabled
                if (
                    self.enable_visual_detection
                    and page_num in self.visual_results
                    and (
                        RegionType.EQUATION in self.region_types
                        or RegionType.FIGURE in self.region_types
                        or RegionType.IMAGE in self.region_types
                    )
                ):
                    result = self.visual_results[page_num]

                    if result["has_visual_content"]:
                        for region_info in result["visual_regions"]:
                            bbox = region_info["bbox"]
                            content = region_info["content"]
                            region_type = region_info["region_type"]

                            if region_type in self.region_types:
                                metadata = region_info["metadata"]
                                region = Region(
                                    region_type=region_type,
                                    bbox=BoundingBox(
                                        x1=bbox.x0, y1=bbox.y0, x2=bbox.x1, y2=bbox.y1
                                    ),
                                    content=content,
                                    metadata=metadata,
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

        # Collect regions to count them before exporting
        region_list = list(regions)

        logger.info(f"Exporting {len(region_list)} regions...")
        with self.writer_class(self.output_path) as writer:
            writer.init_tables()
            # Use a generator with a progress bar for the export process
            with tqdm(
                total=len(region_list), desc="Exporting regions", unit="region"
            ) as pbar:

                def region_generator():
                    for region in region_list:
                        pbar.update(1)
                        yield region.to_json(exclude_fields=self.exclude_fields)

                writer.write_data(region_generator())

        logger.info(f"Export complete: {self.output_path}")

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
        try:
            # Clean up any resources in the equation parser
            if hasattr(self, "eqparser") and self.eqparser:
                if hasattr(self.eqparser, "cleanup"):
                    self.eqparser.cleanup()

            # Close the document file
            if self.file:
                self.file.close()

            # Close PyMuPDF document
            if self.doc:
                self.doc.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        if exc_type is not None:
            logger.error(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        return False  # Propagate exceptions
