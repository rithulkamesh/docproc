"""Abstract DLA engine interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from docproc.doc.regions import Region


class DLAEngine(ABC):
    """Abstract base class for Document Layout Analysis engines.

    Implementations use ML models (e.g., LayoutLMv3, DocLayout-YOLO)
    to detect and classify document regions with high accuracy.
    """

    @abstractmethod
    def analyze_page(self, page_image, page_num: int = 0) -> List[Region]:
        """Analyze a single page image and return detected regions.

        Args:
            page_image: PIL Image or numpy array of the page
            page_num: Zero-based page number for metadata

        Returns:
            List of Region objects with type, bbox, content, confidence
        """
        pass

    @abstractmethod
    def analyze_document(self, pdf_path: str | Path) -> List[Region]:
        """Analyze a full PDF document and return all regions.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Region objects from all pages
        """
        pass
