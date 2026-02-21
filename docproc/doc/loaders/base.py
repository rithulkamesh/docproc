"""Base document loader interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from docproc.doc.regions import Region


@dataclass
class LoadedPage:
    """A single page/slide/sheet from a document."""

    page_num: int
    text: str
    regions: List[Region]
    raw_images: List[bytes]  # PNG/JPEG bytes for vision extraction


class DocumentLoader(ABC):
    """Abstract loader for a document format."""

    @abstractmethod
    def load(self, path: Path) -> Iterator[LoadedPage]:
        """Load document and yield pages."""
        pass

    @abstractmethod
    def get_full_text(self, path: Path) -> str:
        """Extract full text concatenated (for chunking)."""
        pass
