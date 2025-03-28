from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import Optional, Dict
import json


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
    UNCLASSIFIED = auto()

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
    confidence: Optional[float] = 0.0
    content: Optional[str] = None
    metadata: Dict[str, any] = None

    def __post_init__(self):
        """Initialize empty metadata dictionary if none provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_json(self, exclude_fields=None):
        """Convert region to SQLite-compatible dictionary.

        Args:
            exclude_fields (list[str]): Optional list of field names to exclude
        """
        base_dict = {
            "region_type": self.region_type.to_json(),
            "bbox": json.dumps(asdict(self.bbox)),
            "confidence": self.confidence,
            "content": self.content,
            "metadata": json.dumps(self.metadata) if self.metadata else None,
        }

        if exclude_fields:
            return {k: v for k, v in base_dict.items() if k not in exclude_fields}
        return base_dict
