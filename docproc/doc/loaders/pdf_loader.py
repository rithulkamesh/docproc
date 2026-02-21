"""PDF document loader using PyMuPDF."""

from pathlib import Path
from typing import Iterator

import fitz
from PIL import Image
import io

from docproc.doc.loaders.base import DocumentLoader, LoadedPage
from docproc.doc.regions import BoundingBox, Region, RegionType


class PDFLoader(DocumentLoader):
    """Load PDF files via PyMuPDF."""

    def load(self, path: Path) -> Iterator[LoadedPage]:
        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("blocks")
                regions = []
                full_text = []
                for block in blocks:
                    x0, y0, x1, y1, text, *_ = block
                    if text.strip():
                        regions.append(
                            Region(
                                region_type=RegionType.TEXT,
                                bbox=BoundingBox(x1=x0, y1=y0, x2=x1, y2=y1),
                                confidence=0.9,
                                content=text.strip(),
                                metadata={"page_num": page_num},
                            )
                        )
                        full_text.append(text.strip())
                images = []
                for img_ref in page.get_images(full=True):
                    try:
                        xref = img_ref[0]
                        base = doc.extract_image(xref)
                        images.append(base["image"])
                    except Exception:
                        pass
                yield LoadedPage(
                    page_num=page_num,
                    text="\n\n".join(full_text),
                    regions=regions,
                    raw_images=images,
                )
        finally:
            doc.close()

    def get_full_text(self, path: Path) -> str:
        doc = fitz.open(path)
        try:
            return "\n\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
