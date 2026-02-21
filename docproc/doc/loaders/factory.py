"""Document loader factory — dispatch by file extension."""

from pathlib import Path
from typing import Iterator, List

from docproc.doc.loaders.base import DocumentLoader, LoadedPage
from docproc.doc.loaders.pdf_loader import PDFLoader
from docproc.doc.loaders.docx_loader import DOCXLoader
from docproc.doc.loaders.pptx_loader import PPTXLoader
from docproc.doc.loaders.xlsx_loader import XLSXLoader

EXT_TO_LOADER: dict[str, type[DocumentLoader]] = {
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".pptx": PPTXLoader,
    ".xlsx": XLSXLoader,
    ".xlsm": XLSXLoader,  # Excel macro-enabled
}


def get_supported_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return sorted(set(EXT_TO_LOADER.keys()))


def get_loader(path: Path) -> DocumentLoader:
    """Get loader for the given file path."""
    ext = path.suffix.lower()
    loader_cls = EXT_TO_LOADER.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported format: {ext}. Supported: {get_supported_extensions()}")
    return loader_cls()


def load_document(path: Path) -> Iterator[LoadedPage]:
    """Load document and yield pages. Format inferred from extension."""
    loader = get_loader(path)
    return loader.load(path)
