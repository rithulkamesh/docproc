"""Shared document extraction pipeline — full text + optional vision and refine.

No server, no RAG. Simple function chaining for CLI and API reuse.
"""

from pathlib import Path
from typing import Callable, Optional

from docproc.config import get_config
from docproc.doc.loaders import get_full_text


def extract_document_to_text(
    path: Path,
    *,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Extract full text from document, with optional vision (PDF) and LLM refinement.

    Uses config (ingest.use_vision, ingest.use_llm_refine) to decide:
    - PDF: if use_vision, tries vision extraction; else native text
    - All formats: if use_llm_refine, refines output to markdown/LaTeX

    Args:
        path: Input document path (PDF, DOCX, PPTX, XLSX)
        progress_callback: Optional (page, total, message) callback during extraction

    Returns:
        Extracted and optionally refined full text string
    """
    cfg = get_config()
    ext = path.suffix.lower()

    # Step 1: Full text extraction
    if ext == ".pdf":
        use_vision = getattr(cfg.ingest, "use_vision", True)
        if use_vision:
            try:
                from docproc.providers.factory import get_provider
                from docproc.extractors.vision_llm import extract_pdf_text_and_images

                provider = get_provider()
                if provider is not None:
                    full_text = extract_pdf_text_and_images(
                        path, provider, progress_callback=progress_callback
                    )
                    if not full_text or not full_text.strip():
                        full_text = get_full_text(path)
                else:
                    full_text = get_full_text(path)
            except Exception:
                full_text = get_full_text(path)
        else:
            full_text = get_full_text(path)
    else:
        full_text = get_full_text(path)

    # Step 2: Optional LLM refinement
    use_refine = getattr(cfg.ingest, "use_llm_refine", True)
    if use_refine and full_text and full_text.strip():
        try:
            from docproc.providers.factory import get_provider
            from docproc.refiners import refine_extracted_text

            prov = get_provider()
            if prov is not None:
                if progress_callback:
                    progress_callback(0, 1, "Refining content…")
                full_text = refine_extracted_text(full_text, prov)
        except Exception:
            pass

    return full_text
