"""Shared document extraction pipeline — full text + optional vision and refine.

No server, no RAG. Simple function chaining for CLI and API reuse.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from docproc.config import get_config
from docproc.doc.loaders import get_full_text

if TYPE_CHECKING:
    from docproc.config.schema import docprocConfig

logger = logging.getLogger(__name__)

# Exceptions we expect and can safely fall back from (network, provider, config)
_VISION_FALLBACK_EXC: tuple = (
    ConnectionError,
    TimeoutError,
    OSError,
    ValueError,
    KeyError,
    TypeError,
)
try:
    import openai as _openai
    _VISION_FALLBACK_EXC = _VISION_FALLBACK_EXC + (_openai.APIError, _openai.APIConnectionError)
except ImportError:
    pass


def extract_document_to_text(
    path: Path,
    *,
    config: Optional["docprocConfig"] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Extract full text from document, with optional vision (PDF) and LLM refinement.

    Uses config (ingest.use_vision, ingest.use_llm_refine) to decide:
    - PDF: if use_vision, tries vision extraction; else native text
    - All formats: if use_llm_refine, refines output to markdown/LaTeX

    Args:
        path: Input document path (PDF, DOCX, PPTX, XLSX)
        config: Optional config; uses get_config() if None
        progress_callback: Optional (page, total, message) callback during extraction

    Returns:
        Extracted and optionally refined full text string
    """
    cfg = config or get_config()
    ext = path.suffix.lower()

    # Step 1: Full text extraction
    if ext == ".pdf":
        use_vision = getattr(cfg.ingest, "use_vision", True)
        if use_vision:
            try:
                from docproc.providers.factory import get_provider
                from docproc.extractors.vision_llm import extract_pdf_text_and_images

                provider = get_provider(config=cfg) if cfg.ai_providers else None
                if provider is not None:
                    full_text = extract_pdf_text_and_images(
                        path, provider, progress_callback=progress_callback
                    )
                    if not full_text or not full_text.strip():
                        full_text = get_full_text(path)
                else:
                    full_text = get_full_text(path)
            except _VISION_FALLBACK_EXC as e:
                logger.debug("Vision extraction failed, using text fallback: %s", e)
                full_text = get_full_text(path)
            except Exception as e:
                logger.warning("Vision extraction error (unexpected): %s", e, exc_info=True)
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

            prov = get_provider(config=cfg) if cfg.ai_providers else None
            if prov is not None:
                if progress_callback:
                    progress_callback(0, 1, "Refining content…")
                full_text = refine_extracted_text(full_text, prov)
        except _VISION_FALLBACK_EXC as e:
            logger.debug("LLM refine failed, keeping raw text: %s", e)
        except Exception as e:
            logger.warning("LLM refine error (unexpected): %s", e, exc_info=True)

    return full_text
