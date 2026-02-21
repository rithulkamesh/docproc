"""LLM-based refinement of extracted document text.

Converts raw OCR/vision output into clean markdown with:
- LaTeX equations (inline $...$ and display $$...$$)
- Boilerplate removed (logos, repeated headers)
- Figure descriptions merged cleanly (no JSON/confidence artifacts)
- Proper headings and structure
"""

import logging
import re
from typing import Optional

from docproc.providers.base import ChatMessage, ModelProvider

logger = logging.getLogger(__name__)

REFINE_SYSTEM = """You are an expert at cleaning and structuring extracted document content for technical/academic lectures and textbooks.

Your task: convert raw extracted text into clean, publication-ready markdown.

RULES:
1. **Remove boilerplate** – Strip logos, "PES UNIVERSITY ONLINE", "Source: Google images", page footers repeated on every page. Keep only substantive content.
2. **Equations** – Convert all math to LaTeX: inline $E = mc^2$ or display $$\\frac{d}{dx}\\int f = f$$. Preserve subscripts, superscripts, Greek (α, β, ω → \\alpha, \\beta, \\omega).
3. **Figures** – Merge figure descriptions naturally. Remove raw JSON like Figures: {'text': '...', 'confidence': 0.99} and Tags: ... artifacts. Use concise captions or inline: "Figure shows a diagram of...".
4. **Structure** – Use markdown headings (##, ###), lists, bold for key terms. Preserve [Page N] labels only if useful.
5. **Language** – Fix OCR typos where obvious; keep technical terms and equations accurate.
6. **Output** – Return ONLY the refined markdown. No preamble, no "Here is the refined content"."""

REFINE_USER_TEMPLATE = """Refine this extracted document content into clean markdown with proper equations and no boilerplate:

---
{content}
---"""

# Approx 4 chars per token; GPT-4o 128k ~ 500k chars. Chunk at ~80k chars to stay safe.
CHUNK_CHARS = 80_000


def _chunk_by_pages(text: str) -> list[str]:
    """Split text by [Page N] boundaries for chunked refinement."""
    # Match [Page N] or [Page N]\n
    parts = re.split(r'(\[Page \d+\])', text)
    chunks = []
    current = []
    current_len = 0
    for i, p in enumerate(parts):
        if re.match(r'\[Page \d+\]', p):
            if current and current_len + len(p) > CHUNK_CHARS and current_len > 0:
                chunks.append("".join(current))
                current = [p]
                current_len = len(p)
            else:
                current.append(p)
                current_len += len(p)
        else:
            current.append(p)
            current_len += len(p)
    if current:
        chunks.append("".join(current))
    return chunks if chunks else [text]


def refine_extracted_text(
    raw_text: str,
    provider: ModelProvider,
    model: Optional[str] = None,
) -> str:
    """Refine raw extracted text into clean markdown with LaTeX math.

    Removes boilerplate, formats equations, merges figure descriptions.
    Chunks long documents and concatenates results.

    Args:
        raw_text: Raw text from extraction (OCR + vision)
        provider: ModelProvider for chat
        model: Optional model override

    Returns:
        Refined markdown string, or raw_text if refinement fails
    """
    if not raw_text or not raw_text.strip():
        return raw_text
    try:
        chunks = _chunk_by_pages(raw_text)
        refined_parts = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            msg = REFINE_USER_TEMPLATE.format(content=chunk.strip())
            resp = provider.chat(
                messages=[
                    ChatMessage(role="system", content=REFINE_SYSTEM),
                    ChatMessage(role="user", content=msg),
                ],
                model=model,
            )
            out = (resp.content or "").strip()
            if out:
                refined_parts.append(out)
        if refined_parts:
            return "\n\n".join(refined_parts)
        return raw_text
    except Exception as e:
        logger.warning("LLM refinement failed, using raw text: %s", e)
        return raw_text
