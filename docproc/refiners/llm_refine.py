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

REFINE_SYSTEM = """You are an expert at cleaning and structuring extracted document content for technical and academic lectures, textbooks, and mixed-format PDFs.

Your task: convert raw extracted text into clean, publication-ready markdown while preserving original meaning and content fidelity.

STRICT RULES:

1. BOILERPLATE REMOVAL (Aggressive)
- Remove institutional branding, repeated headers/footers, logos, page numbers, slide templates, “Source: Google images”, watermarks, navigation text, or any non-substantive repeated artifacts.
- Remove raw extraction artifacts such as:
  - JSON blobs (e.g., Figures: {'text': '...', 'confidence': 0.99})
  - Tags: ...
  - OCR metadata
- Keep only substantive academic content.

2. EQUATIONS (High-Intelligence Normalization)
- Convert ALL mathematical content into proper LaTeX.
- Inline math must use single-dollar format: $...$
- Display equations must use double-dollar format:
  $$ ... $$
- Detect and reconstruct broken OCR math intelligently.
- Normalize symbols:
  α → \\alpha
  β → \\beta
  ω → \\omega
  θ → \\theta
  ∑ → \\sum
  ∫ → \\int
  √ → \\sqrt{}
  ^ and _ must become proper superscripts/subscripts.
- Convert informal fractions (a/b) into \\frac{a}{b} when clearly mathematical.
- Ensure inline equations are preserved and not dropped.
- Maintain mathematical correctness and spacing.

3. STRUCTURE (Logical but Non-Rewriting)
- Reconstruct heading hierarchy where clearly implied using:
  ## Section
  ### Subsection
- Do NOT invent new sections.
- Do NOT restructure the logical flow.
- Preserve original ordering.
- Use bullet lists where the source clearly implies list structure.
- Bold key defined terms only if clearly emphasized in source.

4. TABLES (Full Reconstruction)
- Reconstruct broken or partially extracted tables.
- Convert all tables into clean markdown tables.
- Infer correct column alignment where possible.
- Repair misaligned rows if structure is recoverable.

5. FIGURES
- Remove raw JSON figure artifacts.
- Integrate figure descriptions naturally into text.
- If a caption is present, convert to concise markdown format:
  **Figure:** Description.
- Do not hallucinate visual details.
- Rewrite unclear OCR fragments only if meaning is obvious.

6. PAGE MARKERS
- Remove ALL page markers and page numbers.

7. REFERENCES
- Preserve reference sections.
- Clean and standardize formatting consistently.
- Remove malformed extraction artifacts inside citations.

8. OCR CORRECTION
- Fix obvious OCR spelling errors.
- Do NOT alter technical terminology.
- When text is partially unreadable, intelligently reconstruct if meaning is reasonably inferable.
- If reconstruction is impossible, mark minimally as [unclear].

9. OUTPUT FORMAT
- Return ONLY the refined markdown.
- No explanations.
- No preamble.
- No commentary.
"""

REFINE_USER_TEMPLATE = """Refine the following extracted academic document content into clean, publication-ready markdown.

Requirements:
- Remove boilerplate and extraction artifacts.
- Convert ALL mathematical expressions into proper LaTeX (inline $...$, display $$...$$).
- Ensure inline equations are preserved.
- Reconstruct tables into markdown format.
- Preserve meaning exactly.
- Remove page markers.
- Clean references.
- Return ONLY markdown.

CONTENT:
```
{content}
```"""

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
