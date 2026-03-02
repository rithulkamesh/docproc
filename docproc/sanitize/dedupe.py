"""Deduplication and boilerplate detection for document chunks/slides."""

import hashlib
import re
from enum import Enum
from typing import List, Optional, Set

from docproc.sanitize.sanitizer import sanitize_text


class Chunk:
    """Minimal chunk for deduplication (content-only; used by CLI pipeline)."""

    def __init__(self, content: str = "", **kwargs: object) -> None:
        self.content = content
        for k, v in kwargs.items():
            setattr(self, k, v)


class BoilerplateKind(Enum):
    """Known boilerplate slide/section types."""

    TITLE = "title"
    THANK_YOU = "thank_you"
    QUESTIONS = "questions"
    APPENDIX = "appendix"
    BLANK = "blank"
    AGENDA = "agenda"
    OBJECTIVES = "objectives"
    NONE = "none"


# Patterns for boilerplate detection (case-insensitive)
BOILERPLATE_PATTERNS: list[tuple[BoilerplateKind, re.Pattern]] = [
    (BoilerplateKind.THANK_YOU, re.compile(r"^(thank\s*you|thanks|gracias|merci)\s*\!?\s*$", re.I)),
    (BoilerplateKind.THANK_YOU, re.compile(r"^(questions\s*\?|q\s*&\s*a|any\s*questions)\s*$", re.I)),
    (BoilerplateKind.QUESTIONS, re.compile(r"^questions\s*\??\s*$", re.I)),
    (BoilerplateKind.QUESTIONS, re.compile(r"^q\s*&\s*a\s*$", re.I)),
    (BoilerplateKind.APPENDIX, re.compile(r"^append(i|x)(ces)?\s*$", re.I)),
    (BoilerplateKind.APPENDIX, re.compile(r"^references?\s*$", re.I)),
    (BoilerplateKind.AGENDA, re.compile(r"^agenda\s*$", re.I)),
    (BoilerplateKind.OBJECTIVES, re.compile(r"^(objectives?|learning\s+outcomes)\s*$", re.I)),
]


def is_boilerplate(
    text: str,
    kind: Optional[BoilerplateKind] = None,
) -> tuple[bool, BoilerplateKind]:
    """Check if text matches known boilerplate (title, thank you, etc.).

    Args:
        text: Content to check
        kind: If set, only check for this kind

    Returns:
        (is_boilerplate, detected_kind)
    """
    cleaned = sanitize_text(text)
    if not cleaned:
        return True, BoilerplateKind.BLANK

    lines = cleaned.split("\n")
    first_line = (lines[0] if lines else "").strip()
    if not first_line:
        return True, BoilerplateKind.BLANK

    for bp_kind, pat in BOILERPLATE_PATTERNS:
        if kind is not None and bp_kind != kind:
            continue
        if pat.search(first_line) or (len(lines) <= 2 and pat.search(cleaned)):
            return True, bp_kind

    # Very short content often boilerplate
    if len(cleaned) < 20 and len(lines) <= 1:
        return True, BoilerplateKind.TITLE

    return False, BoilerplateKind.NONE


def _content_hash(text: str) -> str:
    """Stable hash for exact deduplication."""
    return hashlib.sha256(sanitize_text(text).encode("utf-8")).hexdigest()


def deduplicate_chunks(
    chunks: List[Chunk],
    *,
    drop_exact_duplicates: bool = True,
    drop_boilerplate: bool = True,
    boilerplate_kinds: Optional[set[BoilerplateKind]] = None,
) -> List[Chunk]:
    """Remove duplicate and boilerplate chunks.

    Args:
        chunks: Chunks to deduplicate
        drop_exact_duplicates: Remove chunks with identical sanitized content
        drop_boilerplate: Remove title/thank-you/etc slides
        boilerplate_kinds: Kinds to drop; if None, drops all boilerplate

    Returns:
        Deduplicated list (order preserved)
    """
    seen_hashes: Set[str] = set()
    result: List[Chunk] = []
    kinds_to_drop = boilerplate_kinds or {
        BoilerplateKind.THANK_YOU,
        BoilerplateKind.QUESTIONS,
        BoilerplateKind.BLANK,
    }

    for c in chunks:
        content = c.content or ""
        sanitized = sanitize_text(content)

        if drop_boilerplate:
            is_bp, bp_kind = is_boilerplate(content)
            if is_bp and bp_kind in kinds_to_drop:
                continue

        if drop_exact_duplicates and sanitized:
            h = _content_hash(sanitized)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

        result.append(c)

    return result


def deduplicate_texts(
    texts: List[str],
    *,
    drop_exact_duplicates: bool = True,
    drop_boilerplate: bool = True,
) -> List[str]:
    """Deduplicate a list of text strings.

    Convenience for use before chunking.
    """
    class _FakeChunk:
        def __init__(self, content: str):
            self.content = content
    fake = [_FakeChunk(t) for t in texts]
    deduped = deduplicate_chunks(fake, drop_exact_duplicates=drop_exact_duplicates, drop_boilerplate=drop_boilerplate)
    return [c.content for c in deduped]
