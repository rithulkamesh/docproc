"""Text sanitization: normalize, strip artifacts, fix encoding."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SanitizerConfig:
    """Configuration for text sanitization."""

    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    strip_control_chars: bool = True
    strip_zero_width: bool = True
    min_content_length: int = 2
    max_consecutive_newlines: int = 2


def sanitize_text(
    text: Optional[str],
    config: Optional[SanitizerConfig] = None,
) -> str:
    """Sanitize and normalize text for storage and retrieval.

    - Normalizes unicode (NFC)
    - Collapses whitespace
    - Strips control chars and zero-width characters
    - Limits consecutive newlines

    Args:
        text: Raw text to sanitize
        config: SanitizerConfig; uses defaults if None

    Returns:
        Sanitized text, or empty string if input is None/empty
    """
    if text is None or not isinstance(text, str):
        return ""
    cfg = config or SanitizerConfig()
    s = text

    if cfg.normalize_unicode:
        s = unicodedata.normalize("NFC", s)

    if cfg.strip_control_chars:
        s = "".join(c for c in s if unicodedata.category(c) != "Cc" or c in "\n\t\r")

    if cfg.strip_zero_width:
        zw = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"}
        s = "".join(c for c in s if c not in zw)

    if cfg.collapse_whitespace:
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        if cfg.max_consecutive_newlines >= 0:
            s = re.sub(r"\n{" + str(cfg.max_consecutive_newlines + 1) + ",}", "\n" * (cfg.max_consecutive_newlines + 1), s)
        s = s.strip()

    if cfg.min_content_length > 0 and len(s) < cfg.min_content_length:
        return ""
    return s
