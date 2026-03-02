"""Sanitize student/user text before sending to LLM for evaluation.

- Strip prompt-injection patterns
- Remove system/assistant-style instructions
- Truncate to max token limit (character heuristic)
"""

import re
from typing import Optional


# Patterns that suggest prompt injection or meta-instructions (case-insensitive).
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|above)\s+(instructions|prompts)",
    r"forget\s+(everything|all)\s+(you\s+)?(know|have\s+been\s+told)",
    r"you\s+are\s+now\s+(a|in)\s+",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"user\s*:\s*",
    r"\[system\]",
    r"\[assistant\]",
    r"<\|(system|assistant|user)\|>",
    r"jailbreak",
    r"do\s+not\s+follow\s+(any\s+)?(previous|above)",
    r"new\s+instructions\s*:",
    r"override\s*:",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Approximate chars per token for truncation (conservative for English).
_CHARS_PER_TOKEN = 4


def sanitize_for_llm(
    text: str,
    max_tokens: int = 2000,
    injection_patterns: Optional[list] = None,
) -> str:
    """Sanitize user/student text before sending to an LLM.

    - Strips common prompt-injection phrases (case-insensitive).
    - Truncates to roughly max_tokens using character heuristic.

    Args:
        text: Raw user input (e.g. student answer).
        max_tokens: Maximum allowed tokens (approximate via chars).
        injection_patterns: Optional list of regex patterns to strip; uses defaults if None.

    Returns:
        Sanitized string safe to include in evaluation prompt.
    """
    if not text or not isinstance(text, str):
        return ""
    out = text.strip()
    patterns = injection_patterns or _COMPILED
    if not isinstance(patterns[0], re.Pattern):
        patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    for pat in patterns:
        out = pat.sub(" ", out)
    # Collapse repeated spaces and newlines
    out = re.sub(r"\s+", " ", out).strip()
    # Truncate by character budget
    max_chars = max(1, max_tokens * _CHARS_PER_TOKEN)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] if " " in out[:max_chars] else out[:max_chars]
    return out.strip()
