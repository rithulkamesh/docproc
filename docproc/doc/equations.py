import re
from typing import Set
from docproc.doc.regions import Region
from rapid_latex_ocr import LaTeXOCR as LatexOCR
import fitz


class EquationParser:
    """Takes an equation instance and passes it through a LaTeX OCR Parser to convert into a machine (and human) readable format"""

    def __init__(self):
        """Initialize the class with a LaTeX OCR model instance."""
        self.model = LatexOCR()
        self._cache = {}

    def parse_equation(self, region: Region, page: fitz.Page) -> str:
        """Parse an equation region using LaTeX OCR.

        Args:
            region (Region): Region containing the equation image

        Returns:
            str: LaTeX expression of the equation
        """
        cache_key = f"{page.number}_{region.bbox}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            bbox = region.bbox
            x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))

            pix = page.get_pixmap(clip=(x1, y1, x2, y2))
            from PIL import Image
            import io

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            latex_expression = self.model(img_bytes)
            self._cache[cache_key] = latex_expression
            return latex_expression

        except Exception as e:
            raise Exception(f"Error parsing equation: {str(e)}")


class UnicodeMathDetector:
    """Detector for mathematical content based on Unicode character patterns."""

    # Unicode ranges for mathematical symbols
    MATH_RANGES = [
        (0x2200, 0x22FF),  # Mathematical Operators
        (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
        (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
        (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
        (0x2100, 0x214F),  # Letterlike Symbols
        (0x2150, 0x218F),  # Number Forms
        (0x2190, 0x21FF),  # Arrows
        (0x0391, 0x03C9),  # Greek Letters
        (0x2080, 0x209C),  # Subscripts
        (0x2070, 0x207F),  # Superscripts
        (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    ]

    # Common mathematical symbols that might appear as regular Unicode
    COMMON_MATH_CHARS = {
        "=",
        "+",
        "-",
        "×",
        "÷",
        "±",
        "∓",
        "∑",
        "∏",
        "∫",
        "∬",
        "∭",
        "∮",
        "∯",
        "∰",
        "∇",
        "∆",
        "√",
        "∛",
        "∜",
        "∝",
        "∞",
        "∟",
        "∠",
        "∡",
        "∢",
        "∥",
        "∦",
        "∧",
        "∨",
        "∩",
        "∪",
        "∈",
        "∉",
        "∊",
        "∋",
        "∌",
        "∍",
        "∎",
        "∏",
        "∐",
        "∑",
        "∓",
        "∔",
        "∕",
        "∖",
        "∗",
        "∘",
        "∙",
        "√",
        "∛",
        "∜",
        "∝",
        "∞",
        "∟",
        "∠",
        "∡",
        "∢",
        "∂",
        "∃",
        "∄",
        "∅",
        "∆",
        "∇",
        "∈",
        "∉",
        "∊",
        "∋",
        "∌",
        "∍",
        "∎",
        "∏",
        "≠",
        "≈",
        "≡",
        "≤",
        "≥",
        "⊂",
        "⊃",
        "⊄",
        "⊅",
        "⊆",
        "⊇",
        "⊈",
        "⊉",
        "⊊",
        "⊋",
        "⊌",
        "⊍",
        "⊎",
        "⊏",
        "⊐",
        "⊑",
        "⊒",
        "⊓",
        "⊔",
        "⊕",
        "⊖",
        "⊗",
        "⊘",
        "⊙",
        "⊚",
        "⊛",
        "⊜",
        "⊝",
        "⊞",
        "⊟",
        "⊠",
        "⊡",
        "⊢",
        "⊣",
        "⊤",
        "⊥",
        "⊦",
        "⊧",
        "⊨",
        "⊩",
        "⊪",
        "⊫",
        "⊬",
        "⊭",
        "⊮",
        "⊯",
        "⊰",
        "⊱",
        "⊲",
        "⊳",
        "⊴",
        "⊵",
        "⊶",
        "⊷",
        "⊸",
        "⊹",
        "⊺",
        "⊻",
        "⊼",
        "⊽",
        "⊾",
        "⊿",
        "⋀",
        "⋁",
        "⋂",
        "⋃",
        "⋄",
        "⋅",
        "⋆",
        "⋇",
        "⋈",
        "⋉",
        "⋊",
        "⋋",
        "⋌",
        "⋍",
        "⋎",
        "⋏",
        "⋐",
        "⋑",
        "⋒",
        "⋓",
        "⋔",
        "⋕",
        "⋖",
        "⋗",
        "⋘",
        "⋙",
        "⋚",
        "⋛",
        "⋜",
        "⋝",
        "⋞",
        "⋟",
        "⋠",
        "⋡",
        "⋢",
        "⋣",
        "⋤",
        "⋥",
        "⋦",
        "⋧",
        "⋨",
        "⋩",
        "⋪",
        "⋫",
        "⋬",
        "⋭",
        "⋮",
        "⋯",
        "⋰",
        "⋱",
        "⋲",
        "⋳",
        "⋴",
        "⋵",
        "⋶",
        "⋷",
        "⋸",
        "⋹",
        "⋺",
        "⋻",
        "⋼",
        "⋽",
        "⋾",
        "⋿",
    }

    # Fractions and other number forms
    FRACTIONS = {
        "½",
        "⅓",
        "⅔",
        "¼",
        "¾",
        "⅕",
        "⅖",
        "⅗",
        "⅘",
        "⅙",
        "⅚",
        "⅛",
        "⅜",
        "⅝",
        "⅞",
        "⅟",
        "↉",
    }

    def __init__(self):
        """Initialize the detector with precomputed character sets."""
        self.math_chars = self._get_all_math_chars()

    def _get_all_math_chars(self) -> Set[str]:
        """Get a set of all mathematical Unicode characters."""
        chars = set()

        # Add characters from Unicode ranges
        for start, end in self.MATH_RANGES:
            for code_point in range(start, end + 1):
                try:
                    chars.add(chr(code_point))
                except ValueError:
                    continue

        # Add common math characters and fractions
        chars.update(self.COMMON_MATH_CHARS)
        chars.update(self.FRACTIONS)

        return chars

    def calculate_math_density(self, text: str) -> float:
        """Calculate the density of mathematical symbols in the text.

        Args:
            text (str): Text to analyze

        Returns:
            float: Ratio of mathematical symbols to total characters
        """
        if not text:
            return 0.0

        math_char_count = sum(1 for char in text if char in self.math_chars)
        return math_char_count / len(text)

    def has_math_pattern(self, text: str) -> bool:
        """Check for common mathematical patterns in the text.

        Looks for patterns like:
        - Multiple math symbols in close proximity
        - Superscript/subscript numbers
        - Fraction characters
        - Sequences of operators

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if mathematical patterns are found
        """
        # Check for fractions
        if any(frac in text for frac in self.FRACTIONS):
            return True

        # Look for sequences of math symbols
        math_chars_str = "".join(self.math_chars)
        math_sequence = re.compile(f"[{re.escape(math_chars_str)}]{{2,}}")
        if math_sequence.search(text):
            return True

        # Check for superscript/subscript numbers
        superscript_pattern = re.compile("[\u2070\u00b9\u00b2\u00b3\u2074-\u207f]+")
        subscript_pattern = re.compile("[\u2080-\u208e]+")
        if superscript_pattern.search(text) or subscript_pattern.search(text):
            return True

        return False
