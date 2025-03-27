import re
from typing import Set
from docproc.doc.regions import Region
from rapid_latex_ocr import LaTeXOCR as LatexOCR
import fitz
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class EquationParser:
    """Takes an equation instance and passes it through a LaTeX OCR Parser to convert into a machine (and human) readable format"""

    def __init__(self, max_workers=4):
        """Initialize the class with a LaTeX OCR model instance."""
        self.model = LatexOCR()
        self._cache = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

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
            # Get pixmap with lower DPI if image is large
            pix = page.get_pixmap(clip=(x1, y1, x2, y2), matrix=fitz.Matrix(1, 1))
            # Use numpy for faster image conversion
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            img = Image.fromarray(img_array)
            # Save image to bytes using BytesIO efficiently
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG", optimize=True)
            img_bytes = img_byte_arr.getvalue()

            # Call the OCR model only once and cache its result
            latex_expression = self.model(img_bytes)[0]
            self._cache[cache_key] = latex_expression
            return latex_expression

        except Exception as e:
            raise Exception(f"Error parsing equation: {str(e)}")

    def parse_equations_batch(self, regions, page):
        """Parse multiple equations in parallel"""
        return list(self.executor.map(lambda r: self.parse_equation(r, page), regions))


class UnicodeMathDetector:
    """Detector for mathematical content based on Unicode character patterns."""

    # Constant definitions
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
        """Initialize the detector with precomputed character sets and precompiled regex patterns."""
        # Precompute math_chars only once since the constants never change
        self.math_chars = self._get_all_math_chars()
        # Precompile regex patterns
        math_chars_str = "".join(self.math_chars)
        self.math_sequence = re.compile(f"[{re.escape(math_chars_str)}]{{2,}}")
        self.superscript_pattern = re.compile(
            r"[\u2070\u00b9\u00b2\u00b3\u2074-\u207f]+"
        )
        self.subscript_pattern = re.compile(r"[\u2080-\u208e]+")

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

        Looks for:
        - Fraction characters
        - Sequences of math symbols
        - Superscript/subscript numbers

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if mathematical patterns are found
        """
        # Check for fractions
        if any(frac in text for frac in self.FRACTIONS):
            return True
        # Check for sequences of math symbols using precompiled regex
        if self.math_sequence.search(text):
            return True
        # Check for superscript/subscript numbers via regex
        if self.superscript_pattern.search(text) or self.subscript_pattern.search(text):
            return True
        return False
