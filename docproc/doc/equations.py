import re
from typing import Set
import fitz
from PIL import Image
import io
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging
import cv2
import os

from docproc.doc.regions import Region

logger = logging.getLogger(__name__)


class EquationParser:
    """Takes an equation instance and passes it through IBM Granite 3.2 vision model to convert into LaTeX format"""

    def __init__(self):
        """Initialize the class with the IBM Granite 3.2 model."""
        self._cache = {}
        self.model = None
        self.processor = None
        self._loading_in_progress = False
        self._device = None
        self._instance = None  # Singleton pattern
        # Set a reasonable max image dimension to prevent memory issues
        self._max_image_dim = 768

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if not hasattr(cls, "_instance") or cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _check_gpu_safely(self):
        """Safely check if GPU is available without crashing on NixOS."""
        try:
            import torch
            import os

            # Check for environment variable to force CPU usage
            if os.environ.get("DOCPROC_FORCE_CPU", "").lower() in ("1", "true", "yes"):
                logger.info("GPU usage disabled by environment variable")
                return False

            # First, check if CUDA is available according to PyTorch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                logger.info("CUDA reported as not available by PyTorch")
                return False

            # Check available GPU memory - only use GPU if enough memory is available
            try:
                free_memory = torch.cuda.mem_get_info(0)[0] / (
                    1024**3
                )  # Free memory in GB
                logger.info(f"Free GPU memory: {free_memory:.2f} GB")
                if (
                    free_memory < 0.5
                ):  # Require at least 0.5GB free memory instead of 2GB
                    logger.info(
                        f"Not enough GPU memory available ({free_memory:.2f}GB < 0.5GB)"
                    )
                    return False
            except Exception as e:
                logger.warning(f"Failed to check GPU memory: {e}")

            # Get device count and name
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Found GPU: {device_name}")

                # Try to initialize tensor on GPU to verify it works
                try:
                    test_tensor = torch.tensor([1.0], device="cuda:0")
                    del test_tensor
                    return True
                except Exception as e:
                    logger.warning(f"CUDA device found but failed initialization: {e}")
                    return False
            else:
                logger.info("No CUDA devices found by PyTorch")
                return False

        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False

    def _lazy_load_model(self):
        """Load the model only when needed to save memory and startup time."""
        if self.model is None and not self._loading_in_progress:
            try:
                self._loading_in_progress = True
                logger.info("Loading IBM Granite 3.2 model for equation recognition...")

                # Import torch only when needed
                import torch

                # Safely check for GPU availability
                use_gpu = self._check_gpu_safely()

                # Set memory efficient settings
                if use_gpu and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                        "expandable_segments:True,max_split_size_mb:128"
                    )
                    logger.info(
                        "Set PYTORCH_CUDA_ALLOC_CONF for better memory management"
                    )

                if use_gpu:
                    self._device = (
                        "cuda:0"  # Explicitly specify cuda:0 instead of generic cuda
                    )
                    # Set CUDA device properties to optimize performance
                    torch.backends.cudnn.benchmark = True
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self._device = "cpu"
                    logger.info(
                        "Using CPU for equation processing (no GPU available or disabled)"
                    )

                # Load model components
                model_name = "ibm-granite/granite-vision-3.2-2b"

                # Use fast processor explicitly
                self.processor = AutoProcessor.from_pretrained(
                    model_name, use_fast=True
                )

                # Load model with optimized settings
                if self._device == "cuda:0":
                    try:
                        # Set explicit device map to cuda:0
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            device_map="cuda:0",  # Use explicit device instead of "auto"
                            torch_dtype=torch.float16,  # Use half precision to save GPU memory
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load model on GPU, falling back to CPU: {e}"
                        )
                        self._device = "cpu"
                        # Free GPU memory before loading on CPU
                        torch.cuda.empty_cache()
                        self.model = AutoModelForVision2Seq.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                        )
                else:
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                    )

                # Ensure model is in evaluation mode
                self.model.eval()

                # Clear CUDA cache to free up memory
                if self._device == "cuda:0":
                    torch.cuda.empty_cache()

                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self.model = None
                self.processor = None
                raise
            finally:
                self._loading_in_progress = False

    def _preprocess_image(self, img_array):
        """
        Preprocess image to make it suitable for the model.

        Args:
            img_array (np.ndarray): Input image array

        Returns:
            PIL.Image: Processed image ready for model input
        """
        # Make sure we have a 3-channel RGB image
        if len(img_array.shape) == 2:
            # Convert grayscale to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            # Convert RGBA to RGB
            img_array = img_array[:, :, :3]

        # Ensure the image isn't too small - avoid the "tokens: 0, features" error
        min_size = 32  # Minimum size to prevent token mismatch errors
        h, w = img_array.shape[:2]
        if h < min_size or w < min_size:
            # Scale up small images to prevent token mismatch issues
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_array = cv2.resize(
                img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )
            logger.info(
                f"Scaled up small equation image from {h}x{w} to {new_h}x{new_w}"
            )

        # Resize large images to reduce memory usage
        h, w = img_array.shape[:2]
        max_dim = self._max_image_dim  # Maximum dimension to limit memory usage
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_array = cv2.resize(
                img_array, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            logger.info(
                f"Resized equation image from {h}x{w} to {new_h}x{new_w} to save memory"
            )

        # Ensure consistent dimensions (model often expects multiples of 8 or 16)
        h, w = img_array.shape[:2]
        if h % 16 != 0 or w % 16 != 0:
            new_h = ((h + 15) // 16) * 16
            new_w = ((w + 15) // 16) * 16
            # Create a padded image with black background
            padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            padded[:h, :w] = img_array
            img_array = padded

        # Apply adaptive enhancement for better contrast
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(enhanced_rgb)

        # Verify the image is in a format compatible with the model
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return pil_img

    def parse_equation(self, region: Region, page: fitz.Page) -> str:
        """Parse an equation region using IBM Granite 3.2.

        Args:
            region (Region): Region containing the equation image
            page (fitz.Page): PDF page containing the region

        Returns:
            str: LaTeX expression of the equation
        """
        cache_key = f"{page.number}_{region.bbox}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Ensure model is loaded
            self._lazy_load_model()
            if self.model is None or self.processor is None:
                return "[OCR ERROR: Model not loaded]"

            # Extract image from PDF
            bbox = region.bbox
            x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            pix = page.get_pixmap(clip=(x1, y1, x2, y2), matrix=fitz.Matrix(1, 1))

            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )

            # Preprocess the image
            img = self._preprocess_image(img_array)

            # Create prompt for the model
            prompt = "This image contains a mathematical equation. Convert it to LaTeX format."

            # Process with Granite model - handle device carefully
            import torch

            with torch.no_grad():  # Reduce memory usage during inference
                try:
                    # Process the inputs
                    inputs = self.processor(
                        text=prompt,
                        images=img,
                        return_tensors="pt",
                        do_convert_rgb=True,
                    )

                    # Validate inputs to catch the specific error early
                    if "pixel_values" not in inputs:
                        raise ValueError(
                            "Image processing failed, no pixel values generated"
                        )

                    if (
                        "image_tensors" in inputs
                        and inputs["image_tensors"].shape[0] == 0
                    ):
                        # Fix for the "Image features and image tokens do not match" error
                        raise ValueError(
                            "Empty image tensors detected, image conversion failed"
                        )

                    # Move all tensor inputs to the correct device
                    inputs = {
                        k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }

                    # Generate with reduced parameters to save memory
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,  # Reduced to save memory
                        num_beams=2,  # Reduced beam search
                        length_penalty=1.0,
                    )

                    # Always move to CPU for processing
                    generated_ids = generated_ids.cpu()
                    latex_expression = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                except (RuntimeError, ValueError) as e:
                    if "CUDA out of memory" in str(e) and self._device == "cuda:0":
                        # If we run out of CUDA memory, try again on CPU
                        logger.warning(
                            f"CUDA OOM error, falling back to CPU for this equation: {e}"
                        )
                        self._device = "cpu"  # Temporarily switch to CPU

                        # Move model to CPU
                        self.model = self.model.to("cpu")
                        torch.cuda.empty_cache()

                        # Reprocess on CPU
                        inputs = self.processor(
                            text=prompt,
                            images=img,
                            return_tensors="pt",
                            do_convert_rgb=True,
                        )
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            num_beams=1,  # Reduce to 1 on CPU to save time
                            length_penalty=1.0,
                        )
                        latex_expression = self.processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )[0]
                    elif "Image features and image tokens do not match" in str(e):
                        # Handle the specific error by trying a different approach
                        logger.warning(
                            f"Image processing error, trying alternative method: {e}"
                        )

                        # Try with a different approach - convert to RGB PIL image directly
                        if isinstance(img, np.ndarray):
                            img = Image.fromarray(img_array).convert("RGB")
                        else:
                            img = img.convert("RGB")

                        # Resize to ensure it's not too large
                        current_w, current_h = img.size
                        if current_w > 800 or current_h > 800:
                            scale = min(800 / current_w, 800 / current_h)
                            new_w, new_h = int(current_w * scale), int(
                                current_h * scale
                            )
                            img = img.resize((new_w, new_h), Image.LANCZOS)

                        # Try with simplified processing
                        inputs = self.processor(
                            text=prompt,
                            images=img,
                            return_tensors="pt",
                            do_resize=True,
                            do_convert_rgb=True,
                        )

                        if self._device == "cuda:0":
                            inputs = {
                                k: (
                                    v.to(self._device)
                                    if isinstance(v, torch.Tensor)
                                    else v
                                )
                                for k, v in inputs.items()
                            }

                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            num_beams=1,  # Use simple beam search for reliability
                        )
                        latex_expression = self.processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )[0]
                    else:
                        # Re-raise if it's not a handled error
                        raise

            # Clean up the result to extract just the LaTeX part
            latex_expression = self._clean_latex_output(latex_expression)

            # Cache the result
            self._cache[cache_key] = latex_expression

            # Periodically clear CUDA cache to prevent memory buildup
            if (
                self._device == "cuda:0" and len(self._cache) % 5 == 0
            ):  # More frequent cleanup
                torch.cuda.empty_cache()

            return latex_expression

        except Exception as e:
            error_msg = f"Error parsing equation: {str(e)}"
            logger.error(error_msg)
            return f"[OCR ERROR: {str(e)}]"

    def _clean_latex_output(self, text: str) -> str:
        """Clean the model output to extract proper LaTeX.

        Args:
            text (str): Raw model output

        Returns:
            str: Cleaned LaTeX expression
        """
        # Extract content between $ or $$ if present
        dollar_match = re.search(r"\$(.*?)\$", text)
        if dollar_match:
            return dollar_match.group(1)

        double_dollar_match = re.search(r"\$\$(.*?)\$\$", text)
        if double_dollar_match:
            return double_dollar_match.group(1)

        # If the model outputs "The LaTeX representation is..." or similar
        latex_intro_patterns = [
            r"the latex (?:representation|expression|formula) (?:is|would be)[:\s]+(.+)",
            r"in latex[:\s]+(.+)",
            r"latex[:\s]+(.+)",
        ]

        for pattern in latex_intro_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()

        # If none of the patterns match, return the original text
        return text.strip()


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
