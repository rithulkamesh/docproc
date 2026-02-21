"""LayoutLMv3-based Document Layout Analysis."""

import logging
from pathlib import Path
from typing import List, Optional

import fitz
from PIL import Image
import torch

from docproc.doc.dla.base import DLAEngine
from docproc.doc.regions import BoundingBox, Region, RegionType

logger = logging.getLogger(__name__)

# Map LayoutLMv3/FUNSD token labels to RegionType
# FUNSD labels: O, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER, B-HEADER, I-HEADER
LABEL_TO_REGION: dict[str, RegionType] = {
    "B-HEADER": RegionType.HEADER,
    "I-HEADER": RegionType.HEADER,
    "B-QUESTION": RegionType.TEXT,
    "I-QUESTION": RegionType.TEXT,
    "B-ANSWER": RegionType.TEXT,
    "I-ANSWER": RegionType.TEXT,
    "O": RegionType.TEXT,
}


class LayoutLMv3DLA(DLAEngine):
    """Document Layout Analysis using LayoutLMv3 token classification.

    Uses a fine-tuned LayoutLMv3 model (e.g., on FUNSD or DocLayNet)
    for layout-aware region detection. Falls back to PyMuPDF blocks
    if the model is unavailable.
    """

    def __init__(
        self,
        model_name: str = "nielsr/layoutlmv3-finetuned-funsd",
        device: Optional[str] = None,
        use_fallback: bool = True,
    ):
        """Initialize LayoutLMv3 DLA.

        Args:
            model_name: HuggingFace model ID for LayoutLMv3ForTokenClassification
            device: 'cuda', 'mps', or 'cpu'; auto-detected if None
            use_fallback: If True, fall back to PyMuPDF blocks when model fails
        """
        self.model_name = model_name
        self.use_fallback = use_fallback
        self._model = None
        self._processor = None
        self._id2label = None

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self._load_model()

    def _load_model(self) -> None:
        """Load LayoutLMv3 model and processor."""
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

            logger.info(f"Loading LayoutLMv3 DLA model: {self.model_name}")
            self._processor = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base", apply_ocr=True
            )
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._id2label = self._model.config.id2label
            logger.info("LayoutLMv3 DLA loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LayoutLMv3: {e}")
            self._model = None
            self._processor = None
            self._id2label = None

    def analyze_page(self, page_image, page_num: int = 0) -> List[Region]:
        """Analyze a single page using LayoutLMv3."""
        if self._model is None or self._processor is None:
            if self.use_fallback:
                return self._fallback_from_image(page_image, page_num)
            return []

        try:
            if hasattr(page_image, "numpy"):
                page_image = Image.fromarray(page_image)
            elif not isinstance(page_image, Image.Image):
                page_image = Image.fromarray(page_image)

            encoding = self._processor(
                page_image,
                return_tensors="pt",
                truncation=True,
                return_overflowing_tokens=False,
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self._model(**encoding)

            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            if isinstance(predictions, int):
                predictions = [predictions]

            # Get word_ids: processor may return BatchEncoding with word_ids
            word_ids_list = None
            if "word_ids" in encoding:
                wids = encoding["word_ids"]
                if isinstance(wids, (list, tuple)) and wids:
                    word_ids_list = wids[0] if isinstance(wids[0], (list, tuple)) else wids
            if word_ids_list is None or len(word_ids_list) != len(predictions):
                word_ids_list = list(range(len(predictions)))

            # Get bbox: one per token, shape [seq_len, 4]
            bboxes = None
            if "bbox" in encoding:
                b = encoding["bbox"]
                if hasattr(b, "cpu"):
                    bboxes = b[0].cpu().tolist() if b.dim() > 1 else b.cpu().tolist()
                else:
                    bboxes = b[0] if isinstance(b[0][0], (int, float)) else b
            if bboxes is None or len(bboxes) != len(predictions):
                bboxes = [[0, 0, 100, 100]] * len(predictions)

            # Group consecutive tokens with same label into regions
            return self._aggregate_regions(
                predictions, bboxes, word_ids_list, page_num
            )
        except Exception as e:
            logger.warning(f"LayoutLMv3 inference failed: {e}")
            if self.use_fallback:
                return self._fallback_from_image(page_image, page_num)
            return []

    def _aggregate_regions(
        self,
        predictions: List[int],
        bboxes: List[List[int]],
        word_ids: List,
        page_num: int,
    ) -> List[Region]:
        """Aggregate token predictions into Region objects."""
        regions: List[Region] = []
        prev_label_id = -1
        current_bboxes: List[List[int]] = []
        current_label_id = -1

        for i, (pred_id, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is None:  # Special tokens (CLS, SEP, PAD)
                if current_bboxes:
                    region = self._bboxes_to_region(
                        current_bboxes, current_label_id, page_num
                    )
                    if region:
                        regions.append(region)
                    current_bboxes = []
                    current_label_id = -1
                continue

            label = self._id2label.get(pred_id, "O")
            region_type = LABEL_TO_REGION.get(label, RegionType.TEXT)

            if i < len(bboxes):
                box = bboxes[i]
                # Merge consecutive same-type tokens
                if pred_id == current_label_id or current_label_id == -1:
                    current_bboxes.append(box)
                    current_label_id = pred_id
                else:
                    region = self._bboxes_to_region(
                        current_bboxes, current_label_id, page_num
                    )
                    if region:
                        regions.append(region)
                    current_bboxes = [box]
                    current_label_id = pred_id

        if current_bboxes:
            region = self._bboxes_to_region(
                current_bboxes, current_label_id, page_num
            )
            if region:
                regions.append(region)
        return regions

    def _bboxes_to_region(
        self, bboxes: List[List[int]], label_id: int, page_num: int
    ) -> Optional[Region]:
        """Convert list of token bboxes to a single Region."""
        if not bboxes:
            return None
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        label = self._id2label.get(label_id, "O")
        region_type = LABEL_TO_REGION.get(label, RegionType.TEXT)
        return Region(
            region_type=region_type,
            bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            confidence=0.9,
            content=None,
            metadata={"page_num": page_num, "dla_model": "layoutlmv3"},
        )

    def _fallback_from_image(self, page_image, page_num: int) -> List[Region]:
        """Fallback: use simple heuristics when model unavailable."""
        regions = []
        if hasattr(page_image, "size"):
            w, h = page_image.size
            regions.append(
                Region(
                    region_type=RegionType.TEXT,
                    bbox=BoundingBox(x1=0, y1=0, x2=float(w), y2=float(h)),
                    confidence=0.5,
                    content=None,
                    metadata={"page_num": page_num, "dla_fallback": True},
                )
            )
        return regions

    def _fallback_from_pdf(self, pdf_path: str | Path) -> List[Region]:
        """Fallback: use PyMuPDF blocks when model unavailable."""
        regions = []
        path = Path(pdf_path)
        if not path.exists():
            return regions
        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("blocks")
                for block in blocks:
                    x0, y0, x1, y1, text, *_ = block
                    if text.strip():
                        regions.append(
                            Region(
                                region_type=RegionType.TEXT,
                                bbox=BoundingBox(x1=x0, y1=y0, x2=x1, y2=y1),
                                confidence=0.7,
                                content=text.strip(),
                                metadata={"page_num": page_num},
                            )
                        )
        finally:
            doc.close()
        return regions

    def analyze_document(self, pdf_path: str | Path) -> List[Region]:
        """Analyze full PDF; uses model per page or PyMuPDF fallback."""
        path = Path(pdf_path)
        if not path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        if self._model is None:
            return self._fallback_from_pdf(path)

        regions = []
        doc = fitz.open(path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes(
                    "RGB", (pix.width, pix.height), pix.samples
                )
                page_regions = self.analyze_page(img, page_num)
                regions.extend(page_regions)
        finally:
            doc.close()
        return regions
