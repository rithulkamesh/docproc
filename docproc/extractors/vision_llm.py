"""Vision LLM extractor - route all images to vision-capable LLMs for accurate extraction."""

import base64
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from docproc.providers.base import ChatMessage, ModelProvider

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract ALL content from this document page (lecture slide, textbook page, or diagram).

Include:
1. **text**: Every piece of text - headings, bullet points, paragraphs, captions, labels on figures
2. **equations**: Mathematical expressions in LaTeX
3. **tables**: Tabular data as markdown
4. **figures_descriptions**: What each diagram, chart, or figure shows
5. **data_points**: Key numbers, formulas, or values

Respond with a single JSON object having these keys. Use null for absent fields.
Example:
{
  "text": "Main paragraph text...",
  "equations": ["E = mc^2", "x^2 + y^2 = r^2"],
  "tables": "| Col1 | Col2 |\\n|-----|-----|",
  "figures_descriptions": ["Bar chart showing sales 2020-2024"],
  "data_points": {"metric": "value"}
}"""

IMAGE_PROMPT = """Describe this figure, diagram, or image from a document.
Include: any text visible in the image, what the figure shows (chart, diagram, photo), key labels, and any data or equations.
Respond with a single JSON object: {"text": "description..."}. Use null for absent."""

IMAGE_BATCH_PROMPT = """Describe each of the following %d figures/images from a document.
For each image include: any text visible, what the figure shows (chart, diagram, photo), key labels, and any data or equations.
Respond with a JSON array of exactly %d objects, one per image in order. Each object: {"text": "description..."}. Use null for absent.
Example: [{"text": "..."}, {"text": "..."}]"""


class VisionLLMExtractor:
    """Extract accurate data from document images using vision LLMs."""

    def __init__(
        self,
        provider: ModelProvider,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Initialize extractor.

        Args:
            provider: ModelProvider with chat_with_vision support
            model: Override model name; uses provider default if None
            prompt: Custom extraction prompt; uses default if None
        """
        self.provider = provider
        self.model = model
        self.prompt = prompt or EXTRACTION_PROMPT

    def _image_to_base64(self, image) -> str:
        """Convert image to base64 data URL."""
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
        if hasattr(image, "tobytes"):
            # numpy array
            pil = Image.fromarray(image)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
        if isinstance(image, bytes):
            b64 = base64.b64encode(image).decode()
            return f"data:image/png;base64,{b64}"
        raise ValueError(f"Unsupported image type: {type(image)}")

    def extract(self, image, context: Optional[str] = None) -> Dict[str, Any]:
        """Extract structured data from a single image.

        Args:
            image: PIL Image, numpy array, or bytes
            context: Optional context (e.g., page number, region type)

        Returns:
            Dict with keys: text, equations, tables, figures_descriptions, data_points
        """
        data_url = self._image_to_base64(image)
        content: List[dict] = [
            {"type": "text", "text": self.prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        if context:
            content.insert(0, {"type": "text", "text": f"Context: {context}\n\n"})
        messages = [ChatMessage(role="user", content=content)]
        resp = None
        try:
            resp = self.provider.chat_with_vision(messages, model=self.model)
            raw = resp.content.strip()
            # Try to parse JSON from response (model may wrap in markdown)
            if raw.startswith("```"):
                lines = raw.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                raw = "\n".join(json_lines)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Vision LLM returned non-JSON: {e}")
            fallback = resp.content if resp else ""
            return {"text": fallback, "equations": None, "tables": None, "figures_descriptions": None, "data_points": None}
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return {"text": None, "equations": None, "tables": None, "figures_descriptions": None, "data_points": None}

    def extract_batch(
        self, items: List[Tuple[Any, Optional[str]]], batch_prompt: Optional[str] = None
    ) -> List[str]:
        """Extract from multiple images in one API call. items = [(image, context), ...]. Returns list of description strings."""
        if not items:
            return []
        n = len(items)
        content: List[dict] = [
            {"type": "text", "text": (batch_prompt or IMAGE_BATCH_PROMPT % (n, n))},
        ]
        for image, _ in items:
            data_url = self._image_to_base64(image)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [ChatMessage(role="user", content=content)]
        try:
            resp = self.provider.chat_with_vision(messages, model=self.model)
            raw = resp.content.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                raw = "\n".join(json_lines)
            arr = json.loads(raw)
            if not isinstance(arr, list) or len(arr) != n:
                return [""] * n
            return [VisionLLMExtractor.flatten_extraction(x) if isinstance(x, dict) else str(x or "") for x in arr]
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Vision batch returned invalid JSON: %s", e)
            return [""] * n
        except Exception as e:
            logger.error("Vision batch failed: %s", e)
            return [""] * n

    @staticmethod
    def flatten_extraction(d: Dict[str, Any]) -> str:
        """Turn extraction dict into readable text for RAG indexing."""
        parts = []
        if d.get("text"):
            parts.append(str(d["text"]))
        if d.get("equations"):
            eqs = d["equations"] if isinstance(d["equations"], list) else [d["equations"]]
            parts.append("Equations: " + " ; ".join(str(e) for e in eqs if e))
        if d.get("tables"):
            parts.append(str(d["tables"]))
        if d.get("figures_descriptions"):
            figs = d["figures_descriptions"] if isinstance(d["figures_descriptions"], list) else [d["figures_descriptions"]]
            parts.append("Figures: " + " ; ".join(str(f) for f in figs if f))
        if d.get("data_points") and isinstance(d["data_points"], dict):
            parts.append("Data: " + str(d["data_points"]))
        return "\n".join(parts) if parts else ""


def extract_pdf_with_vision(path: Path, provider: ModelProvider) -> str:
    """Extract full text from PDF by rendering each page and sending to vision LLM."""
    import fitz

    extractor = VisionLLMExtractor(provider=provider)
    page_texts = []
    doc = fitz.open(path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            ctx = f"Page {page_num + 1} of {len(doc)}."
            try:
                result = extractor.extract(png_bytes, context=ctx)
                txt = VisionLLMExtractor.flatten_extraction(result)
                if txt.strip():
                    page_texts.append(f"[Page {page_num + 1}]\n{txt}")
            except Exception as e:
                logger.warning(f"Vision extraction failed for page {page_num + 1}: {e}")
    finally:
        doc.close()
    return "\n\n".join(page_texts)


def _extract_azure_vision_read(image_bytes: bytes, endpoint: str, key: str) -> Optional[str]:
    """OCR text from image using Azure Computer Vision Read API v3.2 (async).

    Extracts text inside images (equations, labels, etc.) - needed when text is
    rendered as graphics in PDFs.
    """
    import time
    import httpx

    if len(image_bytes) < 100:
        return None
    base = endpoint.rstrip("/")
    url = f"{base}/vision/v3.2/read/analyze"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}
    try:
        r = httpx.post(url, content=image_bytes, headers=headers, timeout=30.0)
        if r.status_code == 400:
            return None
        r.raise_for_status()
        op_location = r.headers.get("Operation-Location")
        if not op_location:
            return None
        for _ in range(30):
            time.sleep(1)
            r2 = httpx.get(op_location, headers={"Ocp-Apim-Subscription-Key": key}, timeout=10.0)
            r2.raise_for_status()
            data = r2.json()
            status = data.get("status", "")
            if status == "succeeded":
                lines = []
                for ar in data.get("analyzeResult", {}).get("readResults", []):
                    for line in ar.get("lines", []):
                        t = line.get("text", "").strip()
                        if t:
                            lines.append(t)
                return "\n".join(lines) if lines else None
            if status == "failed":
                break
        return None
    except Exception as e:
        logger.warning("Azure Vision Read failed: %s", e)
        return None


def _extract_azure_vision(image_bytes: bytes, endpoint: str, key: str) -> Optional[Dict[str, Any]]:
    """Extract text/OCR + caption from image using Azure AI Vision (Computer Vision) API.

    Uses Read API (OCR) for text-in-images (equations, labels); falls back to Describe
    for figures without text. Combines both for rich extraction.
    """
    import httpx

    base = endpoint.rstrip("/")
    # 1. Read API (OCR) - extracts text inside images (equations, labels, etc.)
    ocr_text = _extract_azure_vision_read(image_bytes, endpoint, key)
    # 2. Describe API - caption/tags for figures
    url = f"{base}/vision/v3.2/describe"
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}
    try:
        r = httpx.post(url, content=image_bytes, headers=headers, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        captions = data.get("description", {}).get("captions", [])
        tags = data.get("description", {}).get("tags", [])
        text_parts = []
        if ocr_text and ocr_text.strip():
            text_parts.append(ocr_text)
        text_parts.extend(c.get("text", "") for c in captions if c.get("text"))
        if tags:
            text_parts.append("Tags: " + ", ".join(str(t) for t in tags[:20]))
        return {"text": "\n".join(text_parts), "figures_descriptions": captions}
    except Exception as e:
        if ocr_text:
            return {"text": ocr_text, "figures_descriptions": None}
        logger.warning("Azure Vision API failed: %s (endpoint=%s)", e, url, exc_info=True)
        return None


def _content_hash(img_bytes: bytes) -> str:
    """SHA256 of image bytes for deduplication (logos, repeated watermarks)."""
    return hashlib.sha256(img_bytes).hexdigest()


def _extract_single_image(
    item: tuple,
    use_azure_vision: bool,
    azure_vision_endpoint: str,
    azure_vision_key: str,
    llm_extractor: Optional[VisionLLMExtractor],
) -> tuple[int, int, Optional[str]]:
    """Extract from one image. Returns (page_num, img_idx, text or None)."""
    page_num, img_idx, img_bytes, ctx = item
    if not img_bytes:
        return page_num, img_idx, None
    result = None
    if use_azure_vision:
        result = _extract_azure_vision(img_bytes, azure_vision_endpoint, azure_vision_key)
    if result is None and llm_extractor:
        try:
            result = llm_extractor.extract(img_bytes, context=ctx)
        except Exception as e:
            logger.warning("Vision extraction failed for page %s image %s: %s", page_num + 1, img_idx + 1, e)
    if result:
        txt = VisionLLMExtractor.flatten_extraction(result)
        if txt.strip():
            return page_num, img_idx, txt.strip()
    return page_num, img_idx, None


def _extract_unique_image_azure_or_llm(
    hash_img_ctx: Tuple[str, bytes, str],
    use_azure_vision: bool,
    azure_vision_endpoint: str,
    azure_vision_key: str,
    llm_extractor: Optional[VisionLLMExtractor],
) -> Tuple[str, Optional[str]]:
    """Extract from one unique image (by hash). Returns (hash, text). Used for deduplicated extraction."""
    img_hash, img_bytes, ctx = hash_img_ctx
    if not img_bytes:
        return img_hash, None
    result = None
    if use_azure_vision:
        result = _extract_azure_vision(img_bytes, azure_vision_endpoint, azure_vision_key)
    if result is None and llm_extractor:
        try:
            result = llm_extractor.extract(img_bytes, context=ctx)
        except Exception as e:
            logger.warning("Vision extraction failed for hash %s: %s", img_hash[:8], e)
    if result:
        txt = VisionLLMExtractor.flatten_extraction(result)
        if txt.strip():
            return img_hash, txt.strip()
    return img_hash, None


# Batch size for vision LLM (4-5 images per API call)
VISION_BATCH_SIZE = 5
# How many batch requests to run in parallel
VISION_BATCH_PARALLEL = 3


def extract_pdf_text_and_images(
    path: Path,
    provider: Optional[ModelProvider] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Extract full text from PDF: fast text layer + vision only for embedded images.

    Uses native PDF text from load_document (instant). For images, tries:
    1. Azure AI Vision (Computer Vision) if AZURE_VISION_ENDPOINT is set
    2. Vision LLM (chat_with_vision) if provider given

    Deduplicates by content hash (same logo/image on many pages = one extraction).
    LLM path batches 4-5 images per API call to speed up.

    progress_callback(page_idx, total_pages, message) called during processing.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from docproc.doc.loaders import load_document

    azure_vision_endpoint = os.getenv("AZURE_VISION_ENDPOINT", "").strip()
    azure_vision_key = (
        os.getenv("AZURE_VISION_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    use_azure_vision = bool(azure_vision_endpoint and azure_vision_key)
    llm_extractor = VisionLLMExtractor(provider=provider, prompt=IMAGE_PROMPT) if provider else None

    pages = list(load_document(path))
    total = len(pages)
    if progress_callback:
        progress_callback(0, total, "Loading pages…")

    # Build (page_num, img_idx, img_bytes, ctx) and dedupe by content hash
    hash_to_rep: Dict[str, Tuple[bytes, str]] = {}
    page_figure_order: Dict[int, List[Tuple[int, str]]] = {p.page_num: [] for p in pages}
    for page in pages:
        for i, img_bytes in enumerate(page.raw_images or []):
            if img_bytes:
                h = _content_hash(img_bytes)
                if h not in hash_to_rep:
                    hash_to_rep[h] = (img_bytes, f"Page {page.page_num + 1}, image {i + 1}.")
                page_figure_order[page.page_num].append((i, h))

    unique_list = [(h, img_bytes, ctx) for h, (img_bytes, ctx) in hash_to_rep.items()]
    num_total_refs = sum(len(v) for v in page_figure_order.values())
    if num_total_refs > len(unique_list):
        logger.info(
            "Deduplicating images: %d references → %d unique (skipping %d repeats e.g. logos)",
            num_total_refs,
            len(unique_list),
            num_total_refs - len(unique_list),
        )
    fig_by_page: Dict[int, List[str]] = {p.page_num: [] for p in pages}
    hash_to_text: Dict[str, str] = {}

    if not unique_list:
        pass
    elif llm_extractor and not use_azure_vision:
        # LLM path: batch 4-5 images per request, multiple batches in parallel
        batch_size = min(VISION_BATCH_SIZE, len(unique_list))
        num_batches = (len(unique_list) + batch_size - 1) // batch_size
        chunks = [unique_list[b * batch_size : (b + 1) * batch_size] for b in range(num_batches)]

        def run_batch(chunk: List[Tuple[str, bytes, str]]) -> List[Tuple[str, str]]:
            out: List[Tuple[str, str]] = []
            try:
                texts = llm_extractor.extract_batch([(img_bytes, ctx) for _, img_bytes, ctx in chunk])
                for i, (h, _, _) in enumerate(chunk):
                    if i < len(texts) and texts[i].strip():
                        out.append((h, texts[i].strip()))
            except Exception as e:
                logger.warning("Vision batch failed: %s", e)
            return out

        max_workers = min(VISION_BATCH_PARALLEL, len(chunks))
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_batch, chunk): chunk for chunk in chunks}
            for fut in as_completed(futures):
                done += 1
                if progress_callback and total > 0:
                    half = max(0, (total - 1) // 2)
                    pct = int(half * done / num_batches) if num_batches else 0
                    progress_callback(
                        min(pct, half),
                        total,
                        f"Extracting images… batch {done}/{num_batches}",
                    )
                for h, txt in fut.result():
                    hash_to_text[h] = txt
    else:
        # Azure or fallback: one request per unique image (parallel)
        max_workers = min(6, max(1, len(unique_list)))
        result_timeout = 120.0
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _extract_unique_image_azure_or_llm,
                    (h, img_bytes, ctx),
                    use_azure_vision,
                    azure_vision_endpoint,
                    azure_vision_key,
                    llm_extractor,
                ): (h, img_bytes, ctx)
                for h, img_bytes, ctx in unique_list
            }
            for fut in as_completed(futures):
                done += 1
                if progress_callback and total > 0:
                    half = max(0, (total - 1) // 2)
                    pct = int(half * done / len(unique_list)) if unique_list else 0
                    progress_callback(
                        min(pct, half),
                        total,
                        f"Extracting images… {done}/{len(unique_list)} unique",
                    )
                try:
                    h, txt = fut.result(timeout=result_timeout)
                    if txt:
                        hash_to_text[h] = txt
                except TimeoutError:
                    logger.warning("Image extraction timed out after %.0fs (skipped)", result_timeout)
                except Exception as e:
                    logger.warning("Image extraction failed: %s", e)

    # Assemble fig_by_page from page order and hash results
    for page_num in page_figure_order:
        for img_idx, h in sorted(page_figure_order[page_num], key=lambda x: x[0]):
            if h in hash_to_text:
                fig_by_page[page_num].append(hash_to_text[h])

    # Rasterized-page fallback: PDFs from note-taking apps (Notability, GoodNotes, etc.)
    # store each page as a full-page bitmap with no text layer and no embedded image xrefs.
    # Detect those pages and render them as pixmaps for vision extraction.
    empty_page_nums = [
        page.page_num for page in pages
        if not (page.text and page.text.strip()) and not page.raw_images
    ]
    if empty_page_nums and (llm_extractor or use_azure_vision):
        import fitz as _fitz
        page_extractor = VisionLLMExtractor(provider=provider) if provider else None
        _doc = _fitz.open(path)
        try:
            for page_num in empty_page_nums:
                fitz_page = _doc[page_num]
                pix = fitz_page.get_pixmap(dpi=150, alpha=False)
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                ctx = f"Page {page_num + 1} of {total}."
                result = None
                if use_azure_vision:
                    result = _extract_azure_vision(png_bytes, azure_vision_endpoint, azure_vision_key)
                if result is None and page_extractor:
                    try:
                        result = page_extractor.extract(png_bytes, context=ctx)
                    except Exception as e:
                        logger.warning("Vision extraction failed for rasterized page %d: %s", page_num + 1, e)
                if result:
                    txt = VisionLLMExtractor.flatten_extraction(result)
                    if txt.strip():
                        fig_by_page[page_num].append(txt.strip())
        finally:
            _doc.close()

    page_parts = []
    half = max(0, (total - 1) // 2)
    for idx, page in enumerate(pages):
        if progress_callback:
            # Second half of bar = page assembly (half .. total-1)
            pct = half + int((total - 1 - half) * (idx + 1) / total) if total else half
            progress_callback(min(pct, total - 1), total, f"Page {idx + 1}/{total}")
        part = page.text.strip() if page.text else ""
        fig_descs = fig_by_page.get(page.page_num, [])
        if fig_descs:
            part += "\n[Figures: " + " ; ".join(fig_descs) + "]"
        if part:
            page_parts.append(f"[Page {page.page_num + 1}]\n{part}")
    if page_parts:
        return "\n\n".join(page_parts)
    from docproc.doc.loaders import get_full_text
    return get_full_text(path)
