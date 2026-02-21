"""Vision LLM extractor - route all images to vision-capable LLMs for accurate extraction."""

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from docproc.providers.base import ChatMessage, ModelProvider

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analyze this document image and extract all meaningful content as structured data.

For each element you identify, provide:
1. **text**: Any readable text (paragraphs, captions, labels)
2. **equations**: Mathematical expressions in LaTeX
3. **tables**: Tabular data as markdown
4. **figures_descriptions**: Brief descriptions of diagrams, charts, or figures
5. **data_points**: Key numerical or categorical data if present

Respond with a single JSON object having these keys. Use null for absent fields.
Example:
{
  "text": "Main paragraph text...",
  "equations": ["E = mc^2", "x^2 + y^2 = r^2"],
  "tables": "| Col1 | Col2 |\\n|-----|-----|",
  "figures_descriptions": ["Bar chart showing sales 2020-2024"],
  "data_points": {"metric": "value"}
}"""


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
        self,
        images: List,
        contexts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract from multiple images (sequential; batch API could be added)."""
        results = []
        contexts = contexts or [None] * len(images)
        for img, ctx in zip(images, contexts):
            results.append(self.extract(img, context=ctx))
        return results
