"""Document Layout Analysis (DLA) module.

Provides ML-based region detection with 90%+ accuracy.
"""

from docproc.doc.dla.base import DLAEngine
from docproc.doc.dla.layoutlmv3 import LayoutLMv3DLA

__all__ = ["DLAEngine", "LayoutLMv3DLA"]
