"""Document upload and retrieval. Supports PDF, DOCX, PPTX, XLSX."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from docproc.doc.loaders import load_document, get_supported_extensions
from docproc.sanitize import sanitize_text, deduplicate_texts

router = APIRouter()

_documents: dict = {}

SUPPORTED = tuple(get_supported_extensions())


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, PPTX, XLSX) and process it."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(
            400,
            f"Unsupported format. Supported: {', '.join(sorted(SUPPORTED))}",
        )
    content = await file.read()
    doc_id = str(uuid.uuid4())
    _documents[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "status": "pending",
        "regions": [],
        "pages": 0,
    }
    try:
        tmp = Path(f"/tmp/docproc_{doc_id}{ext}")
        tmp.write_bytes(content)
        all_regions = []
        page_count = 0
        for page in load_document(tmp):
            page_count += 1
            for r in page.regions:
                if r.content:
                    c = sanitize_text(r.content)
                    if c:
                        r.content = c
                        all_regions.append(r)
        tmp.unlink(missing_ok=True)
        texts = [r.content for r in all_regions]
        unique_texts = deduplicate_texts(texts)
        seen = set()
        for r in all_regions:
            if r.content in unique_texts and r.content not in seen:
                seen.add(r.content)
                _documents[doc_id]["regions"].append({
                    "region_type": r.region_type.name,
                    "bbox": {"x1": r.bbox.x1, "y1": r.bbox.y1, "x2": r.bbox.x2, "y2": r.bbox.y2},
                    "content": r.content,
                    "confidence": r.confidence,
                    "metadata": r.metadata or {},
                })
        _documents[doc_id]["pages"] = page_count
        _documents[doc_id]["status"] = "completed"
    except Exception as e:
        _documents[doc_id]["status"] = "failed"
        _documents[doc_id]["error"] = str(e)
    return {"id": doc_id, "status": _documents[doc_id]["status"]}


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get document status, metadata, and detected regions."""
    if document_id not in _documents:
        raise HTTPException(404, "Document not found")
    return _documents[document_id]
