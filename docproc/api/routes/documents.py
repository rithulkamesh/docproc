"""Document upload and retrieval. Supports PDF, DOCX, PPTX, XLSX."""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from docproc.doc.loaders import load_document, get_supported_extensions
from docproc.sanitize import sanitize_text, deduplicate_texts

router = APIRouter()

_documents: dict = {}

SUPPORTED = tuple(get_supported_extensions())


def _run_extraction(doc_id: str, tmp: Path, ext: str):
    """Background task: extract text and regions, update _documents."""
    doc = _documents.get(doc_id)
    if not doc:
        return
    doc["status"] = "processing"

    def progress(page: int, total: int, message: str):
        d = _documents.get(doc_id)
        if d:
            d["progress"] = {"page": page, "total": total, "message": message}

    try:
        from docproc.pipeline import extract_document_to_text

        full_text = extract_document_to_text(tmp, progress_callback=progress)
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
                doc["regions"].append({
                    "region_type": r.region_type.name,
                    "bbox": {"x1": r.bbox.x1, "y1": r.bbox.y1, "x2": r.bbox.x2, "y2": r.bbox.y2},
                    "content": r.content,
                    "confidence": r.confidence,
                    "metadata": r.metadata or {},
                })
        doc["full_text"] = full_text
        doc["pages"] = page_count
        doc["status"] = "completed"
        doc.pop("progress", None)
        try:
            from docproc.rag.factory import get_rag
            rag = get_rag()
            if rag is not None and full_text and full_text.strip():
                rag.index(documents=[full_text], document_ids=[doc_id])
        except Exception:
            pass
    except Exception as e:
        doc["status"] = "failed"
        doc["error"] = str(e)
        doc.pop("progress", None)
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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
    tmp = Path(f"/tmp/docproc_{doc_id}{ext}")
    tmp.write_bytes(content)
    _documents[doc_id] = {
        "id": doc_id,
        "filename": file.filename,
        "status": "processing",
        "progress": {"page": 0, "total": 1, "message": "Starting…"},
        "regions": [],
        "full_text": "",
        "pages": 0,
    }
    background_tasks.add_task(_run_extraction, doc_id, tmp, ext)
    return {"id": doc_id, "status": "processing"}


@router.get("")
@router.get("/")
async def list_documents():
    """List all uploaded documents (metadata only, no full_text)."""
    docs = []
    for d in _documents.values():
        docs.append({
            "id": d["id"],
            "filename": d["filename"],
            "status": d["status"],
            "pages": d.get("pages", 0),
        })
    return {"documents": docs}


@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get document status, metadata, full text, and regions."""
    if document_id not in _documents:
        raise HTTPException(404, "Document not found")
    return _documents[document_id]
