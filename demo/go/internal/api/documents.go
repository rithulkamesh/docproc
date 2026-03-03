package api

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"path"
	"strings"

	"github.com/google/uuid"
	"github.com/rithulkamesh/docproc/demo/internal/blob"
	"github.com/rithulkamesh/docproc/demo/internal/db"
	"github.com/rithulkamesh/docproc/demo/internal/mq"
)

// Document routes: upload, list, get, delete, reindex
func (h *Handler) documents(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/documents")
	path = strings.Trim(path, "/")
	parts := strings.Split(path, "/")

	switch r.Method {
	case http.MethodPost:
		if path == "upload" {
			h.uploadDocument(w, r)
			return
		}
		if len(parts) == 2 && parts[1] == "reindex" {
			h.reindexDocument(w, r, parts[0])
			return
		}
		http.NotFound(w, r)
		return
	case http.MethodGet:
		if path == "" {
			h.listDocuments(w, r)
			return
		}
		if len(parts) == 1 {
			h.getDocument(w, r, parts[0])
			return
		}
		http.NotFound(w, r)
		return
	case http.MethodDelete:
		if len(parts) == 1 && parts[0] != "" {
			h.deleteDocument(w, r, parts[0])
			return
		}
		writeError(w, "document id required", http.StatusBadRequest)
		return
	default:
		writeError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
}

var supportedExts = map[string]bool{".pdf": true, ".docx": true, ".pptx": true, ".xlsx": true}

func (h *Handler) uploadDocument(w http.ResponseWriter, r *http.Request) {
	const maxUpload = 50 << 20 // 50 MiB
	if err := r.ParseMultipartForm(maxUpload); err != nil {
		log.Printf("[documents] upload: parse form failed: %v", err)
		writeError(w, "failed to parse form", http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		log.Printf("[documents] upload: form file missing: %v", err)
		writeError(w, "missing file", http.StatusBadRequest)
		return
	}
	defer file.Close()
	ext := strings.ToLower(path.Ext(header.Filename))
	if !supportedExts[ext] {
		log.Printf("[documents] upload: unsupported ext filename=%s", header.Filename)
		writeError(w, "unsupported format; use .pdf, .docx, .pptx, .xlsx", http.StatusBadRequest)
		return
	}
	projectID := r.URL.Query().Get("project_id")
	if projectID == "" {
		projectID = "default"
	}
	docID := uuid.New().String()
	key := blob.UploadKey(docID, ext)
	ctx := r.Context()
	// Stream upload to S3 (avoids buffering entire file in memory)
	contentLength := header.Size
	if contentLength < 0 {
		// Unknown size: read into buffer (fallback for multipart without Content-Length)
		body, err := io.ReadAll(file)
		if err != nil {
			log.Printf("[documents] upload: read file failed doc_id=%s: %v", docID, err)
			writeError(w, "read file", http.StatusInternalServerError)
			return
		}
		if err := h.store.Put(ctx, key, body); err != nil {
			log.Printf("[documents] upload: S3 Put failed doc_id=%s: %v", docID, err)
			writeError(w, "upload to storage failed", http.StatusInternalServerError)
			return
		}
	} else {
		if err := h.store.PutReader(ctx, key, file, contentLength); err != nil {
			log.Printf("[documents] upload: S3 PutReader failed doc_id=%s: %v", docID, err)
			writeError(w, "upload to storage failed", http.StatusInternalServerError)
			return
		}
	}
	if err := h.pool.InsertDocument(ctx, docID, projectID, header.Filename); err != nil {
		log.Printf("[documents] upload: InsertDocument failed doc_id=%s: %v", docID, err)
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	if err := h.pub.Publish(ctx, mq.DocumentJob{DocID: docID, BlobKey: key, ProjectID: projectID}); err != nil {
		log.Printf("[documents] upload: failed to publish job doc_id=%s: %v", docID, err)
		writeError(w, "queue error", http.StatusInternalServerError)
		return
	}
	log.Printf("[documents] upload: doc_id=%s project_id=%s filename=%s key=%s status=processing", docID, projectID, header.Filename, key)
	writeJSON(w, map[string]any{"id": docID, "status": "processing"})
}

func (h *Handler) listDocuments(w http.ResponseWriter, r *http.Request) {
	projectID := r.URL.Query().Get("project_id")
	var list []db.DocumentSummary
	var err error
	ctx := r.Context()
	if projectID != "" {
		list, err = h.pool.ListDocuments(ctx, &projectID)
	} else {
		list, err = h.pool.ListDocuments(ctx, nil)
	}
	if err != nil {
		log.Printf("[documents] list: database error project_id=%s: %v", projectID, err)
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	processingCount := 0
	for _, d := range list {
		if d.Status == "processing" {
			processingCount++
		}
	}
	if processingCount > 0 {
		log.Printf("[documents] list: project_id=%s total=%d processing=%d", projectID, len(list), processingCount)
	}
	docs := make([]any, len(list))
	for i, d := range list {
		docs[i] = map[string]any{
			"id": d.ID, "filename": d.Filename, "status": d.Status, "pages": d.Pages,
			"project_id": d.ProjectID, "error": d.Error, "index_error": d.IndexError,
		}
	}
	writeJSON(w, map[string]any{"documents": docs})
}

func (h *Handler) getDocument(w http.ResponseWriter, r *http.Request, docID string) {
	ctx := r.Context()
	doc, err := h.pool.GetDocument(ctx, docID)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	if doc == nil {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Document not found"})
		return
	}
	out := map[string]any{
		"id": doc.ID, "project_id": doc.ProjectID, "filename": doc.Filename, "status": doc.Status,
		"progress": doc.Progress, "full_text": doc.FullText, "pages": doc.Pages, "regions": doc.Regions,
		"error": doc.Error, "index_error": doc.IndexError,
		"created_at": doc.CreatedAt, "updated_at": doc.UpdatedAt,
	}
	writeJSON(w, out)
}

func (h *Handler) deleteDocument(w http.ResponseWriter, r *http.Request, docID string) {
	ctx := r.Context()
	if h.rag != nil {
		_ = h.rag.DeleteByDocumentID(ctx, docID)
	}
	ok, err := h.pool.DeleteDocument(ctx, docID)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Document not found"})
		return
	}
	writeJSON(w, map[string]any{"ok": true})
}

func (h *Handler) reindexDocument(w http.ResponseWriter, r *http.Request, docID string) {
	ctx := r.Context()
	doc, err := h.pool.GetDocument(ctx, docID)
	if err != nil || doc == nil {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Document not found"})
		return
	}
	if doc.FullText == "" {
		writeError(w, "document has no full_text to index", http.StatusBadRequest)
		return
	}
	if h.rag != nil {
		_ = h.rag.DeleteByDocumentID(ctx, docID)
		if err := h.rag.Index(ctx, docID, doc.FullText); err != nil {
			_ = h.pool.SetDocumentIndexError(ctx, docID, err.Error())
			writeError(w, "reindex failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}
	_ = h.pool.ClearDocumentIndexError(ctx, docID)
	writeJSON(w, map[string]any{"ok": true, "message": "Reindexed successfully"})
}

func parseBody(w http.ResponseWriter, r *http.Request, v any) bool {
	if r.Body == nil {
		return false
	}
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return false
	}
	return true
}
