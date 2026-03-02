package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path"
	"strings"

	"github.com/docproc/demo/internal/blob"
	"github.com/docproc/demo/internal/db"
	"github.com/docproc/demo/internal/mq"
	"github.com/google/uuid"
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
		http.Error(w, "document id required", http.StatusBadRequest)
		return
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
}

var supportedExts = map[string]bool{".pdf": true, ".docx": true, ".pptx": true, ".xlsx": true}

func (h *Handler) uploadDocument(w http.ResponseWriter, r *http.Request) {
	const maxUpload = 50 << 20 // 50 MiB
	if err := r.ParseMultipartForm(maxUpload); err != nil {
		http.Error(w, "failed to parse form", http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing file", http.StatusBadRequest)
		return
	}
	defer file.Close()
	ext := strings.ToLower(path.Ext(header.Filename))
	if !supportedExts[ext] {
		http.Error(w, "unsupported format; use .pdf, .docx, .pptx, .xlsx", http.StatusBadRequest)
		return
	}
	projectID := r.URL.Query().Get("project_id")
	if projectID == "" {
		projectID = "default"
	}
	docID := uuid.New().String()
	key := blob.UploadKey(docID, ext)
	body, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "read file", http.StatusInternalServerError)
		return
	}
	ctx := r.Context()
	if err := h.store.Put(ctx, key, body); err != nil {
		http.Error(w, "upload to storage failed", http.StatusInternalServerError)
		return
	}
	if err := h.pool.InsertDocument(ctx, docID, projectID, header.Filename); err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	if err := h.pub.Publish(ctx, mq.DocumentJob{DocID: docID, BlobKey: key, ProjectID: projectID}); err != nil {
		// Log but still return 202; worker may pick up later if we retry
		http.Error(w, "queue error", http.StatusInternalServerError)
		return
	}
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
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}
	docs := make([]any, len(list))
	for i, d := range list {
		docs[i] = map[string]any{
			"id": d.ID, "filename": d.Filename, "status": d.Status, "pages": d.Pages,
			"project_id": d.ProjectID, "index_error": d.IndexError,
		}
	}
	writeJSON(w, map[string]any{"documents": docs})
}

func (h *Handler) getDocument(w http.ResponseWriter, r *http.Request, docID string) {
	ctx := r.Context()
	doc, err := h.pool.GetDocument(ctx, docID)
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
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
		http.Error(w, "database error", http.StatusInternalServerError)
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
		http.Error(w, "document has no full_text to index", http.StatusBadRequest)
		return
	}
	if h.rag != nil {
		_ = h.rag.DeleteByDocumentID(ctx, docID)
		if err := h.rag.Index(ctx, docID, doc.FullText); err != nil {
			_ = h.pool.SetDocumentIndexError(ctx, docID, err.Error())
			http.Error(w, "reindex failed: "+err.Error(), http.StatusInternalServerError)
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
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return false
	}
	return true
}
