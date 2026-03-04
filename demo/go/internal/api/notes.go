package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/rithulkamesh/docproc/demo/internal/db"
)

func (h *Handler) notes(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/notes")
	path = strings.Trim(path, "/")
	parts := strings.Split(path, "/")

	switch {
	case path == "" && r.Method == http.MethodGet:
		h.listNotes(w, r)
		return
	case path == "" && r.Method == http.MethodPost:
		h.createNote(w, r)
		return
	case len(parts) == 1 && parts[0] != "" && r.Method == http.MethodGet:
		h.getNote(w, r, parts[0])
		return
	case len(parts) == 1 && parts[0] != "" && r.Method == http.MethodPatch:
		h.updateNote(w, r, parts[0])
		return
	case len(parts) == 1 && parts[0] != "" && r.Method == http.MethodDelete:
		h.deleteNote(w, r, parts[0])
		return
	}

	// Stub responses for routes not fully implemented yet
	if path == "notebooks" && (r.Method == http.MethodGet || r.Method == http.MethodPost) {
		if r.Method == http.MethodGet {
			writeJSON(w, map[string]any{"notebooks": []any{}})
			return
		}
		writeJSON(w, map[string]any{"id": "stub", "title": "Notebook", "project_id": "default"})
		return
	}
	if path == "tags" && (r.Method == http.MethodGet || r.Method == http.MethodPost) {
		writeJSON(w, map[string]any{"tags": []any{}})
		return
	}
	if strings.HasPrefix(path, "search") {
		writeJSON(w, map[string]any{"notes": []any{}})
		return
	}
	if len(parts) == 2 && parts[1] == "backlinks" {
		writeJSON(w, map[string]any{"notes": []any{}})
		return
	}
	if path == "generate" && r.Method == http.MethodPost {
		h.generateNote(w, r)
		return
	}

	writeJSON(w, map[string]any{"id": "stub", "content": "", "notebook_id": "", "document_id": nil})
}

func (h *Handler) listNotes(w http.ResponseWriter, r *http.Request) {
	projectID := r.URL.Query().Get("project_id")
	documentID := r.URL.Query().Get("document_id")
	notebookID := r.URL.Query().Get("notebook_id")
	orderBy := r.URL.Query().Get("order_by")
	if orderBy == "" {
		orderBy = "updated_at"
	}
	var pID, dID, nID *string
	if projectID != "" {
		pID = &projectID
	}
	if documentID != "" {
		dID = &documentID
	}
	if notebookID != "" {
		nID = &notebookID
	}
	ctx := r.Context()
	list, err := h.pool.ListNotes(ctx, pID, dID, nID, orderBy)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	out := make([]any, len(list))
	for i, n := range list {
		out[i] = h.noteToMap(&n)
	}
	// Add document filename/display_name so UI doesn't need extra fetches
	for i, n := range list {
		m := out[i].(map[string]any)
		if n.DocumentID != nil && *n.DocumentID != "" {
			if fn, dn, err := h.pool.GetDocumentDisplayInfo(ctx, *n.DocumentID); err == nil {
				m["filename"] = fn
				if dn != "" {
					m["display_name"] = dn
				}
			}
		}
	}
	writeJSON(w, map[string]any{"notes": out})
}

func (h *Handler) getNote(w http.ResponseWriter, r *http.Request, id string) {
	ctx := r.Context()
	n, err := h.pool.GetNote(ctx, id)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	if n == nil {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Note not found"})
		return
	}
	out := h.noteToMap(n)
	if n.DocumentID != nil && *n.DocumentID != "" {
		if fn, dn, err := h.pool.GetDocumentDisplayInfo(ctx, *n.DocumentID); err == nil {
			out["filename"] = fn
			if dn != "" {
				out["display_name"] = dn
			}
		}
	}
	writeJSON(w, out)
}

func (h *Handler) createNote(w http.ResponseWriter, r *http.Request) {
	var body struct {
		DocumentID   *string         `json:"document_id"`
		ProjectID    *string         `json:"project_id"`
		NotebookID   *string         `json:"notebook_id"`
		Title        *string         `json:"title"`
		Content      string          `json:"content"`
		ContentBlocks json.RawMessage `json:"content_blocks"`
		Position     int            `json:"position"`
	}
	if !parseBody(w, r, &body) {
		return
	}
	projectID := "default"
	if body.ProjectID != nil && *body.ProjectID != "" {
		projectID = *body.ProjectID
	}
	id := uuid.New().String()
	content := body.Content
	cb := []byte("null")
	if len(body.ContentBlocks) > 0 {
		cb = body.ContentBlocks
	}
	ctx := r.Context()
	err := h.pool.CreateNote(ctx, id, projectID, body.DocumentID, body.NotebookID, ptrStr(body.Title), content, cb, body.Position)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	n, _ := h.pool.GetNote(ctx, id)
	if n != nil {
		out := h.noteToMap(n)
		if n.DocumentID != nil && *n.DocumentID != "" {
			if fn, dn, err := h.pool.GetDocumentDisplayInfo(ctx, *n.DocumentID); err == nil {
				out["filename"] = fn
				if dn != "" {
					out["display_name"] = dn
				}
			}
		}
		writeJSON(w, out)
		return
	}
	writeJSON(w, map[string]any{"id": id, "content": content, "project_id": projectID})
}

func (h *Handler) updateNote(w http.ResponseWriter, r *http.Request, id string) {
	var body struct {
		Title        *string         `json:"title"`
		Content      *string         `json:"content"`
		ContentBlocks json.RawMessage `json:"content_blocks"`
		NotebookID   *string         `json:"notebook_id"`
		Position     *int            `json:"position"`
	}
	if !parseBody(w, r, &body) {
		return
	}
	var cb []byte
	if len(body.ContentBlocks) > 0 {
		cb = body.ContentBlocks
	}
	ctx := r.Context()
	err := h.pool.UpdateNote(ctx, id, body.Title, body.Content, cb, body.NotebookID, body.Position)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	n, _ := h.pool.GetNote(ctx, id)
	if n != nil {
		out := h.noteToMap(n)
		if n.DocumentID != nil && *n.DocumentID != "" {
			if fn, dn, err := h.pool.GetDocumentDisplayInfo(ctx, *n.DocumentID); err == nil {
				out["filename"] = fn
				if dn != "" {
					out["display_name"] = dn
				}
			}
		}
		writeJSON(w, out)
		return
	}
	writeJSON(w, map[string]any{"id": id})
}

func (h *Handler) deleteNote(w http.ResponseWriter, r *http.Request, id string) {
	ctx := r.Context()
	err := h.pool.DeleteNote(ctx, id)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) noteToMap(n *db.NoteRow) map[string]any {
	m := map[string]any{
		"id":         n.ID,
		"project_id": n.ProjectID,
		"content":    n.Content,
		"position":   n.Position,
		"created_at": n.CreatedAt.Format("2006-01-02T15:04:05.000Z"),
		"updated_at": n.UpdatedAt.Format("2006-01-02T15:04:05.000Z"),
	}
	if n.DocumentID != nil {
		m["document_id"] = *n.DocumentID
	}
	if n.NotebookID != nil {
		m["notebook_id"] = *n.NotebookID
	}
	if n.Title != nil {
		m["title"] = *n.Title
	}
	if len(n.ContentBlocks) > 0 && string(n.ContentBlocks) != "null" {
		var blocks any
		_ = json.Unmarshal(n.ContentBlocks, &blocks)
		m["content_blocks"] = blocks
	}
	return m
}

func ptrStr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func (h *Handler) generateNote(w http.ResponseWriter, r *http.Request) {
	var body struct {
		SourceType string `json:"source_type"`
		DocumentID string `json:"document_id"`
		Text       string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	ctx := r.Context()
	var text string
	switch body.SourceType {
	case "document":
		if body.DocumentID == "" {
			writeError(w, "document_id required", http.StatusBadRequest)
			return
		}
		var err error
		text, err = h.pool.GetDocumentFullText(ctx, body.DocumentID)
		if err != nil {
			writeError(w, "failed to get document text", http.StatusInternalServerError)
			return
		}
		if text == "" {
			writeError(w, "document not found or not yet processed", http.StatusBadRequest)
			return
		}
	case "text":
		text = strings.TrimSpace(body.Text)
		if text == "" {
			writeError(w, "text required", http.StatusBadRequest)
			return
		}
	default:
		writeError(w, "source_type must be 'document' or 'text'", http.StatusBadRequest)
		return
	}
	if h.rag == nil {
		writeError(w, "RAG not configured. Set OPENAI_API_KEY or Azure OpenAI env vars in .env.", http.StatusServiceUnavailable)
		return
	}
	summary, err := h.rag.GenerateNoteSummary(ctx, text)
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"content": summary})
}
