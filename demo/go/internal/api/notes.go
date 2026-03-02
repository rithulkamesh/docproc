package api

import (
	"net/http"
	"strings"
)

// Notes stub: return empty data so frontend can load. Full implementation in step 6.
func (h *Handler) notes(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/notes")
	path = strings.Trim(path, "/")
	if path == "" && r.Method == http.MethodGet {
		writeJSON(w, map[string]any{"notes": []any{}})
		return
	}
	if path == "notebooks" && (r.Method == http.MethodGet || r.Method == http.MethodPost) {
		if r.Method == http.MethodGet {
			writeJSON(w, []any{})
			return
		}
		writeJSON(w, map[string]any{"id": "stub", "name": "Notebook", "notes": []any{}})
		return
	}
	if path == "tags" && (r.Method == http.MethodGet || r.Method == http.MethodPost) {
		writeJSON(w, []any{})
		return
	}
	// /notes/:id, /notes/:id/tags, /notes/generate, /notes/search, /notes/:id/backlinks
	writeJSON(w, map[string]any{"id": "stub", "content": "", "notebook_id": "", "tags": []any{}})
}
