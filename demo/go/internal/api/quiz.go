package api

import (
	"net/http"
	"strings"
)

// Quiz stub: generate returns empty questions. Full implementation in step 6.
func (h *Handler) quiz(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/quiz")
	path = strings.Trim(path, "/")
	if path == "generate" && r.Method == http.MethodPost {
		writeJSON(w, map[string]any{"questions": []any{}})
		return
	}
	http.NotFound(w, r)
}
