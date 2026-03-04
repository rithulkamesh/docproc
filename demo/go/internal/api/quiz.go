package api

import (
	"net/http"
	"strings"
)

// Quiz stub: generate returns empty questions. Full implementation in step 6.
// When implementing AI question generation, use the same rules as flashcard generation in rag.GenerateFlashcardPairs:
// require LaTeX for math ($...$ and $$...$$), no boilerplate (e.g. "syllabus overview"), no reference to "the document" or "the text".
func (h *Handler) quiz(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/quiz")
	path = strings.Trim(path, "/")
	if path == "generate" && r.Method == http.MethodPost {
		writeJSON(w, map[string]any{"questions": []any{}})
		return
	}
	http.NotFound(w, r)
}
