package api

import (
	"net/http"
	"strings"
)

// Flashcards stub: return empty decks. Full implementation in step 6.
func (h *Handler) flashcards(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/flashcards")
	path = strings.Trim(path, "/")
	if path == "decks" && r.Method == http.MethodGet {
		writeJSON(w, map[string]any{"decks": []any{}})
		return
	}
	parts := strings.Split(path, "/")
	if len(parts) == 3 && parts[0] == "decks" && parts[2] == "cards" && r.Method == http.MethodGet {
		writeJSON(w, map[string]any{"cards": []any{}})
		return
	}
	if path == "generate" && r.Method == http.MethodPost {
		writeJSON(w, map[string]any{"id": "stub", "name": "Generated", "cards": []any{}})
		return
	}
	if path == "decks" && r.Method == http.MethodPost {
		writeJSON(w, map[string]any{"id": "stub", "name": "Deck", "cards": []any{}})
		return
	}
	if path == "" && r.Method == http.MethodGet {
		writeJSON(w, map[string]any{"decks": []any{}})
		return
	}
	if len(parts) == 2 && parts[0] == "decks" && parts[1] != "" && r.Method == http.MethodDelete {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	writeJSON(w, map[string]any{"decks": []any{}})
}
