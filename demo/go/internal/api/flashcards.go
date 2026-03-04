package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
)

func (h *Handler) flashcards(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/flashcards")
	path = strings.Trim(path, "/")
	parts := strings.Split(path, "/")

	switch {
	case path == "" && r.Method == http.MethodGet:
		h.listFlashcardDecks(w, r)
		return
	case path == "decks" && r.Method == http.MethodGet:
		h.listFlashcardDecks(w, r)
		return
	case path == "generate" && r.Method == http.MethodPost:
		h.generateFlashcards(w, r)
		return
	case path == "decks" && r.Method == http.MethodPost:
		h.createFlashcardDeckStub(w, r)
		return
	case len(parts) == 3 && parts[0] == "decks" && parts[2] == "cards" && r.Method == http.MethodGet:
		h.listFlashcardCards(w, r, parts[1])
		return
	case len(parts) == 2 && parts[0] == "decks" && parts[1] != "" && r.Method == http.MethodDelete:
		h.deleteFlashcardDeck(w, r, parts[1])
		return
	}
	writeJSON(w, map[string]any{"decks": []any{}})
}

func (h *Handler) listFlashcardDecks(w http.ResponseWriter, r *http.Request) {
	projectID := r.URL.Query().Get("project_id")
	documentID := r.URL.Query().Get("document_id")
	var pID, dID *string
	if projectID != "" {
		pID = &projectID
	}
	if documentID != "" {
		dID = &documentID
	}
	ctx := r.Context()
	list, err := h.pool.ListFlashcardDecks(ctx, pID, dID)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	out := make([]any, len(list))
	for i, d := range list {
		docID := d.DocumentID
		out[i] = map[string]any{
			"id":          d.ID,
			"name":        d.Name,
			"project_id":  d.ProjectID,
			"document_id": docID,
			"created_at":  d.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
			"card_count":  d.CardCount,
		}
	}
	writeJSON(w, map[string]any{"decks": out})
}

func (h *Handler) listFlashcardCards(w http.ResponseWriter, r *http.Request, deckID string) {
	ctx := r.Context()
	list, err := h.pool.ListFlashcardCards(ctx, deckID)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	out := make([]any, len(list))
	for i, c := range list {
		out[i] = map[string]any{
			"id":                   c.ID,
			"deck_id":              c.DeckID,
			"source_document_id":   c.SourceDocumentID,
			"front":                c.Front,
			"back":                 c.Back,
			"position":             c.Position,
			"created_at":           c.CreatedAt.Format("2006-01-02T15:04:05Z07:00"),
		}
	}
	writeJSON(w, map[string]any{"cards": out})
}

func (h *Handler) generateFlashcards(w http.ResponseWriter, r *http.Request) {
	var body struct {
		SourceType   string   `json:"source_type"`
		DocumentID   string   `json:"document_id"`
		DocumentIDs  []string `json:"document_ids"`
		ProjectID    string   `json:"project_id"`
		DeckName     string   `json:"deck_name"`
		Count        int      `json:"count"`
		Text         string   `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	projectID := body.ProjectID
	if projectID == "" {
		projectID = "default"
	}
	ctx := r.Context()

	var text string
	var docIDs []string
	if body.SourceType == "text" && strings.TrimSpace(body.Text) != "" {
		text = strings.TrimSpace(body.Text)
	} else {
		if body.SourceType == "documents" && len(body.DocumentIDs) > 0 {
			docIDs = body.DocumentIDs
		} else if body.SourceType == "document" && body.DocumentID != "" {
			docIDs = []string{body.DocumentID}
		}
		if len(docIDs) == 0 {
			writeError(w, "missing document_ids, document_id, or text", http.StatusBadRequest)
			return
		}
		var combinedText strings.Builder
		for _, id := range docIDs {
			docText, err := h.pool.GetDocumentFullText(ctx, id)
			if err != nil {
				writeError(w, "failed to get document text", http.StatusInternalServerError)
				return
			}
			if docText != "" {
				if combinedText.Len() > 0 {
					combinedText.WriteString("\n\n---\n\n")
				}
				combinedText.WriteString(docText)
			}
		}
		text = combinedText.String()
		if text == "" {
			writeError(w, "no completed document text found for the given documents", http.StatusBadRequest)
			return
		}
	}

	if h.rag == nil {
		writeError(w, "RAG not configured", http.StatusServiceUnavailable)
		return
	}
	pairs, err := h.rag.GenerateFlashcardPairs(ctx, text)
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(pairs) == 0 {
		writeError(w, "no flashcards generated", http.StatusInternalServerError)
		return
	}

	deckID := uuid.New().String()
	deckName := strings.TrimSpace(body.DeckName)
	if deckName == "" {
		deckName = "Deck · " + time.Now().Format("Jan 2, 3:04 PM")
	}
	var docID *string
	if len(docIDs) == 1 {
		docID = &docIDs[0]
	}
	// docID stays nil when source was pasted text
	if err := h.pool.CreateFlashcardDeck(ctx, deckID, projectID, deckName, docID); err != nil {
		writeError(w, "failed to create deck", http.StatusInternalServerError)
		return
	}
	cards := make([]struct{ Front, Back string }, len(pairs))
	for i, p := range pairs {
		cards[i] = struct{ Front, Back string }{Front: p.Front, Back: p.Back}
	}
	if err := h.pool.InsertFlashcardCards(ctx, deckID, cards, docID); err != nil {
		writeError(w, "failed to save cards", http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{
		"id":   deckID,
		"name": deckName,
		"card_count": len(cards),
	})
}

func (h *Handler) deleteFlashcardDeck(w http.ResponseWriter, r *http.Request, deckID string) {
	ctx := r.Context()
	ok, err := h.pool.DeleteFlashcardDeck(ctx, deckID)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	if !ok {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) createFlashcardDeckStub(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name       string `json:"name"`
		DocumentID string `json:"document_id"`
		ProjectID  string `json:"project_id"`
		Pairs      []struct {
			Question string `json:"question"`
			Answer   string `json:"answer"`
		} `json:"pairs"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, map[string]any{"id": "stub", "name": body.Name, "cards": []any{}})
		return
	}
	projectID := body.ProjectID
	if projectID == "" {
		projectID = "default"
	}
	deckID := uuid.New().String()
	deckName := body.Name
	if deckName == "" {
		deckName = "Deck"
	}
	var docID *string
	if body.DocumentID != "" {
		docID = &body.DocumentID
	}
	ctx := r.Context()
	if err := h.pool.CreateFlashcardDeck(ctx, deckID, projectID, deckName, docID); err != nil {
		writeError(w, "failed to create deck", http.StatusInternalServerError)
		return
	}
	if len(body.Pairs) > 0 {
		cards := make([]struct{ Front, Back string }, len(body.Pairs))
		for i, p := range body.Pairs {
			cards[i] = struct{ Front, Back string }{Front: p.Question, Back: p.Answer}
		}
		_ = h.pool.InsertFlashcardCards(ctx, deckID, cards, docID)
	}
	writeJSON(w, map[string]any{"id": deckID, "name": deckName, "card_count": len(body.Pairs)})
}
