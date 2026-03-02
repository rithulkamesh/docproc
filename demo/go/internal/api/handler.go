package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/docproc/demo/internal/blob"
	"github.com/docproc/demo/internal/config"
	"github.com/docproc/demo/internal/db"
	"github.com/docproc/demo/internal/grade"
	"github.com/docproc/demo/internal/mq"
	"github.com/docproc/demo/internal/rag"
)

// Handler is the main HTTP handler for the demo API.
type Handler struct {
	cfg    *config.Config
	pool   *db.Pool
	store  *blob.Store
	pub    *mq.Publisher
	rag    *rag.RAG
	grader *grade.Grader
}

// NewHandler builds the API handler.
func NewHandler(cfg *config.Config, pool *db.Pool, store *blob.Store, pub *mq.Publisher, ragClient *rag.RAG, grader *grade.Grader) *Handler {
	return &Handler{cfg: cfg, pool: pool, store: store, pub: pub, rag: ragClient, grader: grader}
}

// ServeHTTP routes by path and method.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	path := r.URL.Path
	switch {
	case path == "/status" && r.Method == http.MethodGet:
		h.status(w, r)
	case path == "/embed-check" && r.Method == http.MethodGet:
		h.embedCheck(w, r)
	case path == "/query" && r.Method == http.MethodPost:
		h.query(w, r)
	case path == "/models" && r.Method == http.MethodGet:
		h.models(w, r)
	case path == "/documents" || path == "/documents/" || strings.HasPrefix(path, "/documents/"):
		h.documents(w, r)
	case path == "/projects" || path == "/projects/" || strings.HasPrefix(path, "/projects/"):
		h.projects(w, r)
	case path == "/notes" || path == "/notes/" || strings.HasPrefix(path, "/notes/"):
		h.notes(w, r)
	case path == "/flashcards" || path == "/flashcards/" || strings.HasPrefix(path, "/flashcards/"):
		h.flashcards(w, r)
	case strings.HasPrefix(path, "/quiz/"):
		h.quiz(w, r)
	case path == "/assessments" || path == "/assessments/" || strings.HasPrefix(path, "/assessments/"):
		h.assessments(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (h *Handler) status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"ok":                  true,
		"rag_backend":         "embedding",
		"rag_configured":      h.rag != nil,
		"database_provider":   "pgvector",
		"primary_ai":          "openai",
		"namespace":           "default",
		"default_rag_model":   h.cfg.OpenAIModel,
		"embedding_deployment": nil,
	})
}

func (h *Handler) embedCheck(w http.ResponseWriter, r *http.Request) {
	ok := h.cfg.OpenAIKey != ""
	writeJSON(w, map[string]any{"ok": ok})
}

func (h *Handler) query(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Query  string `json:"query"`
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	q := body.Query
	if q == "" {
		q = body.Prompt
	}
	if q == "" {
		http.Error(w, "missing query or prompt", http.StatusBadRequest)
		return
	}
	if h.rag == nil {
		writeJSON(w, map[string]any{"answer": "RAG not configured (set OPENAI_API_KEY).", "sources": []any{}})
		return
	}
	answer, sources, err := h.rag.Query(r.Context(), q)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"answer": answer, "sources": sources})
}

func (h *Handler) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{
		"primary_ai":        "openai",
		"database":          "pgvector",
		"ai_providers":      []any{"openai"},
		"database_options":  []any{"pgvector", "qdrant", "chroma", "faiss", "memory"},
		"ai_options":        []any{"openai", "azure", "anthropic", "ollama", "litellm"},
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
