package api

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/rithulkamesh/docproc/demo/internal/blob"
	"github.com/rithulkamesh/docproc/demo/internal/config"
	"github.com/rithulkamesh/docproc/demo/internal/db"
	"github.com/rithulkamesh/docproc/demo/internal/grade"
	"github.com/rithulkamesh/docproc/demo/internal/mq"
	"github.com/rithulkamesh/docproc/demo/internal/rag"
	"github.com/sashabaranov/go-openai"
)

type Handler struct {
	cfg    *config.Config
	pool   *db.Pool
	store  *blob.Store
	pub    *mq.Publisher
	rag    *rag.RAG
	grader *grade.Grader
}

func NewHandler(cfg *config.Config, pool *db.Pool, store *blob.Store, pub *mq.Publisher, ragClient *rag.RAG, grader *grade.Grader) *Handler {
	return &Handler{cfg: cfg, pool: pool, store: store, pub: pub, rag: ragClient, grader: grader}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// CORS for local/dev; tighten in production
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
	case path == "/query/stream" && r.Method == http.MethodPost:
		h.queryStream(w, r)
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
		"ok":                    true,
		"rag_backend":           "embedding",
		"rag_configured":        h.rag != nil,
		"database_provider":     "pgvector",
		"primary_ai":           h.cfg.PrimaryAI(),
		"namespace":             "default",
		"default_rag_model":     h.cfg.DefaultRAGModel(),
		"embedding_deployment":  nil,
	})
}

func (h *Handler) embedCheck(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{"ok": h.cfg.HasAI()})
}

func (h *Handler) query(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Query    string `json:"query"`
		Prompt   string `json:"prompt"`
		APIKey   string `json:"api_key"`
		Provider string `json:"provider"`
		Model    string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	q := body.Query
	if q == "" {
		q = body.Prompt
	}
	if q == "" {
		writeError(w, "missing query or prompt", http.StatusBadRequest)
		return
	}
	// RAG is required for embeddings and retrieval; api_key/model in body override chat only
	if h.rag == nil {
		writeJSON(w, map[string]any{"answer": "RAG not configured. Set OPENAI_API_KEY or AZURE_OPENAI_* in .env.", "sources": []any{}})
		return
	}
	var chatClient *openai.Client
	model := strings.TrimSpace(body.Model)
	if body.APIKey != "" {
		chatClient = openai.NewClient(strings.TrimSpace(body.APIKey))
	}
	answer, sources, err := h.rag.QueryWithClient(r.Context(), q, chatClient, model)
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	ctx := r.Context()
	for _, s := range sources {
		if docID, ok := s["document_id"].(string); ok && docID != "" {
			if filename, displayName, err := h.pool.GetDocumentDisplayInfo(ctx, docID); err == nil {
				s["filename"] = filename
				if displayName != "" {
					s["display_name"] = displayName
				}
			}
		}
	}
	writeJSON(w, map[string]any{"answer": answer, "sources": sources})
}

func (h *Handler) queryStream(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Query    string `json:"query"`
		Prompt   string `json:"prompt"`
		APIKey   string `json:"api_key"`
		Provider string `json:"provider"`
		Model    string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	q := body.Query
	if q == "" {
		q = body.Prompt
	}
	if q == "" {
		writeError(w, "missing query or prompt", http.StatusBadRequest)
		return
	}
	if h.rag == nil {
		writeError(w, "RAG not configured. Set OPENAI_API_KEY or AZURE_OPENAI_* in .env.", http.StatusServiceUnavailable)
		return
	}
	prompt, sources, err := h.rag.GetContextForQuery(r.Context(), q)
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	ctx := r.Context()
	for _, s := range sources {
		if docID, ok := s["document_id"].(string); ok && docID != "" {
			if filename, displayName, err := h.pool.GetDocumentDisplayInfo(ctx, docID); err == nil {
				s["filename"] = filename
				if displayName != "" {
					s["display_name"] = displayName
				}
			}
		}
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(map[string]any{"sources": sources}); err != nil {
		return
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	var streamClient *openai.Client
	model := strings.TrimSpace(body.Model)
	if body.APIKey != "" {
		streamClient = openai.NewClient(strings.TrimSpace(body.APIKey))
	}
	if err := h.rag.StreamCompletionWithClient(ctx, prompt, w, streamClient, model); err != nil {
		return
	}
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

func writeError(w http.ResponseWriter, detail string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{"detail": detail, "code": http.StatusText(code)})
}
