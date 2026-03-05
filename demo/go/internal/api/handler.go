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
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
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
	case path == "/ai-config" && r.Method == http.MethodGet:
		h.getAIConfig(w, r)
	case path == "/ai-config" && r.Method == http.MethodPut:
		h.putAIConfig(w, r)
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
	// Expose only non-secret server AI config (from .env). API keys are never sent to the client.
	embedDep := h.cfg.DefaultEmbeddingDeployment()
	var embedDepVal any = nil
	if embedDep != "" {
		embedDepVal = embedDep
	}
	writeJSON(w, map[string]any{
		"ok":                   true,
		"rag_backend":          "embedding",
		"rag_configured":       h.rag != nil,
		"database_provider":    "pgvector",
		"primary_ai":           h.cfg.PrimaryAI(),
		"namespace":            "default",
		"default_rag_model":    h.cfg.DefaultRAGModel(),
		"embedding_deployment": embedDepVal,
	})
}

func (h *Handler) embedCheck(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]any{"ok": h.cfg.HasAI()})
}

func (h *Handler) getAIConfig(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	cfg, err := h.pool.GetAIConfig(ctx)
	if err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if cfg == nil {
		writeJSON(w, map[string]any{
			"provider":            "openai",
			"model":               "gpt-4o-mini",
			"api_key_configured":  false,
			"base_url":            "",
			"endpoint":            "",
			"deployment":          "",
			"embedding_deployment": "",
		})
		return
	}
	writeJSON(w, map[string]any{
		"provider":            cfg.Provider,
		"model":               cfg.Model,
		"api_key_configured":  cfg.APIKeyConfigured,
		"base_url":            cfg.BaseURL,
		"endpoint":            cfg.Endpoint,
		"deployment":          cfg.Deployment,
		"embedding_deployment": cfg.EmbeddingDeployment,
	})
}

func (h *Handler) putAIConfig(w http.ResponseWriter, r *http.Request) {
	if len(h.cfg.EncryptionKey) != 32 {
		writeError(w, "DOCPROC_ENCRYPTION_KEY must be set (32 bytes or passphrase) to store API keys securely", http.StatusBadRequest)
		return
	}
	var in db.AIConfigSaveInput
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if in.Provider == "" {
		in.Provider = "openai"
	}
	if in.Model == "" {
		in.Model = "gpt-4o-mini"
	}
	if err := h.pool.SaveAIConfig(r.Context(), h.cfg.EncryptionKey, &in); err != nil {
		writeError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	h.getAIConfig(w, r)
}

// openaiClientFromDBConfig builds an OpenAI client from DB config (for query/stream when no key in body).
// envAzureEndpoint is used when provider is Azure and DB endpoint is empty (e.g. key in Settings, endpoint in .env).
func openaiClientFromDBConfig(cfg *db.AIConfigDecrypted, envAzureEndpoint string) (*openai.Client, string) {
	if cfg == nil || cfg.APIKey == "" {
		return nil, ""
	}
	model := cfg.Model
	if model == "" {
		model = "gpt-4o-mini"
	}
	switch cfg.Provider {
	case "azure":
		endpoint := strings.TrimSpace(cfg.Endpoint)
		if endpoint == "" {
			endpoint = strings.TrimSpace(envAzureEndpoint)
		}
		if endpoint == "" {
			return nil, ""
		}
		clientConfig := openai.DefaultAzureConfig(cfg.APIKey, endpoint)
		return openai.NewClientWithConfig(clientConfig), model
	default:
		clientConfig := openai.DefaultConfig(cfg.APIKey)
		if cfg.BaseURL != "" {
			clientConfig.BaseURL = cfg.BaseURL
		}
		return openai.NewClientWithConfig(clientConfig), model
	}
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
	// RAG is required for embeddings and retrieval; api_key/model in body override; else use DB-stored config.
	if h.rag == nil {
		writeJSON(w, map[string]any{"answer": "RAG not configured. Configure AI in Settings or set OPENAI_API_KEY / AZURE_OPENAI_* in .env.", "sources": []any{}})
		return
	}
	var chatClient *openai.Client
	model := strings.TrimSpace(body.Model)
	if body.APIKey != "" {
		chatClient = openai.NewClient(strings.TrimSpace(body.APIKey))
	} else if len(h.cfg.EncryptionKey) == 32 {
		dbCfg, _ := h.pool.GetAIConfigDecrypted(r.Context(), h.cfg.EncryptionKey)
		if dbCfg != nil && dbCfg.APIKey != "" {
			chatClient, model = openaiClientFromDBConfig(dbCfg, h.cfg.AzureEndpoint)
			if model == "" {
				model = dbCfg.Model
			}
		}
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
	} else if len(h.cfg.EncryptionKey) == 32 {
		dbCfg, _ := h.pool.GetAIConfigDecrypted(ctx, h.cfg.EncryptionKey)
		if dbCfg != nil && dbCfg.APIKey != "" {
			streamClient, model = openaiClientFromDBConfig(dbCfg, h.cfg.AzureEndpoint)
			if model == "" {
				model = dbCfg.Model
			}
		}
	}
	if err := h.rag.StreamCompletionWithClient(ctx, prompt, w, streamClient, model); err != nil {
		_ = enc.Encode(map[string]any{"error": err.Error()})
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
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
