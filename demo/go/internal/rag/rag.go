package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/pgvector/pgvector-go"
	"github.com/rithulkamesh/docproc/demo/internal/db"
	"github.com/sashabaranov/go-openai"
)

const (
	chunkSize  = 512
	namespace  = "default"
	embedDim   = 1536
	topK       = 5
)

// RAG handles chunking, embedding, storage, and query.
type RAG struct {
	pool   *db.Pool
	client *openai.Client
	model  string
}

// New creates a RAG instance (OpenAI embeddings + pgvector).
func New(pool *db.Pool, apiKey, model string) *RAG {
	if model == "" {
		// Default chat model for answering questions over retrieved context
		model = "gpt-4o-mini"
	}
	client := openai.NewClient(apiKey)
	return &RAG{pool: pool, client: client, model: model}
}

// Index chunks fullText, embeds, and upserts into docproc_chunks for the given document ID.
func (r *RAG) Index(ctx context.Context, documentID, fullText string) error {
	chunks := chunkText(fullText, chunkSize)
	if len(chunks) == 0 {
		return nil
	}
	resp, err := r.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: chunks,
		Model: openai.AdaEmbeddingV2,
	})
	if err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if len(resp.Data) != len(chunks) {
		return fmt.Errorf("embed: got %d vectors, expected %d", len(resp.Data), len(chunks))
	}
	for i, c := range chunks {
		vec := pgvector.NewVector(resp.Data[i].Embedding)
		id := fmt.Sprintf("%s-%d", documentID, i)
		_, err := r.pool.Exec(ctx,
			`INSERT INTO docproc_chunks (id, document_id, content, metadata, embedding, namespace, page_ref)
			 VALUES ($1, $2, $3, '{}'::jsonb, $4, $5, $6)
			 ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding`,
			id, documentID, c, vec, namespace, i,
		)
		if err != nil {
			return fmt.Errorf("upsert chunk: %w", err)
		}
	}
	return nil
}

// DeleteByDocumentID removes all chunks for a document.
func (r *RAG) DeleteByDocumentID(ctx context.Context, documentID string) error {
	_, err := r.pool.Exec(ctx, `DELETE FROM docproc_chunks WHERE document_id = $1 AND namespace = $2`, documentID, namespace)
	return err
}

// Query embeds the question, retrieves top_k chunks, and returns an LLM answer + sources.
func (r *RAG) Query(ctx context.Context, question string) (answer string, sources []map[string]interface{}, err error) {
	resp, err := r.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{question},
		Model: openai.AdaEmbeddingV2,
	})
	if err != nil {
		return "", nil, fmt.Errorf("embed query: %w", err)
	}
	if len(resp.Data) == 0 {
		return "", nil, fmt.Errorf("no embedding returned")
	}
	vec := pgvector.NewVector(resp.Data[0].Embedding)

	rows, err := r.pool.Query(ctx,
		`SELECT content, document_id FROM docproc_chunks
		 WHERE namespace = $1
		 ORDER BY embedding <-> $2
		 LIMIT $3`,
		namespace, vec, topK,
	)
	if err != nil {
		return "", nil, fmt.Errorf("search: %w", err)
	}
	defer rows.Close()

	var contents []string
	sources = make([]map[string]interface{}, 0)
	for rows.Next() {
		var content, docID string
		if err := rows.Scan(&content, &docID); err != nil {
			return "", nil, err
		}
		contents = append(contents, content)
		sources = append(sources, map[string]interface{}{"content": content, "document_id": docID})
	}
	if err := rows.Err(); err != nil {
		return "", nil, err
	}

	contextStr := "(No relevant passages found in the indexed documents.)"
	if len(contents) > 0 {
		contextStr = strings.Join(contents, "\n\n")
	}
	prompt := fmt.Sprintf(`Answer the question based only on the following context.

Context:
%s

Question: %s

Answer:`, contextStr, question)

	chatResp, err := r.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		return "", sources, fmt.Errorf("chat: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return "", sources, nil
	}
	answer = strings.TrimSpace(chatResp.Choices[0].Message.Content)
	return answer, sources, nil
}

func chunkText(text string, size int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	var chunks []string
	var current []string
	runes := 0
	for _, w := range words {
		current = append(current, w)
		runes += len(w) + 1
		if runes >= size {
			chunks = append(chunks, strings.Join(current, " "))
			current = nil
			runes = 0
		}
	}
	if len(current) > 0 {
		chunks = append(chunks, strings.Join(current, " "))
	}
	return chunks
}

