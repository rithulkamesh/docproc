package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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

type RAG struct {
	pool             *db.Pool
	client           *openai.Client
	chatModel        string
	embeddingModel   string
}

func New(pool *db.Pool, client *openai.Client, chatModel, embeddingModel string) *RAG {
	if chatModel == "" {
		chatModel = "gpt-4o-mini"
	}
	if embeddingModel == "" {
		embeddingModel = string(openai.AdaEmbeddingV2)
	}
	return &RAG{pool: pool, client: client, chatModel: chatModel, embeddingModel: embeddingModel}
}

func (r *RAG) Index(ctx context.Context, documentID, fullText string) error {
	chunks := chunkText(fullText, chunkSize)
	if len(chunks) == 0 {
		return nil
	}
	resp, err := r.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: chunks,
		Model: openai.EmbeddingModel(r.embeddingModel),
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

func (r *RAG) DeleteByDocumentID(ctx context.Context, documentID string) error {
	_, err := r.pool.Exec(ctx, `DELETE FROM docproc_chunks WHERE document_id = $1 AND namespace = $2`, documentID, namespace)
	return err
}

func (r *RAG) Query(ctx context.Context, question string) (answer string, sources []map[string]interface{}, err error) {
	return r.QueryWithClient(ctx, question, nil, "")
}

func (r *RAG) QueryWithClient(ctx context.Context, question string, client *openai.Client, chatModel string) (answer string, sources []map[string]interface{}, err error) {
	chatClient := r.client
	model := r.chatModel
	if client != nil {
		chatClient = client
	}
	if chatModel != "" {
		model = chatModel
	}
	prompt, sources, err := r.GetContextForQuery(ctx, question)
	if err != nil {
		return "", sources, err
	}
	chatResp, err := chatClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: model,
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

func (r *RAG) GetContextForQuery(ctx context.Context, question string) (prompt string, sources []map[string]interface{}, err error) {
	resp, err := r.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{question},
		Model: openai.EmbeddingModel(r.embeddingModel),
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
	prompt = fmt.Sprintf(`Answer the question based only on the following context.

Context:
%s

Question: %s

Answer in a direct, helpful way. Do not add sign-offs or closing phrases (e.g. "Let me know if...", "Feel free to ask...", "I hope this helps"). End with the answer only.`, contextStr, question)
	return prompt, sources, nil
}

func (r *RAG) StreamCompletion(ctx context.Context, prompt string, w io.Writer) error {
	return r.StreamCompletionWithClient(ctx, prompt, w, nil, "")
}

func (r *RAG) StreamCompletionWithClient(ctx context.Context, prompt string, w io.Writer, client *openai.Client, chatModel string) error {
	chatClient := r.client
	model := r.chatModel
	if client != nil {
		chatClient = client
	}
	if chatModel != "" {
		model = chatModel
	}
	stream, err := chatClient.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		Stream: true,
	})
	if err != nil {
		return fmt.Errorf("chat stream: %w", err)
	}
	defer stream.Close()
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content
		if delta == "" {
			continue
		}
		if encErr := enc.Encode(map[string]string{"delta": delta}); encErr != nil {
			return encErr
		}
		if flusher, ok := w.(interface{ Flush() }); ok {
			flusher.Flush()
		}
	}
	return enc.Encode(map[string]bool{"done": true})
}

func (r *RAG) SuggestDocumentTitle(ctx context.Context, documentExcerpt string) (string, error) {
	excerpt := documentExcerpt
	const maxExcerpt = 2500
	if len(excerpt) > maxExcerpt {
		excerpt = excerpt[:maxExcerpt] + "..."
	}
	prompt := `Based on the following document excerpt, suggest a short, clear title (max 80 characters). Reply with only the title, no quotes or explanation.

---
` + excerpt
	resp, err := r.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.chatModel,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		MaxTokens: 100,
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", nil
	}
	title := strings.TrimSpace(resp.Choices[0].Message.Content)
	if len(title) > 80 {
		title = title[:80]
	}
	return title, nil
}

type FlashcardPair struct {
	Front string
	Back  string
}

const maxTextForFlashcards = 14000

func (r *RAG) GenerateFlashcardPairs(ctx context.Context, text string) ([]FlashcardPair, error) {
	if text == "" {
		return nil, nil
	}
	if len(text) > maxTextForFlashcards {
		text = text[:maxTextForFlashcards] + "\n\n[... truncated]"
	}
	prompt := `Based on the following text, generate flashcard pairs (question and answer) that would help someone study this material.

RULES:
1. LaTeX for all math: use $...$ for inline math and $$...$$ for display math. Write proper LaTeX (e.g. E_0 → $E_0$, exponents as $e^{i(k \\cdot r - \\omega t)}$, \\cdot for dot product, \\omega for omega). Do not output plain-text approximations like e^(...) or E_0 without delimiters.
2. No boilerplate or meta questions: do not ask things like "What is the syllabus overview?", "What does this document cover?", "Summarize the document", or "What are the main topics?". Ask only about the subject matter (concepts, definitions, equations, facts).
3. No reference to the source: do not mention "the document", "the text", "the syllabus", "this chapter", or "the reading" in either the question or the answer. Phrase everything as direct knowledge of the topic.

Cover main concepts, definitions, and important facts. Generate as many cards as appropriate (typically between 5 and 25). Return ONLY a JSON array of objects, each with exactly two keys: "front" (the question) and "back" (the answer). No markdown, no explanation. Example: [{"front":"State the wave equation.","back":"$E(\\\\mathbf{r}, t) = E_0 e^{i(\\\\mathbf{k} \\\\cdot \\\\mathbf{r} - \\\\omega t)}$"}]` + "\n\nText:\n" + text
	resp, err := r.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.chatModel,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		MaxTokens: 4000,
	})
	if err != nil {
		return nil, fmt.Errorf("chat: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, nil
	}
	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	// LLM often wraps JSON in ``` so strip it to parse
	if strings.HasPrefix(raw, "```") {
		if idx := strings.Index(raw, "\n"); idx != -1 {
			raw = raw[idx+1:]
		}
		raw = strings.TrimSuffix(raw, "```")
		raw = strings.TrimSpace(raw)
	}
	var pairs []struct {
		Front string `json:"front"`
		Back  string `json:"back"`
	}
	if err := json.Unmarshal([]byte(raw), &pairs); err != nil {
		return nil, fmt.Errorf("parse flashcard JSON: %w", err)
	}
	out := make([]FlashcardPair, 0, len(pairs))
	for _, p := range pairs {
		f, b := strings.TrimSpace(p.Front), strings.TrimSpace(p.Back)
		if f != "" && b != "" {
			out = append(out, FlashcardPair{Front: f, Back: b})
		}
	}
	return out, nil
}

type AssessmentQuestionOut struct {
	Prompt        string
	CorrectAnswer string
}

const maxTextForAssessment = 14000

func (r *RAG) GenerateAssessmentQuestions(ctx context.Context, text string, count int, difficulty string) ([]AssessmentQuestionOut, error) {
	if text == "" || count <= 0 {
		return nil, nil
	}
	if len(text) > maxTextForAssessment {
		text = text[:maxTextForAssessment] + "\n\n[... truncated]"
	}
	if count > 20 {
		count = 20
	}
	diffHint := "Mix of easy and moderately challenging questions."
	switch difficulty {
	case "easy":
		diffHint = "Focus on straightforward recall: definitions, key terms, and basic facts."
	case "hard":
		diffHint = "Include some application, comparison, or reasoning questions where appropriate."
	}
	prompt := fmt.Sprintf(`Based on the following text, generate exactly %d short-answer practice questions.

RULES:
1. LaTeX for all math: use $...$ for inline and $$...$$ for display. Proper LaTeX only.
2. Ask only about the subject matter. No meta questions (e.g. "What does this document cover?").
3. Do not mention "the document", "the text", or "the reading" in questions or answers.
4. Difficulty: %s

Return ONLY a JSON array of objects, each with two keys: "prompt" (the question) and "correct_answer" (the expected short answer). No markdown, no explanation.
Example: [{"prompt":"What is the wave equation?","correct_answer":"$E = E_0 e^{i(k \\\\cdot r - \\\\omega t)}$"}]`+"\n\nText:\n"+text, count, diffHint)

	resp, err := r.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.chatModel,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		MaxTokens: 4000,
	})
	if err != nil {
		return nil, fmt.Errorf("chat: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, nil
	}
	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	if strings.HasPrefix(raw, "```") {
		if idx := strings.Index(raw, "\n"); idx != -1 {
			raw = raw[idx+1:]
		}
		raw = strings.TrimSuffix(raw, "```")
		raw = strings.TrimSpace(raw)
	}
	var list []struct {
		Prompt        string `json:"prompt"`
		CorrectAnswer string `json:"correct_answer"`
	}
	if err := json.Unmarshal([]byte(raw), &list); err != nil {
		return nil, fmt.Errorf("parse assessment JSON: %w", err)
	}
	out := make([]AssessmentQuestionOut, 0, len(list))
	for _, q := range list {
		p, a := strings.TrimSpace(q.Prompt), strings.TrimSpace(q.CorrectAnswer)
		if p != "" && a != "" {
			out = append(out, AssessmentQuestionOut{Prompt: p, CorrectAnswer: a})
		}
	}
	return out, nil
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

const maxTextForNoteSummary = 24000

func (r *RAG) GenerateNoteSummary(ctx context.Context, text string) (string, error) {
	if text == "" {
		return "", nil
	}
	if len(text) > maxTextForNoteSummary {
		text = text[:maxTextForNoteSummary] + "\n\n[... truncated for length]"
	}
	prompt := `Based on the following text, write a concise study-note summary in Markdown that would help someone review the material.

RULES:
1. Use clear headings (## or ###) for main topics.
2. Use bullet points or short paragraphs for key points.
3. For any math or equations, use LaTeX: $...$ for inline and $$...$$ for display (e.g. $E = mc^2$, $$\\int_0^1 x^2 dx$$).
4. Do not add meta commentary like "This document covers..." or "Summary of the above." Write as if the notes are the primary study material.
5. Focus on concepts, definitions, and important facts. Omit filler.

Return only the markdown content, no surrounding explanation.` + "\n\nText:\n" + text
	resp, err := r.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.chatModel,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		MaxTokens: 4000,
	})
	if err != nil {
		return "", fmt.Errorf("chat: %w", err)
	}
	if len(resp.Choices) == 0 {
		return "", nil
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

