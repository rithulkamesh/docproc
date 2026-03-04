package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/rithulkamesh/docproc/demo/internal/db"
)

// Assessments: CRUD + submit with in-app grading (Go).
func (h *Handler) assessments(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/assessments")
	path = strings.Trim(path, "/")
	parts := strings.Split(path, "/")

	switch r.Method {
	case http.MethodGet:
		if path == "" {
			h.listAssessments(w, r)
			return
		}
		if len(parts) == 1 && parts[0] != "" {
			if parts[0] == "submissions" {
				writeJSON(w, []any{})
				return
			}
			h.getAssessment(w, r, parts[0])
			return
		}
		if len(parts) >= 2 && parts[1] == "submissions" {
			if len(parts) == 2 {
				h.listSubmissions(w, r, parts[0])
				return
			}
			if len(parts) == 3 {
				h.getSubmission(w, r, parts[2])
				return
			}
		}
		writeJSON(w, map[string]any{})
	case http.MethodPost:
		if path == "" {
			h.createAssessment(w, r)
			return
		}
		if path == "paper-upload" {
			writeJSON(w, map[string]any{"total_marks": 0, "sections": []any{}})
			return
		}
		if len(parts) == 2 && parts[1] == "submit" {
			h.submitAssessment(w, r, parts[0])
			return
		}
		if strings.Contains(path, "tutor-feedback") || strings.Contains(path, "re-evaluate") {
			writeJSON(w, map[string]any{"feedback": "", "score": 0})
			return
		}
		writeJSON(w, map[string]any{"id": uuid.New().String()})
	default:
		writeJSON(w, map[string]any{"ok": true})
	}
}

func (h *Handler) listAssessments(w http.ResponseWriter, r *http.Request) {
	projectID := r.URL.Query().Get("project_id")
	var list []db.AssessmentRow
	var err error
	ctx := r.Context()
	if projectID != "" {
		list, err = h.pool.ListAssessments(ctx, &projectID)
	} else {
		list, err = h.pool.ListAssessments(ctx, nil)
	}
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	out := make([]any, len(list))
	for i, row := range list {
		out[i] = map[string]any{
			"id": row.ID, "project_id": row.ProjectID, "title": row.Title, "status": row.Status,
			"marking_scheme": db.MarkingSchemeFromBytes(row.MarkingScheme),
			"created_at": row.CreatedAt, "updated_at": row.UpdatedAt,
		}
	}
	writeJSON(w, out)
}

func (h *Handler) getAssessment(w http.ResponseWriter, r *http.Request, id string) {
	ctx := r.Context()
	ass, err := h.pool.GetAssessment(ctx, id)
	if err != nil || ass == nil {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Assessment not found"})
		return
	}
	questions, err := h.pool.ListQuestions(ctx, id)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	qList := make([]any, len(questions))
	for i, q := range questions {
		var opts []any
		_ = json.Unmarshal(q.Options, &opts)
		qList[i] = map[string]any{
			"id": q.ID, "assessment_id": q.AssessmentID, "type": q.Type, "prompt": q.Prompt,
			"correct_answer": q.CorrectAnswer, "options": opts, "position": q.Position,
		}
	}
	writeJSON(w, map[string]any{
		"id": ass.ID, "project_id": ass.ProjectID, "title": ass.Title, "status": ass.Status,
		"marking_scheme": db.MarkingSchemeFromBytes(ass.MarkingScheme),
		"questions": qList, "created_at": ass.CreatedAt, "updated_at": ass.UpdatedAt,
	})
}

func (h *Handler) createAssessment(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Title      string `json:"title"`
		ProjectID  string `json:"project_id"`
		DocumentID string `json:"document_id"`
		AIConfig   *struct {
			QuestionCount *int   `json:"question_count"`
			Difficulty    string `json:"difficulty"`
		} `json:"ai_config"`
		Questions []struct {
			Type          string   `json:"type"`
			Prompt        string   `json:"prompt"`
			CorrectAnswer *string  `json:"correct_answer"`
			Options       []string `json:"options"`
		} `json:"questions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if body.Title == "" {
		body.Title = "Untitled"
	}
	if body.ProjectID == "" {
		body.ProjectID = "default"
	}
	ctx := r.Context()
	id := uuid.New().String()
	if err := h.pool.CreateAssessment(ctx, id, body.ProjectID, body.Title); err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}

	qList := make([]any, 0)
	questionCount := 10
	difficulty := "mixed"
	if body.AIConfig != nil {
		if body.AIConfig.QuestionCount != nil && *body.AIConfig.QuestionCount > 0 {
			questionCount = *body.AIConfig.QuestionCount
		}
		if body.AIConfig.Difficulty != "" {
			difficulty = body.AIConfig.Difficulty
		}
	}

	if len(body.Questions) > 0 {
		for i, q := range body.Questions {
			qid := uuid.New().String()
			opts, _ := json.Marshal(q.Options)
			_ = h.pool.CreateQuestion(ctx, qid, id, q.Type, q.Prompt, q.CorrectAnswer, opts, i)
			qList = append(qList, map[string]any{
				"id": qid, "assessment_id": id, "type": q.Type, "prompt": q.Prompt,
				"correct_answer": q.CorrectAnswer, "options": q.Options, "position": i,
			})
		}
	} else if body.DocumentID != "" {
		if h.rag == nil {
			writeError(w, "AI question generation is not configured. Set OPENAI_API_KEY or Azure credentials in .env.", http.StatusServiceUnavailable)
			return
		}
		text, err := h.pool.GetDocumentFullText(ctx, body.DocumentID)
		if err != nil {
			writeError(w, "failed to get document text", http.StatusInternalServerError)
			return
		}
		if text == "" {
			writeError(w, "document has no text yet (process it in Sources first) or document not found", http.StatusBadRequest)
			return
		}
		generated, err := h.rag.GenerateAssessmentQuestions(ctx, text, questionCount, difficulty)
		if err != nil {
			writeError(w, "failed to generate questions: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if len(generated) == 0 {
			writeError(w, "no questions could be generated from the document", http.StatusInternalServerError)
			return
		}
		for i, q := range generated {
			qid := uuid.New().String()
			ca := q.CorrectAnswer
			_ = h.pool.CreateQuestion(ctx, qid, id, "short_answer", q.Prompt, &ca, nil, i)
			qList = append(qList, map[string]any{
				"id": qid, "assessment_id": id, "type": "short_answer", "prompt": q.Prompt,
				"correct_answer": q.CorrectAnswer, "options": nil, "position": i,
			})
		}
	}

	now := time.Now().UTC().Format(time.RFC3339)
	w.WriteHeader(http.StatusCreated)
	writeJSON(w, map[string]any{
		"id": id, "project_id": body.ProjectID, "title": body.Title, "status": "draft",
		"marking_scheme": nil,
		"questions": qList,
		"created_at": now, "updated_at": now,
	})
}

func (h *Handler) submitAssessment(w http.ResponseWriter, r *http.Request, assessmentID string) {
	if h.grader == nil {
		writeError(w, "Grading not configured. Set OPENAI_API_KEY or AZURE_OPENAI_* in .env.", http.StatusServiceUnavailable)
		return
	}
	var body struct {
		Answers map[string]any `json:"answers"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	ctx := r.Context()
	questions, err := h.pool.ListQuestions(ctx, assessmentID)
	if err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	if len(questions) == 0 {
		writeJSON(w, map[string]any{"submission_id": uuid.New().String(), "score_pct": 0, "graded_count": 0, "pending_ai_count": 0, "ai_status": "completed"})
		return
	}

	questionResults := make(map[string]any)
	var totalScore float64
	graded := 0
	for _, q := range questions {
		rawAns, ok := body.Answers[q.ID]
		if !ok {
			continue
		}
		var studentAnswer any = rawAns
		if s, ok := rawAns.(string); ok {
			studentAnswer = s
		} else if arr, ok := rawAns.([]any); ok && len(arr) > 0 {
			studentAnswer = arr[0]
		}
		question := map[string]any{
			"id": q.ID, "type": q.Type, "prompt": q.Prompt, "correct_answer": q.CorrectAnswer,
			"options": nil,
		}
		if len(q.Options) > 0 {
			var opts []any
			_ = json.Unmarshal(q.Options, &opts)
			question["options"] = opts
		}
		var rubric map[string]any
		result, err := h.grader.Grade(r.Context(), question, studentAnswer, rubric)
		if err != nil {
			questionResults[q.ID] = map[string]any{"score": 0, "feedback": err.Error()}
			continue
		}
		graded++
		totalScore += result.Score
		questionResults[q.ID] = map[string]any{
			"score": result.Score, "feedback": result.Feedback, "justification": result.Justification,
			"confidence_score": result.ConfidenceScore, "is_correct": result.IsCorrect,
			"strengths": result.Strengths, "missing_concepts": result.MissingConcepts,
		}
	}
	scorePct := totalScore / float64(len(questions))
	subID := uuid.New().String()
	answersJSON, _ := json.Marshal(body.Answers)
	resultsJSON, _ := json.Marshal(questionResults)
	if err := h.pool.InsertSubmission(ctx, subID, assessmentID, answersJSON, resultsJSON, &scorePct); err != nil {
		writeError(w, "database error", http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{
		"submission_id": subID, "score_pct": scorePct, "graded_count": graded,
		"pending_ai_count": 0, "ai_status": "completed",
	})
}

func (h *Handler) listSubmissions(w http.ResponseWriter, r *http.Request, assessmentID string) {
	writeJSON(w, []any{})
}

func (h *Handler) getSubmission(w http.ResponseWriter, r *http.Request, submissionID string) {
	ctx := r.Context()
	sub, err := h.pool.GetSubmission(ctx, submissionID)
	if err != nil || sub == nil {
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, map[string]any{"detail": "Submission not found"})
		return
	}
	var results, answers map[string]any
	_ = json.Unmarshal(sub.QuestionResults, &results)
	_ = json.Unmarshal(sub.Answers, &answers)
	scorePct := 0.0
	if sub.ScorePct != nil {
		scorePct = *sub.ScorePct
	}
	writeJSON(w, map[string]any{
		"id": sub.ID, "assessment_id": sub.AssessmentID, "answers": answers,
		"status": sub.Status, "score_pct": scorePct, "question_results": results, "created_at": sub.CreatedAt,
	})
}
