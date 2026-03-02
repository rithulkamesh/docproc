package grade

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// Result is the grading result (same shape as frontend expects).
type Result struct {
	Score           float64  `json:"score"`
	RubricBreakdown []any    `json:"rubric_breakdown"`
	Feedback        string   `json:"feedback"`
	Justification   string   `json:"justification"`
	ConfidenceScore float64  `json:"confidence_score"`
	IsCorrect       bool     `json:"is_correct"`
	Strengths       []string `json:"strengths,omitempty"`
	MissingConcepts []string `json:"missing_concepts,omitempty"`
}

// Grader grades answers using mode (single_select, formula, conceptual, derivation).
type Grader struct {
	client *openai.Client
	model  string
}

// NewGrader creates a grader that uses the given OpenAI client and model.
func NewGrader(client *openai.Client, model string) *Grader {
	if model == "" {
		model = "gpt-4o-mini"
	}
	return &Grader{client: client, model: model}
}

// Mode is the grading mode.
type Mode string

const (
	ModeSingleSelect Mode = "single_select"
	ModeFormula      Mode = "formula"
	ModeConceptual   Mode = "conceptual"
	ModeDerivation   Mode = "derivation"
)

// InferMode infers grading mode from question type and optional grading_mode / correct_answer.
func InferMode(question map[string]any) Mode {
	qtype := strings.ToLower(strings.TrimSpace(getStr(question, "type")))
	explicit := strings.ToLower(strings.TrimSpace(getStr(question, "grading_mode")))
	ref := getStr(question, "correct_answer")

	switch qtype {
	case "long_answer":
		return ModeDerivation
	case "single_select", "mcq", "multi":
		return ModeSingleSelect
	case "short_answer":
		if explicit == "formula" || explicit == "equation" {
			return ModeFormula
		}
		if explicit == "conceptual" {
			return ModeConceptual
		}
		if looksLikeEquation(ref) {
			return ModeFormula
		}
		return ModeConceptual
	}
	return ModeConceptual
}

func getStr(m map[string]any, key string) string {
	v, ok := m[key]
	if !ok || v == nil {
		return ""
	}
	s, _ := v.(string)
	return s
}

var equationPat = regexp.MustCompile(`[=+\-*/^()\[\]\d\s]|\\frac|\\sqrt|\\lambda|\\pi|\\alpha|\\beta|\\Gamma`)

func looksLikeEquation(s string) bool {
	if len(s) == 0 || len(s) > 500 {
		return false
	}
	// Heuristic: contains LaTeX or math symbols
	return equationPat.MatchString(s) || strings.Contains(s, "=")
}

// Grade grades one answer and returns a Result.
func (g *Grader) Grade(ctx context.Context, question map[string]any, studentAnswer any, rubric map[string]any) (Result, error) {
	mode := InferMode(question)
	stuStr := normalizeStudentAnswer(studentAnswer)

	if stuStr == "" {
		return Result{Score: 0, IsCorrect: false, Feedback: "No answer provided.", ConfidenceScore: 0}, nil
	}

	switch mode {
	case ModeSingleSelect:
		return g.gradeSingleSelect(question, stuStr)
	case ModeFormula:
		return g.gradeFormula(ctx, question, stuStr)
	case ModeConceptual:
		return g.gradeConceptual(ctx, question, stuStr, rubric)
	case ModeDerivation:
		return g.gradeDerivation(ctx, question, stuStr, rubric)
	}
	return g.gradeConceptual(ctx, question, stuStr, rubric)
}

func normalizeStudentAnswer(a any) string {
	if a == nil {
		return ""
	}
	if s, ok := a.(string); ok {
		return strings.TrimSpace(s)
	}
	if arr, ok := a.([]any); ok && len(arr) > 0 {
		return strings.TrimSpace(fmt.Sprint(arr[0]))
	}
	return strings.TrimSpace(fmt.Sprint(a))
}

func (g *Grader) gradeSingleSelect(question map[string]any, studentAnswer string) (Result, error) {
	ref := strings.TrimSpace(getStr(question, "correct_answer"))
	if ref == "" {
		return Result{Score: 100, IsCorrect: true, Feedback: "No correct answer defined.", ConfidenceScore: 1}, nil
	}
	refNorm := strings.ToLower(ref)
	stuNorm := strings.ToLower(studentAnswer)
	correct := refNorm == stuNorm
	score := 0.0
	if correct {
		score = 100
	}
	fb := "Incorrect."
	if correct {
		fb = "Correct."
	}
	return Result{
		Score:           score,
		IsCorrect:       correct,
		Feedback:        fb,
		ConfidenceScore: 1,
	}, nil
}

func (g *Grader) gradeFormula(ctx context.Context, question map[string]any, studentAnswer string) (Result, error) {
	ref := getStr(question, "correct_answer")
	prompt := fmt.Sprintf(`Are these two mathematical or physics expressions equivalent? Answer ONLY with a JSON object: {"equivalent": true or false, "score": 0-100}.

Reference: %s
Student: %s`, ref, studentAnswer)
	resp, err := g.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: g.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
	})
	if err != nil {
		return Result{Score: 0, Feedback: err.Error()}, err
	}
	text := ""
	if len(resp.Choices) > 0 {
		text = strings.TrimSpace(resp.Choices[0].Message.Content)
	}
	score, equivalent := parseFormulaResponse(text)
	return Result{
		Score:           score,
		IsCorrect:       equivalent,
		Feedback:        text,
		ConfidenceScore: score / 100,
	}, nil
}

func parseFormulaResponse(text string) (score float64, equivalent bool) {
	score = 0
	equivalent = false
	re := regexp.MustCompile(`\{[^}]+\}`)
	m := re.FindString(text)
	if m == "" {
		return 0, false
	}
	var v struct {
		Equivalent bool    `json:"equivalent"`
		Score      float64 `json:"score"`
	}
	if json.Unmarshal([]byte(m), &v) != nil {
		return 0, false
	}
	if v.Score > 0 {
		score = v.Score
	} else if v.Equivalent {
		score = 100
	}
	equivalent = v.Equivalent || score >= 99
	return score, equivalent
}

const conceptualPrompt = `You are a fair grader for physics/engineering conceptual short-answer questions.
Grade based on meaning, not wording. Paraphrases and equivalent explanations must receive full or partial credit.

Question: %s

Reference answer (key ideas): %s

Rubric key concepts (must be covered for full credit): %s

Student response:
---
%s
---

Score the response 0-100 based on coverage of the key concepts. Same meaning in different words = full credit.
- All key ideas present (possibly paraphrased): 85-100
- Most key ideas: 50-84
- Some key ideas or partially correct: 20-49
- Wrong or irrelevant: 0-19
- Empty or no meaningful content: 0

Output ONLY a single JSON object with these exact keys: "score" (0-100), "feedback" (string), "key_points_covered" (array of strings), "key_points_missing" (array of strings). No markdown.`

func (g *Grader) gradeConceptual(ctx context.Context, question map[string]any, studentAnswer string, rubric map[string]any) (Result, error) {
	prompt := getStr(question, "prompt")
	ref := getStr(question, "correct_answer")
	concepts := ""
	if rubric != nil {
		if kc, ok := rubric["key_concepts"].([]any); ok {
			var ss []string
			for _, c := range kc {
				ss = append(ss, fmt.Sprint(c))
			}
			concepts = strings.Join(ss, ", ")
		}
	}
	if concepts == "" && ref != "" {
		concepts = ref
	}
	body := fmt.Sprintf(conceptualPrompt, prompt, ref, concepts, studentAnswer)
	resp, err := g.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: g.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: body},
		},
	})
	if err != nil {
		return Result{Score: 0, Feedback: err.Error()}, err
	}
	text := ""
	if len(resp.Choices) > 0 {
		text = strings.TrimSpace(resp.Choices[0].Message.Content)
	}
	return parseConceptualResponse(text)
}

func parseConceptualResponse(text string) (Result, error) {
	re := regexp.MustCompile(`\{[\s\S]*\}`)
	m := re.FindString(text)
	if m == "" {
		return Result{Score: 50, Feedback: "Could not parse grader response."}, nil
	}
	var v struct {
		Score            *float64 `json:"score"`
		Feedback         string  `json:"feedback"`
		KeyPointsCovered []string `json:"key_points_covered"`
		KeyPointsMissing []string `json:"key_points_missing"`
	}
	if json.Unmarshal([]byte(m), &v) != nil {
		return Result{Score: 50, Feedback: "Could not parse grader response."}, nil
	}
	score := 50.0
	if v.Score != nil {
		score = *v.Score
		if score <= 1 {
			score = score * 100
		}
		if score < 0 {
			score = 0
		}
		if score > 100 {
			score = 100
		}
	}
	fb := strings.TrimSpace(v.Feedback)
	if fb == "" {
		fb = "No feedback."
	}
	return Result{
		Score:           score,
		IsCorrect:       score >= 85,
		Feedback:        fb,
		Justification:   fb,
		Strengths:       v.KeyPointsCovered,
		MissingConcepts: v.KeyPointsMissing,
		ConfidenceScore: score / 100,
	}, nil
}

const derivationPrompt = `You are a fair grader for physics/engineering derivation and long-answer questions.
Award partial credit for each rubric step that is correctly addressed. Score each step independently.

Question: %s

Reference answer / key steps: %s

Rubric (score each step; total max = %d):
%s

Student response:
---
%s
---

Output ONLY a single JSON object with these exact keys:
- "score" (number 0-%d)
- "max_score" (number, same as above)
- "missing_concepts" (array of strings: rubric steps not adequately addressed)
- "feedback" (string, brief feedback for the student)
- "rubric_breakdown" (array of objects: { "step": string, "points": number, "max_points": number, "comment": string })

No markdown, no explanation outside the JSON.`

func (g *Grader) gradeDerivation(ctx context.Context, question map[string]any, studentAnswer string, rubric map[string]any) (Result, error) {
	prompt := getStr(question, "prompt")
	ref := getStr(question, "correct_answer")
	maxScore := 10.0
	rubricText := "Complete derivation (10 points)"
	if rubric != nil {
		if steps, ok := rubric["steps"].([]any); ok && len(steps) > 0 {
			var sb strings.Builder
			for i, s := range steps {
				stepMap, _ := s.(map[string]any)
				step := fmt.Sprint(stepMap["step"])
				pts := 10.0 / float64(len(steps))
				if p, ok := stepMap["points"].(float64); ok {
					pts = p
				}
				sb.WriteString(fmt.Sprintf("- %s (max %.0f)\n", step, pts))
				_ = i
			}
			rubricText = sb.String()
		}
	}
	body := fmt.Sprintf(derivationPrompt, prompt, ref, 10, rubricText, studentAnswer, 10)
	resp, err := g.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: g.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: body},
		},
	})
	if err != nil {
		return Result{Score: 0, Feedback: err.Error()}, err
	}
	text := ""
	if len(resp.Choices) > 0 {
		text = strings.TrimSpace(resp.Choices[0].Message.Content)
	}
	return parseDerivationResponse(text, maxScore)
}

func parseDerivationResponse(text string, maxScore float64) (Result, error) {
	re := regexp.MustCompile(`\{[\s\S]*\}`)
	m := re.FindString(text)
	if m == "" {
		return Result{Score: 0, Feedback: "Could not parse grader response."}, nil
	}
	var v struct {
		Score           float64  `json:"score"`
		MaxScore        float64  `json:"max_score"`
		MissingConcepts []string `json:"missing_concepts"`
		Feedback        string   `json:"feedback"`
		RubricBreakdown []any    `json:"rubric_breakdown"`
	}
	if json.Unmarshal([]byte(m), &v) != nil {
		return Result{Score: 0, Feedback: "Could not parse grader response."}, nil
	}
	if v.MaxScore <= 0 {
		v.MaxScore = maxScore
	}
	score := v.Score
	if score > v.MaxScore {
		score = v.MaxScore
	}
	if score < 0 {
		score = 0
	}
	scorePct := 0.0
	if v.MaxScore > 0 {
		scorePct = 100 * score / v.MaxScore
	}
	fb := strings.TrimSpace(v.Feedback)
	if fb == "" {
		fb = "No feedback."
	}
	r := Result{
		Score:           scorePct,
		IsCorrect:       scorePct >= 85,
		Feedback:        fb,
		Justification:   fb,
		RubricBreakdown: v.RubricBreakdown,
		MissingConcepts: v.MissingConcepts,
		ConfidenceScore: scorePct / 100,
	}
	return r, nil
}
