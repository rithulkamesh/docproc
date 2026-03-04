package db

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// AssessmentRow is an assessment record.
type AssessmentRow struct {
	ID            string
	ProjectID     string
	Title         string
	Status        string
	MarkingScheme []byte
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// QuestionRow is a question record.
type QuestionRow struct {
	ID           string
	AssessmentID string
	Type         string
	Prompt       string
	CorrectAnswer *string
	Options      []byte
	Position     int
	Metadata     []byte
}

// SubmissionRow is a submission record.
type SubmissionRow struct {
	ID              string
	AssessmentID    string
	Answers         []byte
	Status          string
	ScorePct        *float64
	GradedAt        *time.Time
	QuestionResults []byte
	CreatedAt       time.Time
}

// ListAssessments returns assessments for a project (or all).
func (p *Pool) ListAssessments(ctx context.Context, projectID *string) ([]AssessmentRow, error) {
	var r pgx.Rows
	var err error
	if projectID != nil && *projectID != "" {
		r, err = p.Query(ctx, `SELECT id, project_id, title, status, marking_scheme, created_at, updated_at FROM docproc_assessments WHERE project_id = $1 ORDER BY updated_at DESC`, *projectID)
	} else {
		r, err = p.Query(ctx, `SELECT id, project_id, title, status, marking_scheme, created_at, updated_at FROM docproc_assessments ORDER BY updated_at DESC`)
	}
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var out []AssessmentRow
	for r.Next() {
		var row AssessmentRow
		if err := r.Scan(&row.ID, &row.ProjectID, &row.Title, &row.Status, &row.MarkingScheme, &row.CreatedAt, &row.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, r.Err()
}

// GetAssessment returns an assessment by ID.
func (p *Pool) GetAssessment(ctx context.Context, id string) (*AssessmentRow, error) {
	var row AssessmentRow
	err := p.QueryRow(ctx, `SELECT id, project_id, title, status, marking_scheme, created_at, updated_at FROM docproc_assessments WHERE id = $1`, id).
		Scan(&row.ID, &row.ProjectID, &row.Title, &row.Status, &row.MarkingScheme, &row.CreatedAt, &row.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	return &row, nil
}

// CreateAssessment inserts an assessment.
func (p *Pool) CreateAssessment(ctx context.Context, id, projectID, title string) error {
	_, err := p.Exec(ctx, `INSERT INTO docproc_assessments (id, project_id, title, status, created_at, updated_at) VALUES ($1, $2, $3, 'draft', NOW(), NOW())`, id, projectID, title)
	return err
}

// DeleteAssessment removes an assessment and its questions/submissions (CASCADE).
func (p *Pool) DeleteAssessment(ctx context.Context, id string) error {
	_, err := p.Exec(ctx, `DELETE FROM docproc_assessments WHERE id = $1`, id)
	return err
}

// ListQuestions returns questions for an assessment.
func (p *Pool) ListQuestions(ctx context.Context, assessmentID string) ([]QuestionRow, error) {
	r, err := p.Query(ctx, `SELECT id, assessment_id, type, prompt, correct_answer, options, position, metadata FROM docproc_questions WHERE assessment_id = $1 ORDER BY position`, assessmentID)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var out []QuestionRow
	for r.Next() {
		var row QuestionRow
		if err := r.Scan(&row.ID, &row.AssessmentID, &row.Type, &row.Prompt, &row.CorrectAnswer, &row.Options, &row.Position, &row.Metadata); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, r.Err()
}

// CreateQuestion inserts a question.
func (p *Pool) CreateQuestion(ctx context.Context, id, assessmentID, qtype, prompt string, correctAnswer *string, options []byte, position int) error {
	_, err := p.Exec(ctx, `INSERT INTO docproc_questions (id, assessment_id, type, prompt, correct_answer, options, position, created_at) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW())`, id, assessmentID, qtype, prompt, correctAnswer, options, position)
	return err
}

// InsertSubmission inserts a submission with optional question_results and score.
func (p *Pool) InsertSubmission(ctx context.Context, id, assessmentID string, answers, questionResults []byte, scorePct *float64) error {
	_, err := p.Exec(ctx, `INSERT INTO docproc_submissions (id, assessment_id, answers, status, score_pct, graded_at, question_results, created_at) VALUES ($1, $2, $3::jsonb, 'submitted', $4, NOW(), $5::jsonb, NOW())`, id, assessmentID, answers, scorePct, questionResults)
	return err
}

// GetSubmission returns a submission by ID.
func (p *Pool) GetSubmission(ctx context.Context, id string) (*SubmissionRow, error) {
	var row SubmissionRow
	err := p.QueryRow(ctx, `SELECT id, assessment_id, answers, status, score_pct, graded_at, question_results, created_at FROM docproc_submissions WHERE id = $1`, id).
		Scan(&row.ID, &row.AssessmentID, &row.Answers, &row.Status, &row.ScorePct, &row.GradedAt, &row.QuestionResults, &row.CreatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	return &row, nil
}

// Helper to unmarshal marking_scheme for API response.
func MarkingSchemeFromBytes(b []byte) map[string]any {
	if len(b) == 0 {
		return nil
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}
