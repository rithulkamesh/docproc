package db

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// DocumentRow is a document record (metadata + optional full_text).
type DocumentRow struct {
	ID          string
	ProjectID   string
	Filename    string
	Status      string
	Progress    map[string]interface{}
	FullText    string
	Pages       int
	Regions     []interface{}
	Error       *string
	IndexError  *string
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// InsertDocument creates a new document in "processing" status.
func (p *Pool) InsertDocument(ctx context.Context, id, projectID, filename string) error {
	prog := `{"page":0,"total":1,"message":"Starting…"}`
	_, err := p.Exec(ctx,
		`INSERT INTO docproc_documents (id, project_id, filename, status, progress, created_at, updated_at)
		 VALUES ($1, $2, $3, 'processing', $4::jsonb, NOW(), NOW())`,
		id, projectID, filename, prog,
	)
	return err
}

// UpdateDocumentCompleted sets status=completed, full_text, pages, regions.
func (p *Pool) UpdateDocumentCompleted(ctx context.Context, id, fullText string, pages int, regions []interface{}) error {
	regJSON, _ := json.Marshal(regions)
	_, err := p.Exec(ctx,
		`UPDATE docproc_documents SET status = 'completed', full_text = $2, pages = $3, regions = $4::jsonb,
		 progress = NULL, error = NULL, index_error = NULL, updated_at = NOW() WHERE id = $1`,
		id, fullText, pages, regJSON,
	)
	return err
}

// UpdateDocumentFailed sets status=failed and error message.
func (p *Pool) UpdateDocumentFailed(ctx context.Context, id, errMsg string) error {
	_, err := p.Exec(ctx,
		`UPDATE docproc_documents SET status = 'failed', error = $2, progress = NULL, updated_at = NOW() WHERE id = $1`,
		id, errMsg,
	)
	return err
}

// UpdateDocumentProgress updates progress JSON.
func (p *Pool) UpdateDocumentProgress(ctx context.Context, id string, progress map[string]interface{}) error {
	prog, _ := json.Marshal(progress)
	_, err := p.Exec(ctx,
		`UPDATE docproc_documents SET progress = $2::jsonb, updated_at = NOW() WHERE id = $1`,
		id, prog,
	)
	return err
}

// DocumentSummary is a list-view document (no full_text).
type DocumentSummary struct {
	ID         string
	Filename   string
	Status     string
	Pages      int
	ProjectID  string
	IndexError string
}

// ListDocuments returns document summaries for a project (or all).
func (p *Pool) ListDocuments(ctx context.Context, projectID *string) ([]DocumentSummary, error) {
	var rows []DocumentSummary
	var r pgx.Rows
	var err error
	if projectID != nil && *projectID != "" {
		r, err = p.Query(ctx, `SELECT id, filename, status, pages, project_id, COALESCE(index_error, '')
			FROM docproc_documents WHERE project_id = $1 ORDER BY updated_at DESC`, *projectID)
	} else {
		r, err = p.Query(ctx, `SELECT id, filename, status, pages, project_id, COALESCE(index_error, '')
			FROM docproc_documents ORDER BY updated_at DESC`)
	}
	if err != nil {
		return nil, err
	}
	defer r.Close()
	for r.Next() {
		var d DocumentSummary
		if err = r.Scan(&d.ID, &d.Filename, &d.Status, &d.Pages, &d.ProjectID, &d.IndexError); err != nil {
			return nil, err
		}
		rows = append(rows, d)
	}
	return rows, r.Err()
}

// GetDocument returns full document by ID.
func (p *Pool) GetDocument(ctx context.Context, id string) (*DocumentRow, error) {
	var d DocumentRow
	var progress, regions []byte
	err := p.QueryRow(ctx,
		`SELECT id, project_id, filename, status, progress, full_text, pages, regions, error, index_error, created_at, updated_at
		 FROM docproc_documents WHERE id = $1`,
		id,
	).Scan(&d.ID, &d.ProjectID, &d.Filename, &d.Status, &progress, &d.FullText, &d.Pages, &regions, &d.Error, &d.IndexError, &d.CreatedAt, &d.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	_ = json.Unmarshal(progress, &d.Progress)
	_ = json.Unmarshal(regions, &d.Regions)
	return &d, nil
}

// DeleteDocument deletes a document by ID. Returns true if deleted.
func (p *Pool) DeleteDocument(ctx context.Context, id string) (bool, error) {
	tag, err := p.Exec(ctx, `DELETE FROM docproc_documents WHERE id = $1`, id)
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() > 0, nil
}

// ClearDocumentIndexError clears index_error for a document.
func (p *Pool) ClearDocumentIndexError(ctx context.Context, id string) error {
	_, err := p.Exec(ctx, `UPDATE docproc_documents SET index_error = NULL, updated_at = NOW() WHERE id = $1`, id)
	return err
}

// SetDocumentIndexError sets index_error for a document.
func (p *Pool) SetDocumentIndexError(ctx context.Context, id, errMsg string) error {
	_, err := p.Exec(ctx, `UPDATE docproc_documents SET index_error = $2, updated_at = NOW() WHERE id = $1`, id, errMsg)
	return err
}

