package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// NoteRow is a note record from docproc_notes.
type NoteRow struct {
	ID            string
	ProjectID     string
	DocumentID    *string
	NotebookID    *string
	Title         *string
	Content       string
	ContentBlocks []byte // JSONB
	Position      int
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// ListNotes returns notes for the given filters. orderBy is "position" or "updated_at".
func (p *Pool) ListNotes(ctx context.Context, projectID *string, documentID *string, notebookID *string, orderBy string) ([]NoteRow, error) {
	if orderBy != "position" && orderBy != "updated_at" {
		orderBy = "updated_at"
	}
	orderClause := "ORDER BY position ASC, created_at ASC"
	if orderBy == "updated_at" {
		orderClause = "ORDER BY updated_at DESC, created_at DESC"
	}
	var rows []NoteRow
	var r pgx.Rows
	var err error
	args := []interface{}{}
	argNum := 1
	if projectID != nil && *projectID != "" {
		args = append(args, *projectID)
		argNum++
	}
	if documentID != nil && *documentID != "" {
		args = append(args, *documentID)
		argNum++
	}
	if notebookID != nil && *notebookID != "" {
		args = append(args, *notebookID)
		argNum++
	}
	q := `SELECT id, project_id, document_id, notebook_id, title, content, content_blocks, position, created_at, updated_at
		FROM docproc_notes WHERE 1=1`
	n := 1
	if projectID != nil && *projectID != "" {
		q += fmt.Sprintf(" AND project_id = $%d", n)
		n++
	}
	if documentID != nil && *documentID != "" {
		q += fmt.Sprintf(" AND document_id = $%d", n)
		n++
	}
	if notebookID != nil && *notebookID != "" {
		q += fmt.Sprintf(" AND notebook_id = $%d", n)
		n++
	}
	q += ` ` + orderClause
	r, err = p.Query(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	for r.Next() {
		var n NoteRow
		var docID, nbID, title *string
		var cb []byte
		if err = r.Scan(&n.ID, &n.ProjectID, &docID, &nbID, &title, &n.Content, &cb, &n.Position, &n.CreatedAt, &n.UpdatedAt); err != nil {
			return nil, err
		}
		n.DocumentID = docID
		n.NotebookID = nbID
		n.Title = title
		n.ContentBlocks = cb
		rows = append(rows, n)
	}
	return rows, r.Err()
}

// GetNote returns a note by ID.
func (p *Pool) GetNote(ctx context.Context, id string) (*NoteRow, error) {
	var n NoteRow
	var docID, nbID, title *string
	var cb []byte
	err := p.QueryRow(ctx,
		`SELECT id, project_id, document_id, notebook_id, title, content, content_blocks, position, created_at, updated_at
		 FROM docproc_notes WHERE id = $1`,
		id,
	).Scan(&n.ID, &n.ProjectID, &docID, &nbID, &title, &n.Content, &cb, &n.Position, &n.CreatedAt, &n.UpdatedAt)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	n.DocumentID = docID
	n.NotebookID = nbID
	n.Title = title
	n.ContentBlocks = cb
	return &n, nil
}

// CreateNote inserts a new note.
func (p *Pool) CreateNote(ctx context.Context, id, projectID string, documentID, notebookID *string, title, content string, contentBlocks []byte, position int) error {
	cb := contentBlocks
	if cb == nil {
		cb = []byte("null")
	}
	_, err := p.Exec(ctx,
		`INSERT INTO docproc_notes (id, project_id, document_id, notebook_id, title, content, content_blocks, position, created_at, updated_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, NOW(), NOW())`,
		id, projectID, documentID, notebookID, nullIfEmpty(title), content, cb, position,
	)
	return err
}

// UpdateNote updates a note's fields. Nil pointer means leave existing value.
func (p *Pool) UpdateNote(ctx context.Context, id string, title *string, content *string, contentBlocks []byte, notebookID *string, position *int) error {
	cur, err := p.GetNote(ctx, id)
	if err != nil || cur == nil {
		return err
	}
	t := title
	if t == nil {
		t = cur.Title
	}
	c := content
	if c == nil {
		c = &cur.Content
	}
	cb := contentBlocks
	if cb == nil {
		cb = cur.ContentBlocks
	}
	if cb == nil {
		cb = []byte("null")
	}
	nb := notebookID
	if nb == nil {
		nb = cur.NotebookID
	}
	pos := position
	if pos == nil {
		pos = &cur.Position
	}
	_, err = p.Exec(ctx,
		`UPDATE docproc_notes SET updated_at = NOW(), title = $2, content = $3, content_blocks = $4::jsonb, notebook_id = $5, position = $6 WHERE id = $1`,
		id, t, *c, cb, nb, *pos,
	)
	return err
}

// DeleteNote deletes a note by ID.
func (p *Pool) DeleteNote(ctx context.Context, id string) error {
	_, err := p.Exec(ctx, `DELETE FROM docproc_notes WHERE id = $1`, id)
	return err
}

func nullIfEmpty(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
