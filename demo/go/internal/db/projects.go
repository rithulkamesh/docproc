package db

import (
	"context"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// ProjectRow is a project record.
type ProjectRow struct {
	ID        string
	Name      string
	IsDefault bool
	CreatedAt time.Time
	UpdatedAt time.Time
}

// ListProjects returns all projects.
func (p *Pool) ListProjects(ctx context.Context) ([]ProjectRow, error) {
	r, err := p.Query(ctx, `SELECT id, name, is_default, created_at, updated_at FROM docproc_projects ORDER BY is_default DESC, name`)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var out []ProjectRow
	for r.Next() {
		var row ProjectRow
		if err := r.Scan(&row.ID, &row.Name, &row.IsDefault, &row.CreatedAt, &row.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	return out, r.Err()
}

// GetProject returns a project by ID.
func (p *Pool) GetProject(ctx context.Context, id string) (*ProjectRow, error) {
	var row ProjectRow
	err := p.QueryRow(ctx, `SELECT id, name, is_default, created_at, updated_at FROM docproc_projects WHERE id = $1`, id).
		Scan(&row.ID, &row.Name, &row.IsDefault, &row.CreatedAt, &row.UpdatedAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	return &row, nil
}

// CreateProject inserts a new project.
func (p *Pool) CreateProject(ctx context.Context, name string) (string, error) {
	id := uuid.New().String()
	_, err := p.Exec(ctx, `INSERT INTO docproc_projects (id, name, is_default, created_at, updated_at) VALUES ($1, $2, FALSE, NOW(), NOW())`, id, name)
	return id, err
}

// UpdateProject updates project name.
func (p *Pool) UpdateProject(ctx context.Context, id, name string) error {
	_, err := p.Exec(ctx, `UPDATE docproc_projects SET name = $2, updated_at = NOW() WHERE id = $1`, id, name)
	return err
}

// DeleteProject deletes a project (optional; may be unused).
func (p *Pool) DeleteProject(ctx context.Context, id string) error {
	_, err := p.Exec(ctx, `DELETE FROM docproc_projects WHERE id = $1`, id)
	return err
}
