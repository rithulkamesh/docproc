package db

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// isDuplicateExtensionError returns true if err is a unique violation on pg_extension (extension already exists).
func isDuplicateExtensionError(err error) bool {
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) {
		return false
	}
	return pgErr.Code == "23505" && pgErr.ConstraintName == "pg_extension_name_index"
}

// Pool is the PostgreSQL connection pool.
type Pool struct {
	*pgxpool.Pool
}

// NewPool creates a connection pool and ensures schema exists.
func NewPool(ctx context.Context, connString string) (*Pool, error) {
	config, err := pgxpool.ParseConfig(connString)
	if err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	// Create vector extension before any connection uses pgxvec.RegisterTypes.
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		return nil, fmt.Errorf("bootstrap connect: %w", err)
	}
	_, err = conn.Exec(ctx, `CREATE EXTENSION IF NOT EXISTS vector`)
	conn.Close(ctx)
	if err != nil {
		// Another process (e.g. API and worker both starting) may have created it; duplicate key is OK.
		if !isDuplicateExtensionError(err) {
			return nil, fmt.Errorf("vector extension: %w", err)
		}
	}
	config.AfterConnect = func(ctx context.Context, c *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, c)
	}
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("pgxpool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping: %w", err)
	}
	p := &Pool{Pool: pool}
	if err := p.initSchema(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return p, nil
}

func (p *Pool) initSchema(ctx context.Context) error {
	_, err := p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_documents (
			id VARCHAR(255) PRIMARY KEY,
			project_id VARCHAR(255) NOT NULL DEFAULT 'default',
			filename TEXT NOT NULL,
			display_name TEXT,
			status VARCHAR(64) NOT NULL DEFAULT 'processing',
			progress JSONB,
			full_text TEXT,
			pages INTEGER DEFAULT 0,
			regions JSONB,
			error TEXT,
			index_error TEXT,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_documents: %w", err)
	}
	_, err = p.Exec(ctx, `ALTER TABLE docproc_documents ADD COLUMN IF NOT EXISTS display_name TEXT`)
	if err != nil {
		return fmt.Errorf("docproc_documents display_name: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_projects (
			id VARCHAR(255) PRIMARY KEY,
			name TEXT NOT NULL,
			is_default BOOLEAN NOT NULL DEFAULT FALSE,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_projects: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_chunks (
			id VARCHAR(255) PRIMARY KEY,
			document_id VARCHAR(255),
			content TEXT,
			metadata JSONB,
			embedding vector(1536),
			namespace VARCHAR(255),
			page_ref INTEGER
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_chunks: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_assessments (
			id VARCHAR(255) PRIMARY KEY,
			project_id VARCHAR(255) NOT NULL DEFAULT 'default',
			title TEXT NOT NULL,
			status VARCHAR(64) NOT NULL DEFAULT 'draft',
			marking_scheme JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_assessments: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_questions (
			id VARCHAR(255) PRIMARY KEY,
			assessment_id VARCHAR(255) NOT NULL REFERENCES docproc_assessments(id) ON DELETE CASCADE,
			type VARCHAR(64) NOT NULL,
			prompt TEXT NOT NULL,
			correct_answer TEXT,
			options JSONB,
			position INTEGER NOT NULL DEFAULT 0,
			metadata JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_questions: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_submissions (
			id VARCHAR(255) PRIMARY KEY,
			assessment_id VARCHAR(255) NOT NULL REFERENCES docproc_assessments(id) ON DELETE CASCADE,
			answers JSONB NOT NULL,
			status VARCHAR(64) NOT NULL DEFAULT 'submitted',
			score_pct NUMERIC,
			graded_at TIMESTAMPTZ,
			question_results JSONB,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_submissions: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_notebooks (
			id VARCHAR(255) PRIMARY KEY,
			project_id VARCHAR(255) NOT NULL,
			parent_id VARCHAR(255),
			title VARCHAR(512) NOT NULL DEFAULT '',
			position INTEGER NOT NULL DEFAULT 0,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_notebooks: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_notes (
			id VARCHAR(255) PRIMARY KEY,
			project_id VARCHAR(255) NOT NULL DEFAULT 'default',
			document_id VARCHAR(255),
			notebook_id VARCHAR(255),
			title TEXT,
			content TEXT NOT NULL DEFAULT '',
			content_blocks JSONB,
			position INTEGER DEFAULT 0,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_notes: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_tags (
			id VARCHAR(255) PRIMARY KEY,
			project_id VARCHAR(255) NOT NULL,
			name VARCHAR(255) NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_tags: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_flashcard_decks (
			id VARCHAR(255) PRIMARY KEY,
			project_id VARCHAR(255) NOT NULL DEFAULT 'default',
			document_id VARCHAR(255),
			name TEXT NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_flashcard_decks: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_flashcard_cards (
			id VARCHAR(255) PRIMARY KEY,
			deck_id VARCHAR(255) NOT NULL,
			source_document_id VARCHAR(255),
			front TEXT NOT NULL,
			back TEXT NOT NULL,
			position INTEGER NOT NULL DEFAULT 0,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_flashcard_cards: %w", err)
	}
	_, err = p.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS docproc_ai_config (
			id VARCHAR(255) PRIMARY KEY DEFAULT 'default',
			provider VARCHAR(64) NOT NULL DEFAULT 'openai',
			model VARCHAR(255) NOT NULL DEFAULT 'gpt-4o-mini',
			api_key_encrypted TEXT,
			base_url TEXT,
			endpoint TEXT,
			deployment TEXT,
			embedding_deployment TEXT,
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		return fmt.Errorf("docproc_ai_config: %w", err)
	}
	return nil
}
