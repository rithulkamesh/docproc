package db

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// FlashcardDeckRow is a deck record.
type FlashcardDeckRow struct {
	ID         string
	ProjectID  string
	DocumentID *string
	Name       string
	CreatedAt  time.Time
	CardCount  int
}

// FlashcardCardRow is a card record.
type FlashcardCardRow struct {
	ID                 string
	DeckID             string
	SourceDocumentID   *string
	Front              string
	Back               string
	Position           int
	CreatedAt          time.Time
}

// CreateFlashcardDeck inserts a deck and returns it. CardCount is 0 until cards are added.
func (p *Pool) CreateFlashcardDeck(ctx context.Context, id, projectID, name string, documentID *string) error {
	_, err := p.Exec(ctx,
		`INSERT INTO docproc_flashcard_decks (id, project_id, document_id, name, created_at)
		 VALUES ($1, $2, $3, $4, NOW())`,
		id, projectID, documentID, name,
	)
	return err
}

// ListFlashcardDecks returns decks, optionally filtered by project_id and document_id. CardCount is filled.
func (p *Pool) ListFlashcardDecks(ctx context.Context, projectID, documentID *string) ([]FlashcardDeckRow, error) {
	query := `
		SELECT d.id, d.project_id, d.document_id, d.name, d.created_at,
		       COALESCE((SELECT COUNT(*) FROM docproc_flashcard_cards c WHERE c.deck_id = d.id), 0) AS card_count
		FROM docproc_flashcard_decks d
		WHERE 1=1`
	args := []interface{}{}
	argNum := 1
	if projectID != nil && *projectID != "" {
		query += fmt.Sprintf(" AND d.project_id = $%d", argNum)
		args = append(args, *projectID)
		argNum++
	}
	if documentID != nil && *documentID != "" {
		query += fmt.Sprintf(" AND d.document_id = $%d", argNum)
		args = append(args, *documentID)
		argNum++
	}
	query += " ORDER BY d.created_at DESC"
	rows, err := p.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []FlashcardDeckRow
	for rows.Next() {
		var d FlashcardDeckRow
		var docID *string
		if err := rows.Scan(&d.ID, &d.ProjectID, &docID, &d.Name, &d.CreatedAt, &d.CardCount); err != nil {
			return nil, err
		}
		d.DocumentID = docID
		list = append(list, d)
	}
	return list, rows.Err()
}

// ListFlashcardCards returns cards for a deck ordered by position.
func (p *Pool) ListFlashcardCards(ctx context.Context, deckID string) ([]FlashcardCardRow, error) {
	rows, err := p.Query(ctx,
		`SELECT id, deck_id, source_document_id, front, back, position, created_at
		 FROM docproc_flashcard_cards WHERE deck_id = $1 ORDER BY position, created_at`,
		deckID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []FlashcardCardRow
	for rows.Next() {
		var c FlashcardCardRow
		var docID *string
		if err := rows.Scan(&c.ID, &c.DeckID, &docID, &c.Front, &c.Back, &c.Position, &c.CreatedAt); err != nil {
			return nil, err
		}
		c.SourceDocumentID = docID
		list = append(list, c)
	}
	return list, rows.Err()
}

// InsertFlashcardCards inserts multiple cards for a deck.
func (p *Pool) InsertFlashcardCards(ctx context.Context, deckID string, cards []struct{ Front, Back string }, sourceDocumentID *string) error {
	for i, c := range cards {
		id := fmt.Sprintf("%s-%d", deckID, i)
		_, err := p.Exec(ctx,
			`INSERT INTO docproc_flashcard_cards (id, deck_id, source_document_id, front, back, position, created_at)
			 VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
			id, deckID, sourceDocumentID, c.Front, c.Back, i,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

// DeleteFlashcardDeck deletes a deck and its cards. Returns true if deleted.
func (p *Pool) DeleteFlashcardDeck(ctx context.Context, deckID string) (bool, error) {
	_, err := p.Exec(ctx, `DELETE FROM docproc_flashcard_cards WHERE deck_id = $1`, deckID)
	if err != nil {
		return false, err
	}
	tag, err := p.Exec(ctx, `DELETE FROM docproc_flashcard_decks WHERE id = $1`, deckID)
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() > 0, nil
}

// GetDocumentFullText returns full_text for a document (only if status = 'completed'). Empty string if not found or not completed.
func (p *Pool) GetDocumentFullText(ctx context.Context, id string) (string, error) {
	var fullText string
	err := p.QueryRow(ctx,
		`SELECT COALESCE(full_text, '') FROM docproc_documents WHERE id = $1 AND status = 'completed'`,
		id,
	).Scan(&fullText)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return "", nil
		}
		return "", err
	}
	return fullText, nil
}
