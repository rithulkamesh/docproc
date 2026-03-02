import { useEffect, useState, useCallback } from 'react'
import {
  deleteDeck,
  generateFlashcardsFromDocument,
  generateFlashcardsFromText,
  listCards,
  listDecks,
  type FlashcardCard,
  type FlashcardDeck,
} from '../api/flashcards'
import { Button } from '../components/Button'

interface FlashcardsViewProps {
  selectedDocumentId: string | null
}

export function FlashcardsView({ selectedDocumentId }: FlashcardsViewProps) {
  const [decks, setDecks] = useState<FlashcardDeck[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [mode, setMode] = useState<'document' | 'text'>('document')
  const [count, setCount] = useState(5)
  const [deckName, setDeckName] = useState('')
  const [pastedText, setPastedText] = useState('')
  const [generating, setGenerating] = useState(false)

  const [activeDeckId, setActiveDeckId] = useState<string | null>(null)
  const [cards, setCards] = useState<FlashcardCard[]>([])
  const [cardIndex, setCardIndex] = useState(0)
  const [showBack, setShowBack] = useState(false)
  const [cardsLoading, setCardsLoading] = useState(false)
  const [flipKey, setFlipKey] = useState(0)

  const loadDecks = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await listDecks(selectedDocumentId ?? undefined)
      setDecks(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load decks')
    } finally {
      setLoading(false)
    }
  }, [selectedDocumentId])

  useEffect(() => {
    void loadDecks()
  }, [loadDecks])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (cards.length === 0) return
      if (e.key === 'ArrowRight') {
        if (showBack) {
          setCardIndex((i) => (i + 1) % cards.length)
          setShowBack(false)
          setFlipKey((k) => k + 1)
        } else {
          setShowBack(true)
        }
      } else if (e.key === 'ArrowLeft') {
        if (showBack) {
          setShowBack(false)
        } else {
          setCardIndex((i) => (i - 1 + cards.length) % cards.length)
          setFlipKey((k) => k + 1)
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [cards.length, showBack])

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (mode === 'document') {
        if (!selectedDocumentId) {
          setError('Select a document in the sidebar first.')
          return
        }
        await generateFlashcardsFromDocument({
          documentId: selectedDocumentId,
          count,
          deckName: deckName || undefined,
        })
      } else {
        if (!pastedText.trim()) return
        await generateFlashcardsFromText({
          text: pastedText.trim(),
          count,
          deckName: deckName || undefined,
        })
      }
      await loadDecks()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate flashcards')
    } finally {
      setGenerating(false)
    }
  }

  const handleSelectDeck = async (deckId: string) => {
    setActiveDeckId(deckId)
    setShowBack(false)
    setCardIndex(0)
    setFlipKey((k) => k + 1)
    try {
      setCardsLoading(true)
      const data = await listCards(deckId)
      setCards(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load cards')
    } finally {
      setCardsLoading(false)
    }
  }

  const handleDeleteDeck = async (deckId: string) => {
    try {
      await deleteDeck(deckId)
      if (activeDeckId === deckId) {
        setActiveDeckId(null)
        setCards([])
      }
      await loadDecks()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete deck')
    }
  }

  const currentCard = cards.length > 0 ? cards[cardIndex % cards.length] : null

  const inputBase = {
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--color-border-strong)',
    fontFamily: 'var(--font-family)',
    fontSize: 'var(--text-base)',
    transition: `border-color ${'120ms ease'}, box-shadow ${'120ms ease'}`,
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '40px', height: '100%' }}>
      {/* Page header */}
      <header>
        <h1
          style={{
            fontFamily: 'var(--font-family)',
            fontSize: 'var(--text-2xl)',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            margin: 0,
          }}
        >
          Flashcards
        </h1>
        <p
          style={{
            fontSize: 'var(--text-sm)',
            color: 'var(--color-text-muted)',
            marginTop: 'var(--space-sm)',
            marginBottom: 0,
          }}
        >
          Generate decks from documents or pasted text, then review with flip cards.
        </p>
        <div
          style={{
            marginTop: 'var(--space-lg)',
            height: 1,
            backgroundColor: 'var(--color-border-light)',
          }}
        />
      </header>

      {/* Generate card */}
      <section
        id="generate-flashcards"
        style={{
          backgroundColor: 'var(--color-bg-alt)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--color-border-strong)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            padding: 'var(--space-lg)',
            borderBottom: `1px solid ${'var(--color-border-light)'}`,
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--color-text-muted)',
          }}
        >
          GENERATE FLASHCARDS
        </div>
        <div style={{ padding: 'var(--space-xl)' }}>
          {/* Segmented control */}
          <div
            style={{
              display: 'inline-flex',
              borderRadius: 'var(--radius-md)',
              border: `1px solid ${'var(--color-border-light)'}`,
              padding: 2,
              backgroundColor: 'var(--color-bg)',
              marginBottom: 'var(--space-xl)',
            }}
          >
            {(['document', 'text'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                style={{
                  padding: '8px 16px',
                  borderRadius: 'var(--radius-sm)',
                  border: 'none',
                  fontSize: 'var(--text-sm)',
                  fontWeight: 500,
                  cursor: 'pointer',
                  backgroundColor: mode === m ? 'var(--color-bg-alt)' : 'transparent',
                  color: mode === m ? 'var(--color-text)' : 'var(--color-text-muted)',
                  boxShadow: mode === m ? 'var(--shadow-card)' : 'none',
                  transition: `background-color ${'120ms ease'}, color ${'120ms ease'}`,
                }}
              >
                {m === 'document' ? 'From current document' : 'From pasted text'}
              </button>
            ))}
          </div>

          {mode === 'document' ? (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginTop: 0, marginBottom: 'var(--space-xl)' }}>
              {selectedDocumentId ? 'Cards will be grounded in the selected document.' : 'Select a document in the sidebar.'}
            </p>
          ) : (
            <textarea
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder="Paste or type content…"
              rows={4}
              style={{
                width: '100%',
                padding: 'var(--space-md)',
                marginBottom: 'var(--space-xl)',
                ...inputBase,
              }}
            />
          )}

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-lg)', alignItems: 'center' }}>
            <label style={{ fontSize: 'var(--text-sm)', fontWeight: 500 }}>
              Cards{' '}
              <input
                type="number"
                min={3}
                max={15}
                value={count}
                onChange={(e) => setCount(Number(e.target.value) || 3)}
                style={{
                  width: 56,
                  marginLeft: 8,
                  padding: '8px 10px',
                  ...inputBase,
                }}
              />
            </label>
            <input
              value={deckName}
              onChange={(e) => setDeckName(e.target.value)}
              placeholder="Deck name (optional)"
              style={{
                flex: 1,
                minWidth: 200,
                padding: '10px 12px',
                ...inputBase,
              }}
            />
            <Button
              loading={generating}
              onClick={handleGenerate}
              disabled={generating || (mode === 'text' && !pastedText.trim())}
            >
              Generate
            </Button>
          </div>
          {error && (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-danger)', marginTop: 'var(--space-md)' }}>
              {error}
            </p>
          )}
        </div>
      </section>

      {/* Decks + Review */}
      <section
        className="flashcards-decks-review"
        style={{
          flex: 1,
          minHeight: 0,
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 1.2fr)',
          gap: 'var(--space-xl)',
        }}
      >
        {/* Your decks */}
        <div
          style={{
            backgroundColor: 'var(--color-bg-alt)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--color-border-strong)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              padding: 'var(--space-lg)',
              borderBottom: `1px solid ${'var(--color-border-light)'}`,
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.06em',
              color: 'var(--color-text-muted)',
            }}
          >
            YOUR DECKS
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-md)' }}>
            {loading && (
              <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>Loading decks…</p>
            )}
            {!loading && decks.length === 0 && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '48px',
                  textAlign: 'center',
                }}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 'var(--radius-lg)',
                    backgroundColor: 'var(--color-bg)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 24,
                    marginBottom: 'var(--space-md)',
                  }}
                >
                  📚
                </div>
                <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
                  No decks yet. Generate one above to get started.
                </p>
                <Button
                  variant="ghost"
                  style={{ marginTop: 'var(--space-lg)' }}
                  onClick={() => document.getElementById('generate-flashcards')?.scrollIntoView?.({ behavior: 'smooth' })}
                >
                  Generate flashcards
                </Button>
              </div>
            )}
            {!loading && decks.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                {decks.map((deck) => (
                  <div
                    key={deck.id}
                    style={{
                      display: 'flex',
                      alignItems: 'stretch',
                      gap: 'var(--space-md)',
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => void handleSelectDeck(deck.id)}
                      style={{
                        flex: 1,
                        textAlign: 'left',
                        padding: 'var(--space-md)',
                        borderRadius: 'var(--radius-md)',
                        border: `1px solid ${deck.id === activeDeckId ? 'var(--color-accent)' : 'var(--color-border-light)'}`,
                        backgroundColor: deck.id === activeDeckId ? 'var(--color-accent-soft)' : 'var(--color-bg-alt)',
                        boxShadow: 'var(--shadow-card)',
                        cursor: 'pointer',
                        transition: `border-color ${'120ms ease'}, box-shadow ${'120ms ease'}`,
                      }}
                      onMouseEnter={(e) => {
                        if (deck.id !== activeDeckId) {
                          e.currentTarget.style.boxShadow = 'var(--shadow-card)'
                          e.currentTarget.style.borderColor = 'var(--color-text-muted)'
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (deck.id !== activeDeckId) {
                          e.currentTarget.style.boxShadow = 'var(--shadow-card)'
                          e.currentTarget.style.borderColor = 'var(--color-border-light)'
                        }
                      }}
                    >
                      <div style={{ fontWeight: 600, fontSize: 'var(--text-sm)' }}>{deck.name}</div>
                      <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginTop: 4 }}>
                        {deck.card_count ?? 0} cards
                        {deck.created_at && ` · ${new Date(deck.created_at).toLocaleDateString()}`}
                      </div>
                    </button>
                    <Button variant="danger" onClick={() => void handleDeleteDeck(deck.id)}>
                      Delete
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Review panel */}
        <div
          style={{
            backgroundColor: 'var(--color-bg-alt)',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--color-border-strong)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              padding: 'var(--space-lg)',
              borderBottom: `1px solid ${'var(--color-border-light)'}`,
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.06em',
              color: 'var(--color-text-muted)',
            }}
          >
            REVIEW
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 'var(--space-lg)', minHeight: 0 }}>
            {cardsLoading && (
              <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>Loading cards…</p>
            )}
            {!cardsLoading && !currentCard && (
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                  padding: 'var(--space-xl)',
                }}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: 'var(--radius-lg)',
                    backgroundColor: 'var(--color-bg)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 24,
                    marginBottom: 'var(--space-md)',
                  }}
                >
                  🃏
                </div>
                <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
                  Select a deck on the left to start reviewing.
                </p>
                <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginTop: 'var(--space-sm)' }}>
                  Use ← → keys to navigate.
                </p>
              </div>
            )}
            {currentCard && (
              <>
                <div
                  key={flipKey}
                  className="flashcard-flip"
                  style={{
                    flex: 1,
                    borderRadius: 'var(--radius-lg)',
                    border: `1px solid ${'var(--color-border-light)'}`,
                    backgroundColor: 'var(--color-bg)',
                    padding: '40px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    fontSize: 'var(--text-lg)',
                    fontWeight: 600,
                    lineHeight: 1.5,
                  }}
                >
                  {showBack ? currentCard.back : currentCard.front}
                </div>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginTop: 'var(--space-lg)',
                    flexWrap: 'wrap',
                    gap: 'var(--space-md)',
                  }}
                >
                  <div style={{ display: 'flex', gap: 'var(--space-md)' }}>
                    <Button
                      variant="ghost"
                      onClick={() => {
                        if (showBack) {
                          setShowBack(false)
                        } else {
                          setCardIndex((i) => (i - 1 + cards.length) % cards.length)
                          setFlipKey((k) => k + 1)
                        }
                      }}
                    >
                      ← Prev
                    </Button>
                    {!showBack ? (
                      <Button onClick={() => setShowBack(true)}>Show answer</Button>
                    ) : (
                      <Button
                        onClick={() => {
                          setCardIndex((i) => (i + 1) % cards.length)
                          setShowBack(false)
                          setFlipKey((k) => k + 1)
                        }}
                      >
                        Next →
                      </Button>
                    )}
                  </div>
                  <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
                    Card {cardIndex + 1} of {cards.length}
                  </span>
                </div>
                <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginTop: 'var(--space-md)' }}>
                  Keyboard: ← → to navigate
                </p>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
