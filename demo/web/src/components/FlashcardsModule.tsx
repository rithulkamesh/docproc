import { useCallback, useEffect, useRef, useState } from 'react'
import type { DocumentSummary } from '../types'
import {
  deleteDeck,
  generateFlashcardsFromDocument,
  generateFlashcardsFromText,
  listCards,
  listDecks,
  type FlashcardCard,
  type FlashcardDeck,
} from '../api/flashcards'
import { Button } from './Button'
import { LatexText } from './LatexText'

type StudyMode = 'classic' | 'timed' | 'reverse'
type Difficulty = 'new' | 'review' | 'mastered'

interface FlashcardsModuleProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

export function FlashcardsModule({ selectedDocumentId, projectId }: FlashcardsModuleProps) {
  const [decks, setDecks] = useState<FlashcardDeck[]>([])
  const [loadingDecks, setLoadingDecks] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [mode, setMode] = useState<'document' | 'text'>('document')
  const [count, setCount] = useState(8)
  const [deckName, setDeckName] = useState('')
  const [pastedText, setPastedText] = useState('')
  const [generating, setGenerating] = useState(false)

  const [activeDeckId, setActiveDeckId] = useState<string | null>(null)
  const [cards, setCards] = useState<FlashcardCard[]>([])
  const [cardIndex, setCardIndex] = useState(0)
  const [showBack, setShowBack] = useState(false)
  const [studyMode, setStudyMode] = useState<StudyMode>('classic')
  const [ratings, setRatings] = useState<Record<string, Difficulty>>({})
  const [cardsLoading, setCardsLoading] = useState(false)

  const [timedSeconds] = useState(120)
  const [timeLeft, setTimeLeft] = useState<number | null>(null)

  const dragStartX = useRef<number | null>(null)
  const [dragX, setDragX] = useState(0)

  const currentCard = cards.length > 0 ? cards[cardIndex % cards.length] : null

  const loadDecks = useCallback(async () => {
    try {
      setLoadingDecks(true)
      setError(null)
      const data = await listDecks(undefined, projectId)
      setDecks(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load decks')
    } finally {
      setLoadingDecks(false)
    }
  }, [projectId])

  useEffect(() => {
    void loadDecks()
  }, [loadDecks])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!currentCard || cards.length === 0) return
      if (e.key === 'ArrowRight') {
        handleNext()
      } else if (e.key === 'ArrowLeft') {
        handlePrev()
      } else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault()
        setShowBack((s) => !s)
      } else if (e.key === '1') {
        handleRate('review')
      } else if (e.key === '2') {
        handleRate('review')
      } else if (e.key === '3') {
        handleRate('mastered')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  useEffect(() => {
    if (timeLeft == null || timeLeft <= 0 || studyMode !== 'timed') return
    const id = window.setTimeout(() => setTimeLeft((t) => (t == null ? null : t - 1)), 1000)
    return () => window.clearTimeout(id)
  }, [timeLeft, studyMode])

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (mode === 'document') {
        if (!selectedDocumentId) {
          setError('Select a document in the left column first.')
          return
        }
        await generateFlashcardsFromDocument({
          documentId: selectedDocumentId,
          count,
          deckName: deckName || undefined,
          projectId,
        })
      } else {
        if (!pastedText.trim()) return
        await generateFlashcardsFromText({
          text: pastedText.trim(),
          deckName: deckName || undefined,
          projectId,
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
    setRatings({})
    setTimeLeft(studyMode === 'timed' ? timedSeconds : null)
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
        setRatings({})
      }
      await loadDecks()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete deck')
    }
  }

  const handlePrev = () => {
    if (cards.length === 0) return
    setShowBack(false)
    setCardIndex((i) => (i - 1 + cards.length) % cards.length)
    setDragX(0)
  }

  const handleNext = () => {
    if (cards.length === 0) return
    setShowBack(false)
    setCardIndex((i) => (i + 1) % cards.length)
    setDragX(0)
  }

  const handleRate = (level: Difficulty) => {
    if (!currentCard) return
    setRatings((prev) => ({ ...prev, [currentCard.id]: level }))
    handleNext()
  }

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    dragStartX.current = event.clientX
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragStartX.current == null) return
    const delta = event.clientX - dragStartX.current
    setDragX(delta)
  }

  const handlePointerUp = () => {
    if (dragStartX.current == null) return
    const threshold = 64
    if (dragX > threshold) {
      handlePrev()
    } else if (dragX < -threshold) {
      handleNext()
    } else {
      setDragX(0)
    }
    dragStartX.current = null
  }

  const totalCards = cards.length
  const masteredCount = Object.values(ratings).filter((r) => r === 'mastered').length
  const reviewCount = Object.values(ratings).filter((r) => r === 'review').length
  const newCount = totalCards - masteredCount - reviewCount
  const completedCount = masteredCount + reviewCount
  const progressPct = totalCards > 0 ? Math.round((completedCount / totalCards) * 100) : 0

  const displayFront = () => {
    if (!currentCard) return ''
    if (studyMode === 'reverse') return currentCard.back
    return currentCard.front
  }

  const displayBack = () => {
    if (!currentCard) return ''
    if (studyMode === 'reverse') return currentCard.front
    return currentCard.back
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      <div
        style={{
          fontSize: 'var(--text-xs)',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: 'var(--color-text-muted)',
        }}
      >
        Project decks with spaced review, stack-based cards, and keyboard controls.
      </div>
      {error && (
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-danger)', margin: 0 }}>
          {error}
        </p>
      )}

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          backgroundColor: 'var(--color-bg)',
          padding: 'var(--space-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-sm)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--space-sm)',
          }}
        >
          <div
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
            }}
          >
            Your decks
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
            {decks.length} deck{decks.length === 1 ? '' : 's'}
          </span>
        </div>
        {loadingDecks && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
            Loading decks…
          </p>
        )}
        {!loadingDecks && decks.length === 0 && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
            No decks yet. Generate a small deck from the current document or any pasted text.
          </p>
        )}
        {!loadingDecks && decks.length > 0 && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
              maxHeight: 120,
              overflowY: 'auto',
            }}
          >
            {decks.map((deck) => (
              <div
                key={deck.id}
                style={{
                  display: 'flex',
                  alignItems: 'stretch',
                  gap: 4,
                }}
              >
                <button
                  type="button"
                  onClick={() => void handleSelectDeck(deck.id)}
                  style={{
                    flex: 1,
                    textAlign: 'left',
                    padding: '6px 8px',
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${deck.id === activeDeckId ? 'var(--color-accent)' : 'var(--color-border-light)'}`,
                    backgroundColor: deck.id === activeDeckId ? 'var(--color-accent-soft)' : 'var(--color-bg-alt)',
                    cursor: 'pointer',
                    fontSize: 'var(--text-xs)',
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{deck.name}</div>
                  <div style={{ color: 'var(--color-text-muted)' }}>
                    {deck.card_count ?? 0} cards
                    {deck.created_at && ` · ${new Date(deck.created_at).toLocaleDateString()}`}
                  </div>
                </button>
                <Button type="button" variant="danger" onClick={() => void handleDeleteDeck(deck.id)}>
                  ✕
                </Button>
              </div>
            ))}
          </div>
        )}

        <div
          style={{
            marginTop: 'var(--space-sm)',
            paddingTop: 'var(--space-sm)',
            borderTop: `1px solid ${'var(--color-border-light)'}`,
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>Generate:</span>
            <button
              type="button"
              onClick={() => setMode('document')}
              style={{
                padding: '2px 6px',
                fontSize: 'var(--text-xs)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border-strong)',
                backgroundColor: mode === 'document' ? 'var(--color-bg-alt)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              From document
            </button>
            <button
              type="button"
              onClick={() => setMode('text')}
              style={{
                padding: '2px 6px',
                fontSize: 'var(--text-xs)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border-strong)',
                backgroundColor: mode === 'text' ? 'var(--color-bg-alt)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              From text
            </button>
          </div>
          {mode === 'document' ? (
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', margin: 0 }}>
              {selectedDocumentId
                ? 'Use the selected document as the source.'
                : 'Select a document in the left column first.'}
            </p>
          ) : (
            <textarea
              value={pastedText}
              onChange={(e) => setPastedText(e.target.value)}
              placeholder="Paste or type content…"
              rows={2}
              style={{
                width: '100%',
                padding: 'var(--space-sm)',
                fontFamily: 'var(--font-family)',
                fontSize: 'var(--text-base)',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--color-border-strong)',
              }}
            />
          )}
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <input
              type="number"
              min={3}
              max={20}
              value={count}
              onChange={(e) => setCount(Number(e.target.value) || 3)}
              style={{
                width: 56,
                padding: '4px 6px',
                fontSize: 'var(--text-xs)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border-strong)',
              }}
            />
            <input
              value={deckName}
              onChange={(e) => setDeckName(e.target.value)}
              placeholder="Deck name (optional)"
              style={{
                flex: 1,
                padding: '4px 6px',
                fontSize: 'var(--text-xs)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border-strong)',
              }}
            />
            <Button
              type="button"
              loading={generating}
              disabled={generating || (mode === 'text' && !pastedText.trim())}
              onClick={handleGenerate}
            >
              Go
            </Button>
          </div>
        </div>
      </section>

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          backgroundColor: 'var(--color-bg)',
          padding: 'var(--space-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-sm)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--space-sm)',
          }}
        >
          <div style={{ display: 'flex', gap: 4 }}>
            {(['classic', 'timed', 'reverse'] as StudyMode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => {
                  setStudyMode(m)
                  if (m === 'timed' && cards.length > 0) {
                    setTimeLeft(timedSeconds)
                  } else {
                    setTimeLeft(null)
                  }
                }}
                style={{
                  padding: '2px 8px',
                  fontSize: 'var(--text-xs)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--color-border-strong)',
                  backgroundColor: studyMode === m ? 'var(--color-bg-alt)' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                {m === 'classic' ? 'Classic' : m === 'timed' ? 'Timed' : 'Reverse'}
              </button>
            ))}
          </div>
          <div
            style={{
              flex: 1,
              marginLeft: 'var(--space-sm)',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 'var(--text-xs)',
            }}
          >
            <div
              style={{
                flex: 1,
                height: 6,
                backgroundColor: 'var(--color-bg-alt)',
                border: '1px solid var(--color-border-strong)',
                position: 'relative',
              }}
            >
              <div
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: `${progressPct}%`,
                  backgroundColor: 'var(--color-accent-soft)',
                  transition: 'width 120ms ease',
                }}
              />
            </div>
            <span style={{ color: 'var(--color-text-muted)' }}>
              {completedCount}/{totalCards}
            </span>
          </div>
        </div>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
          <span>New: {newCount}</span> · <span>Review: {reviewCount}</span> · <span>Mastered: {masteredCount}</span>
          {studyMode === 'timed' && timeLeft != null && (
            <span>
              {' '}
              · Time left: {Math.max(timeLeft, 0)}s
            </span>
          )}
        </div>

        <div
          style={{
            flex: 1,
            minHeight: 160,
            display: 'flex',
            alignItems: 'stretch',
            justifyContent: 'center',
            marginTop: 'var(--space-sm)',
          }}
        >
          {cardsLoading && (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 'auto' }}>
              Loading cards…
            </p>
          )}
          {!cardsLoading && !currentCard && (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 'auto' }}>
              Select a deck above to start a session.
            </p>
          )}
          {!cardsLoading && currentCard && (
            <div
              style={{
                position: 'relative',
                width: '100%',
                maxWidth: 360,
                height: 200,
              }}
            >
              {cards.slice(cardIndex + 1, cardIndex + 3).map((card, idx) => {
                const depth = idx + 1
                return (
                  <div
                    key={card.id}
                    style={{
                      position: 'absolute',
                      inset: depth * 6,
                      borderRadius: 'var(--radius-lg)',
                      border: '1px solid var(--color-border-strong)',
                      backgroundColor: 'var(--color-bg-alt)',
                      opacity: 0.4 - depth * 0.1,
                      transform: `scale(${1 - depth * 0.03})`,
                      transition: 'transform 120ms ease, opacity 120ms ease',
                    }}
                  />
                )
              })}
              <div
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
                style={{
                  position: 'absolute',
                  inset: 0,
                  borderRadius: 'var(--radius-lg)',
                  border: '1px solid var(--color-border-strong)',
                  backgroundColor: 'var(--color-bg-alt)',
                  padding: 'var(--space-md)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  textAlign: 'center',
                  fontSize: 'var(--text-sm)',
                  fontWeight: 600,
                  lineHeight: 1.5,
                  cursor: 'grab',
                  transform: `translateX(${dragX}px)`,
                  transition: dragStartX.current == null ? 'transform 120ms ease' : 'none',
                }}
              >
                <LatexText text={showBack ? displayBack() : displayFront()} />
              </div>
            </div>
          )}
        </div>

        {currentCard && (
          <>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                marginTop: 'var(--space-sm)',
                flexWrap: 'wrap',
              }}
            >
              <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                <Button type="button" variant="ghost" onClick={handlePrev}>
                  ← Prev
                </Button>
                <Button type="button" variant="ghost" onClick={() => setShowBack((s) => !s)}>
                  {showBack ? 'Hide answer' : 'Show answer'}
                </Button>
                <Button type="button" variant="ghost" onClick={handleNext}>
                  Next →
                </Button>
              </div>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                Card {cardIndex + 1} of {totalCards}
              </span>
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                marginTop: 'var(--space-sm)',
                flexWrap: 'wrap',
              }}
            >
              <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                <Button type="button" variant="ghost" onClick={() => handleRate('review')}>
                  Hard (1)
                </Button>
                <Button type="button" variant="ghost" onClick={() => handleRate('review')}>
                  Medium (2)
                </Button>
                <Button type="button" variant="ghost" onClick={() => handleRate('mastered')}>
                  Easy (3)
                </Button>
              </div>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                Keys: ← → navigate, Space flip, 1/2/3 rate
              </span>
            </div>
          </>
        )}
      </section>
    </div>
  )
}

