import { useCallback, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import type { FlashcardCard, FlashcardDeck } from '../api/flashcards'
import {
  deleteDeck,
  generateFlashcardsFromDocument,
  generateFlashcardsFromText,
  listCards,
  listDecks,
} from '../api/flashcards'
import { LatexText } from './LatexText'
import { theme } from '../design/theme'
import { SoftButton } from './SoftButton'
import { useWorkspace } from '../context/WorkspaceContext'

const CARD_WIDTH = 'min(40rem, 60vw)'
const FLIP_DURATION = 0.4

type Difficulty = 'new' | 'review' | 'mastered'

export function FlashcardsCanvas() {
  const { documents, selectedDocumentId, currentProjectId } = useWorkspace()
  const [decks, setDecks] = useState<FlashcardDeck[]>([])
  const [loadingDecks, setLoadingDecks] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [mode, setMode] = useState<'document' | 'text'>('document')
  const [count, setCount] = useState(8)
  const [deckName, setDeckName] = useState('')
  const [pastedText, setPastedText] = useState('')
  const [generating, setGenerating] = useState(false)

  const [studyActive, setStudyActive] = useState(false)
  const [studyDeckId, setStudyDeckId] = useState<string | null>(null)
  const [studyDeckName, setStudyDeckName] = useState('')
  const [cards, setCards] = useState<FlashcardCard[]>([])
  const [cardIndex, setCardIndex] = useState(0)
  const [showBack, setShowBack] = useState(false)
  const [ratings, setRatings] = useState<Record<string, Difficulty>>({})
  const [cardsLoading, setCardsLoading] = useState(false)

  const currentDoc = documents.find((d) => d.id === selectedDocumentId) ?? null
  const currentCard = cards.length > 0 ? cards[cardIndex % cards.length] : null

  const loadDecks = useCallback(async () => {
    try {
      setLoadingDecks(true)
      setError(null)
      const data = await listDecks(undefined, currentProjectId)
      setDecks(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load decks')
    } finally {
      setLoadingDecks(false)
    }
  }, [currentProjectId])

  useEffect(() => {
    void loadDecks()
  }, [loadDecks])

  useEffect(() => {
    if (!studyActive || !studyDeckId) return
    let cancelled = false
    setCardsLoading(true)
    listCards(studyDeckId)
      .then((data) => {
        if (!cancelled) {
          setCards(data)
          setCardIndex(0)
          setShowBack(false)
          setRatings({})
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load cards')
      })
      .finally(() => {
        if (!cancelled) setCardsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [studyActive, studyDeckId])

  useEffect(() => {
    if (!currentCard || cards.length === 0) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') handleNext()
      else if (e.key === 'ArrowLeft') handlePrev()
      else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault()
        setShowBack((s) => !s)
      } else if (e.key === '1') handleRate('review')
      else if (e.key === '2') handleRate('review')
      else if (e.key === '3') handleRate('mastered')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (mode === 'document') {
        if (!selectedDocumentId) {
          setError('Select a document in Sources first.')
          return
        }
        await generateFlashcardsFromDocument({
          documentId: selectedDocumentId,
          count,
          deckName: deckName || undefined,
          projectId: currentProjectId,
        })
      } else {
        if (!pastedText.trim()) return
        await generateFlashcardsFromText({
          text: pastedText.trim(),
          deckName: deckName || undefined,
          projectId: currentProjectId,
        })
      }
      await loadDecks()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate flashcards')
    } finally {
      setGenerating(false)
    }
  }

  const handleStartStudy = (deck: FlashcardDeck) => {
    setStudyDeckId(deck.id)
    setStudyDeckName(deck.name)
    setStudyActive(true)
  }

  const handleExitStudy = () => {
    setStudyActive(false)
    setStudyDeckId(null)
    setCards([])
    setRatings({})
  }

  const handleDeleteDeck = async (deckId: string) => {
    try {
      await deleteDeck(deckId)
      if (studyDeckId === deckId) handleExitStudy()
      await loadDecks()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete deck')
    }
  }

  const handlePrev = () => {
    if (cards.length === 0) return
    setShowBack(false)
    setCardIndex((i) => (i - 1 + cards.length) % cards.length)
  }

  const handleNext = () => {
    if (cards.length === 0) return
    setShowBack(false)
    setCardIndex((i) => (i + 1) % cards.length)
  }

  const handleRate = (level: Difficulty) => {
    if (!currentCard) return
    setRatings((prev) => ({ ...prev, [currentCard.id]: level }))
    handleNext()
  }

  const totalCards = cards.length
  const masteredCount = Object.values(ratings).filter((r) => r === 'mastered').length
  const reviewCount = Object.values(ratings).filter((r) => r === 'review').length
  const completedCount = masteredCount + reviewCount
  const progressPct = totalCards > 0 ? Math.round((completedCount / totalCards) * 100) : 0

  const displayFront = () => (currentCard ? currentCard.front : '')
  const displayBack = () => (currentCard ? currentCard.back : '')

  if (studyActive) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: '60vh',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '3rem',
          padding: 'var(--space-lg)',
          backgroundColor: 'var(--color-bg)',
        }}
      >
        {/* 2px top progress bar */}
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            height: '2px',
            background: 'var(--color-bg-interactive)',
            zIndex: 10,
          }}
        >
          <motion.div
            style={{
              height: '100%',
              background: 'var(--color-accent)',
            }}
            initial={{ width: 0 }}
            animate={{ width: `${progressPct}%` }}
            transition={{ duration: theme.motion.durationPanel / 1000, ease: [0.4, 0, 0.2, 1] }}
          />
        </div>
        <div
          style={{
            position: 'absolute',
            top: 'var(--space-lg)',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-md)',
            fontSize: 'var(--text-sm)',
            color: 'var(--color-text-muted)',
          }}
        >
          <span>{progressPct}%</span>
          <span>{studyDeckName}</span>
        </div>

        {cardsLoading && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>Loading cards…</p>
        )}
        {!cardsLoading && cards.length === 0 && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>No cards in this deck.</p>
        )}
        {!cardsLoading && currentCard && (
          <>
            <div
              style={{
                position: 'relative',
                width: '100%',
                maxWidth: CARD_WIDTH,
                aspectRatio: '4/3',
                perspective: '1000px',
              }}
            >
              <motion.div
                onClick={() => setShowBack((s) => !s)}
                style={{
                  position: 'absolute',
                  inset: 0,
                  cursor: 'pointer',
                  transformStyle: 'preserve-3d',
                }}
                animate={{ rotateY: showBack ? 180 : 0 }}
                transition={{ duration: FLIP_DURATION, ease: 'easeInOut' }}
              >
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: 'var(--radius-lg)',
                    border: 'var(--border-subtle)',
                    backgroundColor: 'var(--color-bg-alt)',
                    padding: 'var(--space-xl)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    fontSize: 'var(--text-lg)',
                    fontWeight: 500,
                    lineHeight: theme.lineHeight.body,
                    backfaceVisibility: 'hidden',
                    WebkitBackfaceVisibility: 'hidden',
                  }}
                >
                  <LatexText text={displayFront()} />
                </div>
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: 'var(--radius-lg)',
                    border: 'var(--border-subtle)',
                    backgroundColor: 'var(--color-bg-alt)',
                    padding: 'var(--space-xl)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    fontSize: 'var(--text-lg)',
                    fontWeight: 500,
                    lineHeight: theme.lineHeight.body,
                    backfaceVisibility: 'hidden',
                    WebkitBackfaceVisibility: 'hidden',
                    transform: 'rotateY(180deg)',
                  }}
                >
                  <LatexText text={displayBack()} />
                </div>
              </motion.div>
            </div>

            <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap', justifyContent: 'center' }}>
              <SoftButton onClick={() => handleRate('review')}>Easy</SoftButton>
              <SoftButton onClick={() => handleRate('review')}>Medium</SoftButton>
              <SoftButton onClick={() => handleRate('mastered')}>Hard</SoftButton>
              <SoftButton onClick={handleNext}>Skip</SoftButton>
            </div>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', margin: 0 }}>
              ← → navigate · Space flip · 1/2/3 rate
            </p>
            <SoftButton onClick={handleExitStudy}>Exit study</SoftButton>
          </>
        )}
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      <p style={{ fontSize: 'var(--text-sm)', lineHeight: theme.lineHeight.body, color: 'var(--color-text-muted)', margin: 0 }}>
        Decks and spaced review. Generate from a document or pasted text, then study.
      </p>
      {error && (
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-danger)', margin: 0 }}>{error}</p>
      )}

      <section
        style={{
          borderRadius: 'var(--radius-md)',
          border: 'var(--border-subtle)',
          padding: 'var(--space-lg)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-md)',
          }}
        >
          Your decks
        </div>
        {loadingDecks && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>Loading decks…</p>
        )}
        {!loadingDecks && decks.length === 0 && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
            No decks yet. Generate one below from a document or pasted text.
          </p>
        )}
        {!loadingDecks && decks.length > 0 && (
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
            {decks.map((deck) => (
              <li
                key={deck.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 'var(--space-md)',
                  borderRadius: 'var(--radius-md)',
                  border: 'var(--border-subtle)',
                  padding: 'var(--space-lg)',
                  backgroundColor: 'var(--color-bg)',
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: 'var(--text-sm)' }}>{deck.name}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                    {deck.card_count ?? 0} cards
                    {deck.created_at && ` · ${new Date(deck.created_at).toLocaleDateString()}`}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-md)' }}>
                  <SoftButton onClick={() => handleStartStudy(deck)}>Study</SoftButton>
                  <SoftButton onClick={() => void handleDeleteDeck(deck.id)} style={{ color: 'var(--color-danger)' }}>
                    Delete
                  </SoftButton>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section
        style={{
          borderRadius: 'var(--radius-md)',
          border: 'var(--border-subtle)',
          padding: 'var(--space-lg)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-md)',
          }}
        >
          Generate deck
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-sm)', flexWrap: 'wrap', marginBottom: 'var(--space-md)' }}>
          <SoftButton active={mode === 'document'} onClick={() => setMode('document')}>
            From document
          </SoftButton>
          <SoftButton active={mode === 'text'} onClick={() => setMode('text')}>
            From text
          </SoftButton>
        </div>
        {mode === 'document' ? (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-md)' }}>
            {currentDoc ? `Using: ${currentDoc.display_name ?? currentDoc.filename}` : 'Select a document in the sidebar.'}
          </p>
        ) : (
          <textarea
            value={pastedText}
            onChange={(e) => setPastedText(e.target.value)}
            placeholder="Paste or type content…"
            rows={3}
            style={{
              width: '100%',
              padding: 'var(--space-md)',
              marginBottom: 'var(--space-md)',
              borderRadius: 'var(--radius-md)',
              fontFamily: 'var(--font-family)',
              fontSize: 'var(--text-base)',
              border: 'var(--border-subtle)',
              backgroundColor: 'var(--color-bg)',
              color: 'var(--color-text)',
            }}
          />
        )}
        <div style={{ display: 'flex', gap: 'var(--space-md)', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="number"
            min={3}
            max={20}
            value={count}
            onChange={(e) => setCount(Number(e.target.value) || 3)}
            style={{
              width: '4rem',
              padding: 'var(--space-sm)',
              fontSize: 'var(--text-sm)',
              border: 'var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
            }}
          />
          <input
            value={deckName}
            onChange={(e) => setDeckName(e.target.value)}
            placeholder="Deck name (optional)"
            style={{
              flex: 1,
              minWidth: '10rem',
              padding: 'var(--space-sm)',
              fontSize: 'var(--text-sm)',
              border: 'var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
            }}
          />
          <SoftButton
            disabled={
              generating ||
              (mode === 'document' && (!selectedDocumentId || currentDoc?.status !== 'completed')) ||
              (mode === 'text' && !pastedText.trim())
            }
            onClick={handleGenerate}
          >
            {generating ? 'Generating…' : 'Generate'}
          </SoftButton>
        </div>
      </section>
    </div>
  )
}
