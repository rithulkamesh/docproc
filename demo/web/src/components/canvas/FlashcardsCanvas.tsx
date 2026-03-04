import { useCallback, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import type { FlashcardCard, FlashcardDeck } from '@/api/flashcards'
import {
  deleteDeck,
  generateFlashcardsFromDocuments,
  listCards,
  listDecks,
} from '@/api/flashcards'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Button } from '@/components/ui/button'
import { LatexText } from '@/components/LatexText'
import { Loader2, Play, Trash2, X } from 'lucide-react'
import { motion as motionTokens } from '@/design/tokens'

const FLIP_DURATION = 0.4

type Difficulty = 'new' | 'review' | 'mastered'

export function FlashcardsCanvas() {
  const { documents, currentProjectId } = useWorkspace()
  const [decks, setDecks] = useState<FlashcardDeck[]>([])
  const [loadingDecks, setLoadingDecks] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)
  const [studyActive, setStudyActive] = useState(false)
  const [studyDeckId, setStudyDeckId] = useState<string | null>(null)
  const [studyDeckName, setStudyDeckName] = useState('')
  const [cards, setCards] = useState<FlashcardCard[]>([])
  const [cardIndex, setCardIndex] = useState(0)
  const [showBack, setShowBack] = useState(false)
  const [ratings, setRatings] = useState<Record<string, Difficulty>>({})
  const [cardsLoading, setCardsLoading] = useState(false)
  const [successToast, setSuccessToast] = useState<string | null>(null)

  const completedDocs = documents.filter((d) => d.status === 'completed')
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
      else if (e.key === '4') handleRate('mastered')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  const handleGenerate = async () => {
    if (completedDocs.length === 0) {
      setError('Add and process at least one document in Sources first.')
      return
    }
    try {
      setGenerating(true)
      setError(null)
      await generateFlashcardsFromDocuments({
        documentIds: completedDocs.map((d) => d.id),
        projectId: currentProjectId,
      })
      await loadDecks()
      setSuccessToast('Flashcards generated successfully.')
      setTimeout(() => setSuccessToast(null), 4000)
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
  const completedCount = Object.keys(ratings).length
  const progressPct = totalCards > 0 ? Math.round((completedCount / totalCards) * 100) : 0
  const toReviewAgain = Object.values(ratings).filter((r) => r === 'review').length
  const sessionComplete = totalCards > 0 && completedCount >= totalCards

  if (studyActive) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-8 p-6">
        <div className="fixed left-0 right-0 top-0 z-10 h-0.5 bg-muted">
          <motion.div
            className="h-full bg-primary"
            initial={{ width: 0 }}
            animate={{ width: `${progressPct}%` }}
            transition={{
              duration: motionTokens.durationPanel / 1000,
              ease: motionTokens.easingFramer,
            }}
          />
        </div>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>{cardIndex + 1} / {totalCards}</span>
          <span>{studyDeckName}</span>
        </div>

        {cardsLoading && <p className="text-sm text-muted-foreground">Loading cards…</p>}
        {!cardsLoading && cards.length === 0 && (
          <p className="text-sm text-muted-foreground">No cards in this deck.</p>
        )}

        {!cardsLoading && sessionComplete && (
          <div className="flex w-full max-w-md flex-col items-center gap-6 rounded-xl border border-border bg-card p-8 text-center">
            <p className="text-lg font-semibold text-foreground">Session complete</p>
            <p className="text-sm text-muted-foreground">
              You reviewed {totalCards} card{totalCards === 1 ? '' : 's'}.
              {toReviewAgain > 0 && ` ${toReviewAgain} to review again.`}
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              <Button onClick={() => { setCardIndex(0); setShowBack(false); setRatings({}) }}>
                Study again
              </Button>
              <Button variant="secondary" onClick={handleExitStudy}>
                <X className="mr-2 h-4 w-4" />
                Back to decks
              </Button>
            </div>
          </div>
        )}

        {!cardsLoading && currentCard && !sessionComplete && (
          <>
            <div
              className="relative w-full max-w-[min(42rem,60vw)]"
              style={{ aspectRatio: '4/3', perspective: '1000px' }}
            >
              <div
                className="absolute inset-0 rounded-2xl opacity-50 blur-3xl pointer-events-none overflow-hidden"
                style={{
                  transform: 'scale(1.15)',
                  background: 'conic-gradient(from 0deg at 50% 50%, hsl(var(--primary) / 0.25), transparent 30%, hsl(var(--primary) / 0.15), transparent 70%)',
                  animation: 'flashcard-ambient-spin 12s linear infinite',
                }}
                aria-hidden
              />
              <motion.div
                onClick={() => setShowBack((s) => !s)}
                className="absolute inset-0 cursor-pointer"
                style={{ transformStyle: 'preserve-3d' }}
                animate={{ rotateY: showBack ? 180 : 0 }}
                transition={{ duration: FLIP_DURATION, ease: 'easeInOut' }}
              >
                <div
                  className="absolute inset-0 flex items-center justify-center rounded-xl border border-border bg-card p-8 text-center text-lg font-medium [backface-visibility:hidden] overflow-auto"
                  style={{ WebkitBackfaceVisibility: 'hidden' }}
                >
                  <LatexText text={currentCard.front} className="inline-block" />
                </div>
                <div
                  className="absolute inset-0 flex items-center justify-center rounded-xl border border-border bg-card p-8 text-center text-lg font-medium [backface-visibility:hidden] overflow-auto"
                  style={{
                    WebkitBackfaceVisibility: 'hidden',
                    transform: 'rotateY(180deg)',
                  }}
                >
                  <LatexText text={currentCard.back} className="inline-block" />
                </div>
              </motion.div>
            </div>

            <p className="text-xs text-muted-foreground">Tap card to flip</p>
            <div className="flex flex-wrap justify-center gap-2">
              <Button variant="secondary" size="sm" onClick={() => handleRate('review')}>
                Again
              </Button>
              <Button variant="secondary" size="sm" onClick={() => handleRate('review')}>
                Hard
              </Button>
              <Button variant="secondary" size="sm" onClick={() => handleRate('mastered')}>
                Good
              </Button>
              <Button variant="secondary" size="sm" onClick={() => handleRate('mastered')}>
                Easy
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              ← → navigate · Space flip · 1–4 rate
            </p>
            <Button variant="ghost" size="sm" onClick={handleExitStudy}>
              <X className="mr-2 h-4 w-4" />
              Exit study
            </Button>
          </>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col space-y-8">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Flashcards
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Decks and spaced review. Generate from all project documents or pasted text, then study.
        </p>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {successToast && (
        <p className="text-sm text-muted-foreground rounded-md bg-muted px-3 py-2">{successToast}</p>
      )}

      <section className="space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Your decks
        </h3>
        {loadingDecks && (
          <p className="text-sm text-muted-foreground">Loading decks…</p>
        )}
        {!loadingDecks && decks.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No decks yet. Generate from all documents or pasted text below.
          </p>
        )}
        {!loadingDecks && decks.length > 0 && (
          <ul className="space-y-2">
            {decks.map((deck) => (
              <li key={deck.id}>
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-muted/10 px-4 py-3">
                  <div>
                    <p className="font-medium">{deck.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {deck.card_count ?? 0} cards
                      {deck.created_at
                        ? ` · ${new Date(deck.created_at).toLocaleDateString()}`
                        : ''}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" onClick={() => handleStartStudy(deck)}>
                      <Play className="mr-1 h-4 w-4" />
                      Study
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive"
                      onClick={() => void handleDeleteDeck(deck.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Generate deck
        </h3>
        <p className="text-sm text-muted-foreground">
          {completedDocs.length > 0
            ? `Generate flashcards from all ${completedDocs.length} document${completedDocs.length === 1 ? '' : 's'} in this project.`
            : 'Add and process documents in Sources first.'}
        </p>
        <Button onClick={handleGenerate} disabled={generating || completedDocs.length === 0}>
          {generating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
          Generate
        </Button>
      </section>
    </div>
  )
}
