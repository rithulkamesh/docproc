import { useCallback, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import type { FlashcardCard, FlashcardDeck } from '@/api/flashcards'
import {
  deleteDeck,
  generateFlashcardsFromDocument,
  generateFlashcardsFromText,
  listCards,
  listDecks,
} from '@/api/flashcards'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Loader2, Play, Trash2, X } from 'lucide-react'
import { motion as motionTokens } from '@/design/tokens'

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
          count,
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
  const completedCount = Object.keys(ratings).length
  const progressPct = totalCards > 0 ? Math.round((completedCount / totalCards) * 100) : 0

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
          <span>{progressPct}%</span>
          <span>{studyDeckName}</span>
        </div>

        {cardsLoading && <p className="text-sm text-muted-foreground">Loading cards…</p>}
        {!cardsLoading && cards.length === 0 && (
          <p className="text-sm text-muted-foreground">No cards in this deck.</p>
        )}

        {!cardsLoading && currentCard && (
          <>
            <div
              className="relative w-full max-w-[min(42rem,60vw)]"
              style={{ aspectRatio: '4/3', perspective: '1000px' }}
            >
              <motion.div
                onClick={() => setShowBack((s) => !s)}
                className="absolute inset-0 cursor-pointer"
                style={{ transformStyle: 'preserve-3d' }}
                animate={{ rotateY: showBack ? 180 : 0 }}
                transition={{ duration: FLIP_DURATION, ease: 'easeInOut' }}
              >
                <div
                  className="absolute inset-0 flex items-center justify-center rounded-xl border border-border bg-card p-8 text-center text-lg font-medium [backface-visibility:hidden]"
                  style={{ WebkitBackfaceVisibility: 'hidden' }}
                >
                  {currentCard.front}
                </div>
                <div
                  className="absolute inset-0 flex items-center justify-center rounded-xl border border-border bg-card p-8 text-center text-lg font-medium [backface-visibility:hidden]"
                  style={{
                    WebkitBackfaceVisibility: 'hidden',
                    transform: 'rotateY(180deg)',
                  }}
                >
                  {currentCard.back}
                </div>
              </motion.div>
            </div>

            <div className="flex flex-wrap justify-center gap-2">
              <Button variant="secondary" size="sm" onClick={() => handleRate('review')}>
                Easy
              </Button>
              <Button variant="secondary" size="sm" onClick={() => handleRate('review')}>
                Medium
              </Button>
              <Button variant="secondary" size="sm" onClick={() => handleRate('mastered')}>
                Hard
              </Button>
              <Button variant="secondary" size="sm" onClick={handleNext}>
                Skip
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              ← → navigate · Space flip · 1/2/3 rate
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
    <div className="flex flex-col gap-8">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Flashcards
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Decks and spaced review. Generate from a document or pasted text, then study.
        </p>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <Card className="p-6">
        <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Your decks
        </h3>
        {loadingDecks && (
          <p className="text-sm text-muted-foreground">Loading decks…</p>
        )}
        {!loadingDecks && decks.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No decks yet. Generate one below from a document or pasted text.
          </p>
        )}
        {!loadingDecks && decks.length > 0 && (
          <ul className="space-y-2">
            {decks.map((deck) => (
              <li key={deck.id}>
                <Card className="flex items-center justify-between gap-4 p-4">
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
                </Card>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card className="p-6">
        <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Generate deck
        </h3>
        <div className="flex gap-2">
          <Button
            variant={mode === 'document' ? 'default' : 'secondary'}
            size="sm"
            onClick={() => setMode('document')}
          >
            From document
          </Button>
          <Button
            variant={mode === 'text' ? 'default' : 'secondary'}
            size="sm"
            onClick={() => setMode('text')}
          >
            From text
          </Button>
        </div>
        {mode === 'document' && (
          <p className="mt-2 text-sm text-muted-foreground">
            {currentDoc ? `Using: ${currentDoc.filename}` : 'Select a document in Sources first.'}
          </p>
        )}
        {mode === 'text' && (
          <textarea
            value={pastedText}
            onChange={(e) => setPastedText(e.target.value)}
            placeholder="Paste content…"
            rows={4}
            className="mt-2 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
        )}
        <div className="mt-4 flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <Label htmlFor="count">Count (3–20)</Label>
            <Input
              id="count"
              type="number"
              min={3}
              max={20}
              value={count}
              onChange={(e) => setCount(Number(e.target.value) || 8)}
              className="w-20"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="deckName">Deck name (optional)</Label>
            <Input
              id="deckName"
              value={deckName}
              onChange={(e) => setDeckName(e.target.value)}
              placeholder="My deck"
              className="w-40"
            />
          </div>
          <Button onClick={handleGenerate} disabled={generating}>
            {generating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Generate
          </Button>
        </div>
      </Card>
    </div>
  )
}
