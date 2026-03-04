import { useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import type { FlashcardCard } from '@/api/flashcards'
import { listCards } from '@/api/flashcards'
import { Button } from '@/components/ui/button'
import { LatexText } from '@/components/LatexText'
import { X } from 'lucide-react'
import { motion as motionTokens } from '@/design/tokens'

const FLIP_DURATION = 0.4
type Difficulty = 'new' | 'review' | 'mastered'

interface FlashcardStudyViewProps {
  deckId: string
  deckName: string
  onExit: () => void
}

export function FlashcardStudyView({ deckId, deckName, onExit }: FlashcardStudyViewProps) {
  const [cards, setCards] = useState<FlashcardCard[]>([])
  const [cardIndex, setCardIndex] = useState(0)
  const [showBack, setShowBack] = useState(false)
  const [ratings, setRatings] = useState<Record<string, Difficulty>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const currentCard = cards.length > 0 ? cards[cardIndex % cards.length] : null
  const totalCards = cards.length
  const completedCount = Object.keys(ratings).length
  const progressPct = totalCards > 0 ? Math.round((completedCount / totalCards) * 100) : 0
  const toReviewAgain = Object.values(ratings).filter((r) => r === 'review').length
  const sessionComplete = totalCards > 0 && completedCount >= totalCards

  const loadCards = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await listCards(deckId)
      setCards(data)
      setCardIndex(0)
      setShowBack(false)
      setRatings({})
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load cards')
    } finally {
      setLoading(false)
    }
  }, [deckId])

  useEffect(() => {
    void loadCards()
  }, [loadCards])

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

  useEffect(() => {
    if (!currentCard || cards.length === 0) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') handleNext()
      else if (e.key === 'ArrowLeft') handlePrev()
      else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault()
        setShowBack((s) => !s)
      } else if (e.key === '1' || e.key === '2') handleRate('review')
      else if (e.key === '3' || e.key === '4') handleRate('mastered')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [currentCard?.id, cards.length])

  if (loading) return <p className="text-sm text-muted-foreground">Loading cards…</p>
  if (error) return <p className="text-sm text-destructive">{error}</p>
  if (cards.length === 0) return <p className="text-sm text-muted-foreground">No cards in this deck.</p>

  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-8 p-6">
      <div className="absolute left-0 right-0 top-0 z-10 h-0.5 bg-muted">
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
        <span>{deckName}</span>
      </div>

      {sessionComplete ? (
        <div className="flex w-full max-w-md flex-col items-center gap-6 rounded-xl border border-border bg-card p-8 text-center">
          <p className="text-lg font-semibold text-foreground">Session complete</p>
          <p className="text-sm text-muted-foreground">
            You reviewed {totalCards} card{totalCards === 1 ? '' : 's'}.
            {toReviewAgain > 0 && ` ${toReviewAgain} to review again.`}
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            <Button onClick={() => { setCardIndex(0); setShowBack(false); setRatings({}); void loadCards() }}>
              Study again
            </Button>
            <Button variant="secondary" onClick={onExit}>
              <X className="mr-2 h-4 w-4" />
              Back to home
            </Button>
          </div>
        </div>
      ) : currentCard ? (
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
            <Button variant="secondary" size="sm" onClick={() => handleRate('review')}>Again</Button>
            <Button variant="secondary" size="sm" onClick={() => handleRate('review')}>Hard</Button>
            <Button variant="secondary" size="sm" onClick={() => handleRate('mastered')}>Good</Button>
            <Button variant="secondary" size="sm" onClick={() => handleRate('mastered')}>Easy</Button>
          </div>
          <p className="text-xs text-muted-foreground">← → navigate · Space flip · 1–4 rate</p>
          <Button variant="ghost" size="sm" onClick={onExit}>
            <X className="mr-2 h-4 w-4" />
            Exit study
          </Button>
        </>
      ) : null}
    </div>
  )
}
