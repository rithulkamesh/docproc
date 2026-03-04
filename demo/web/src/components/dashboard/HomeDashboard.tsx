import { useEffect, useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { useWorkspace } from '@/context/WorkspaceContext'
import { generateFlashcardsFromDocuments, listDecks, deleteDeck } from '@/api/flashcards'
import type { FlashcardDeck } from '@/api/flashcards'
import { listAssessments, deleteAssessment } from '@/api/assessments'
import type { Assessment } from '@/api/assessments'
import { Button } from '@/components/ui/button'
import { DocumentRow } from './DocumentRow'
import { FlashcardStudyView } from '@/components/FlashcardStudyView'
import { ConfirmModal } from '@/components/ConfirmModal'
import { Upload, Plus, Layers, ClipboardList, ArrowRight, Loader2, Trash2 } from 'lucide-react'

export function HomeDashboard() {
  const {
    currentProject,
    documents,
    setCanvasMode,
    setSelectedDocumentId,
    currentProjectId,
  } = useWorkspace()
  const [decks, setDecks] = useState<FlashcardDeck[]>([])
  const { data: assessments = [], isLoading: loadingAssessments, mutate: mutateAssessments } = useSWR<Assessment[]>(
    ['assessments-dashboard', currentProjectId],
    () => listAssessments(currentProjectId),
    { revalidateOnFocus: true }
  )
  const [loadingDecks, setLoadingDecks] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [flashcardError, setFlashcardError] = useState<string | null>(null)
  const [flashcardToast, setFlashcardToast] = useState<string | null>(null)
  const [studyDeck, setStudyDeck] = useState<FlashcardDeck | null>(null)
  const [deckToDelete, setDeckToDelete] = useState<FlashcardDeck | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [assessmentToDelete, setAssessmentToDelete] = useState<Assessment | null>(null)
  const [deletingAssessmentId, setDeletingAssessmentId] = useState<string | null>(null)

  const loadDecks = useCallback(async () => {
    setLoadingDecks(true)
    try {
      const data = await listDecks(undefined, currentProjectId)
      setDecks(data)
    } finally {
      setLoadingDecks(false)
    }
  }, [currentProjectId])

  useEffect(() => {
    void loadDecks()
  }, [loadDecks])

  const completedDocs = documents.filter((d) => d.status === 'completed')
  const firstDoc = completedDocs[0]

  const handleGenerateFlashcards = async () => {
    if (completedDocs.length === 0) {
      setFlashcardError('Add and process at least one document in Sources first.')
      return
    }
    setFlashcardError(null)
    setGenerating(true)
    try {
      await generateFlashcardsFromDocuments({
        documentIds: completedDocs.map((d) => d.id),
        projectId: currentProjectId,
      })
      await loadDecks()
      setFlashcardToast('Flashcards generated successfully.')
      setTimeout(() => setFlashcardToast(null), 4000)
    } catch (e) {
      setFlashcardError(e instanceof Error ? e.message : 'Failed to generate flashcards')
    } finally {
      setGenerating(false)
    }
  }

  const handleConfirmDeleteDeck = async () => {
    if (!deckToDelete) return
    setDeletingId(deckToDelete.id)
    setFlashcardError(null)
    try {
      await deleteDeck(deckToDelete.id)
      await loadDecks()
      setFlashcardToast('Deck deleted.')
      setTimeout(() => setFlashcardToast(null), 3000)
    } catch (e) {
      setFlashcardError(e instanceof Error ? e.message : 'Failed to delete deck')
    } finally {
      setDeletingId(null)
    }
  }

  const formatDeckDate = (createdAt: string | undefined) => {
    if (!createdAt) return ''
    try {
      const d = new Date(createdAt)
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    } catch {
      return ''
    }
  }

  const formatAssessmentDate = (iso: string | undefined) => {
    if (!iso) return ''
    try {
      const d = new Date(iso)
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    } catch {
      return ''
    }
  }

  const handleConfirmDeleteAssessment = async () => {
    if (!assessmentToDelete) return
    setDeletingAssessmentId(assessmentToDelete.id)
    setFlashcardError(null)
    try {
      await deleteAssessment(assessmentToDelete.id)
      await mutateAssessments()
      setFlashcardToast('Test deleted.')
      setTimeout(() => setFlashcardToast(null), 3000)
    } catch (e) {
      setFlashcardError(e instanceof Error ? e.message : 'Failed to delete test')
    } finally {
      setDeletingAssessmentId(null)
      setAssessmentToDelete(null)
    }
  }

  return (
    <>
      <ConfirmModal
        open={deckToDelete !== null}
        onOpenChange={(open) => !open && setDeckToDelete(null)}
        title="Delete deck"
        description={
          deckToDelete
            ? `"${deckToDelete.name}" will be permanently deleted. This cannot be undone.`
            : ''
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={handleConfirmDeleteDeck}
        loading={deletingId !== null}
      />
      <ConfirmModal
        open={assessmentToDelete !== null}
        onOpenChange={(open) => !open && setAssessmentToDelete(null)}
        title="Delete test"
        description={
          assessmentToDelete
            ? `"${assessmentToDelete.title}" will be permanently deleted, including all submissions. This cannot be undone.`
            : ''
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={handleConfirmDeleteAssessment}
        loading={deletingAssessmentId !== null}
      />
      {studyDeck ? (
        <div className="flex flex-col gap-4">
          <Button variant="ghost" size="sm" onClick={() => setStudyDeck(null)} className="self-start">
            ← Back to home
          </Button>
          <FlashcardStudyView
            deckId={studyDeck.id}
            deckName={studyDeck.name}
            onExit={() => setStudyDeck(null)}
          />
        </div>
      ) : (
    <div className="flex flex-col space-y-8">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Study workspace
        </h2>
        <p className="mt-1 text-base text-muted-foreground">
          {currentProject?.name ?? '—'}
        </p>
      </div>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-foreground">
            Documents ({documents.length})
          </h3>
          <Button variant="outline" size="sm" onClick={() => setCanvasMode('sources')}>
            <Upload className="mr-1.5 h-4 w-4" />
            Add document
          </Button>
        </div>
        {documents.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border bg-muted/30 px-4 py-6 text-center">
            <p className="text-sm text-muted-foreground">
              No documents yet. Add a document to start chatting and generating study material.
            </p>
            <Button className="mt-2" onClick={() => setCanvasMode('sources')}>
              <Upload className="mr-2 h-4 w-4" />
              Add document
            </Button>
          </div>
        ) : (
          <ul className="space-y-1.5">
            {documents.map((doc) => (
              <li key={doc.id}>
                <DocumentRow doc={doc} showActions compact />
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-foreground">Flashcard decks</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={handleGenerateFlashcards}
            disabled={generating || completedDocs.length === 0}
          >
            {generating ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> : <Layers className="mr-1.5 h-4 w-4" />}
            Generate
          </Button>
        </div>
        {flashcardError && <p className="text-sm text-destructive">{flashcardError}</p>}
        {flashcardToast && (
          <p className="text-sm text-muted-foreground rounded-md bg-muted px-3 py-2">{flashcardToast}</p>
        )}
        {loadingDecks ? (
          <p className="text-sm text-muted-foreground">Loading decks…</p>
        ) : decks.length === 0 ? (
          <div className="rounded-lg border border-border bg-card px-4 py-3">
            <p className="text-sm text-muted-foreground">
              No decks yet. Use Generate above or create from Chat with “Turn into flashcards”.
            </p>
          </div>
        ) : (
          <ul className="space-y-1.5">
            {decks.slice(0, 10).map((deck) => (
              <li key={deck.id}>
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-card px-3 py-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <Layers className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0">
                      <span className="font-medium truncate block">{deck.name}</span>
                      <span className="text-xs text-muted-foreground">
                        {deck.card_count ?? 0} cards
                        {deck.created_at ? ` · ${formatDeckDate(deck.created_at)}` : ''}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setStudyDeck(deck)}
                    >
                      Study
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={() => setDeckToDelete(deck)}
                      disabled={deletingId === deck.id}
                      title="Delete deck"
                    >
                      {deletingId === deck.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-foreground">Tests</h3>
          <Button variant="outline" size="sm" asChild>
            <Link to="/assessments/create">
              <Plus className="mr-1.5 h-4 w-4" />
              Create assessment
            </Link>
          </Button>
        </div>
        {loadingAssessments ? (
          <p className="text-sm text-muted-foreground">Loading assessments…</p>
        ) : assessments.length === 0 ? (
          <div className="rounded-lg border border-border bg-card px-4 py-3">
            <p className="text-sm text-muted-foreground">
              No assessments yet. Create one from your documents.
            </p>
            <Button variant="ghost" size="sm" className="mt-2" asChild>
              <Link to="/assessments/create">
                <ClipboardList className="mr-2 h-4 w-4" />
                Create assessment
              </Link>
            </Button>
          </div>
        ) : (
          <ul className="space-y-1.5">
            {assessments.slice(0, 8).map((a) => (
              <li key={a.id}>
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-card px-3 py-2">
                  <div className="min-w-0 flex-1">
                    <p className="font-medium truncate" title={a.title}>{a.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatAssessmentDate(a.updated_at || a.created_at)}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <Button variant="secondary" size="sm" asChild>
                      <Link to={`/assessments/${a.id}/take`}>Take</Link>
                    </Button>
                    <Button variant="ghost" size="sm" asChild>
                      <Link to={`/assessments/${a.id}/submissions`}>Submissions</Link>
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={(e) => { e.stopPropagation(); setAssessmentToDelete(a) }}
                      disabled={deletingAssessmentId === a.id}
                      title="Delete test"
                    >
                      {deletingAssessmentId === a.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
        {assessments.length > 8 && (
          <p className="text-xs text-muted-foreground">
            <Link to="/assessments" className="underline hover:text-foreground">View all {assessments.length} tests</Link>
          </p>
        )}
      </section>

      <section className="space-y-4">
        <h3 className="text-sm font-semibold text-foreground">Recommended</h3>
        <ul className="space-y-1.5">
          {firstDoc && (
            <li>
              <button
                type="button"
                onClick={() => {
                  setSelectedDocumentId(firstDoc.id)
                  setCanvasMode('converse')
                }}
                className="flex w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:bg-muted/50"
              >
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                Continue studying {firstDoc.display_name ?? firstDoc.filename}
              </button>
            </li>
          )}
          <li>
            <button
              type="button"
              onClick={() => setCanvasMode('converse')}
              className="flex w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:bg-muted/50"
            >
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              Ask a question in Chat
            </button>
          </li>
          {decks.length > 0 && (
            <li>
              <button
                type="button"
                onClick={() => setStudyDeck(decks[0])}
                className="flex w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:bg-muted/50"
              >
                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                Study flashcards
              </button>
            </li>
          )}
          <li>
            <Link
              to="/assessments"
              className="flex w-full items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-left text-sm transition-colors hover:bg-muted/50"
            >
              <ArrowRight className="h-4 w-4 text-muted-foreground" />
              View all tests
            </Link>
          </li>
        </ul>
      </section>
    </div>
      )}
    </>
  )
}
