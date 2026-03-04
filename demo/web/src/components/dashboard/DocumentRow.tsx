import { Link } from 'react-router-dom'
import { useWorkspace } from '@/context/WorkspaceContext'
import type { DocumentSummary } from '@/types'
import { Button } from '@/components/ui/button'
import { FileText, Loader2, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DocumentRowProps {
  doc: DocumentSummary
  showActions?: boolean
  compact?: boolean
}

export function DocumentRow({ doc, showActions = true, compact }: DocumentRowProps) {
  const {
    setCanvasMode,
    setSelectedDocumentId,
    selectedDocumentId,
  } = useWorkspace()

  const isSelected = selectedDocumentId === doc.id
  const isProcessing = doc.status === 'processing'
  const isCompleted = doc.status === 'completed'
  const isFailed = doc.status === 'failed'

  const statusLabel = (() => {
    if (isProcessing) {
      const p = doc.progress
      const pct =
        p?.percent ??
        (p?.total != null && p.total > 0 && p?.page != null
          ? Math.min(100, Math.round((p.page / p.total) * 100))
          : null)
      return pct != null ? `${pct}%` : (p?.message ?? 'Processing…')
    }
    if (isCompleted) return `Ready · ${doc.pages ?? '?'} pages`
    if (isFailed) return 'Failed'
    return doc.status
  })()

  const handleAction = (mode: 'converse' | 'notes' | 'flashcards' | 'tests', navigateToCreate?: boolean) => {
    setSelectedDocumentId(doc.id)
    if (navigateToCreate && mode === 'tests') {
      return // Link to /assessments/create handles navigation
    }
    setCanvasMode(mode === 'flashcards' ? 'home' : mode)
  }

  const name = (doc.display_name ?? doc.filename) || 'Processing…'

  return (
    <div
      className={cn(
        'grid items-center gap-x-3 gap-y-2 rounded-lg border border-border bg-card px-3 py-2 transition-colors',
        showActions && isCompleted ? 'grid-cols-[1fr_auto_auto]' : 'grid-cols-[1fr_auto]',
        compact ? 'py-1.5' : 'py-2',
        isSelected && 'ring-1 ring-primary bg-primary/5'
      )}
    >
      <button
        type="button"
        onClick={() => setSelectedDocumentId(doc.id)}
        className="flex min-w-0 items-center gap-2 text-left"
      >
        <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
        <span className="min-w-0 truncate font-medium" title={name}>
          {name}
        </span>
      </button>
      <div className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground">
        {isProcessing && (
          <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
            <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
            {statusLabel}
          </span>
        )}
        {isCompleted && <span className="whitespace-nowrap">{statusLabel}</span>}
        {isFailed && (
          <span className="inline-flex items-center gap-1 text-destructive whitespace-nowrap">
            <AlertCircle className="h-3 w-3" />
            {statusLabel}
          </span>
        )}
      </div>
      {showActions && isCompleted && (
        <div className="flex shrink-0 flex-wrap items-center gap-1 justify-end">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={() => handleAction('converse')}
          >
            Chat
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={() => handleAction('notes')}
          >
            Summary
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={() => handleAction('flashcards')}
          >
            Flashcards
          </Button>
          <Button variant="ghost" size="sm" className="h-7 text-xs" asChild>
            <Link to="/assessments/create" onClick={() => setSelectedDocumentId(doc.id)}>
              Test
            </Link>
          </Button>
        </div>
      )}
    </div>
  )
}
