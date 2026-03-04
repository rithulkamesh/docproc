import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Loader2 } from 'lucide-react'
import { generateNoteFromDocument } from '@/api/notes'
import type { DocumentSummary } from '@/types'

export type GenerateSource = 'all' | 'selected'

export interface GenerateSummaryModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Callback with generated markdown; caller should insert at cursor. */
  onInsert: (markdown: string) => void
  documents: DocumentSummary[]
  selectedDocumentId: string | null
}

export function GenerateSummaryModal({
  open,
  onOpenChange,
  onInsert,
  documents,
  selectedDocumentId,
}: GenerateSummaryModalProps) {
  const [source, setSource] = useState<GenerateSource>('selected')
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const completedDocs = documents.filter((d) => d.status === 'completed')
  const selectedDoc = documents.find((d) => d.id === selectedDocumentId)
  const canGenerateAll = completedDocs.length > 0
  const canGenerateSelected = selectedDoc?.status === 'completed'

  const handleGenerate = async () => {
    setError(null)
    setGenerating(true)
    try {
      if (source === 'all') {
        if (!canGenerateAll) {
          setError('Add and process at least one document in Sources first.')
          setGenerating(false)
          return
        }
        const parts: string[] = []
        for (const doc of completedDocs) {
          const content = await generateNoteFromDocument(doc.id)
          parts.push(`## ${doc.display_name ?? doc.filename}\n\n${content}`)
        }
        const markdown = parts.join('\n\n---\n\n')
        if (!markdown.trim()) {
          setError('No content was generated. The service may not be configured yet.')
          setGenerating(false)
          return
        }
        onInsert(markdown)
      } else {
        if (!canGenerateSelected || !selectedDocumentId) {
          setError('Select a processed document in the sidebar first.')
          setGenerating(false)
          return
        }
        const markdown = await generateNoteFromDocument(selectedDocumentId)
        if (!markdown.trim()) {
          setError('No content was generated. The service may not be configured yet.')
          setGenerating(false)
          return
        }
        onInsert(markdown)
      }
      onOpenChange(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate summary')
    } finally {
      setGenerating(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !generating && onOpenChange(next)}>
      <DialogContent showClose={!generating}>
        <DialogHeader>
          <DialogTitle>Generate AI summary</DialogTitle>
          <DialogDescription>
            Insert a summary at your cursor. Choose the source below.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-4 py-2">
          <fieldset className="flex flex-col gap-3">
            <span className="text-sm font-medium text-foreground">Generate summary from</span>
            <div className="flex flex-col gap-3">
              <label
                className={`flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 transition-colors ${
                  source === 'selected'
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:bg-muted/50'
                } ${!canGenerateSelected ? 'cursor-not-allowed opacity-60' : ''}`}
              >
                <input
                  type="radio"
                  name="source"
                  checked={source === 'selected'}
                  onChange={() => setSource('selected')}
                  disabled={!canGenerateSelected}
                  className="peer sr-only"
                />
                <span
                  className="mt-0.5 flex h-4 w-4 shrink-0 rounded-full border-2 border-input bg-background ring-offset-2 ring-offset-background transition-colors peer-checked:border-primary peer-checked:bg-primary peer-checked:ring-2 peer-checked:ring-primary/30"
                  aria-hidden
                />
                <span className="text-sm text-foreground">
                  Selected document
                  {selectedDoc && (
                    <span className="ml-1 text-muted-foreground">
                      — {selectedDoc.display_name ?? selectedDoc.filename}
                    </span>
                  )}
                </span>
              </label>
              <label
                className={`flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 transition-colors ${
                  source === 'all'
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:bg-muted/50'
                } ${!canGenerateAll ? 'cursor-not-allowed opacity-60' : ''}`}
              >
                <input
                  type="radio"
                  name="source"
                  checked={source === 'all'}
                  onChange={() => setSource('all')}
                  disabled={!canGenerateAll}
                  className="peer sr-only"
                />
                <span
                  className="mt-0.5 flex h-4 w-4 shrink-0 rounded-full border-2 border-input bg-background ring-offset-2 ring-offset-background transition-colors peer-checked:border-primary peer-checked:bg-primary peer-checked:ring-2 peer-checked:ring-primary/30"
                  aria-hidden
                />
                <span className="text-sm text-foreground">
                  All documents
                  {completedDocs.length > 0 && (
                    <span className="ml-1 text-muted-foreground">
                      ({completedDocs.length} processed)
                    </span>
                  )}
                </span>
              </label>
            </div>
          </fieldset>
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
        </div>
        <DialogFooter className="gap-3 sm:gap-2">
          <Button
            type="button"
            variant="outline"
            size="default"
            className="min-w-[100px] rounded-lg border-2 font-medium shadow-sm transition-all hover:bg-accent hover:shadow"
            onClick={() => onOpenChange(false)}
            disabled={generating}
          >
            Cancel
          </Button>
          <Button
            type="button"
            size="default"
            className="min-w-[160px] rounded-lg bg-primary px-5 py-2 font-medium shadow-md transition-all hover:opacity-90 hover:shadow-lg disabled:opacity-70"
            onClick={handleGenerate}
            disabled={
              generating ||
              (source === 'selected' && !canGenerateSelected) ||
              (source === 'all' && !canGenerateAll)
            }
          >
            {generating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                Generating…
              </>
            ) : (
              'Generate and insert'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
