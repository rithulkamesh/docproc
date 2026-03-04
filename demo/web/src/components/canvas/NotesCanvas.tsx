import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createNote,
  listNotes,
  updateNote,
  type Note,
} from '@/api/notes'
import { jsPDF } from 'jspdf'
import { useWorkspace } from '@/context/WorkspaceContext'
import { MarkdownWithMathRenderer } from '@/components/MarkdownWithMathRenderer'
import { GenerateSummaryModal } from '@/components/GenerateSummaryModal'
import { Button } from '@/components/ui/button'
import { Download, Sparkles } from 'lucide-react'

const DOC_MAX_WIDTH = 760
const DOC_PADDING = 32
const DOC_MIN_HEIGHT = 500
const DOC_BORDER_RADIUS = 12
const SAVE_DEBOUNCE_MS = 600

/** Single-document notes workspace: one note per project, document-style layout. */
export function NotesCanvas() {
  const { documents, selectedDocumentId, currentProjectId } = useWorkspace()
  const [note, setNote] = useState<Note | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [documentContent, setDocumentContent] = useState('')
  const [documentTitle, setDocumentTitle] = useState('')
  const [generateModalOpen, setGenerateModalOpen] = useState(false)
  const saveTimerRef = useRef<number | null>(null)
  const latestContentRef = useRef(documentContent)
  const latestTitleRef = useRef(documentTitle)
  latestContentRef.current = documentContent
  latestTitleRef.current = documentTitle

  const loadOrCreateNote = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const list = await listNotes({ projectId: currentProjectId, orderBy: 'updated_at' })
      if (list.length > 0) {
        const n = list[0]
        setNote(n)
        setDocumentContent(n.content ?? '')
        setDocumentTitle(n.title?.trim() ?? '')
      } else {
        const created = await createNote({
          projectId: currentProjectId,
          title: 'Untitled',
          content: '',
        })
        setNote(created)
        setDocumentContent(created.content ?? '')
        setDocumentTitle(created.title?.trim() ?? 'Untitled')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load notes')
    } finally {
      setLoading(false)
    }
  }, [currentProjectId])

  useEffect(() => {
    void loadOrCreateNote()
  }, [loadOrCreateNote])

  const scheduleSave = useCallback(() => {
    if (!note) return
    if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current)
    saveTimerRef.current = window.setTimeout(async () => {
      try {
        const content = latestContentRef.current
        const title = latestTitleRef.current
        await updateNote(note.id, { content, title: title || undefined })
        saveTimerRef.current = null
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to save')
      }
    }, SAVE_DEBOUNCE_MS)
  }, [note])

  const handleContentChange = useCallback(
    (markdown: string) => {
      setDocumentContent(markdown)
      scheduleSave()
    },
    [scheduleSave]
  )

  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDocumentTitle(e.target.value)
    scheduleSave()
  }

  const handleInsertSummary = useCallback(
    (markdown: string) => {
      if (!markdown.trim()) return
      const combined = documentContent.trim()
        ? `${documentContent.trim()}\n\n${markdown.trim()}`
        : markdown.trim()
      setDocumentContent(combined)
      latestContentRef.current = combined
      if (note) scheduleSave()
    },
    [documentContent, note, scheduleSave]
  )

  const handleExportPdf = useCallback(() => {
    const doc = new jsPDF({ format: 'a4', unit: 'mm' })
    const margin = 20
    const pageW = doc.internal.pageSize.getWidth()
    const pageH = doc.internal.pageSize.getHeight()
    const maxW = pageW - margin * 2
    let y = margin
    const lineHeight = 5

    const addText = (text: string, fontSize: number, isBold = false) => {
      doc.setFontSize(fontSize)
      doc.setFont('helvetica', isBold ? 'bold' : 'normal')
      const lines = doc.splitTextToSize(text, maxW)
      for (const line of lines) {
        if (y > pageH - margin) {
          doc.addPage()
          y = margin
        }
        doc.text(line, margin, y)
        y += lineHeight
      }
    }

    addText('docproc // edu', 18, true)
    y += 2
    addText(documentTitle || 'Project Notes', 14, true)
    y += 4
    addText(`Project: ${currentProjectId}  ·  ${new Date().toLocaleString()}`, 9)
    y += 6
    addText(documentContent.trim() || '(No content)', 10)
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
    doc.save(`docproc-notes-${currentProjectId}-${stamp}.pdf`)
  }, [currentProjectId, documentTitle, documentContent])

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        Loading notes…
      </div>
    )
  }

  return (
    <div className="flex flex-1 flex-col min-h-0">
      <header
        className="flex shrink-0 items-center justify-between gap-4 border-b border-border bg-background/95 px-4 py-2"
        style={{ minHeight: 44 }}
      >
        <span className="text-sm font-medium text-foreground">Notes</span>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={handleExportPdf}>
            <Download className="mr-2 h-4 w-4" />
            Export PDF
          </Button>
          <Button size="sm" onClick={() => setGenerateModalOpen(true)}>
            <Sparkles className="mr-2 h-4 w-4" />
            Generate AI summary
          </Button>
        </div>
      </header>

      {error && (
        <div className="shrink-0 px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <main className="flex-1 overflow-auto py-8">
        <div
          className="mx-auto bg-card border border-border shadow-sm"
          style={{
            maxWidth: DOC_MAX_WIDTH * 1.6,
            padding: DOC_PADDING,
            minHeight: DOC_MIN_HEIGHT,
            borderRadius: DOC_BORDER_RADIUS,
          }}
        >
          <input
            type="text"
            value={documentTitle}
            onChange={handleTitleChange}
            placeholder="Document title"
            className="mb-4 w-full border-0 bg-transparent p-0 text-[26px] font-semibold leading-tight text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-0"
          />
          <div
            className="grid gap-6 md:grid-cols-2"
            style={{ minHeight: 400 }}
          >
            <div className="flex flex-col min-h-0">
              <label className="text-xs font-medium text-muted-foreground mb-1">
                Source (Markdown + LaTeX)
              </label>
              <textarea
                value={documentContent}
                onChange={(e) => handleContentChange(e.target.value)}
                placeholder="Start writing… Use Markdown and LaTeX. Inline math: $...$  Block math: $$...$$"
                className="flex-1 w-full resize-none rounded-md border border-border bg-background px-3 py-2 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/20"
                style={{ minHeight: 320 }}
              />
            </div>
            <div className="flex flex-col min-h-0 overflow-auto">
              <span className="text-xs font-medium text-muted-foreground mb-1">
                Rendered
              </span>
              <div
                className="flex-1 rounded-md border border-border bg-muted/30 px-3 py-2 overflow-auto"
                style={{ minHeight: 320 }}
              >
                <MarkdownWithMathRenderer content={documentContent} />
              </div>
            </div>
          </div>
        </div>
      </main>

      <GenerateSummaryModal
        open={generateModalOpen}
        onOpenChange={setGenerateModalOpen}
        onInsert={handleInsertSummary}
        documents={documents}
        selectedDocumentId={selectedDocumentId}
      />
    </div>
  )
}
