import { useEffect, useRef, useState } from 'react'
import {
  createNote,
  generateNoteFromDocument,
  generateNoteFromText,
  listNotes,
  updateNote,
  type Note,
} from '@/api/notes'
import { jsPDF } from 'jspdf'
import { useWorkspace } from '@/context/WorkspaceContext'
import { NoteContent } from '@/components/NoteContent'
import { Button } from '@/components/ui/button'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { ChevronDown, ChevronUp, Plus, Download, Loader2 } from 'lucide-react'

type SavingState = 'idle' | 'saving' | 'saved'

export function NotesCanvas() {
  const { documents, selectedDocumentId, currentProjectId } = useWorkspace()
  const [notes, setNotes] = useState<Note[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [generatedContent, setGeneratedContent] = useState('')
  const [generatedForDocument, setGeneratedForDocument] = useState<string | null>(null)
  const [generateMode, setGenerateMode] = useState<'document' | 'text'>('document')
  const [pastedText, setPastedText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [summaryOpen, setSummaryOpen] = useState(true)
  const [savingById, setSavingById] = useState<Record<string, SavingState>>({})
  const [localContent, setLocalContent] = useState<Record<string, string>>({})
  const [notesSuccessToast, setNotesSuccessToast] = useState<string | null>(null)
  const saveTimers = useRef<Record<string, number>>({})

  const currentDoc = documents.find((d) => d.id === selectedDocumentId) ?? null

  const loadNotes = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await listNotes({ projectId: currentProjectId })
      setNotes(data)
      setLocalContent(
        data.reduce<Record<string, string>>((acc, n) => {
          acc[n.id] = n.content
          return acc
        }, {})
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load notes')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadNotes()
  }, [currentProjectId])

  const completedDocs = documents.filter((d) => d.status === 'completed')

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (generateMode === 'document') {
        if (completedDocs.length === 0) {
          setError('Add and process at least one document in Sources first.')
          return
        }
        const parts: string[] = []
        for (const doc of completedDocs) {
          const content = await generateNoteFromDocument(doc.id)
          parts.push(`## ${doc.display_name ?? doc.filename}\n\n${content}`)
        }
        setGeneratedContent(parts.join('\n\n---\n\n'))
        setGeneratedForDocument(null)
        setSummaryOpen(true)
        setNotesSuccessToast('Summary generated. Save as section when ready.')
      } else {
        if (!pastedText.trim()) return
        const content = await generateNoteFromText(pastedText.trim())
        setGeneratedContent(content)
        setGeneratedForDocument(null)
        setSummaryOpen(true)
        setNotesSuccessToast('Summary generated. Save as section when ready.')
      }
      setTimeout(() => setNotesSuccessToast(null), 4000)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate notes')
    } finally {
      setGenerating(false)
    }
  }

  const handleSaveGenerated = async () => {
    if (!generatedContent.trim()) return
    try {
      await createNote({
        content: generatedContent,
        documentId: generatedForDocument ?? undefined,
        projectId: currentProjectId,
      })
      setGeneratedContent('')
      setGeneratedForDocument(null)
      setNotesSuccessToast('Saved as section.')
      setTimeout(() => setNotesSuccessToast(null), 3000)
      void loadNotes()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save summary as note')
    }
  }

  const scheduleSave = (noteId: string, content: string) => {
    window.clearTimeout(saveTimers.current[noteId])
    setSavingById((prev) => ({ ...prev, [noteId]: 'saving' }))
    saveTimers.current[noteId] = window.setTimeout(async () => {
      try {
        const updated = await updateNote(noteId, { content })
        setNotes((prev) =>
          prev.map((n) =>
            n.id === noteId ? { ...n, content: updated.content, updated_at: updated.updated_at } : n
          )
        )
        setSavingById((prev) => ({ ...prev, [noteId]: 'saved' }))
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to auto-save note')
        setSavingById((prev) => ({ ...prev, [noteId]: 'idle' }))
      }
    }, 600)
  }

  const handleChangeNote = (note: Note, value: string) => {
    setLocalContent((prev) => ({ ...prev, [note.id]: value }))
    scheduleSave(note.id, value)
  }

  const handleAddSection = async () => {
    try {
      const baseContent = currentDoc ? `Section for: ${currentDoc.display_name ?? currentDoc.filename}\n` : ''
      const created = await createNote({
        content: baseContent,
        documentId: currentDoc?.id ?? undefined,
        projectId: currentProjectId,
      })
      setNotes((prev) => [created, ...prev])
      setLocalContent((prev) => ({ ...prev, [created.id]: created.content }))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to add section')
    }
  }

  const handleDownloadPdf = () => {
    const doc = new jsPDF({ format: 'a4', unit: 'mm' })
    const margin = 20
    const pageW = doc.internal.pageSize.getWidth()
    const pageH = doc.internal.pageSize.getHeight()
    const maxW = pageW - margin * 2
    let y = margin
    const lineHeight = 5
    const sectionGap = 8

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

    const ensureSpace = (needed: number) => {
      if (y + needed > pageH - margin) {
        doc.addPage()
        y = margin
      }
    }

    addText('docproc // edu', 18, true)
    y += 2
    addText('Project Notes', 12, true)
    y += lineHeight
    addText(`Project: ${currentProjectId}  ·  Generated: ${new Date().toLocaleString()}`, 9)
    y += sectionGap

    if (generatedContent.trim()) {
      ensureSpace(lineHeight * 2)
      addText('Generated summary', 11, true)
      y += 2
      addText(generatedContent.trim(), 10)
      y += sectionGap
    }

    notes.forEach((note, idx) => {
      const linkedDoc = documents.find((d) => d.id === note.document_id) ?? null
      const sectionTitle = linkedDoc
        ? `Section ${idx + 1} — ${linkedDoc.display_name ?? linkedDoc.filename}`
        : `Section ${idx + 1}`
      ensureSpace(lineHeight * 3)
      addText(sectionTitle, 11, true)
      y += 2
      addText((localContent[note.id] ?? note.content).trim() || '(No content)', 10)
      y += sectionGap
    })

    if (notes.length === 0 && !generatedContent.trim()) {
      ensureSpace(lineHeight)
      addText('No saved notes yet for this project.', 10)
    }

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
    doc.save(`docproc-notes-${currentProjectId}-${stamp}.pdf`)
  }

  return (
    <div className="flex flex-col space-y-8">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Notes
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Sections linked to your documents. Generate a summary from all project documents or pasted text.
        </p>
      </div>

      {loading && <p className="text-sm text-muted-foreground">Loading notes…</p>}
      {error && <p className="text-sm text-destructive">{error}</p>}
      {notesSuccessToast && (
        <p className="text-sm text-muted-foreground rounded-md bg-muted px-3 py-2">{notesSuccessToast}</p>
      )}

      {/* NotesSectionList first so notes feel like a document editor */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={handleAddSection}>
            <Plus className="mr-2 h-4 w-4" />
            Add section
          </Button>
          <Button size="sm" variant="secondary" onClick={handleDownloadPdf}>
            <Download className="mr-2 h-4 w-4" />
            Download PDF
          </Button>
        </div>
        <ul className="space-y-4">
          {notes.map((note) => {
            const content = localContent[note.id] ?? note.content
            const saving = savingById[note.id]
            const linkedDoc = documents.find((d) => d.id === note.document_id)
            return (
              <li key={note.id}>
                <div className="rounded-md border border-border bg-muted/10 p-4 space-y-2">
                  <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                    <span>{linkedDoc ? (linkedDoc.display_name ?? linkedDoc.filename) : 'No document'}</span>
                    <span>{saving === 'saving' ? 'Saving…' : saving === 'saved' ? 'Saved' : ''}</span>
                  </div>
                  <textarea
                    value={content}
                    onChange={(e) => handleChangeNote(note, e.target.value)}
                    className="min-h-[120px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm"
                    placeholder="Note content…"
                  />
                </div>
              </li>
            )
          })}
        </ul>
      </section>

      {/* GenerateSummaryPanel below NotesSectionList */}
      <section className="space-y-4">
        <Collapsible open={summaryOpen} onOpenChange={setSummaryOpen}>
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="flex w-full items-center justify-between rounded-md border border-border/60 bg-muted/20 px-3 py-2 text-left transition-colors hover:bg-muted/40"
            >
              <span className="text-xs font-semibold uppercase tracking-wider text-foreground">
                Generate summary
              </span>
              {summaryOpen ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-2">
            <p className="text-xs text-muted-foreground">
              Generate an overview from all project documents or pasted text. Save as a section when ready.
            </p>
            {generatedContent ? (
              <div className="space-y-4 rounded-md border border-border bg-muted/20 p-4">
                <NoteContent content={generatedContent} className="text-sm" />
                <div className="flex gap-2">
                  <Button size="sm" onClick={handleSaveGenerated}>
                    Save as section
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      setGeneratedContent('')
                      setGeneratedForDocument(null)
                    }}
                  >
                    Clear
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                <div className="flex gap-2">
                  <Button
                    variant={generateMode === 'document' ? 'default' : 'secondary'}
                    size="sm"
                    onClick={() => setGenerateMode('document')}
                  >
                    From all documents
                  </Button>
                  <Button
                    variant={generateMode === 'text' ? 'default' : 'secondary'}
                    size="sm"
                    onClick={() => setGenerateMode('text')}
                  >
                    From text
                  </Button>
                </div>
                {generateMode === 'document' && (
                  <p className="text-sm text-muted-foreground">
                    {completedDocs.length > 0
                      ? `Using all ${completedDocs.length} document${completedDocs.length === 1 ? '' : 's'} in this project.`
                      : 'Add and process documents in Sources first.'}
                  </p>
                )}
                {generateMode === 'text' && (
                  <textarea
                    value={pastedText}
                    onChange={(e) => setPastedText(e.target.value)}
                    placeholder="Paste content to summarize…"
                    rows={4}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  />
                )}
                <Button size="sm" onClick={handleGenerate} disabled={generating}>
                  {generating ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : null}
                  Generate summary
                </Button>
              </div>
            )}
          </CollapsibleContent>
        </Collapsible>
      </section>
    </div>
  )
}
