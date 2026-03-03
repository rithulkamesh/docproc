import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { DocumentSummary } from '../types'
import {
  createNote,
  generateNoteFromDocument,
  generateNoteFromText,
  listNotes,
  type Note,
  updateNote,
} from '../api/notes'
import { jsPDF } from 'jspdf'
import { Button } from './Button'
import { NoteContent } from './NoteContent'

interface NotesModuleProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

type SavingState = 'idle' | 'saving' | 'saved'

export function NotesModule({ documents, selectedDocumentId, projectId }: NotesModuleProps) {
  const [notes, setNotes] = useState<Note[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [generatedContent, setGeneratedContent] = useState<string>('')
  const [generatedForDocument, setGeneratedForDocument] = useState<string | null>(null)
  const [generateMode, setGenerateMode] = useState<'document' | 'text'>('document')
  const [pastedText, setPastedText] = useState('')
  const [generating, setGenerating] = useState(false)
  const [summaryOpen, setSummaryOpen] = useState(true)

  const [savingById, setSavingById] = useState<Record<string, SavingState>>({})
  const [localContent, setLocalContent] = useState<Record<string, string>>({})
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null)
  const saveTimers = useRef<Record<string, number>>({})

  const currentDoc = documents.find((d) => d.id === selectedDocumentId) ?? null

  const loadNotes = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await listNotes({ projectId })
      setNotes(data)
      setLocalContent(
        data.reduce<Record<string, string>>((acc, n) => {
          acc[n.id] = n.content
          return acc
        }, {}),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load notes')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadNotes()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId])

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (generateMode === 'document') {
        if (!selectedDocumentId) {
          setError('Select a document in the left column first.')
          return
        }
        const content = await generateNoteFromDocument(selectedDocumentId)
        setGeneratedContent(content)
        setGeneratedForDocument(selectedDocumentId)
        setSummaryOpen(true)
      } else {
        if (!pastedText.trim()) return
        const content = await generateNoteFromText(pastedText.trim())
        setGeneratedContent(content)
        setGeneratedForDocument(null)
        setSummaryOpen(true)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to generate notes')
    } finally {
      setGenerating(false)
    }
  }

  const handleSaveGenerated = async () => {
    if (!generatedContent.trim()) return
    try {
      await createNote({ content: generatedContent, documentId: generatedForDocument ?? undefined, projectId })
      setGeneratedContent('')
      setGeneratedForDocument(null)
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
        setNotes((prev) => prev.map((n) => (n.id === noteId ? { ...n, content: updated.content, updated_at: updated.updated_at } : n)))
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
      const baseContent = currentDoc ? `Section for: ${currentDoc.filename}\n` : ''
      const created = await createNote({ content: baseContent, documentId: currentDoc?.id ?? undefined, projectId })
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

    // Branding header
    addText('docproc // edu', 18, true)
    y += 2
    addText('Project Notes', 12, true)
    y += lineHeight
    addText(`Project: ${projectId}  ·  Generated: ${new Date().toLocaleString()}`, 9)
    y += sectionGap

    if (generatedContent.trim()) {
      ensureSpace(lineHeight * 2)
      addText('Generated summary', 11, true)
      y += 2
      addText(generatedContent.trim(), 10)
      y += sectionGap
    }

    if (notes.length > 0) {
      notes.forEach((note, idx) => {
        const linkedDoc = documents.find((d) => d.id === note.document_id) ?? null
        const sectionTitle = linkedDoc
          ? `Section ${idx + 1} — ${linkedDoc.filename}`
          : `Section ${idx + 1}`
        ensureSpace(lineHeight * 3)
        addText(sectionTitle, 11, true)
        y += 2
        addText(note.content.trim() || '(No content)', 10)
        y += sectionGap
      })
    } else {
      ensureSpace(lineHeight)
      addText('No saved notes yet for this project.', 10)
    }

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
    doc.save(`docproc-notes-${projectId}-${stamp}.pdf`)
  }

  const renderSummaryBody = () => {
    if (!generatedContent) return null
    return (
      <div
        style={{
          border: `1px solid ${'var(--color-border-light)'}`,
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'var(--color-bg)',
          padding: 'var(--space-md)',
        }}
      >
        <NoteContent content={generatedContent} />
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      <div
        style={{
          fontSize: 'var(--text-xs)',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: 'var(--color-text-muted)',
          marginBottom: 'var(--space-sm)',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--space-md)',
        }}
      >
        <span>Notes live at the project level. Sections can still be linked to specific documents.</span>
        <Link to="/notes" style={{ fontSize: 'var(--text-sm)', fontWeight: 400, color: 'var(--color-accent)', textTransform: 'none' }}>Open in Notes</Link>
      </div>
      {loading && (
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>Loading notes…</p>
      )}
      {error && (
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-danger)', margin: 0 }}>
          {error}
        </p>
      )}

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          padding: 'var(--space-md)',
          backgroundColor: 'var(--color-bg)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--space-sm)',
            marginBottom: 'var(--space-sm)',
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
            Generated summary
          </div>
          <button
            type="button"
            onClick={() => setSummaryOpen((o) => !o)}
            style={{
              border: '1px solid var(--color-border-strong)',
              backgroundColor: 'var(--color-bg-alt)',
              fontSize: 'var(--text-xs)',
              padding: '2px 6px',
              cursor: 'pointer',
            }}
          >
            {summaryOpen ? 'HIDE' : 'SHOW'}
          </button>
        </div>
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginTop: 0 }}>
          Auto-generated overview of the current document or pasted text. Save it into your sections when it looks right.
        </p>
        {summaryOpen && generatedContent && (
          <>
            {renderSummaryBody()}
            <div style={{ display: 'flex', gap: 'var(--space-sm)', marginTop: 'var(--space-sm)' }}>
              <Button type="button" onClick={handleSaveGenerated}>
                Save as section
              </Button>
              <Button
                type="button"
                variant="ghost"
                onClick={() => {
                  setGeneratedContent('')
                  setGeneratedForDocument(null)
                }}
              >
                Clear
              </Button>
            </div>
          </>
        )}
        {!generatedContent && (
          <div
            style={{
              marginTop: 'var(--space-sm)',
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-sm)',
            }}
          >
            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
              <Button
                type="button"
                variant={generateMode === 'document' ? 'primary' : 'ghost'}
                onClick={() => setGenerateMode('document')}
              >
                From document
              </Button>
              <Button
                type="button"
                variant={generateMode === 'text' ? 'primary' : 'ghost'}
                onClick={() => setGenerateMode('text')}
              >
                From text
              </Button>
            </div>
            {generateMode === 'document' ? (
              <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', margin: 0 }}>
                {currentDoc
                  ? `Using: ${currentDoc.filename}`
                  : 'Select a document in the left column to generate a summary.'}
              </p>
            ) : (
              <textarea
                value={pastedText}
                onChange={(event) => setPastedText(event.target.value)}
                placeholder="Paste content to summarize…"
                rows={3}
                style={{
                  width: '100%',
                  padding: 'var(--space-sm)',
                  borderRadius: 'var(--radius-md)',
                  border: `1px solid ${'var(--color-border-light)'}`,
                  fontFamily: 'var(--font-family)',
                  fontSize: 'var(--text-base)',
                }}
              />
            )}
            <Button
              type="button"
              onClick={handleGenerate}
              loading={generating}
              disabled={
                generating ||
                (generateMode === 'document' && (!selectedDocumentId || currentDoc?.status !== 'completed')) ||
                (generateMode === 'text' && !pastedText.trim())
              }
            >
              Generate summary
            </Button>
          </div>
        )}
      </section>

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          padding: 'var(--space-md)',
          backgroundColor: 'var(--color-bg)',
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
            Sections
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
            <Button type="button" variant="ghost" onClick={handleAddSection}>
              + Add section
            </Button>
            <Button type="button" variant="ghost" onClick={handleDownloadPdf}>
              Download PDF
            </Button>
          </div>
        </div>
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', margin: 0 }}>
          Each section is an editable note. Changes auto-save.
        </p>
        {notes.length === 0 ? (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginTop: 'var(--space-sm)' }}>
            No sections yet. Start from the summary above or add a blank section.
          </p>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-sm)',
              maxHeight: 260,
              overflowY: 'auto',
            }}
          >
            {notes.map((note) => {
              const content = localContent[note.id] ?? note.content
              const saving = savingById[note.id] ?? 'idle'
              const linkedDoc = documents.find((d) => d.id === note.document_id) ?? null
              return (
                <article
                  key={note.id}
                  style={{
                    border: '1px solid var(--color-border-strong)',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--color-bg-alt)',
                    padding: 'var(--space-sm)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 4,
                  }}
                >
                  {editingNoteId === note.id ? (
                    <textarea
                      value={content}
                      onChange={(event) => handleChangeNote(note, event.target.value)}
                      onBlur={() => setEditingNoteId(null)}
                      autoFocus
                      rows={5}
                      style={{
                        width: '100%',
                        resize: 'vertical',
                        borderRadius: 'var(--radius-md)',
                        border: `1px solid ${'var(--color-border-light)'}`,
                        padding: 'var(--space-sm)',
                        fontFamily: 'var(--font-family)',
                        fontSize: 'var(--text-base)',
                        lineHeight: 1.5,
                        backgroundColor: 'var(--color-bg)',
                      }}
                    />
                  ) : (
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => setEditingNoteId(note.id)}
                      onKeyDown={(e) => e.key === 'Enter' && setEditingNoteId(note.id)}
                      style={{ cursor: 'text', padding: 'var(--space-xs)', borderRadius: 'var(--radius-sm)' }}
                    >
                      {content ? (
                        <NoteContent content={content} style={{ fontSize: 'var(--text-sm)' }} />
                      ) : (
                        <span style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-xs)' }}>
                          Click to edit (Markdown + $...$ math)
                        </span>
                      )}
                    </div>
                  )}
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      fontSize: 'var(--text-xs)',
                      color: 'var(--color-text-muted)',
                    }}
                  >
                    <div>
                      {linkedDoc && <span>Source: {linkedDoc.filename}</span>}
                      {note.updated_at && (
                        <span>
                          {' '}
                          · Updated {note.updated_at.slice(0, 19).replace('T', ' ')}
                        </span>
                      )}
                    </div>
                    <div>
                      {saving === 'saving' && <span>Saving…</span>}
                      {saving === 'saved' && <span>Saved</span>}
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}

