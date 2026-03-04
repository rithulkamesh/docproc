import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import {
  createNote,
  generateNoteFromDocument,
  generateNoteFromText,
  listNotes,
  type Note,
  updateNote,
} from '../api/notes'
import { jsPDF } from 'jspdf'
import { theme } from '../design/theme'
import { useWorkspace } from '../context/WorkspaceContext'
import { BlockSection } from './BlockSection'
import { NoteContent } from './NoteContent'
import { SoftButton } from './SoftButton'

type SavingState = 'idle' | 'saving' | 'saved'

const SECTION_GAP = '1.2rem'

function getTitleAndBody(content: string): [string, string] {
  const idx = content.indexOf('\n')
  if (idx === -1) return [content.trim() || 'Untitled', '']
  return [content.slice(0, idx).trim() || 'Untitled', content.slice(idx + 1).trimStart()]
}

export function NotesCanvas() {
  const { documents, selectedDocumentId, currentProjectId } = useWorkspace()
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
      const data = await listNotes({ projectId: currentProjectId })
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
  }, [currentProjectId])

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (generateMode === 'document') {
        if (!selectedDocumentId) {
          setError('Select a document in Sources first.')
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
      await createNote({
        content: generatedContent,
        documentId: generatedForDocument ?? undefined,
        projectId: currentProjectId,
      })
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
        setNotes((prev) =>
          prev.map((n) => (n.id === noteId ? { ...n, content: updated.content, updated_at: updated.updated_at } : n)),
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

  const handleTitleChange = (note: Note, newTitle: string) => {
    const raw = localContent[note.id] ?? note.content
    const [, body] = getTitleAndBody(raw)
    const newContent = newTitle.trim() ? `${newTitle.trim()}\n\n${body}` : body
    handleChangeNote(note, newContent)
  }

  const handleAddSection = async () => {
    try {
      const baseContent = currentDoc ? `Section for: ${currentDoc.display_name ?? currentDoc.filename}\n` : ''
      const created = await createNote({ content: baseContent, documentId: currentDoc?.id ?? undefined, projectId: currentProjectId })
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

    if (notes.length > 0) {
      notes.forEach((note, idx) => {
        const linkedDoc = documents.find((d) => d.id === note.document_id) ?? null
        const sectionTitle = linkedDoc ? `Section ${idx + 1} — ${linkedDoc.display_name ?? linkedDoc.filename}` : `Section ${idx + 1}`
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
    doc.save(`docproc-notes-${currentProjectId}-${stamp}.pdf`)
  }

  const renderSummaryBody = () => {
    if (!generatedContent) return null
    return (
      <div
        style={{
          border: 'var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'var(--color-bg)',
          padding: 'var(--space-lg)',
        }}
      >
        <NoteContent content={generatedContent} />
      </div>
    )
  }

  const surfaceStyle: React.CSSProperties = {
    borderRadius: 'var(--radius-md)',
    border: 'var(--border-subtle)',
    padding: 'var(--space-lg)',
    backgroundColor: 'var(--color-bg-alt)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      <p
        style={{
          margin: 0,
          fontSize: 'var(--text-sm)',
          lineHeight: theme.lineHeight.body,
          color: 'var(--color-text-muted)',
        }}
      >
        Notes live at the project level. Sections can be linked to documents (sidebar).
      </p>
      {loading && (
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
          Loading notes…
        </p>
      )}
      {error && (
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-danger)', margin: 0 }}>{error}</p>
      )}

      <motion.section layout style={surfaceStyle}>
        <button
          type="button"
          onClick={() => setSummaryOpen((o) => !o)}
          style={{
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            border: 'none',
            background: 'none',
            cursor: 'pointer',
            padding: 0,
            marginBottom: 'var(--space-md)',
          }}
        >
          <span
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text)',
            }}
          >
            AI-generated summary
          </span>
          <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
            {summaryOpen ? 'Hide' : 'Show'}
          </span>
        </button>
        <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', marginTop: 0, marginBottom: 'var(--space-md)' }}>
          Generate an overview from a document or pasted text. Save it as a section when ready.
        </p>
        {summaryOpen && generatedContent && (
          <>
            {renderSummaryBody()}
            <div style={{ display: 'flex', gap: 'var(--space-md)', marginTop: 'var(--space-lg)' }}>
              <SoftButton onClick={handleSaveGenerated}>Save as section</SoftButton>
              <SoftButton
                onClick={() => {
                  setGeneratedContent('')
                  setGeneratedForDocument(null)
                }}
              >
                Clear
              </SoftButton>
            </div>
          </>
        )}
        {!generatedContent && summaryOpen && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
            <div style={{ display: 'flex', gap: 'var(--space-md)' }}>
              <SoftButton
                active={generateMode === 'document'}
                onClick={() => setGenerateMode('document')}
              >
                From document
              </SoftButton>
              <SoftButton
                active={generateMode === 'text'}
                onClick={() => setGenerateMode('text')}
              >
                From text
              </SoftButton>
            </div>
            {generateMode === 'document' ? (
              <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
                {currentDoc
                  ? `Using: ${currentDoc.display_name ?? currentDoc.filename}`
                  : 'Switch to Sources and select a document to generate a summary.'}
              </p>
            ) : (
              <textarea
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                placeholder="Paste content to summarize…"
                rows={4}
                style={{
                  width: '100%',
                  padding: 'var(--space-md)',
                  border: '1px solid var(--color-border-light)',
                  borderRadius: 'var(--radius-input)',
                  fontFamily: 'var(--font-family)',
                  fontSize: 'var(--text-base)',
                  background: 'var(--color-bg)',
                  color: 'var(--color-text)',
                }}
              />
            )}
            <SoftButton
              onClick={handleGenerate}
              disabled={
                generating ||
                (generateMode === 'document' &&
                  (!selectedDocumentId || currentDoc?.status !== 'completed')) ||
                (generateMode === 'text' && !pastedText.trim())
              }
            >
              {generating ? 'Generating…' : 'Generate summary'}
            </SoftButton>
          </div>
        )}
      </motion.section>

      <section style={{ display: 'flex', flexDirection: 'column', gap: SECTION_GAP }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
          <span
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
            }}
          >
            Sections
          </span>
          <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
            <SoftButton onClick={handleAddSection}>+ Add section</SoftButton>
            <SoftButton onClick={handleDownloadPdf}>Download PDF</SoftButton>
          </div>
        </div>
        {notes.length === 0 ? (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
            No sections yet. Generate a summary above or add a blank section.
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: SECTION_GAP }}>
            {notes.map((note) => {
              const content = localContent[note.id] ?? note.content
              const [title, body] = getTitleAndBody(content)
              const saving = savingById[note.id] ?? 'idle'
              const linkedDoc = documents.find((d) => d.id === note.document_id) ?? null
              return (
                <BlockSection
                  key={note.id}
                  id={note.id}
                  title={title}
                  onTitleChange={(t) => handleTitleChange(note, t)}
                  saving={saving}
                  meta={
                    <>
                      {linkedDoc && <span>Source: {linkedDoc.display_name ?? linkedDoc.filename}</span>}
                      {note.updated_at && (
                        <span> · Updated {note.updated_at.slice(0, 19).replace('T', ' ')}</span>
                      )}
                    </>
                  }
                >
                  {editingNoteId === note.id ? (
                    <textarea
                      value={body}
                      onChange={(e) => handleChangeNote(note, title ? `${title}\n\n${e.target.value}` : e.target.value)}
                      onBlur={() => setEditingNoteId(null)}
                      autoFocus
                      rows={10}
                      style={{
                        width: '100%',
                        resize: 'vertical',
                        minHeight: '10rem',
                        border: 'var(--border-subtle)',
                        borderRadius: 'var(--radius-md)',
                        padding: 'var(--space-md)',
                        fontFamily: 'var(--font-family)',
                        fontSize: 'var(--text-base)',
                        lineHeight: theme.lineHeight.body,
                        backgroundColor: 'var(--color-bg)',
                        color: 'var(--color-text)',
                      }}
                    />
                  ) : (
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => setEditingNoteId(note.id)}
                      onKeyDown={(e) => e.key === 'Enter' && setEditingNoteId(note.id)}
                      style={{
                        minHeight: '4rem',
                        cursor: 'text',
                        padding: 'var(--space-sm)',
                        borderRadius: 'var(--radius-sm)',
                        outline: 'none',
                      }}
                      className="note-section-body-preview"
                    >
                      {body ? (
                        <NoteContent content={body} />
                      ) : (
                        <span style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-sm)' }}>
                          Click to add content… (Markdown and $...$ / $$...$$ math supported)
                        </span>
                      )}
                    </div>
                  )}
                  {editingNoteId === note.id && (
                    <SoftButton
                      onClick={() => setEditingNoteId(null)}
                      style={{ marginTop: 'var(--space-sm)' }}
                    >
                      Done
                    </SoftButton>
                  )}
                </BlockSection>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}
