import type { FormEvent } from 'react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { DocumentSummary } from '../types'
import {
  createNote,
  generateNoteFromDocument,
  generateNoteFromText,
  listNotes,
  type Note,
} from '../api/notes'
import { Button } from '../components/Button'
import { NoteContent } from '../components/NoteContent'

interface NotesViewProps {
  selectedDocumentId: string | null
  documents: DocumentSummary[]
}

export function NotesView({ selectedDocumentId, documents }: NotesViewProps) {
  const [notes, setNotes] = useState<Note[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [generatedContent, setGeneratedContent] = useState<string>('')
  const [generatedForDocument, setGeneratedForDocument] = useState<string | null>(null)
  const [generateMode, setGenerateMode] = useState<'document' | 'text'>('document')
  const [pastedText, setPastedText] = useState('')
  const [generating, setGenerating] = useState(false)

  const [manualContent, setManualContent] = useState('')
  const [linkToDoc, setLinkToDoc] = useState(true)

  const loadNotes = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await listNotes(selectedDocumentId ? { documentId: selectedDocumentId } : undefined)
      setNotes(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load notes')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadNotes()
  }, [selectedDocumentId])

  const handleGenerate = async () => {
    try {
      setGenerating(true)
      setError(null)
      if (generateMode === 'document') {
        if (!selectedDocumentId) {
          setError('Select a document in the sidebar first.')
          return
        }
        const content = await generateNoteFromDocument(selectedDocumentId)
        setGeneratedContent(content)
        setGeneratedForDocument(selectedDocumentId)
      } else {
        if (!pastedText.trim()) return
        const content = await generateNoteFromText(pastedText.trim())
        setGeneratedContent(content)
        setGeneratedForDocument(null)
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
      })
      setGeneratedContent('')
      setGeneratedForDocument(null)
      void loadNotes()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
    }
  }

  const handleSaveManual = async (event: FormEvent) => {
    event.preventDefault()
    if (!manualContent.trim()) return
    try {
      await createNote({
        content: manualContent.trim(),
        documentId: linkToDoc ? selectedDocumentId ?? undefined : undefined,
      })
      setManualContent('')
      void loadNotes()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
    }
  }

  const currentDoc = documents.find((d) => d.id === selectedDocumentId)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-sm)',
          padding: 'var(--space-xl)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-sm)',
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-md)',
          }}
        >
          <span>NOTES</span>
          <Link to="/notes" style={{ fontSize: 'var(--text-sm)', fontWeight: 400, color: 'var(--color-accent)' }}>Open in Notes</Link>
        </div>
        {currentDoc && (
          <p style={{ fontSize: 13, marginTop: 0, marginBottom: 'var(--space-sm)' }}>
            Showing notes for: <strong>{currentDoc.display_name ?? currentDoc.filename}</strong>
          </p>
        )}
        <p style={{ fontSize: 13, marginTop: 0 }}>
          Mix your own notes with AI-generated study notes grounded in your documents.
        </p>
        {loading && <p style={{ fontSize: 13 }}>Loading notes…</p>}
        {error && (
          <p style={{ fontSize: 13, color: 'var(--color-danger)' }}>
            {error}
          </p>
        )}
      </section>

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-sm)',
          padding: 'var(--space-xl)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-sm)',
          }}
        >
          GENERATE NOTES (AI)
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
          <Button
            type="button"
            variant={generateMode === 'document' ? 'primary' : 'ghost'}
            onClick={() => setGenerateMode('document')}
          >
            From current document
          </Button>
          <Button
            type="button"
            variant={generateMode === 'text' ? 'primary' : 'ghost'}
            onClick={() => setGenerateMode('text')}
          >
            From pasted text
          </Button>
        </div>
        {generateMode === 'document' ? (
          <p style={{ fontSize: 13, marginTop: 0 }}>
            {currentDoc
              ? `Document: ${currentDoc.display_name ?? currentDoc.filename}`
              : 'Select a document in the sidebar to generate notes from its content.'}
          </p>
        ) : (
          <textarea
            value={pastedText}
            onChange={(event) => setPastedText(event.target.value)}
            placeholder="Paste or type content…"
            rows={4}
            style={{
              width: '100%',
              marginTop: 'var(--space-sm)',
              marginBottom: 'var(--space-md)',
              padding: 'var(--space-md)',
              borderRadius: 'var(--radius-md)',
              border: `1px solid ${'var(--color-border-light)'}`,
              fontFamily: 'var(--font-family)',
              fontSize: 'var(--text-base)',
            }}
          />
        )}
        {generateMode === 'document' && currentDoc && currentDoc.status !== 'completed' && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-md)' }}>
            Wait until the document finishes processing (see sidebar).
          </p>
        )}
        <Button
          type="button"
          onClick={handleGenerate}
          loading={generating}
          disabled={
            generating ||
            (generateMode === 'text' && !pastedText.trim()) ||
            (generateMode === 'document' && (!selectedDocumentId || currentDoc?.status !== 'completed'))
          }
        >
          Generate notes
        </Button>

        {generatedContent && (
          <div
            style={{
              marginTop: 'var(--space-lg)',
              borderTop: '1px solid var(--color-border-strong)',
              paddingTop: 'var(--space-md)',
            }}
          >
            <div
              style={{
                fontSize: 11,
                textTransform: 'uppercase',
                letterSpacing: '0.12em',
                marginBottom: 'var(--space-sm)',
              }}
            >
              Generated notes
            </div>
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
            <div style={{ display: 'flex', gap: 'var(--space-md)', marginTop: 'var(--space-md)' }}>
              <Button type="button" onClick={handleSaveGenerated}>
                Save as note
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
          </div>
        )}
      </section>

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-sm)',
          padding: 'var(--space-xl)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-sm)',
          }}
        >
          ADD NOTE (MANUAL)
        </div>
        <form onSubmit={handleSaveManual} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          <textarea
            value={manualContent}
            onChange={(event) => setManualContent(event.target.value)}
            placeholder="Write your note…"
            rows={4}
            style={{
              width: '100%',
              padding: 'var(--space-md)',
              borderRadius: 'var(--radius-md)',
              border: `1px solid ${'var(--color-border-light)'}`,
              fontFamily: 'var(--font-family)',
              fontSize: 'var(--text-base)',
            }}
          />
          <label style={{ fontSize: 13 }}>
            <input
              type="checkbox"
              checked={linkToDoc}
              onChange={(event) => setLinkToDoc(event.target.checked)}
              style={{ marginRight: 6 }}
            />
            Link to current document
          </label>
          <Button type="submit" disabled={!manualContent.trim()}>
            Save note
          </Button>
        </form>
      </section>

      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-sm)',
          padding: 'var(--space-xl)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-sm)',
          }}
        >
          YOUR NOTES
        </div>
        {notes.length === 0 ? (
          <p style={{ fontSize: 'var(--text-sm)' }}>No notes yet.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
            {notes.map((note) => (
              <article
                key={note.id}
                style={{
                  border: '1px solid var(--color-border-strong)',
                  borderRadius: 'var(--radius-sm)',
                  padding: 'var(--space-md)',
                  backgroundColor: 'var(--color-bg)',
                }}
              >
                <NoteContent content={note.content} style={{ fontSize: 14 }} />
                <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 'var(--space-sm)' }}>
                  {(note.display_name ?? note.filename) && <span>Source: {note.display_name ?? note.filename}</span>}
                  {note.updated_at && <span> · Updated: {note.updated_at.slice(0, 19)}</span>}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

