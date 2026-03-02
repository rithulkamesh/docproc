import { useCallback, useEffect, useState } from 'react'
import { Routes, Route, useNavigate, useParams, Link } from 'react-router-dom'
import { useWorkspace } from '../context/WorkspaceContext'
import {
  createNotebook,
  createNote,
  createTag,
  getBacklinks,
  getNote,
  listNotebooks,
  listNotes,
  listTags,
  searchNotes,
  updateNote,
  type Notebook,
  type Note,
  type Tag,
} from '../api/notes'
import { noteSnippet } from '../lib/noteContent'
import { BlockNoteEditor } from '../components/BlockNoteEditor'
import { Button } from '../components/Button'
import { Spinner } from '../components/Spinner'

const SIDEBAR_WIDTH = 220
const LIST_WIDTH = 280

function NotesSidebar({
  projectId,
  notebooks,
  tags,
  loading,
  selectedNotebookId,
  selectedTagId,
  searchActive,
  onAllNotes,
  onNotebook,
  onTag,
  onSearch,
  onNewNotebook,
  onNewNote,
  onRefresh,
}: {
  projectId: string
  notebooks: Notebook[]
  tags: Tag[]
  loading: boolean
  selectedNotebookId: string | null
  selectedTagId: string | null
  searchActive: boolean
  onAllNotes: () => void
  onNotebook: (id: string) => void
  onTag: (id: string) => void
  onSearch: () => void
  onNewNotebook: () => void
  onNewNote: () => void
  onRefresh: () => void
}) {
  const [newTagName, setNewTagName] = useState('')
  const [addingTag, setAddingTag] = useState(false)
  const rootNotebooks = notebooks.filter((n) => !n.parent_id)
  const hasSelection = selectedNotebookId !== null || selectedTagId !== null || searchActive

  const handleCreateTag = async () => {
    const name = newTagName.trim()
    if (!name) return
    try {
      await createTag({ project_id: projectId, name })
      setNewTagName('')
      setAddingTag(false)
      onRefresh()
    } catch {
      // ignore
    }
  }

  return (
    <aside
      style={{
        width: SIDEBAR_WIDTH,
        minWidth: SIDEBAR_WIDTH,
        borderRight: `1px solid ${'var(--color-border-light)'}`,
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'var(--color-bg-alt)',
        padding: 'var(--space-md)',
      }}
    >
      <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, letterSpacing: '0.06em', color: 'var(--color-text-muted)', marginBottom: 'var(--space-md)' }}>
        NOTES
      </div>
      <nav style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
        <button
          type="button"
          onClick={onAllNotes}
          style={{
            textAlign: 'left',
            padding: 'var(--space-sm)',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: !hasSelection ? 'var(--color-bg-hover)' : 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
            fontSize: 'var(--text-sm)',
          }}
        >
          All notes
        </button>
        <button
          type="button"
          onClick={onSearch}
          style={{
            textAlign: 'left',
            padding: 'var(--space-sm)',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: searchActive ? 'var(--color-bg-hover)' : 'transparent',
            color: 'var(--color-text)',
            cursor: 'pointer',
            fontSize: 'var(--text-sm)',
          }}
        >
          Search
        </button>
      </nav>

      <div style={{ marginTop: 'var(--space-md)', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text-muted)' }}>
        Notebooks
      </div>
      {loading ? (
        <Spinner />
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, marginTop: 'var(--space-xs)' }}>
          {rootNotebooks.map((nb) => (
            <li key={nb.id}>
              <button
                type="button"
                onClick={() => onNotebook(nb.id)}
                style={{
                  textAlign: 'left',
                  width: '100%',
                  padding: 'var(--space-sm)',
                  borderRadius: 'var(--radius-sm)',
                  border: 'none',
                  background: selectedNotebookId === nb.id ? 'var(--color-bg-hover)' : 'transparent',
                  color: 'var(--color-text)',
                  cursor: 'pointer',
                  fontSize: 'var(--text-sm)',
                }}
              >
                {nb.title || 'Untitled'}
              </button>
            </li>
          ))}
        </ul>
      )}
      <div style={{ marginTop: 'var(--space-md)', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text-muted)' }}>
        Tags
      </div>
      <ul style={{ listStyle: 'none', padding: 0, margin: 0, marginTop: 'var(--space-xs)' }}>
        {tags.map((t) => (
          <li key={t.id}>
            <button
              type="button"
              onClick={() => onTag(t.id)}
              style={{
                textAlign: 'left',
                width: '100%',
                padding: 'var(--space-sm)',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: selectedTagId === t.id ? 'var(--color-bg-hover)' : 'transparent',
                color: 'var(--color-text)',
                cursor: 'pointer',
                fontSize: 'var(--text-sm)',
              }}
            >
              #{t.name}
            </button>
          </li>
        ))}
      </ul>
      {addingTag ? (
        <div style={{ marginTop: 'var(--space-md)', display: 'flex', gap: 'var(--space-sm)' }}>
          <input
            value={newTagName}
            onChange={(e) => setNewTagName(e.target.value)}
            placeholder="Tag name"
            style={{
              flex: 1,
              padding: 'var(--space-sm)',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${'var(--color-border-light)'}`,
              fontSize: 'var(--text-sm)',
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateTag()}
          />
          <Button type="button" onClick={handleCreateTag}>Add</Button>
          <Button type="button" variant="ghost" onClick={() => { setAddingTag(false); setNewTagName('') }}>Cancel</Button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setAddingTag(true)}
          style={{
            marginTop: 'var(--space-sm)',
            padding: 'var(--space-sm)',
            fontSize: 'var(--text-xs)',
            color: 'var(--color-text-muted)',
            border: 'none',
            background: 'none',
            cursor: 'pointer',
          }}
        >
          + New tag
        </button>
      )}

      <div style={{ marginTop: 'auto', paddingTop: 'var(--space-md)', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        <Button type="button" variant="primary" onClick={onNewNote}>New note</Button>
        <Button type="button" variant="ghost" onClick={onNewNotebook}>New notebook</Button>
      </div>
    </aside>
  )
}

function NotesList({
  notes,
  loading,
  selectedNoteId,
  onSelectNote,
}: {
  notes: Note[]
  loading: boolean
  selectedNoteId: string | null
  onSelectNote: (id: string) => void
}) {
  if (loading) return <div style={{ padding: 'var(--space-md)' }}><Spinner /></div>
  if (notes.length === 0) {
    return (
      <div style={{ padding: 'var(--space-md)', color: 'var(--color-text-muted)', fontSize: 'var(--text-sm)' }}>
        No notes
      </div>
    )
  }
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0, overflow: 'auto' }}>
      {notes.map((n) => (
        <li key={n.id}>
          <button
            type="button"
            onClick={() => onSelectNote(n.id)}
            style={{
              display: 'block',
              width: '100%',
              textAlign: 'left',
              padding: 'var(--space-md)',
              border: 'none',
              borderBottom: `1px solid ${'var(--color-border-light)'}`,
              background: selectedNoteId === n.id ? 'var(--color-bg-hover)' : 'transparent',
              color: 'var(--color-text)',
              cursor: 'pointer',
              fontSize: 'var(--text-sm)',
            }}
          >
            <div style={{ fontWeight: 600 }}>{noteSnippet(n.title, n.content, n.content_blocks ?? undefined, 60)}</div>
            {n.updated_at && (
              <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 2 }}>
                {new Date(n.updated_at).toLocaleDateString()}
              </div>
            )}
          </button>
        </li>
      ))}
    </ul>
  )
}

function NoteEditorArea({
  noteId,
  projectId,
  onClose,
}: {
  noteId: string
  projectId: string
  onClose: () => void
  onDeleted?: () => void
}) {
  const [note, setNote] = useState<Note | null>(null)
  const [backlinks, setBacklinks] = useState<Note[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      setError(null)
      try {
        const [n, bl] = await Promise.all([getNote(noteId), getBacklinks(noteId, projectId)])
        if (!cancelled) {
          setNote(n)
          setBacklinks(bl)
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load note')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => { cancelled = true }
  }, [noteId, projectId])

  const handleSaveBlocks = useCallback(
    async (blocks: unknown) => {
      if (!noteId) return
      try {
        const updated = await updateNote(noteId, { content_blocks: blocks as Note['content_blocks'] })
        setNote(updated)
      } catch {
        // ignore save error
      }
    },
    [noteId],
  )

  if (loading) return <div style={{ padding: 'var(--space-xl)' }}><Spinner /></div>
  if (error || !note) {
    return (
      <div style={{ padding: 'var(--space-xl)', color: 'var(--color-danger)' }}>
        {error || 'Note not found'}
        <Button type="button" variant="ghost" onClick={onClose} style={{ marginLeft: 'var(--space-md)' }}>Back</Button>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flex: 1, minWidth: 0, overflow: 'hidden' }}>
      <div style={{ flex: 1, overflow: 'auto', padding: 'var(--space-xl)' }}>
        <BlockNoteEditor
          initialBlocks={note.content_blocks ?? undefined}
          initialMarkdown={note.content ?? undefined}
          title={note.title ?? undefined}
          onSave={handleSaveBlocks}
        />
      </div>
      {backlinks.length > 0 && (
        <aside
          style={{
            width: 200,
            minWidth: 200,
            borderLeft: `1px solid ${'var(--color-border-light)'}`,
            padding: 'var(--space-md)',
            backgroundColor: 'var(--color-bg-alt)',
            fontSize: 'var(--text-sm)',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 'var(--space-sm)' }}>Backlinks</div>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {backlinks.map((n) => (
              <li key={n.id}>
                <Link to={`/notes/${n.id}`} style={{ color: 'var(--color-accent)', textDecoration: 'none' }}>
                  {noteSnippet(n.title, n.content, n.content_blocks ?? undefined, 40)}
                </Link>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  )
}

export function NotesShell() {
  const navigate = useNavigate()
  const { currentProjectId } = useWorkspace()
  const params = useParams<{ notebookId?: string; tagId?: string; noteId?: string }>()
  const [notebooks, setNotebooks] = useState<Notebook[]>([])
  const [tags, setTags] = useState<Tag[]>([])
  const [notes, setNotes] = useState<Note[]>([])
  const [loadingNotebooks, setLoadingNotebooks] = useState(true)
  const [loadingNotes, setLoadingNotes] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState(false)
  const [listMode, setListMode] = useState<'all' | 'notebook' | 'tag' | 'search'>('all')

  const selectedNotebookId = params.notebookId ?? null
  const selectedTagId = params.tagId ?? null
  const selectedNoteId = params.noteId ?? null

  // Sync list mode and search from URL (e.g. direct navigate to /notes/notebook/123)
  useEffect(() => {
    if (params.notebookId) setListMode('notebook')
    else if (params.tagId) setListMode('tag')
    else if (params.noteId) setListMode('all') // viewing a note
    else setListMode('all')
    const isSearch = typeof window !== 'undefined' && window.location.pathname === '/notes/search'
    setSearchMode(isSearch)
  }, [params.notebookId, params.tagId, params.noteId])

  const loadNotebooksAndTags = useCallback(async () => {
    setLoadingNotebooks(true)
    try {
      const [nbs, tgs] = await Promise.all([listNotebooks(currentProjectId), listTags(currentProjectId)])
      setNotebooks(nbs)
      setTags(tgs)
    } finally {
      setLoadingNotebooks(false)
    }
  }, [currentProjectId])

  const loadNotes = useCallback(async () => {
    setLoadingNotes(true)
    try {
      if (listMode === 'search' && searchQuery.trim()) {
        const items = await searchNotes({ q: searchQuery.trim(), projectId: currentProjectId })
        setNotes(items)
      } else if (listMode === 'notebook' && selectedNotebookId) {
        const items = await listNotes({ notebookId: selectedNotebookId, projectId: currentProjectId, orderBy: 'position' })
        setNotes(items)
      } else if (listMode === 'tag' && selectedTagId) {
        const items = await listNotes({ tagId: selectedTagId, projectId: currentProjectId })
        setNotes(items)
      } else {
        const items = await listNotes({ projectId: currentProjectId, orderBy: 'updated_at' })
        setNotes(items)
      }
    } finally {
      setLoadingNotes(false)
    }
  }, [currentProjectId, listMode, searchQuery, selectedNotebookId, selectedTagId])

  useEffect(() => { void loadNotebooksAndTags() }, [loadNotebooksAndTags])
  useEffect(() => { void loadNotes() }, [loadNotes])

  const goAllNotes = () => { setSearchMode(false); setListMode('all'); navigate('/notes') }
  const goNotebook = (id: string) => { setSearchMode(false); setListMode('notebook'); navigate(`/notes/notebook/${id}`) }
  const goTag = (id: string) => { setSearchMode(false); setListMode('tag'); navigate(`/notes/tag/${id}`) }
  const goSearch = () => { setSearchMode(true); setListMode('search'); navigate('/notes/search') }
  const goNote = (id: string) => navigate(`/notes/${id}`)

  const handleNewNote = async () => {
    try {
      const note = await createNote({
        projectId: currentProjectId,
        title: 'Untitled',
        content_blocks: [{ id: crypto.randomUUID(), type: 'paragraph', data: { text: '' } }],
      })
      goNote(note.id)
      await loadNotes()
    } catch {
      // ignore
    }
  }

  const handleNewNotebook = async () => {
    try {
      const nb = await createNotebook({ project_id: currentProjectId, title: 'New notebook' })
      await loadNotebooksAndTags()
      goNotebook(nb.id)
    } catch {
      // ignore
    }
  }

  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
      <NotesSidebar
        projectId={currentProjectId}
        notebooks={notebooks}
        tags={tags}
        loading={loadingNotebooks}
        selectedNotebookId={selectedNotebookId}
        selectedTagId={selectedTagId}
        searchActive={searchMode}
        onAllNotes={goAllNotes}
        onNotebook={goNotebook}
        onTag={goTag}
        onSearch={goSearch}
        onNewNotebook={handleNewNotebook}
        onNewNote={handleNewNote}
        onRefresh={loadNotebooksAndTags}
      />

      <section
        style={{
          width: LIST_WIDTH,
          minWidth: LIST_WIDTH,
          borderRight: `1px solid ${'var(--color-border-light)'}`,
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--color-bg)',
        }}
      >
        {searchMode ? (
          <div style={{ padding: 'var(--space-md)', borderBottom: `1px solid ${'var(--color-border-light)'}` }}>
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search notes…"
              style={{
                width: '100%',
                padding: 'var(--space-sm)',
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${'var(--color-border-light)'}`,
                fontSize: 'var(--text-sm)',
              }}
            />
          </div>
        ) : null}
        <NotesList
          notes={notes}
          loading={loadingNotes}
          selectedNoteId={selectedNoteId}
          onSelectNote={goNote}
        />
      </section>

      <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Routes>
          <Route path="/notes" element={
            <div style={{ padding: 'var(--space-xl)', color: 'var(--color-text-muted)', fontSize: 'var(--text-sm)' }}>
              Select a note or create a new one.
            </div>
          } />
          <Route path="/notes/notebook/:notebookId" element={
            <div style={{ padding: 'var(--space-xl)', color: 'var(--color-text-muted)' }}>Select a note from the list.</div>
          } />
          <Route path="/notes/tag/:tagId" element={
            <div style={{ padding: 'var(--space-xl)', color: 'var(--color-text-muted)' }}>Select a note from the list.</div>
          } />
          <Route path="/notes/search" element={
            <div style={{ padding: 'var(--space-xl)', color: 'var(--color-text-muted)' }}>Search and select a note.</div>
          } />
          <Route path="/notes/:noteId" element={
            <NoteEditorArea
              noteId={params.noteId!}
              projectId={currentProjectId}
              onClose={() => navigate('/notes')}
              onDeleted={loadNotes}
            />
          } />
        </Routes>
      </main>
    </div>
  )
}
