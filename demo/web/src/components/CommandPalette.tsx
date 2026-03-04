import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useWorkspace } from '../context/WorkspaceContext'

interface Command {
  id: string
  label: string
  keywords: string[]
  run: () => void
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const {
    setCanvasMode,
    setFocusMode,
    focusMode,
    projects,
    setCurrentProjectId,
  } = useWorkspace()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
        if (!open) {
          setQuery('')
          setSelectedIndex(0)
          setTimeout(() => inputRef.current?.focus(), 0)
        }
      }
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open])

  const commands: Command[] = [
    { id: 'converse', label: 'Switch to Converse', keywords: ['chat', 'converse'], run: () => setCanvasMode('converse') },
    { id: 'notes', label: 'Switch to Notes', keywords: ['notes'], run: () => setCanvasMode('notes') },
    { id: 'flashcards', label: 'Go to Home (flashcards)', keywords: ['flashcards', 'cards', 'home'], run: () => setCanvasMode('home') },
    { id: 'tests', label: 'Switch to Tests', keywords: ['tests', 'quiz'], run: () => setCanvasMode('tests') },
    { id: 'sources', label: 'Switch to Sources', keywords: ['sources', 'documents'], run: () => setCanvasMode('sources') },
    { id: 'focus', label: focusMode ? 'Exit Focus Mode' : 'Enter Focus Mode', keywords: ['focus'], run: () => setFocusMode(!focusMode) },
    { id: 'add-doc', label: 'Add document', keywords: ['add', 'document', 'upload'], run: () => { setCanvasMode('sources'); setOpen(false) } },
    { id: 'settings', label: 'Open Settings', keywords: ['settings'], run: () => { navigate('/settings'); setOpen(false) } },
    ...projects.map((p) => ({
      id: `project-${p.id}`,
      label: `Switch to ${p.name}`,
      keywords: [p.name, 'project', 'switch'],
      run: () => {
        setCurrentProjectId(p.id)
        setOpen(false)
      },
    })),
  ]

  const q = query.trim().toLowerCase()
  const filtered = q
    ? commands.filter(
        (c) =>
          c.label.toLowerCase().includes(q) || c.keywords.some((k) => k.includes(q))
      )
    : commands
  const selected = filtered[Math.min(selectedIndex, filtered.length - 1)]

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  useEffect(() => {
    if (!selected) return
    const el = listRef.current?.querySelector(`[data-index="${filtered.indexOf(selected)}"]`)
    el?.scrollIntoView({ block: 'nearest' })
  }, [selectedIndex, filtered, selected])

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-label="Command palette"
      aria-modal="true"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '15vh',
        backgroundColor: 'rgba(0,0,0,0.4)',
        backdropFilter: 'blur(4px)',
      }}
      onClick={() => setOpen(false)}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '40rem',
          backgroundColor: 'var(--color-bg-alt)',
          border: `${'1px'} solid ${'var(--color-border-strong)'}`,
          borderRadius: 'var(--radius-panel)',
          boxShadow: 'var(--shadow-card)',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '70vh',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search commands…"
          aria-label="Command search"
          style={{
            padding: 'var(--space-lg)',
            border: 'none',
            borderBottom: '1px solid var(--color-border-light)',
            borderRadius: 0,
            fontSize: 'var(--text-sm)',
            fontFamily: 'var(--font-family)',
            background: 'var(--color-bg)',
            color: 'var(--color-text)',
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              setSelectedIndex((i) => Math.min(i + 1, filtered.length - 1))
            }
            if (e.key === 'ArrowUp') {
              e.preventDefault()
              setSelectedIndex((i) => Math.max(i - 1, 0))
            }
            if (e.key === 'Enter' && selected) {
              e.preventDefault()
              selected.run()
            }
          }}
        />
        <div
          ref={listRef}
          style={{
            overflowY: 'auto',
            maxHeight: '50vh',
          }}
        >
          {filtered.map((cmd, idx) => (
            <button
              key={cmd.id}
              type="button"
              data-index={idx}
              onClick={() => {
                cmd.run()
                setOpen(false)
              }}
              style={{
                width: '100%',
                padding: 'var(--space-md)',
                textAlign: 'left',
                border: 'none',
                borderBottom: '1px solid var(--color-border-light)',
                background: selected === cmd ? 'var(--color-accent-soft)' : 'transparent',
                color: 'var(--color-text)',
                cursor: 'pointer',
                fontSize: 'var(--text-sm)',
              }}
            >
              {cmd.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
