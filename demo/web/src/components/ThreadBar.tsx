import { useState, useRef, useEffect, type KeyboardEvent } from 'react'
import type { ChatSession } from '../lib/chatSessions'

const DEFAULT_TITLE = 'New chat'
const MAX_TITLE_LENGTH = 40

export interface ThreadBarProps {
  threads: ChatSession[]
  activeId: string | null
  onSelect: (id: string | null) => void
  onNew: () => void
  onRename: (id: string, title: string) => void
  onClose: (id: string) => void
}

export function ThreadBar({
  threads,
  activeId,
  onSelect,
  onNew,
  onRename,
  onClose,
}: ThreadBarProps) {
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editingId) {
      const t = threads.find((s) => s.id === editingId)
      setEditTitle(t?.title ?? '')
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [editingId, threads])

  const handleCommitRename = () => {
    if (editingId == null) return
    const trimmed = editTitle.trim().slice(0, MAX_TITLE_LENGTH) || DEFAULT_TITLE
    onRename(editingId, trimmed)
    setEditingId(null)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleCommitRename()
    }
    if (e.key === 'Escape') {
      setEditingId(null)
    }
  }

  return (
    <div
      className="thread-bar"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-sm)',
        padding: 'var(--space-md) var(--space-lg)',
        borderBottom: '1px solid var(--color-border-light)',
        background: 'var(--color-bg)',
        overflowX: 'auto',
        overflowY: 'hidden',
        flexShrink: 0,
        minHeight: 44,
        WebkitOverflowScrolling: 'touch',
      }}
    >
      {threads.map((t) => (
        <div
          key={t.id}
          role="tab"
          aria-selected={activeId === t.id}
          className="thread-pill"
          data-active={activeId === t.id}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 10px',
            borderRadius: 'var(--radius-full)',
            border: '1px solid var(--color-border-light)',
            background: activeId === t.id ? 'var(--color-primary)' : 'var(--color-bg-alt)',
            color: activeId === t.id ? 'var(--color-primary-foreground)' : 'var(--color-text)',
            fontSize: 'var(--text-sm)',
            cursor: 'pointer',
            flexShrink: 0,
            maxWidth: 200,
          }}
          onClick={() => editingId !== t.id && onSelect(t.id)}
        >
          {editingId === t.id ? (
            <input
              ref={inputRef}
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              onBlur={handleCommitRename}
              onKeyDown={handleKeyDown}
              onClick={(e) => e.stopPropagation()}
              style={{
                flex: 1,
                minWidth: 80,
                maxWidth: 160,
                padding: '2px 6px',
                fontSize: 'var(--text-sm)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--color-bg)',
                color: 'var(--color-text)',
              }}
            />
          ) : (
            <>
              <span
                style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                onDoubleClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setEditingId(t.id)
                }}
              >
                {t.title}
              </span>
              <button
                type="button"
                aria-label="Close thread"
                onClick={(e) => {
                  e.stopPropagation()
                  onClose(t.id)
                }}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: 18,
                  height: 18,
                  padding: 0,
                  border: 'none',
                  borderRadius: 'var(--radius-full)',
                  background: 'transparent',
                  color: 'inherit',
                  opacity: 0.8,
                  cursor: 'pointer',
                  fontSize: 14,
                  lineHeight: 1,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = '1'
                  e.currentTarget.style.background = 'rgba(0,0,0,0.12)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = '0.8'
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                ×
              </button>
            </>
          )}
        </div>
      ))}
      <button
        type="button"
        className="thread-pill-new"
        onClick={onNew}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 4,
          padding: '6px 12px',
          borderRadius: 'var(--radius-full)',
          border: '1px dashed var(--color-border)',
          background: 'transparent',
          color: 'var(--color-text-muted)',
          fontSize: 'var(--text-sm)',
          cursor: 'pointer',
          flexShrink: 0,
        }}
      >
        <span aria-hidden>+</span> New
      </button>
    </div>
  )
}
