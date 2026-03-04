import { useState, useEffect } from 'react'
import { useWorkspace, type CanvasMode } from '../context/WorkspaceContext'

const NARROW_BREAKPOINT = '48rem'
const RAIL_WIDTH = '4.5rem'
const RAIL_LEFT = 'clamp(1.5rem, 3vw, 3rem)'
const RAIL_GAP = '1.5rem'

const MODES: { id: CanvasMode; label: string }[] = [
  { id: 'converse', label: 'Converse' },
  { id: 'notes', label: 'Notes' },
  { id: 'tests', label: 'Tests' },
  { id: 'sources', label: 'Sources' },
]

function IconConverse() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  )
}
function IconNotes() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  )
}
function IconTests() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 11l3 3L22 4" />
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    </svg>
  )
}
function IconSources() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
      <line x1="12" y1="11" x2="12" y2="17" />
      <line x1="9" y1="14" x2="15" y2="14" />
    </svg>
  )
}

function getIcon(mode: CanvasMode) {
  switch (mode) {
    case 'converse':
      return <IconConverse />
    case 'notes':
      return <IconNotes />
    case 'tests':
      return <IconTests />
    case 'sources':
      return <IconSources />
  }
}

export function ToolRail() {
  const { canvasMode, setCanvasMode, focusMode } = useWorkspace()
  const [isNarrow, setIsNarrow] = useState(false)

  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${NARROW_BREAKPOINT})`)
    const fn = () => setIsNarrow(mq.matches)
    fn()
    mq.addEventListener('change', fn)
    return () => mq.removeEventListener('change', fn)
  }, [])

  if (focusMode) return null

  if (isNarrow) {
    return (
      <nav
        aria-label="Tool navigation"
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 50,
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-evenly',
          alignItems: 'center',
          padding: 'var(--space-md)',
          borderTop: '1px solid var(--color-border-strong)',
          backgroundColor: 'var(--color-bg-alt)',
        }}
      >
        {MODES.map((m) => {
          const isActive = canvasMode === m.id
          return (
            <button
              key={m.id}
              type="button"
              title={m.label}
              aria-label={m.label}
              onClick={() => setCanvasMode(m.id)}
              style={{
                width: '2.75rem',
                height: '2.75rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: 'none',
                borderTop: 'none',
                background: isActive ? 'var(--color-accent-soft)' : 'transparent',
                color: 'var(--color-text)',
                cursor: 'pointer',
                fontWeight: isActive ? 700 : 400,
              }}
            >
              {getIcon(m.id)}
            </button>
          )
        })}
      </nav>
    )
  }

  return (
    <nav
      aria-label="Tool rail"
      style={{
        position: 'fixed',
        top: '50%',
        transform: 'translateY(-50%)',
        left: RAIL_LEFT,
        width: RAIL_WIDTH,
        display: 'flex',
        flexDirection: 'column',
        gap: RAIL_GAP,
        zIndex: 50,
        padding: 'var(--space-md)',
        border: '1px solid var(--color-border-strong)',
        backgroundColor: 'var(--color-bg-alt)',
      }}
    >
      {MODES.map((m) => {
        const isActive = canvasMode === m.id
        return (
          <button
            key={m.id}
            type="button"
            title={m.label}
            aria-label={m.label}
            onClick={() => setCanvasMode(m.id)}
            style={{
              position: 'relative',
              width: '2.75rem',
              height: '2.75rem',
              marginInline: 'auto',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: 'none',
              background: isActive ? 'var(--color-accent-soft)' : 'transparent',
              color: 'var(--color-text)',
              cursor: 'pointer',
              fontWeight: isActive ? 700 : 400,
            }}
          >
            {getIcon(m.id)}
            {isActive && (
              <span
                style={{
                  position: 'absolute',
                  right: 0,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '0.25rem',
                  height: '1.25rem',
                  backgroundColor: 'var(--color-accent)',
                }}
              />
            )}
          </button>
        )
      })}
    </nav>
  )
}
