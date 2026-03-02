import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { theme } from '../design/theme'
import { useWorkspace, type CanvasMode } from '../context/WorkspaceContext'

const NARROW_BREAKPOINT = '48rem'
const DOCK_WIDTH_COLLAPSED = '4.5rem'
const DOCK_WIDTH_EXPANDED = '11rem'
const RAIL_LEFT = 'clamp(1.5rem, 3vw, 3rem)'

const MODES: { id: CanvasMode; label: string }[] = [
  { id: 'converse', label: 'Converse' },
  { id: 'notes', label: 'Notes' },
  { id: 'flashcards', label: 'Flashcards' },
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
function IconFlashcards() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="4" width="20" height="14" rx="2" />
      <line x1="2" y1="10" x2="22" y2="10" />
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
    case 'flashcards':
      return <IconFlashcards />
    case 'tests':
      return <IconTests />
    case 'sources':
      return <IconSources />
  }
}

const springPanel = {
  type: 'spring' as const,
  stiffness: 300,
  damping: 30,
}

export function SmartDock() {
  const { canvasMode, setCanvasMode, focusMode } = useWorkspace()
  const [isNarrow, setIsNarrow] = useState(false)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${NARROW_BREAKPOINT})`)
    const fn = () => setIsNarrow(mq.matches)
    fn()
    mq.addEventListener('change', fn)
    return () => mq.removeEventListener('change', fn)
  }, [])

  if (focusMode) return null

  const sharedNavStyle: React.CSSProperties = {
    backgroundColor: 'var(--color-bg-alt)',
    border: '1px solid var(--color-border-strong)',
    borderRadius: 'var(--radius-panel)',
    boxShadow: 'var(--shadow-card)',
    zIndex: 50,
  }

  if (isNarrow) {
    return (
      <motion.nav
        aria-label="Tool navigation"
        initial={false}
        animate={{
          padding: 'var(--space-md)',
          height: expanded ? 'auto' : 'auto',
        }}
        transition={springPanel}
        style={{
          position: 'fixed',
          bottom: 'var(--space-md)',
          left: 'var(--space-md)',
          right: 'var(--space-md)',
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-evenly',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 'var(--space-sm)',
          ...sharedNavStyle,
        }}
      >
        {MODES.map((m) => {
          const isActive = canvasMode === m.id
          return (
            <motion.button
              key={m.id}
              type="button"
              title={m.label}
              aria-label={m.label}
              aria-current={isActive ? 'true' : undefined}
              onClick={() => setCanvasMode(m.id)}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.98 }}
              transition={{ duration: theme.motion.durationMicro / 1000, ease: theme.motion.easeFramer }}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 'var(--space-sm)',
                padding: 'var(--space-sm)',
                minWidth: '2.75rem',
                minHeight: '2.75rem',
                border: 'none',
                borderRadius: theme.radius.button,
                background: isActive ? 'var(--color-accent-soft)' : 'transparent',
                color: 'var(--color-text)',
                cursor: 'pointer',
                fontWeight: isActive ? 700 : 400,
              }}
            >
              {getIcon(m.id)}
              <AnimatePresence>
                {expanded && (
                  <motion.span
                    initial={{ width: 0, opacity: 0 }}
                    animate={{ width: 'auto', opacity: 1 }}
                    exit={{ width: 0, opacity: 0 }}
                    transition={springPanel}
                    style={{
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                      fontSize: 'var(--text-xs)',
                    }}
                  >
                    {m.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </motion.button>
          )
        })}
        <button
          type="button"
          aria-label={expanded ? 'Collapse dock' : 'Expand dock'}
          onClick={() => setExpanded((e) => !e)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 'var(--space-sm)',
            border: 'none',
            borderRadius: theme.radius.badge,
            background: 'var(--color-bg)',
            color: 'var(--color-text-muted)',
            cursor: 'pointer',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0)' }}>
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
      </motion.nav>
    )
  }

  return (
    <motion.nav
      aria-label="Tool dock"
      initial={false}
      animate={{
        width: expanded ? DOCK_WIDTH_EXPANDED : DOCK_WIDTH_COLLAPSED,
      }}
      transition={springPanel}
      style={{
        position: 'fixed',
        top: '50%',
        transform: 'translateY(-50%)',
        left: RAIL_LEFT,
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-sm)',
        padding: 'var(--space-md)',
        ...sharedNavStyle,
      }}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      {MODES.map((m) => {
        const isActive = canvasMode === m.id
        return (
          <motion.button
            key={m.id}
            type="button"
            title={m.label}
            aria-label={m.label}
            aria-current={isActive ? 'true' : undefined}
            onClick={() => setCanvasMode(m.id)}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.98 }}
            transition={{ duration: theme.motion.durationMicro / 1000, ease: theme.motion.easeFramer }}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: expanded ? 'flex-start' : 'center',
              gap: 'var(--space-md)',
              padding: 'var(--space-sm)',
              width: '100%',
              minHeight: '2.75rem',
              border: 'none',
              borderRadius: theme.radius.button,
              background: isActive ? 'var(--color-accent-soft)' : 'transparent',
              color: 'var(--color-text)',
              cursor: 'pointer',
              fontWeight: isActive ? 700 : 400,
            }}
          >
            <span style={{ flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {getIcon(m.id)}
            </span>
            <AnimatePresence>
              {expanded && (
                <motion.span
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 'auto', opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={springPanel}
                  style={{
                    overflow: 'hidden',
                    whiteSpace: 'nowrap',
                    fontSize: 'var(--text-sm)',
                  }}
                >
                  {m.label}
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>
        )
      })}
      <button
        type="button"
        aria-label={expanded ? 'Collapse dock' : 'Expand dock'}
        onClick={() => setExpanded((e) => !e)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginTop: 'var(--space-sm)',
          padding: 'var(--space-sm)',
          border: 'none',
          borderRadius: theme.radius.badge,
          background: 'var(--color-bg)',
          color: 'var(--color-text-muted)',
          cursor: 'pointer',
          alignSelf: 'center',
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0)' }}>
          <polyline points="9 18 15 12 9 6" />
        </svg>
      </button>
    </motion.nav>
  )
}
