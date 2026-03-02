import type { ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { theme } from '../design/theme'

interface AdaptivePanelProps {
  /** Unique key for the current panel content (e.g. canvasMode) */
  panelKey: string
  children: ReactNode
  /** Optional: show a minimal placeholder when content is converse (chat is main) */
  showPlaceholder?: boolean
}

const springTransition = {
  type: 'spring' as const,
  stiffness: theme.motion.springPanel.stiffness,
  damping: theme.motion.springPanel.damping,
}

export function AdaptivePanel({ panelKey, children, showPlaceholder }: AdaptivePanelProps) {
  return (
    <motion.div
      layout
      style={{
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 'var(--radius-panel)',
        boxShadow: 'var(--shadow-card)',
        backgroundColor: 'var(--color-bg-alt)',
        border: '1px solid var(--color-border-strong)',
        overflow: 'hidden',
      }}
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={panelKey}
          layout
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={springTransition}
          style={{
            flex: 1,
            minHeight: 0,
            overflow: 'auto',
            padding: 'var(--content-gap)',
          }}
        >
          {showPlaceholder && panelKey === 'converse' ? (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '12rem',
                fontSize: 'var(--text-sm)',
                color: 'var(--color-text-muted)',
              }}
            >
              Chat is in the main area. Use the dock to open Sources, Notes, Flashcards, or Tests.
            </div>
          ) : (
            children
          )}
        </motion.div>
      </AnimatePresence>
    </motion.div>
  )
}
