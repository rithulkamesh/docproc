import type { ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { theme } from '../design/theme'

interface UtilityPanelProps {
  isOpen: boolean
  onClose: () => void
  children: ReactNode
  title?: string
}

const panelTransition = {
  duration: theme.motion.durationPanel / 1000,
  ease: [0.4, 0, 0.2, 1] as const,
}

export function UtilityPanel({ isOpen, onClose, title, children }: UtilityPanelProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            role="presentation"
            aria-hidden
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={panelTransition}
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 40,
              background: 'rgba(0,0,0,0.2)',
            }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-label={title ?? 'Utility panel'}
            initial={{ opacity: 0, x: '100%' }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: '100%' }}
            transition={panelTransition}
            style={{
              position: 'fixed',
              top: 0,
              right: 0,
              bottom: 0,
              width: 'var(--utility-width, min(28rem, 35vw))',
              maxWidth: '100%',
              zIndex: 50,
              display: 'flex',
              flexDirection: 'column',
              background: 'var(--color-bg-alt)',
              borderLeft: 'var(--border-subtle)',
              boxShadow: '-4px 0 24px rgba(0,0,0,0.06)',
            }}
          >
            <div
              style={{
                flexShrink: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: 'var(--space-lg)',
                paddingTop: 'var(--space-md)',
                paddingBottom: 'var(--space-md)',
                borderBottom: 'var(--border-subtle)',
              }}
            >
              {title && (
                <h2
                  style={{
                    margin: 0,
                    fontFamily: 'var(--font-family)',
                    fontSize: 'var(--text-lg)',
                    fontWeight: 600,
                    lineHeight: theme.lineHeight.heading,
                    color: 'var(--color-text)',
                  }}
                >
                  {title}
                </h2>
              )}
              <button
                type="button"
                aria-label="Close panel"
                onClick={onClose}
                style={{
                  padding: 'var(--space-sm)',
                  border: 'none',
                  background: 'none',
                  color: 'var(--color-text-muted)',
                  cursor: 'pointer',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 'var(--text-lg)',
                  lineHeight: 1,
                }}
              >
                ×
              </button>
            </div>
            <div
              style={{
                flex: 1,
                minHeight: 0,
                overflowY: 'auto',
                padding: 'var(--space-lg)',
                paddingTop: 'var(--space-md)',
                paddingBottom: 'var(--space-xl)',
              }}
            >
              {children}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
