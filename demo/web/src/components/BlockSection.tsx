import type { ReactNode } from 'react'
import { useState } from 'react'
import { InlineEditableTitle } from './InlineEditableTitle'

interface BlockSectionProps {
  id: string
  title: string
  onTitleChange: (title: string) => void
  children: ReactNode
  meta?: ReactNode
  saving?: 'idle' | 'saving' | 'saved'
  controls?: ReactNode
}

/** Block-style section: inline editable title, hover-revealed controls, optional auto-save indicator */
export function BlockSection({
  id,
  title,
  onTitleChange,
  children,
  meta,
  saving = 'idle',
  controls,
}: BlockSectionProps) {
  const [hover, setHover] = useState(false)

  return (
    <section
      id={id}
      aria-labelledby={`section-title-${id}`}
      style={{
        padding: 'var(--space-lg)',
        borderRadius: 'var(--radius-md)',
        background: 'var(--color-bg)',
        border: 'var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.2rem',
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 'var(--space-md)',
          flexWrap: 'wrap',
        }}
      >
        <InlineEditableTitle id={`section-title-${id}`} value={title} onChange={onTitleChange} level={2} />
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-sm)',
            opacity: hover ? 1 : 0.6,
            transition: 'opacity 120ms ease',
          }}
        >
          {controls}
          {saving === 'saving' && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--color-text-muted)',
              }}
            >
              Saving…
            </span>
          )}
          {saving === 'saved' && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--color-success)',
              }}
            >
              Saved
            </span>
          )}
        </div>
      </div>
      <div
        style={{
          fontSize: 'var(--text-base)',
          lineHeight: 'var(--line-height-body)',
          color: 'var(--color-text)',
          minHeight: '4em',
        }}
      >
        {children}
      </div>
      {meta && (
        <div
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--color-text-muted)',
            marginTop: 'auto',
          }}
        >
          {meta}
        </div>
      )}
    </section>
  )
}
