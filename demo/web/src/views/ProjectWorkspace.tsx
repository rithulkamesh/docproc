import { useCallback, useEffect, useState } from 'react'
import type { DocumentSummary } from '../types'
import { ChatConsole } from '../components/ChatConsole'
import { StudyDock } from '../components/StudyDock'

const MIN_RIGHT_WIDTH = 260
const MAX_RIGHT_WIDTH = 520
const KEYBOARD_STEP = 24

interface ProjectWorkspaceProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

export function ProjectWorkspace({ documents, selectedDocumentId, projectId }: ProjectWorkspaceProps) {
  const [rightWidth, setRightWidth] = useState<number>(360)
  const [dragStartX, setDragStartX] = useState<number | null>(null)
  const [initialRightWidth, setInitialRightWidth] = useState<number>(rightWidth)

  const clampWidth = useCallback((w: number) => Math.min(MAX_RIGHT_WIDTH, Math.max(MIN_RIGHT_WIDTH, w)), [])

  const handleDragStart = (event: React.MouseEvent<HTMLDivElement>) => {
    setDragStartX(event.clientX)
    setInitialRightWidth(rightWidth)
  }

  const handleDragEnd = () => {
    setDragStartX(null)
  }

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        setRightWidth((w) => clampWidth(w - KEYBOARD_STEP))
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        setRightWidth((w) => clampWidth(w + KEYBOARD_STEP))
      } else if (e.key === 'Home') {
        e.preventDefault()
        setRightWidth(MIN_RIGHT_WIDTH)
      } else if (e.key === 'End') {
        e.preventDefault()
        setRightWidth(MAX_RIGHT_WIDTH)
      }
    },
    [clampWidth]
  )

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (dragStartX != null) {
        const delta = e.clientX - dragStartX
        setRightWidth(clampWidth(initialRightWidth - delta))
      }
    }
    const onMouseUp = () => handleDragEnd()
    if (dragStartX != null) {
      window.addEventListener('mousemove', onMouseMove)
      window.addEventListener('mouseup', onMouseUp)
    }
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [dragStartX, initialRightWidth, clampWidth])

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `minmax(0, 1fr) 8px ${rightWidth}px`,
        gap: 'var(--space-sm)',
        height: '100%',
        alignItems: 'stretch',
      }}
    >
      <section
        aria-label="Chat console"
        style={{
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-sm)',
          backgroundColor: 'var(--color-bg-alt)',
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          minHeight: 0,
        }}
      >
        <header
          style={{
            padding: 'var(--space-lg)',
            borderBottom: '1px solid var(--color-border-light)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 'var(--space-md)',
          }}
        >
          <div className="section-label" style={{ letterSpacing: '0.12em' }}>
            RESEARCH CONSOLE
          </div>
          <div className="text-xs text-muted" style={{ textAlign: 'right' }}>
            {documents.length === 0 ? (
              <span>No documents in corpus</span>
            ) : (
              <span>
                {documents.length} document{documents.length === 1 ? '' : 's'} in corpus
              </span>
            )}
          </div>
        </header>
        <div
          style={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <ChatConsole documents={documents} selectedDocumentId={selectedDocumentId} projectId={projectId} />
        </div>
      </section>

      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize study panel"
        tabIndex={0}
        onMouseDown={handleDragStart}
        onKeyDown={handleKeyDown}
        style={{
          cursor: 'col-resize',
          backgroundColor: 'var(--color-border-strong)',
        }}
      />

      <section
        aria-label="Study modules"
        className="study-dock-wrapper"
        style={{
          border: '1px solid var(--color-border-strong)',
          borderRadius: 'var(--radius-sm)',
          backgroundColor: 'var(--color-bg-alt)',
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          minHeight: 0,
        }}
      >
        <StudyDock documents={documents} selectedDocumentId={selectedDocumentId} projectId={projectId} />
      </section>
    </div>
  )
}

