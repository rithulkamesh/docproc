import { useState } from 'react'
import type { DocumentSummary } from '../types'
import { ChatConsole } from '../components/ChatConsole'
import { StudyDock } from '../components/StudyDock'

interface ProjectWorkspaceProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

export function ProjectWorkspace({ documents, selectedDocumentId, projectId }: ProjectWorkspaceProps) {
  const [rightWidth, setRightWidth] = useState<number>(360)
  const [dragStartX, setDragStartX] = useState<number | null>(null)
  const [initialRightWidth, setInitialRightWidth] = useState<number>(rightWidth)

  const handleDragStart = (event: React.MouseEvent<HTMLDivElement>) => {
    setDragStartX(event.clientX)
    setInitialRightWidth(rightWidth)
  }

  const handleDragMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (dragStartX == null) return
    const delta = event.clientX - dragStartX
    const next = Math.min(520, Math.max(260, initialRightWidth - delta))
    setRightWidth(next)
  }

  const handleDragEnd = () => {
    setDragStartX(null)
  }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `minmax(0, 1fr) 6px ${rightWidth}px`,
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
        onMouseDown={handleDragStart}
        onMouseMove={handleDragMove}
        onMouseUp={handleDragEnd}
        onMouseLeave={handleDragEnd}
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

