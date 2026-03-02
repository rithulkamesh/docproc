import { useState } from 'react'
import type { DocumentSummary } from '../types'
import { NotesModule } from './NotesModule'
import { FlashcardsModule } from './FlashcardsModule'
import { TestsModule } from './TestsModule'

interface StudyDockProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

interface PanelProps {
  title: string
  badge?: string
  isOpen: boolean
  onToggle: () => void
  children: React.ReactNode
}

function DockPanel({ title, badge, isOpen, onToggle, children }: PanelProps) {
  return (
    <section
      className="dock-panel"
      style={{
        borderBottom: '1px solid var(--color-border-strong)',
        backgroundColor: 'var(--color-bg-alt)',
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        style={{
          width: '100%',
          padding: 'var(--space-md)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          textAlign: 'left',
          border: 'none',
          borderBottom: '1px solid var(--color-border-light)',
          backgroundColor: 'var(--color-bg-alt)',
          cursor: 'pointer',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 700,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
          }}
        >
          {title}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          {badge && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                padding: '2px 6px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border-strong)',
                backgroundColor: 'var(--color-bg)',
              }}
            >
              {badge}
            </span>
          )}
          <span
            aria-hidden="true"
            style={{
              fontSize: 'var(--text-sm)',
            }}
          >
            {isOpen ? '−' : '+'}
          </span>
        </div>
      </button>
      <div
        style={{
          maxHeight: isOpen ? 420 : 0,
          overflow: 'hidden',
          transition: 'max-height 120ms ease',
        }}
      >
        <div
          className="dock-panel-body"
          style={{
            padding: isOpen ? 'var(--space-md)' : 0,
          }}
        >
          {isOpen && children}
        </div>
      </div>
    </section>
  )
}

export function StudyDock({ documents, selectedDocumentId, projectId }: StudyDockProps) {
  const [notesOpen, setNotesOpen] = useState(true)
  const [flashcardsOpen, setFlashcardsOpen] = useState(true)
  const [testsOpen, setTestsOpen] = useState(false)

  const hasDocs = documents.length > 0

  return (
    <div
      className="study-dock"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
        background: 'var(--color-bg-alt)',
      }}
    >
      <DockPanel
        title="NOTES"
        badge={hasDocs ? `${documents.length} source${documents.length === 1 ? '' : 's'}` : undefined}
        isOpen={notesOpen}
        onToggle={() => setNotesOpen((o) => !o)}
      >
        <NotesModule documents={documents} selectedDocumentId={selectedDocumentId} projectId={projectId} />
      </DockPanel>

      <DockPanel
        title="FLASHCARDS"
        badge={undefined}
        isOpen={flashcardsOpen}
        onToggle={() => setFlashcardsOpen((o) => !o)}
      >
        <FlashcardsModule documents={documents} selectedDocumentId={selectedDocumentId} projectId={projectId} />
      </DockPanel>

      <DockPanel
        title="TESTS"
        badge={undefined}
        isOpen={testsOpen}
        onToggle={() => setTestsOpen((o) => !o)}
      >
        <TestsModule documents={documents} selectedDocumentId={selectedDocumentId} />
      </DockPanel>
    </div>
  )
}

