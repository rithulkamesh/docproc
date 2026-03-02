import { useWorkspace } from '../context/WorkspaceContext'
import { DocumentThread } from './DocumentThread'

export function ConverseCanvas() {
  const { documents, selectedDocumentId, currentProjectId } = useWorkspace()

  if (documents.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '40vh',
          gap: '3rem',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
          }}
        >
          No documents yet
        </div>
        <p style={{ fontSize: 'var(--text-lg)', lineHeight: 'var(--line-height-body)', maxWidth: '36ch', margin: 0, color: 'var(--color-text)' }}>
          Add a document in the sidebar (Documents) to start chatting and generating study material.
        </p>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
          <kbd>⌘K</kbd> command palette
        </p>
      </div>
    )
  }

  return (
    <DocumentThread
      documents={documents}
      selectedDocumentId={selectedDocumentId}
      projectId={currentProjectId}
    />
  )
}
