import { useMemo } from 'react'
import type { DocumentSummary } from '../types'
import { theme } from '../design/theme'
import { Button } from './Button'
import { Spinner } from './Spinner'

interface ProjectSidebarProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  onSelectDocument: (id: string) => void
  onUploadFile: (file: File) => void
  apiStatusLabel: string
  isCollapsed: boolean
  isNarrow: boolean
  projectName?: string
}

export function ProjectSidebar({
  documents,
  selectedDocumentId,
  onSelectDocument,
  onUploadFile,
  apiStatusLabel,
  isCollapsed,
  isNarrow,
  projectName = '—',
}: ProjectSidebarProps) {
  const processingCount = useMemo(
    () => documents.filter((d) => d.status === 'processing').length,
    [documents],
  )

  const handleUploadChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files?.length) return
    event.target.value = ''
    for (const file of Array.from(files)) {
      await onUploadFile(file)
    }
  }

  const collapsedView = isCollapsed && !isNarrow
  if (collapsedView) {
    return null
  }

  return (
    <>
      <section
        className="project-sidebar-section"
        style={{
          marginBottom: 'var(--space-lg)',
          border: '1px solid var(--color-border-strong)',
          padding: 'var(--space-md)',
          backgroundColor: 'var(--color-bg)',
        }}
      >
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-sm)',
          }}
        >
          PROJECT
        </div>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>{projectName}</div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
            {documents.length === 0
              ? 'Attach sources to begin building this project.'
              : `${documents.length} source${documents.length === 1 ? '' : 's'} · ${
                  processingCount > 0 ? `${processingCount} processing` : 'ready'
                }`}
          </div>
        </div>
      </section>

      <section
        className="project-sidebar-section"
        style={{
          marginBottom: 'var(--space-lg)',
          border: '1px solid var(--color-border-strong)',
          padding: 'var(--space-md)',
          backgroundColor: 'var(--color-bg)',
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 'var(--space-sm)',
          }}
        >
          <span
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
            }}
          >
            PROJECT DATA
          </span>
          {documents.length > 0 && (
            <span
              style={{
                fontSize: 'var(--text-xs)',
                padding: '2px 6px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: theme.colors.badge,
                color: 'var(--color-text-muted)',
              }}
            >
              {documents.length} doc{documents.length === 1 ? '' : 's'}
            </span>
          )}
        </div>
        <p
          style={{
            fontSize: 'var(--text-sm)',
            color: 'var(--color-text-muted)',
            marginTop: 0,
            marginBottom: 'var(--space-md)',
          }}
        >
          Documents attached to this project ground chat, notes, flashcards, and tests.
        </p>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-sm)',
            maxHeight: 220,
            overflowY: 'auto',
          }}
        >
          {documents.length === 0 ? (
            <div
              style={{
                padding: 'var(--space-lg)',
                fontSize: 'var(--text-sm)',
                color: 'var(--color-text-muted)',
                textAlign: 'center',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--color-bg-alt)',
              }}
            >
              No documents yet.
            </div>
          ) : (
            documents.map((doc) => {
              const isSelected = doc.id === selectedDocumentId
              const isProcessing = doc.status === 'processing'
              const isFailed = doc.status === 'failed'
              return (
                <button
                  key={doc.id}
                  type="button"
                  className={`project-doc-item${isSelected ? ' project-doc-item--selected' : ''}`}
                  onClick={() => onSelectDocument(doc.id)}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: 'var(--space-md)',
                    border: '1px solid var(--color-border-strong)',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: isSelected ? 'var(--color-accent-soft)' : 'var(--color-bg-alt)',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: 'var(--text-sm)' }}>{doc.display_name ?? doc.filename}</div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      marginTop: 4,
                      fontSize: 'var(--text-xs)',
                      color: 'var(--color-text-muted)',
                    }}
                  >
                    {isProcessing && (
                      <>
                        <Spinner size="sm" />
                        <span>{doc.progress?.message ?? 'Processing…'}</span>
                        {doc.progress?.heartbeat && <span title="Worker is processing"> ● live</span>}
                      </>
                    )}
                    {doc.status === 'completed' && doc.pages != null && <span>Ready · {doc.pages} pages</span>}
                    {isFailed && <span style={{ color: 'var(--color-danger)' }}>Failed</span>}
                    {doc.status === 'completed' && doc.pages == null && <span>Ready</span>}
                  </div>
                </button>
              )
            })
          )}
        </div>
      </section>

      <section
        className="project-sidebar-section"
        style={{
          marginTop: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-md)',
        }}
      >
        <Button
          type="button"
          fullWidth
          onClick={() => {
            const input = document.getElementById('doc-upload-input') as HTMLInputElement | null
            input?.click()
          }}
        >
          Add document
        </Button>
        <input
          id="doc-upload-input"
          type="file"
          accept=".pdf,.docx,.pptx,.xlsx"
          multiple
          style={{ display: 'none' }}
          onChange={handleUploadChange}
        />
        <div
          style={{
            border: '1px solid var(--color-border-strong)',
            padding: 'var(--space-md)',
            backgroundColor: 'var(--color-bg)',
            fontSize: 'var(--text-xs)',
            color: 'var(--color-text-muted)',
            lineHeight: 1.4,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Corpus status</div>
          <div>
            {documents.length} document{documents.length === 1 ? '' : 's'}
            {processingCount > 0 && ` · ${processingCount} processing`}
          </div>
          <div style={{ marginTop: 4 }}>{apiStatusLabel}</div>
        </div>
      </section>
    </>
  )
}

