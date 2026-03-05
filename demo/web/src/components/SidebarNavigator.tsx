import { useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useWorkspace } from '../context/WorkspaceContext'
import type { DocumentSummary } from '../types'
import { Spinner } from './Spinner'
import { SoftButton } from './SoftButton'

function Collapsible({
  label,
  defaultOpen,
  children,
}: {
  label: string
  defaultOpen: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="sidebar-section">
      <button
        type="button"
        className="sidebar-section-header"
        onClick={() => setOpen((o) => !o)}
      >
        <span>{label}</span>
        <span style={{ opacity: 0.7 }}>{open ? '−' : '+'}</span>
      </button>
      {open && children}
    </section>
  )
}

export function SidebarNavigator() {
  const {
    projects,
    currentProjectId,
    setCurrentProjectId,
    documents,
    selectedDocumentId,
    setSelectedDocumentId,
    activePanel,
    setActivePanel,
    handleUploadFile,
    handleDeleteDocument,
    apiStatusLabel,
  } = useWorkspace()
  const navigate = useNavigate()
  const location = useLocation()
  const isNotesRoute = location.pathname === '/notes' || location.pathname.startsWith('/notes/')
  const [uploadError, setUploadError] = useState<string | null>(null)

  const handleUploadChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (!files?.length) return
      e.target.value = ''
      setUploadError(null)
      for (const file of Array.from(files)) {
        try {
          await handleUploadFile(file)
        } catch (err) {
          setUploadError(err instanceof Error ? err.message : 'Upload failed')
        }
      }
    },
    [handleUploadFile]
  )

  const processingCount = documents.filter((d) => d.status === 'processing').length

  return (
    <aside className="sidebar-nav" aria-label="Notebook sidebar">
      <Collapsible label="Projects" defaultOpen={true}>
        <div className="sidebar-section-body">
          {projects.length === 0 ? (
            <p className="text-xs text-muted" style={{ margin: 0 }}>No projects</p>
          ) : (
            projects.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`sidebar-nav-item ${currentProjectId === p.id ? 'sidebar-nav-item--active' : ''}`}
                onClick={() => setCurrentProjectId(p.id)}
              >
                {p.name}
              </button>
            ))
          )}
        </div>
      </Collapsible>

      <Collapsible label="Documents" defaultOpen={true}>
        <div className="sidebar-section-body">
          <label className="sidebar-upload-zone">
            <input
              type="file"
              accept=".pdf,.docx,.pptx,.xlsx"
              multiple
              style={{ display: 'none' }}
              onChange={handleUploadChange}
            />
            Add document
          </label>
          {uploadError && (
            <p className="text-xs" style={{ color: 'var(--color-danger)', margin: '0 0 0.5rem 0' }}>
              {uploadError}
            </p>
          )}
          {documents.length === 0 ? (
            <p className="text-xs text-muted" style={{ margin: 0 }}>No documents yet</p>
          ) : (
            <ul className="sidebar-doc-list">
              {documents.map((doc) => (
                <li key={doc.id}>
                  <DocumentItem
                    doc={doc}
                    isSelected={doc.id === selectedDocumentId}
                    onSelect={() => setSelectedDocumentId(doc.id)}
                    onDelete={() => handleDeleteDocument(doc.id)}
                  />
                </li>
              ))}
            </ul>
          )}
          <p className="text-xs text-muted mt-sm" style={{ lineHeight: 1.4 }}>
            {documents.length} document{documents.length === 1 ? '' : 's'}
            {processingCount > 0 && ` · ${processingCount} processing`}
          </p>
          <p className="text-xs text-muted mt-xs">{apiStatusLabel}</p>
        </div>
      </Collapsible>

      <Collapsible label="Study tools" defaultOpen={true}>
        <div className="sidebar-study-tools">
          <SoftButton
            active={isNotesRoute}
            onClick={() => navigate('/notes')}
          >
            Notes
          </SoftButton>
          <SoftButton
            active={activePanel === 'tests'}
            onClick={() => setActivePanel(activePanel === 'tests' ? null : 'tests')}
          >
            Tests
          </SoftButton>
          <SoftButton onClick={() => navigate('/assessments')}>
            Assessments
          </SoftButton>
          <SoftButton onClick={() => navigate('/assessments/create')}>
            Create assessment
          </SoftButton>
        </div>
      </Collapsible>
    </aside>
  )
}

function DocumentItem({
  doc,
  isSelected,
  onSelect,
  onDelete,
}: {
  doc: DocumentSummary
  isSelected: boolean
  onSelect: () => void
  onDelete: () => void | Promise<void>
}) {
  const [deleting, setDeleting] = useState(false)
  const isProcessing = doc.status === 'processing'
  const isFailed = doc.status === 'failed'

  return (
    <div className="sidebar-doc-item flex items-center gap-2">
      <button
        type="button"
        className={`sidebar-nav-item flex min-w-0 flex-1 flex-col items-stretch gap-1 ${isSelected ? 'sidebar-nav-item--active' : ''}`}
        onClick={onSelect}
      >
        <div className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap font-medium">
          {doc.display_name ?? doc.filename}
        </div>
        <div className="sidebar-doc-meta flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
          {isProcessing && (
            <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
              <Spinner size="sm" />
              <span>
                {(() => {
                  const p = doc.progress
                  const pct =
                    p?.percent ??
                    (p?.total != null && p.total > 0 && p?.page != null
                      ? Math.min(100, Math.round((p.page / p.total) * 100))
                      : null)
                  return pct != null ? `${pct}%` : (p?.message ?? 'Processing…')
                })()}
              </span>
              {doc.progress?.heartbeat && <span title="Worker is processing">● live</span>}
            </span>
          )}
          {doc.status === 'completed' && doc.pages != null && !doc.index_error && <span>{doc.pages} pages</span>}
          {doc.index_error && <span className="status-pill--danger">Index failed</span>}
          {isFailed && <span className="status-pill--danger">Failed</span>}
          {doc.status === 'completed' && doc.pages == null && !doc.index_error && <span>Ready</span>}
        </div>
      </button>
      <button
        type="button"
        title="Remove document"
        disabled={deleting}
        className="sidebar-doc-delete"
        onClick={(e) => {
          e.stopPropagation()
          setDeleting(true)
          void Promise.resolve(onDelete()).finally(() => setDeleting(false))
        }}
      >
        {deleting ? <Spinner size="sm" /> : '×'}
      </button>
    </div>
  )
}
