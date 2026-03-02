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
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        setUploadError(null)
        handleUploadFile(file).catch((err) => {
          setUploadError(err instanceof Error ? err.message : 'Upload failed')
        })
        e.target.value = ''
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
            active={activePanel === 'flashcards'}
            onClick={() => setActivePanel(activePanel === 'flashcards' ? null : 'flashcards')}
          >
            Flashcards
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
    <div className="sidebar-doc-item">
      <button
        type="button"
        className={`sidebar-nav-item ${isSelected ? 'sidebar-nav-item--active' : ''}`}
        onClick={onSelect}
      >
        <div style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {doc.filename}
        </div>
        <div className="sidebar-doc-meta">
          {isProcessing && (
            <>
              <Spinner size="sm" />
              <span>Processing…</span>
            </>
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
