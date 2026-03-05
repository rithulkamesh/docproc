import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useWorkspace } from '../context/WorkspaceContext'
import { theme } from '../design/theme'
import { Button } from './Button'
import { Spinner } from './Spinner'

const ACCEPT_TYPES = '.pdf,.docx,.pptx,.xlsx'

export function SourcesCanvas() {
  const {
    documents,
    selectedDocumentId,
    setSelectedDocumentId,
    handleUploadFile,
    handleDeleteDocument,
    handleReindexDocument,
    apiStatusLabel,
  } = useWorkspace()
  const [dragOver, setDragOver] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [reindexingId, setReindexingId] = useState<string | null>(null)

  const processingCount = documents.filter((d) => d.status === 'processing').length

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return
      setUploadError(null)
      for (const file of Array.from(files)) {
        try {
          await handleUploadFile(file)
        } catch (e) {
          setUploadError(e instanceof Error ? e.message : 'Upload failed')
        }
      }
    },
    [handleUploadFile]
  )

  const handleUploadChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    void handleFiles(event.target.files)
    event.target.value = ''
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }

  const cardStyle: React.CSSProperties = {
    borderRadius: 'var(--radius-panel)',
    boxShadow: 'var(--shadow-card)',
    border: '1px solid var(--color-border-strong)',
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--content-gap)',
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
        Sources
      </div>
      <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', margin: 0 }}>
        Documents in this project ground chat, notes, flashcards, and tests.
      </p>

      <motion.div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        animate={{
          borderColor: dragOver ? 'var(--color-accent)' : 'var(--color-border-light)',
          backgroundColor: dragOver ? 'var(--color-accent-soft)' : 'var(--color-bg)',
        }}
        transition={{ duration: theme.motion.durationMicro / 1000, ease: theme.motion.easeFramer }}
        style={{
          ...cardStyle,
          padding: 'var(--space-xl)',
          border: `2px dashed ${'var(--color-border-light)'}`,
          textAlign: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          <Button
            type="button"
            onClick={() => {
              const input = document.getElementById('canvas-doc-upload') as HTMLInputElement | null
              input?.click()
            }}
          >
            Add document
          </Button>
          <input
            id="canvas-doc-upload"
            type="file"
            accept={ACCEPT_TYPES}
            multiple
            style={{ display: 'none' }}
            onChange={handleUploadChange}
          />
          <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
            or drag and drop PDF, DOCX, PPTX, XLSX
          </span>
        </div>
      </motion.div>

      {uploadError && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{ fontSize: 'var(--text-xs)', color: 'var(--color-danger)', margin: 0 }}
        >
          {uploadError}
        </motion.p>
      )}

      {documents.length === 0 ? (
        <div
          style={{
            ...cardStyle,
            padding: theme.spacing(6),
            fontSize: 'var(--text-sm)',
            color: 'var(--color-text-muted)',
            textAlign: 'center',
            backgroundColor: 'var(--color-bg-alt)',
          }}
        >
          No documents yet. Add a PDF or document to get started.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {documents.map((doc) => {
            const isSelected = doc.id === selectedDocumentId
            const isProcessing = doc.status === 'processing'
            const isFailed = doc.status === 'failed'
            const hasIndexError = Boolean(doc.index_error)
            const isDeleting = deletingId === doc.id
            const isReindexing = reindexingId === doc.id
            return (
              <li key={doc.id}>
                <motion.div
                  layout
                  style={{
                    display: 'flex',
                    alignItems: 'stretch',
                    gap: 'var(--space-sm)',
                    ...cardStyle,
                    padding: 0,
                    overflow: 'hidden',
                    backgroundColor: isSelected ? 'var(--color-accent-soft)' : 'var(--color-bg-alt)',
                  }}
                >
                  <motion.button
                    type="button"
                    onClick={() => setSelectedDocumentId(doc.id)}
                    whileHover={{ y: -1 }}
                    transition={{ duration: theme.motion.durationMicro / 1000, ease: theme.motion.easeFramer }}
                    style={{
                      flex: 1,
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: 'var(--space-lg)',
                      border: 'none',
                      background: 'none',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: 'var(--text-sm)' }}>{doc.display_name ?? doc.filename}</div>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 'var(--space-sm)',
                        marginTop: 'var(--space-sm)',
                        fontSize: 'var(--text-xs)',
                        color: 'var(--color-text-muted)',
                      }}
                    >
                      {isProcessing && (
                        <motion.span
                          animate={{ opacity: [0.7, 1, 0.7] }}
                          transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-sm)' }}
                        >
                          <Spinner size="sm" />
                          <span>Processing…</span>
                        </motion.span>
                      )}
                      {doc.status === 'completed' && doc.pages != null && !hasIndexError && (
                        <span>Ready · {doc.pages} pages</span>
                      )}
                      {hasIndexError && (
                        <span style={{ color: 'var(--color-danger)' }}>Index failed</span>
                      )}
                      {isFailed && <span style={{ color: 'var(--color-danger)' }}>Failed</span>}
                      {doc.status === 'completed' && doc.pages == null && !hasIndexError && <span>Ready</span>}
                    </div>
                  </motion.button>
                  {hasIndexError && (
                    <button
                      type="button"
                      title="Reindex for RAG (e.g. after fixing embedding config)"
                      disabled={isReindexing}
                      onClick={(e) => {
                        e.stopPropagation()
                        setReindexingId(doc.id)
                        void Promise.resolve(handleReindexDocument(doc.id)).finally(() => setReindexingId(null))
                      }}
                      style={{
                        flexShrink: 0,
                        alignSelf: 'center',
                        marginRight: 'var(--space-sm)',
                        padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
                        border: `1px solid ${'var(--color-border-light)'}`,
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--color-bg)',
                        color: 'var(--color-text)',
                        cursor: isReindexing ? 'wait' : 'pointer',
                        fontSize: 'var(--text-xs)',
                      }}
                    >
                      {isReindexing ? <Spinner size="sm" /> : 'Reindex'}
                    </button>
                  )}
                  <button
                    type="button"
                    title="Remove document"
                    disabled={isDeleting}
                    onClick={(e) => {
                      e.stopPropagation()
                      setDeletingId(doc.id)
                      void Promise.resolve(handleDeleteDocument(doc.id)).finally(() => setDeletingId(null))
                    }}
                    style={{
                      flexShrink: 0,
                      alignSelf: 'center',
                      marginRight: 'var(--space-md)',
                      padding: 'var(--space-sm)',
                      border: 'none',
                      borderRadius: 'var(--radius-sm)',
                      background: 'transparent',
                      color: 'var(--color-text-muted)',
                      cursor: isDeleting ? 'wait' : 'pointer',
                      fontSize: 'var(--text-sm)',
                    }}
                  >
                    {isDeleting ? <Spinner size="sm" /> : '×'}
                  </button>
                </motion.div>
              </li>
            )
          })}
        </ul>
      )}

      <div
        style={{
          ...cardStyle,
          padding: 'var(--space-lg)',
          backgroundColor: 'var(--color-bg-alt)',
          fontSize: 'var(--text-xs)',
          color: 'var(--color-text-muted)',
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 'var(--space-sm)' }}>Status</div>
        <div>
          {documents.length} document{documents.length === 1 ? '' : 's'}
          {processingCount > 0 && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              style={{ marginLeft: 'var(--space-sm)' }}
            >
              · {processingCount} processing
            </motion.span>
          )}
        </div>
        <div style={{ marginTop: 'var(--space-sm)' }}>{apiStatusLabel}</div>
      </div>
    </div>
  )
}
