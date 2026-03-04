import { useState, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Upload, FileText, Loader2, AlertCircle, RefreshCw, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { motion as motionTokens } from '@/design/tokens'

const ACCEPT_TYPES = '.pdf,.docx,.pptx,.xlsx'

export function SourcesCanvas() {
  const {
    documents,
    selectedDocumentId,
    setSelectedDocumentId,
    setCanvasMode,
    handleUploadFile,
    handleDeleteDocument,
    handleReindexDocument,
    apiStatusLabel,
  } = useWorkspace()
  const [dragOver, setDragOver] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [reindexingId, setReindexingId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files?.length) return
      const file = files[0]
      setUploadError(null)
      handleUploadFile(file).catch((e) => {
        setUploadError(e instanceof Error ? e.message : 'Upload failed')
      })
    },
    [handleUploadFile]
  )

  const handleUploadChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files)
    e.target.value = ''
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

  const handleDelete = async (documentId: string) => {
    setDeletingId(documentId)
    try {
      await handleDeleteDocument(documentId)
    } finally {
      setDeletingId(null)
    }
  }

  const handleReindex = async (documentId: string) => {
    setReindexingId(documentId)
    try {
      await handleReindexDocument(documentId)
    } finally {
      setReindexingId(null)
    }
  }

  return (
    <div className="flex flex-col space-y-8">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Sources
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Documents in this project ground chat, notes, flashcards, and tests.
        </p>
      </div>

      <motion.div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current?.click()}
        initial={{ opacity: 0, scale: 0.96, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{
          duration: motionTokens.durationPanel / 1000,
          ease: motionTokens.easingFramer,
        }}
      >
        <Card
          className={cn(
            'cursor-pointer border-2 border-dashed p-8 transition-colors',
            dragOver ? 'border-primary bg-primary/10' : 'border-border'
          )}
        >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_TYPES}
          onChange={handleUploadChange}
          className="hidden"
          aria-hidden
        />
        <div className="flex flex-col items-center justify-center gap-2 text-center">
          <Upload className="h-10 w-10 text-muted-foreground" />
          <p className="text-sm font-medium text-foreground">Add document</p>
          <p className="text-xs text-muted-foreground">PDF, DOCX, PPTX, XLSX</p>
        </div>
        </Card>
      </motion.div>

      {uploadError && (
        <p className="flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {uploadError}
        </p>
      )}

      {documents.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No documents yet. Add a PDF or document to get started.
        </p>
      ) : (
        <ul className="space-y-2">
          <AnimatePresence initial={false}>
            {documents.map((doc) => {
            const isSelected = selectedDocumentId === doc.id
            const isProcessing = doc.status === 'processing'
            const isCompleted = doc.status === 'completed'
            const isFailed = doc.status === 'failed'
            return (
              <motion.li
                key={doc.id}
                layout
                initial={{ opacity: 0, y: 8, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.97 }}
                transition={{
                  duration: motionTokens.durationStandard / 1000,
                  ease: motionTokens.easingFramer,
                }}
              >
                <Card
                  className={cn(
                    'flex items-center gap-4 p-3 transition-colors',
                    isSelected && 'bg-primary/10 ring-1 ring-primary'
                  )}
                >
                  <button
                    type="button"
                    onClick={() => setSelectedDocumentId(doc.id)}
                    className="flex min-w-[6rem] flex-1 items-center gap-2 text-left"
                  >
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 truncate font-medium" title={doc.display_name ?? doc.filename ?? undefined}>
                      {(doc.display_name ?? doc.filename) || 'Processing…'}
                    </span>
                  </button>
                  <div className="flex shrink-0 items-center gap-2 text-xs text-muted-foreground" style={{ minWidth: '7rem' }}>
                    {isProcessing && (
                      <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                        <Loader2 className="h-3 w-3 shrink-0 animate-spin" />
                        {(() => {
                          const p = doc.progress
                          const pct =
                            p?.percent ??
                            (p?.total != null && p.total > 0 && p?.page != null
                              ? Math.min(100, Math.round((p.page / p.total) * 100))
                              : null)
                          const status =
                            pct != null ? `${pct}%` : (p?.message ?? 'Processing…')
                          return (
                            <>
                              {status}
                              {p?.heartbeat && (
                                <span className="text-muted-foreground/80" title="Worker is updating this">
                                  {' '}
                                  ● live
                                </span>
                              )}
                            </>
                          )
                        })()}
                      </span>
                    )}
                    {isCompleted && <span className="whitespace-nowrap">Ready · {doc.pages ?? '?'} pages</span>}
                    {isFailed && (
                      <span className="text-destructive whitespace-nowrap" title={(doc as { error?: string }).error ?? ''}>
                        Failed
                      </span>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-wrap items-center gap-1">
                    {isCompleted && (
                      <>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedDocumentId(doc.id)
                            setCanvasMode('notes')
                          }}
                        >
                          Summary
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedDocumentId(doc.id)
                            setCanvasMode('home')
                          }}
                        >
                          Flashcards
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 text-xs" asChild>
                          <Link to="/assessments/create" onClick={(e) => { e.stopPropagation(); setSelectedDocumentId(doc.id) }}>
                            Test
                          </Link>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedDocumentId(doc.id)
                            setCanvasMode('converse')
                          }}
                        >
                          Chat
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={(e) => {
                            e.stopPropagation()
                            void handleReindex(doc.id)
                          }}
                          disabled={reindexingId === doc.id}
                          aria-label="Reindex"
                        >
                          {reindexingId === doc.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <RefreshCw className="h-4 w-4" />
                          )}
                        </Button>
                      </>
                    )}
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => {
                          e.stopPropagation()
                          void handleDelete(doc.id)
                        }}
                        disabled={deletingId === doc.id}
                        aria-label="Delete"
                        className="text-destructive hover:text-destructive"
                      >
                        {deletingId === doc.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </Button>
                  </div>
                </Card>
                {isFailed && (doc as { error?: string }).error && (
                  <p className="mt-1 truncate pl-5 text-xs text-destructive" title={(doc as { error?: string }).error}>
                    {(doc as { error?: string }).error}
                  </p>
                )}
              </motion.li>
            )
          })}
          </AnimatePresence>
        </ul>
      )}

      <p className="text-xs text-muted-foreground">{apiStatusLabel}</p>
    </div>
  )
}
