import type { ReactNode } from 'react'
import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { deleteDocument, listDocuments, reindexDocument, uploadDocument } from '../api/documents'
import { getProject, listProjects, updateProject, type Project } from '../api/projects'
import { fetchStatus, type ApiStatus } from '../api/status'
import type { DocumentSummary } from '../types'

export type CanvasMode = 'converse' | 'notes' | 'flashcards' | 'tests' | 'sources'

/** Which utility panel is open (right slide-over). null = none. */
export type ActivePanel = 'notes' | 'flashcards' | 'tests' | null

interface WorkspaceContextValue {
  projects: Project[]
  currentProjectId: string
  setCurrentProjectId: (id: string) => void
  currentProject: Project | null
  setCurrentProjectName: (name: string) => Promise<void>
  documents: DocumentSummary[]
  setDocuments: React.Dispatch<React.SetStateAction<DocumentSummary[]>>
  selectedDocumentId: string | null
  setSelectedDocumentId: (id: string | null) => void
  loadProjects: () => Promise<void>
  loadDocuments: () => Promise<void>
  handleUploadFile: (file: File) => Promise<void>
  handleDeleteDocument: (documentId: string) => Promise<void>
  handleReindexDocument: (documentId: string) => Promise<void>
  status: ApiStatus | null
  apiStatusLabel: string
  themeMode: 'light' | 'dark'
  setThemeMode: (mode: 'light' | 'dark') => void
  focusMode: boolean
  setFocusMode: (on: boolean) => void
  canvasMode: CanvasMode
  setCanvasMode: (mode: CanvasMode) => void
  /** Which right panel is open. Replaces canvasMode for layout. */
  activePanel: ActivePanel
  setActivePanel: (panel: ActivePanel) => void
  lastIndexedLabel: string
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null)

export function useWorkspace() {
  const ctx = useContext(WorkspaceContext)
  if (!ctx) throw new Error('useWorkspace must be used within WorkspaceProvider')
  return ctx
}

interface WorkspaceProviderProps {
  children: ReactNode
}

export function WorkspaceProvider({ children }: WorkspaceProviderProps) {
  const [projects, setProjects] = useState<Project[]>([])
  const [currentProjectId, setCurrentProjectIdState] = useState<string>('default')
  const [currentProject, setCurrentProject] = useState<Project | null>(null)
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [themeMode, setThemeMode] = useState<'light' | 'dark'>(() => {
    if (typeof window === 'undefined') return 'light'
    const stored = window.localStorage.getItem('docproc-theme')
    if (stored === 'light' || stored === 'dark') return stored
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  const [focusMode, setFocusMode] = useState(false)
  const [canvasMode, setCanvasMode] = useState<CanvasMode>('converse')
  const [activePanel, setActivePanelState] = useState<ActivePanel>(null)

  const setActivePanel = useCallback((panel: ActivePanel) => {
    setActivePanelState(panel)
  }, [])

  const setCanvasModeAndPanel = useCallback((mode: CanvasMode) => {
    setCanvasMode(mode)
    if (mode === 'notes' || mode === 'flashcards' || mode === 'tests') {
      setActivePanelState(mode)
    } else if (mode === 'converse' || mode === 'sources') {
      setActivePanelState(null)
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode
    if (themeMode === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    window.localStorage.setItem('docproc-theme', themeMode)
  }, [themeMode])

  const loadProjects = useCallback(async () => {
    try {
      const list = await listProjects()
      setProjects(list)
      const current = list.find((p) => p.id === currentProjectId) ?? list[0]
      if (current && current.id !== currentProjectId) setCurrentProjectIdState(current.id)
    } catch {
      setProjects([])
    }
  }, [currentProjectId])

  const loadDocuments = useCallback(async () => {
    try {
      const [docs, proj] = await Promise.all([
        listDocuments(currentProjectId),
        getProject(currentProjectId).catch(() => null),
      ])
      setDocuments(docs)
      setCurrentProject(proj)
      setSelectedDocumentId((prev) => (prev || (docs.length > 0 ? docs[0].id : null)))
    } catch {
      setDocuments([])
      setCurrentProject(null)
    }
  }, [currentProjectId])

  useEffect(() => {
    void loadProjects()
  }, [])

  useEffect(() => {
    const load = async () => {
      try {
        const [docs, stat, proj] = await Promise.all([
          listDocuments(currentProjectId),
          fetchStatus(),
          getProject(currentProjectId).catch(() => null),
        ])
        setDocuments(docs)
        setStatus(stat)
        setCurrentProject(proj)
        if (docs.length === 0) {
          setActivePanelState(null)
        }
        if (!selectedDocumentId && docs.length > 0) {
          setSelectedDocumentId(docs[0].id)
        }
      } catch {
        setDocuments([])
        setCurrentProject(null)
      }
    }
    void load()
  }, [currentProjectId])

  useEffect(() => {
    const hasProcessing = documents.some((d) => d.status === 'processing')
    if (!hasProcessing) return
    const POLL_INTERVAL_MS = 2000
    const POLL_TIMEOUT_MS = 10 * 60 * 1000 // 10 min max, then stop so loop doesn't run forever
    const startedAt = Date.now()
    let intervalId: ReturnType<typeof setInterval> | null = null
    const stopPolling = () => {
      if (intervalId != null) {
        clearInterval(intervalId)
        intervalId = null
      }
    }
    const tick = async () => {
      if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
        stopPolling()
        return
      }
      try {
        const docs = await listDocuments(currentProjectId)
        setDocuments(docs)
        const stillProcessing = docs.some((d) => d.status === 'processing')
        if (!stillProcessing) {
          stopPolling()
        }
      } catch {
        // ignore
      }
    }
    intervalId = setInterval(tick, POLL_INTERVAL_MS)
    return () => stopPolling()
  }, [documents, currentProjectId])

  const setCurrentProjectId = useCallback((id: string) => {
    setCurrentProjectIdState(id)
    setSelectedDocumentId(null)
  }, [])

  const setCurrentProjectName = useCallback(
    async (name: string) => {
      if (!currentProjectId) return
      try {
        const updated = await updateProject(currentProjectId, { name })
        setCurrentProject(updated)
        setProjects((prev) => prev.map((p) => (p.id === currentProjectId ? updated : p)))
      } catch {
        // revert or show error
      }
    },
    [currentProjectId]
  )

  const handleUploadFile = useCallback(
    async (file: File) => {
      try {
        await uploadDocument(file, currentProjectId)
        const docs = await listDocuments(currentProjectId)
        setDocuments(docs)
        if (!selectedDocumentId && docs.length > 0) {
          setSelectedDocumentId(docs[0].id)
        }
      } catch (e) {
        throw e
      }
    },
    [currentProjectId, selectedDocumentId]
  )

  const handleDeleteDocument = useCallback(
    async (documentId: string) => {
      await deleteDocument(documentId)
      const docs = await listDocuments(currentProjectId)
      setDocuments(docs)
      if (selectedDocumentId === documentId) {
        setSelectedDocumentId(docs.length > 0 ? docs[0].id : null)
      }
    },
    [currentProjectId, selectedDocumentId]
  )

  const handleReindexDocument = useCallback(
    async (documentId: string) => {
      await reindexDocument(documentId)
      const docs = await listDocuments(currentProjectId)
      setDocuments(docs)
    },
    [currentProjectId]
  )

  const completedDocs = documents.filter((d) => d.status === 'completed')
  const lastUpdated = completedDocs.length
    ? completedDocs.reduce((best, d) => {
        const doc = d as DocumentSummary & { updated_at?: string }
        const t = doc.updated_at
        if (!t) return best
        return !best || t > best ? t : best
      }, null as string | null)
    : null
  const lastIndexedLabel = lastUpdated
    ? new Date(lastUpdated).toLocaleDateString(undefined, { dateStyle: 'short', timeStyle: 'short' })
    : '—'

  const apiStatusLabel =
    status && status.ok
      ? `API connected · RAG: ${status.rag_backend ?? '?'} · DB: ${status.database_provider ?? '?'}`
      : 'API unreachable. Ensure the API is running (e.g. docker compose up).'

  const value: WorkspaceContextValue = {
    projects,
    currentProjectId,
    setCurrentProjectId,
    currentProject,
    setCurrentProjectName,
    documents,
    setDocuments,
    selectedDocumentId,
    setSelectedDocumentId,
    loadProjects,
    loadDocuments,
    handleUploadFile,
    handleDeleteDocument,
    handleReindexDocument,
    status,
    apiStatusLabel,
    themeMode,
    setThemeMode,
    focusMode,
    setFocusMode,
    canvasMode,
    setCanvasMode: setCanvasModeAndPanel,
    activePanel,
    setActivePanel,
    lastIndexedLabel,
  }

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>
}
