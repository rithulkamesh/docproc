import type { ReactNode } from 'react'
import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { deleteDocument, listDocuments, reindexDocument, uploadDocument } from '../api/documents'
import { createProject, getProject, listProjects, updateProject, type Project } from '../api/projects'
import { fetchStatus, type ApiStatus } from '../api/status'
import type { DocumentSummary, RagSource } from '../types'
import { loadTheme, saveTheme, applyTheme, type ThemeId } from '../lib/themeStorage'

export type CanvasMode = 'home' | 'converse' | 'notes' | 'tests' | 'sources'

export type ActivePanel = 'notes' | 'tests' | null

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
  themeId: ThemeId
  setThemeId: (id: ThemeId) => void
  focusMode: boolean
  setFocusMode: (on: boolean) => void
  canvasMode: CanvasMode
  setCanvasMode: (mode: CanvasMode) => void
  activePanel: ActivePanel
  setActivePanel: (panel: ActivePanel) => void
  lastIndexedLabel: string
  contextPanelSources: RagSource[] | null
  setContextPanelSources: (sources: RagSource[] | null) => void
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null)

const CURRENT_PROJECT_STORAGE_KEY = 'docproc-current-project-id'

function getStoredProjectId(): string {
  if (typeof window === 'undefined') return 'default'
  const stored = window.localStorage.getItem(CURRENT_PROJECT_STORAGE_KEY)
  return stored?.trim() || 'default'
}

function setStoredProjectId(id: string): void {
  try {
    window.localStorage.setItem(CURRENT_PROJECT_STORAGE_KEY, id)
  } catch {}
}

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
  const [currentProjectId, setCurrentProjectIdState] = useState<string>(getStoredProjectId)
  const [currentProject, setCurrentProject] = useState<Project | null>(null)
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [themeId, setThemeIdState] = useState<ThemeId>(loadTheme)
  const [focusMode, setFocusMode] = useState(false)
  const [canvasMode, setCanvasMode] = useState<CanvasMode>('home')
  const [activePanel, setActivePanelState] = useState<ActivePanel>(null)
  const [contextPanelSources, setContextPanelSources] = useState<RagSource[] | null>(null)

  const setActivePanel = useCallback((panel: ActivePanel) => {
    setActivePanelState(panel)
  }, [])

  const setCanvasModeAndPanel = useCallback((mode: CanvasMode) => {
    setCanvasMode(mode)
    if (mode === 'notes' || mode === 'tests') {
      setActivePanelState(mode)
    } else if (mode === 'home' || mode === 'converse' || mode === 'sources') {
      setActivePanelState(null)
    }
  }, [])

  useEffect(() => {
    applyTheme(themeId)
    saveTheme(themeId)
  }, [themeId])

  const loadProjects = useCallback(async () => {
    try {
      let list = await listProjects()
      if (list.length === 0) {
        const created = await createProject({ name: 'My First Project' })
        list = await listProjects()
        setProjects(list)
        const createdInList = list.find((p) => p.id === created.id) ?? created
        setCurrentProjectIdState(createdInList.id)
        setStoredProjectId(createdInList.id)
        return
      }
      setProjects(list)
      setCurrentProjectIdState((prev) => {
        const valid = list.find((p) => p.id === prev)
        const next = valid ? prev : list[0]?.id ?? prev
        setStoredProjectId(next)
        return next
      })
    } catch {
      setProjects([])
    }
  }, [])

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
    const POLL_TIMEOUT_MS = 10 * 60 * 1000 // stop polling after 10min so loop doesn't run forever
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
      } catch {}
    }
    intervalId = setInterval(tick, POLL_INTERVAL_MS)
    return () => stopPolling()
  }, [documents, currentProjectId])

  const setCurrentProjectId = useCallback((id: string) => {
    setCurrentProjectIdState(id)
    setStoredProjectId(id)
    setSelectedDocumentId(null)
  }, [])

  const setCurrentProjectName = useCallback(
    async (name: string) => {
      if (!currentProjectId) return
      try {
        const updated = await updateProject(currentProjectId, { name })
        setCurrentProject(updated)
        setProjects((prev) => prev.map((p) => (p.id === currentProjectId ? updated : p)))
      } catch {}
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
    themeId,
    setThemeId: (id: ThemeId) => {
      setThemeIdState(id)
      saveTheme(id)
    },
    focusMode,
    setFocusMode,
    canvasMode,
    setCanvasMode: setCanvasModeAndPanel,
    activePanel,
    setActivePanel,
    lastIndexedLabel,
    contextPanelSources,
    setContextPanelSources,
  }

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>
}
