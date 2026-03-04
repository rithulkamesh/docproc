/**
 * Multiple chat threads per project, persisted in localStorage.
 */

const CONVERSE_SESSIONS_KEY = 'docproc-converse-sessions'

export interface ChatSessionMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{ document_id?: string; filename?: string; display_name?: string | null; content?: string }>
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatSessionMessage[]
  updatedAt: number
}

export interface SessionsState {
  sessions: ChatSession[]
  activeId: string | null
}

const DEFAULT_TITLE = 'New chat'
const MAX_TITLE_LENGTH = 40

function storageKey(projectId: string): string {
  return `${CONVERSE_SESSIONS_KEY}-${projectId}`
}

function createEmptySession(): ChatSession {
  return {
    id: crypto.randomUUID(),
    title: DEFAULT_TITLE,
    messages: [],
    updatedAt: Date.now(),
  }
}

/** Derive a short title from the first user message. */
export function sessionTitleFromMessage(content: string): string {
  const trimmed = content.trim()
  if (!trimmed) return DEFAULT_TITLE
  const firstLine = trimmed.split(/\n/)[0]?.trim() ?? trimmed
  if (firstLine.length <= MAX_TITLE_LENGTH) return firstLine
  return firstLine.slice(0, MAX_TITLE_LENGTH - 1) + '…'
}

export function loadSessions(projectId: string): SessionsState {
  try {
    const raw = localStorage.getItem(storageKey(projectId))
    if (!raw) return getDefaultState()
    const parsed = JSON.parse(raw) as { sessions?: ChatSession[]; activeId?: string | null }
    const sessions = Array.isArray(parsed.sessions) ? parsed.sessions : []
    const rawActiveId = parsed.activeId
    const activeId =
      rawActiveId === null
        ? null
        : typeof rawActiveId === 'string' && sessions.some((s) => s.id === rawActiveId)
          ? rawActiveId
          : sessions[0]?.id ?? null
    return { sessions, activeId }
  } catch {
    return getDefaultState()
  }
}

function getDefaultState(): SessionsState {
  return { sessions: [], activeId: null }
}

export function saveSessions(projectId: string, state: SessionsState): void {
  try {
    localStorage.setItem(storageKey(projectId), JSON.stringify(state))
  } catch {}
}

export function createNewSession(): ChatSession {
  return createEmptySession()
}

/** Document-contextual chat: one thread per (project, document). */
const CONVERSE_DOC_KEY = 'docproc-converse-doc'

function docStorageKey(projectId: string, documentId: string | null): string {
  return `${CONVERSE_DOC_KEY}-${projectId}-${documentId ?? 'project'}`
}

export function loadDocumentMessages(
  projectId: string,
  documentId: string | null
): ChatSessionMessage[] {
  try {
    const raw = localStorage.getItem(docStorageKey(projectId, documentId))
    if (!raw) return []
    const parsed = JSON.parse(raw) as ChatSessionMessage[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveDocumentMessages(
  projectId: string,
  documentId: string | null,
  messages: ChatSessionMessage[]
): void {
  try {
    localStorage.setItem(docStorageKey(projectId, documentId), JSON.stringify(messages))
  } catch {}
}
