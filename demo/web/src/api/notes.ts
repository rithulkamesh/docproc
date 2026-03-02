import { apiClient } from './client'
import { stripNoteCodeFence } from '../lib/noteContent'

/** Block shape for block-based editor (e.g. paragraph, heading, bulletList, code). */
export interface ContentBlock {
  id: string
  type: string
  data?: Record<string, unknown>
  children?: ContentBlock[]
}

export interface Note {
  id: string
  content: string
  document_id?: string | null
  project_id?: string | null
  filename?: string
  created_at?: string
  updated_at?: string
  /** New fields */
  notebook_id?: string | null
  title?: string | null
  position?: number
  content_blocks?: ContentBlock[] | null
  tag_ids?: string[]
}

export interface Notebook {
  id: string
  project_id: string
  parent_id?: string | null
  title: string
  position?: number
  created_at?: string
  updated_at?: string
}

export interface Tag {
  id: string
  project_id: string
  name: string
  created_at?: string
}

interface ListNotesResponse {
  notes: Note[]
}

interface ListNotebooksResponse {
  notebooks: Notebook[]
}

interface ListTagsResponse {
  tags: Tag[]
}

interface GenerateNoteResponse {
  content: string
}

export interface ListNotesParams {
  documentId?: string
  projectId?: string
  notebookId?: string
  tagId?: string
  orderBy?: 'position' | 'updated_at'
}

export async function listNotes(params?: ListNotesParams): Promise<Note[]> {
  const search = new URLSearchParams()
  if (params?.documentId) search.set('document_id', params.documentId)
  if (params?.projectId) search.set('project_id', params.projectId)
  if (params?.notebookId) search.set('notebook_id', params.notebookId)
  if (params?.tagId) search.set('tag_id', params.tagId)
  if (params?.orderBy) search.set('order_by', params.orderBy)
  const qs = search.toString() ? `?${search.toString()}` : ''
  const data = await apiClient.get<ListNotesResponse>(`/notes${qs}`)
  return data.notes ?? []
}

export async function getNote(noteId: string): Promise<Note> {
  return apiClient.get<Note>(`/notes/${encodeURIComponent(noteId)}`)
}

export async function createNote(payload: {
  content?: string
  content_blocks?: ContentBlock[]
  documentId?: string | null
  projectId?: string | null
  source_quote?: string | null
  notebook_id?: string | null
  title?: string | null
  position?: number
  tag_ids?: string[]
}): Promise<Note> {
  const body: Record<string, unknown> = {
    document_id: payload.documentId ?? undefined,
    project_id: payload.projectId ?? undefined,
    source_quote: payload.source_quote ?? undefined,
    notebook_id: payload.notebook_id ?? undefined,
    title: payload.title ?? undefined,
    position: payload.position ?? 0,
    tag_ids: payload.tag_ids,
  }
  if (payload.content_blocks != null) {
    body.content_blocks = payload.content_blocks
  } else if (payload.content != null) {
    body.content = stripNoteCodeFence(payload.content)
  }
  return apiClient.post<Note>('/notes', body)
}

export async function updateNote(
  noteId: string,
  patch: {
    content?: string
    content_blocks?: ContentBlock[] | null
    source_quote?: string | null
    notebook_id?: string | null
    title?: string | null
    position?: number | null
    tag_ids?: string[] | null
  },
): Promise<Note> {
  const payload: Record<string, unknown> = { ...patch }
  if (payload.content !== undefined) payload.content = stripNoteCodeFence(payload.content as string)
  if (patch.content_blocks !== undefined) payload.content_blocks = patch.content_blocks
  return apiClient.patch<Note>(`/notes/${encodeURIComponent(noteId)}`, payload)
}

export async function deleteNote(noteId: string): Promise<void> {
  await apiClient.delete<void>(`/notes/${encodeURIComponent(noteId)}`)
}

// ---- Notebooks ----
export async function listNotebooks(projectId: string = 'default'): Promise<Notebook[]> {
  const data = await apiClient.get<ListNotebooksResponse>(`/notes/notebooks?project_id=${encodeURIComponent(projectId)}`)
  return data.notebooks ?? []
}

export async function createNotebook(payload: {
  project_id?: string
  parent_id?: string | null
  title?: string
}): Promise<Notebook> {
  return apiClient.post<Notebook>('/notes/notebooks', {
    project_id: payload.project_id ?? 'default',
    parent_id: payload.parent_id ?? undefined,
    title: payload.title ?? '',
  })
}

export async function getNotebook(notebookId: string): Promise<Notebook> {
  return apiClient.get<Notebook>(`/notes/notebooks/${encodeURIComponent(notebookId)}`)
}

export async function updateNotebook(
  notebookId: string,
  patch: { title?: string; parent_id?: string | null; position?: number },
): Promise<Notebook> {
  return apiClient.patch<Notebook>(`/notes/notebooks/${encodeURIComponent(notebookId)}`, patch)
}

export async function deleteNotebook(notebookId: string): Promise<void> {
  await apiClient.delete<void>(`/notes/notebooks/${encodeURIComponent(notebookId)}`)
}

// ---- Tags ----
export async function listTags(projectId: string = 'default'): Promise<Tag[]> {
  const data = await apiClient.get<ListTagsResponse>(`/notes/tags?project_id=${encodeURIComponent(projectId)}`)
  return data.tags ?? []
}

export async function createTag(payload: { project_id?: string; name: string }): Promise<Tag> {
  return apiClient.post<Tag>('/notes/tags', {
    project_id: payload.project_id ?? 'default',
    name: payload.name,
  })
}

export async function deleteTag(tagId: string): Promise<void> {
  await apiClient.delete<void>(`/notes/tags/${encodeURIComponent(tagId)}`)
}

export async function setNoteTags(noteId: string, tagIds: string[]): Promise<Note> {
  return apiClient.patch<Note>(`/notes/${encodeURIComponent(noteId)}/tags`, { tag_ids: tagIds })
}

// ---- Generate ----
export async function generateNoteFromDocument(documentId: string): Promise<string> {
  const data = await apiClient.post<GenerateNoteResponse>('/notes/generate', {
    source_type: 'document',
    document_id: documentId,
  })
  return data.content
}

export async function generateNoteFromText(text: string): Promise<string> {
  const data = await apiClient.post<GenerateNoteResponse>('/notes/generate', {
    source_type: 'text',
    text,
  })
  return data.content
}

// ---- Backlinks & search ----
export async function getBacklinks(noteId: string, projectId?: string): Promise<Note[]> {
  const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''
  const data = await apiClient.get<{ notes: Note[] }>(`/notes/${encodeURIComponent(noteId)}/backlinks${qs}`)
  return data.notes ?? []
}

export async function searchNotes(params: {
  q: string
  projectId?: string
  notebookId?: string
  tagId?: string
}): Promise<Note[]> {
  const search = new URLSearchParams()
  search.set('q', params.q)
  if (params.projectId) search.set('project_id', params.projectId)
  if (params.notebookId) search.set('notebook_id', params.notebookId)
  if (params.tagId) search.set('tag_id', params.tagId)
  const data = await apiClient.get<ListNotesResponse>(`/notes/search?${search.toString()}`)
  return data.notes ?? []
}
