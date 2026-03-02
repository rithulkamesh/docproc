import { apiClient } from './client'
import type { DocumentDetail, DocumentSummary } from '../types'

interface ListDocumentsResponse {
  documents: DocumentSummary[]
}

export async function listDocuments(projectId?: string): Promise<DocumentSummary[]> {
  const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : ''
  const data = await apiClient.get<ListDocumentsResponse>(`/documents/${query}`)
  return data.documents ?? []
}

export async function getDocument(documentId: string): Promise<DocumentDetail> {
  return apiClient.get<DocumentDetail>(`/documents/${encodeURIComponent(documentId)}`)
}

export async function deleteDocument(documentId: string): Promise<void> {
  await apiClient.delete<void>(`/documents/${encodeURIComponent(documentId)}`)
}

export async function reindexDocument(documentId: string): Promise<{ ok: boolean; message?: string }> {
  return apiClient.post<{ ok: boolean; message?: string }>(
    `/documents/${encodeURIComponent(documentId)}/reindex`,
    {}
  )
}

export async function uploadDocument(
  file: File,
  projectId?: string,
): Promise<{ id: string; status: string }> {
  const form = new FormData()
  form.append('file', file)
  const url = new URL(`${apiClient.baseUrl}/documents/upload`)
  if (projectId) {
    url.searchParams.set('project_id', projectId)
  }
  const res = await fetch(url.toString(), {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    let message = res.statusText
    try {
      const data = (await res.json()) as { detail?: unknown }
      if (typeof data.detail === 'string') {
        message = data.detail
      }
    } catch {
      // ignore
    }
    throw new Error(message || `Upload failed with status ${res.status}`)
  }
  return (await res.json()) as { id: string; status: string }
}

