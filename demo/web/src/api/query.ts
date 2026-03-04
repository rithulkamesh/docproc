import { apiClient } from './client'
import type { RagResponse, RagSource } from '../types'

interface QueryRequestBody {
  prompt: string
  top_k?: number
  model?: string | null
}

export async function runQuery(prompt: string, topK = 10, model?: string | null): Promise<RagResponse> {
  return apiClient.post<RagResponse>('/query', { prompt, top_k: topK, model: model ?? undefined } satisfies QueryRequestBody)
}

export interface QueryStreamCallbacks {
  onSources: (sources: RagSource[]) => void
  onDelta: (delta: string) => void
  onDone: () => void
  onError: (message: string) => void
}

/** Run RAG query with streaming; calls onSources, onDelta, onDone or onError. Returns true if stream was used, false if caller should fallback (e.g. 404). */
export async function runQueryStream(
  prompt: string,
  callbacks: QueryStreamCallbacks
): Promise<boolean> {
  const res = await fetch(`${apiClient.baseUrl}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  })
  if (!res.ok) {
    if (res.status === 404) return false
    const err = await res.json().catch(() => ({})) as { detail?: string }
    callbacks.onError(err.detail ?? res.statusText)
    return false
  }
  const reader = res.body?.getReader()
  if (!reader) {
    callbacks.onError('No response body')
    return false
  }
  const dec = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += dec.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        try {
          const data = JSON.parse(trimmed) as Record<string, unknown>
          if (Array.isArray(data.sources)) {
            callbacks.onSources(data.sources as RagSource[])
          } else if (typeof data.delta === 'string') {
            callbacks.onDelta(data.delta)
          } else if (data.done === true) {
            callbacks.onDone()
            return true
          }
        } catch {
          // skip malformed line
        }
      }
    }
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer.trim()) as Record<string, unknown>
        if (data.done === true) callbacks.onDone()
      } catch {
        // ignore
      }
    } else {
      callbacks.onDone()
    }
    return true
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : 'Stream failed')
    return false
  }
}

