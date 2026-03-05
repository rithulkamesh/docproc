import { apiClient } from './client'
import type { RagResponse, RagSource } from '../types'

/** Optional per-request AI config. Sent to backend; backend uses for chat when provided. */
export interface AIRequestConfig {
  api_key?: string | null
  model?: string | null
  provider?: string | null
}

interface QueryRequestBody {
  prompt: string
  query?: string
  top_k?: number
  model?: string | null
  api_key?: string | null
  provider?: string | null
}

export async function runQuery(
  prompt: string,
  topK = 10,
  options?: { model?: string | null; api_key?: string | null; provider?: string | null } | null
): Promise<RagResponse> {
  const body: QueryRequestBody = {
    prompt,
    top_k: topK,
    model: options?.model ?? undefined,
    api_key: options?.api_key ?? undefined,
    provider: options?.provider ?? undefined,
  }
  return apiClient.post<RagResponse>('/query', body)
}

export interface QueryStreamCallbacks {
  onSources: (sources: RagSource[]) => void
  onDelta: (delta: string) => void
  onDone: () => void
  onError: (message: string) => void
}

export async function runQueryStream(
  prompt: string,
  callbacks: QueryStreamCallbacks,
  options?: AIRequestConfig | null
): Promise<boolean> {
  const body: Record<string, unknown> = { prompt }
  if (options?.api_key != null && options.api_key !== '') body.api_key = options.api_key
  if (options?.model != null && options.model !== '') body.model = options.model
  if (options?.provider != null && options.provider !== '') body.provider = options.provider
  const res = await fetch(`${apiClient.baseUrl}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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
          if (typeof data.error === 'string') {
            callbacks.onError(data.error)
            return false
          }
          if (Array.isArray(data.sources)) {
            callbacks.onSources(data.sources as RagSource[])
          } else if (typeof data.delta === 'string') {
            callbacks.onDelta(data.delta)
          } else if (data.done === true) {
            callbacks.onDone()
            return true
          }
        } catch {
          // ignore malformed NDJSON lines
        }
      }
    }
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer.trim()) as Record<string, unknown>
        if (data.done === true) callbacks.onDone()
      } catch {}
    } else {
      callbacks.onDone()
    }
    return true
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : 'Stream failed')
    return false
  }
}

