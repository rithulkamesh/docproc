const DEFAULT_API_BASE = 'http://localhost:8080'
const raw = (import.meta.env.VITE_DOCPROC_API_URL ?? '').toString().trim().replace(/\/+$/, '')
const API_BASE =
  raw && (raw.startsWith('http://') || raw.startsWith('https://'))
    ? raw
    : DEFAULT_API_BASE

const REQUEST_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  readonly status: number
  readonly detail?: string
  readonly code?: string
  constructor(message: string, status: number, detail?: string, code?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.code = code
    Object.setPrototypeOf(this, ApiError.prototype)
  }
}

async function request<T>(
  path: string,
  options?: RequestInit & { skipJsonContentType?: boolean; timeoutMs?: number }
): Promise<T> {
  const { skipJsonContentType: skipJson, timeoutMs = REQUEST_TIMEOUT_MS, ...fetchOptions } = options ?? {}
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions,
      signal: controller.signal,
      headers: skipJson ? (fetchOptions.headers ?? {}) : {
        'Content-Type': 'application/json',
        ...(fetchOptions.headers ?? {}),
      },
    })
    clearTimeout(timeoutId)

    if (!res.ok) {
      let message = res.statusText
      let detail: string | undefined
      let code: string | undefined
      try {
        const data = (await res.json()) as { detail?: unknown; code?: string }
        if (typeof data.detail === 'string') {
          message = data.detail
          detail = data.detail
        } else if (Array.isArray(data.detail) && typeof data.detail[0] === 'string') {
          message = data.detail[0]
          detail = data.detail[0]
        }
        if (typeof data.code === 'string') code = data.code
      } catch {}
      throw new ApiError(message || `Request failed with status ${res.status}`, res.status, detail, code)
    }
    if (res.status === 204) {
      return undefined as unknown as T
    }
    return (await res.json()) as T
  } catch (err) {
    clearTimeout(timeoutId)
    if (err instanceof ApiError) throw err
    if (err instanceof Error && err.name === 'AbortError') {
      throw new ApiError(`Request timed out after ${timeoutMs}ms`, 408)
    }
    throw err
  }
}

export const apiClient = {
  baseUrl: API_BASE,
  get: request,
  post: <T>(path: string, body: unknown, fetchOptions?: RequestInit) =>
    request<T>(path, {
      method: 'POST',
      body: JSON.stringify(body),
      ...fetchOptions,
    }),
  postForm: <T>(path: string, formData: FormData) =>
    request<T>(path, {
      method: 'POST',
      body: formData,
      skipJsonContentType: true,
    } as RequestInit & { skipJsonContentType?: boolean }),
  delete: <T>(path: string) =>
    request<T>(path, {
      method: 'DELETE',
    }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
}

