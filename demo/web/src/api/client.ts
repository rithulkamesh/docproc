const API_BASE = import.meta.env.VITE_DOCPROC_API_URL?.replace(/\/+$/, '') || 'http://localhost:8080'

async function request<T>(path: string, options?: RequestInit & { skipJsonContentType?: boolean }): Promise<T> {
  const { skipJsonContentType: skipJson, ...fetchOptions } = options ?? {}
  const res = await fetch(`${API_BASE}${path}`, {
    ...fetchOptions,
    headers: skipJson ? (fetchOptions.headers ?? {}) : {
      'Content-Type': 'application/json',
      ...(fetchOptions.headers ?? {}),
    },
  })
  if (!res.ok) {
    let message = res.statusText
    try {
      const data = (await res.json()) as { detail?: unknown }
      if (typeof data.detail === 'string') {
        message = data.detail
      } else if (Array.isArray(data.detail) && typeof data.detail[0] === 'string') {
        message = data.detail[0]
      }
    } catch {
      // ignore
    }
    throw new Error(message || `Request failed with status ${res.status}`)
  }
  if (res.status === 204) {
    return undefined as unknown as T
  }
  return (await res.json()) as T
}

export const apiClient = {
  baseUrl: API_BASE,
  get: request,
  post: <T>(path: string, body: unknown) =>
    request<T>(path, {
      method: 'POST',
      body: JSON.stringify(body),
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
}

