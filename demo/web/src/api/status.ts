import { apiClient } from './client'

/** Server AI config from .env. Only non-secret values (provider, model names). API keys never sent. */
export interface ApiStatus {
  ok: boolean
  rag_backend: string | null
  rag_configured: boolean
  database_provider: string | null
  primary_ai: string | null
  namespace: string | null
  default_rag_model?: string | null
  /** Azure embedding deployment name when server uses Azure; never exposes keys or endpoints. */
  embedding_deployment?: string | null
  error?: string
}

export async function fetchStatus(): Promise<ApiStatus> {
  return apiClient.get<ApiStatus>('/status')
}

