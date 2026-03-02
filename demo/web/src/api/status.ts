import { apiClient } from './client'

export interface ApiStatus {
  ok: boolean
  rag_backend: string | null
  rag_configured: boolean
  database_provider: string | null
  primary_ai: string | null
  namespace: string | null
  default_rag_model?: string | null
  embedding_deployment?: string | null
  error?: string
}

export async function fetchStatus(): Promise<ApiStatus> {
  return apiClient.get<ApiStatus>('/status')
}

