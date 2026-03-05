import { apiClient } from './client'

/** Server-stored AI config (key never returned). */
export interface AIConfigFromServer {
  provider: string
  model: string
  api_key_configured: boolean
  base_url?: string
  endpoint?: string
  deployment?: string
  embedding_deployment?: string
}

/** Payload for saving AI config. Omit api_key to leave unchanged; send "" to clear. */
export interface AIConfigSavePayload {
  provider: string
  model: string
  api_key?: string | null
  base_url?: string
  endpoint?: string
  deployment?: string
  embedding_deployment?: string
}

export async function fetchAIConfig(): Promise<AIConfigFromServer> {
  return apiClient.get<AIConfigFromServer>('/ai-config')
}

export async function saveAIConfig(payload: AIConfigSavePayload): Promise<AIConfigFromServer> {
  const body: Record<string, unknown> = {
    provider: payload.provider,
    model: payload.model,
    base_url: payload.base_url ?? '',
    endpoint: payload.endpoint ?? '',
    deployment: payload.deployment ?? '',
    embedding_deployment: payload.embedding_deployment ?? '',
  }
  if (payload.api_key !== undefined) {
    body.api_key = payload.api_key === null ? undefined : payload.api_key
  }
  return apiClient.put<AIConfigFromServer>('/ai-config', body)
}
