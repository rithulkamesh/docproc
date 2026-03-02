import { apiClient } from './client'
import type { RagResponse } from '../types'

interface QueryRequestBody {
  prompt: string
  top_k?: number
  model?: string | null
}

export async function runQuery(prompt: string, topK = 10, model?: string | null): Promise<RagResponse> {
  return apiClient.post<RagResponse>('/query', { prompt, top_k: topK, model: model ?? undefined } satisfies QueryRequestBody)
}

