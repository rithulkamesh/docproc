import { apiClient } from './client'

export interface QuizPair {
  question: string
  answer: string
}

interface QuizGenerateResponse {
  pairs: QuizPair[]
}

export async function generateQuiz(documentId: string, count: number): Promise<QuizPair[]> {
  const data = await apiClient.post<QuizGenerateResponse>('/quiz/generate', {
    document_id: documentId,
    count,
  })
  return data.pairs ?? []
}

