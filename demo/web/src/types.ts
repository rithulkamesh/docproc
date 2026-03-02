export interface DocumentSummary {
  id: string
  filename: string
  status: string
  pages?: number
  project_id?: string | null
  /** Set when extraction succeeded but RAG indexing failed (e.g. embedding API error). */
  index_error?: string
}

export interface DocumentDetail extends DocumentSummary {
  full_text?: string
}

export interface RagSource {
  document_id?: string
  filename?: string
  content?: string
}

export interface RagResponse {
  answer: string
  sources: RagSource[]
}

