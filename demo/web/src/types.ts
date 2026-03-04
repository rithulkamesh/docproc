export interface DocumentProgress {
  message?: string
  page?: number
  total?: number
  percent?: number
  heartbeat?: string
}

export interface DocumentSummary {
  id: string
  filename: string
  display_name?: string | null
  status: string
  pages?: number
  project_id?: string | null
  error?: string
  index_error?: string
  progress?: DocumentProgress
}

export interface DocumentDetail extends DocumentSummary {
  full_text?: string
}

export interface RagSource {
  document_id?: string
  filename?: string
  display_name?: string | null
  content?: string
}

export interface RagResponse {
  answer: string
  sources: RagSource[]
}

