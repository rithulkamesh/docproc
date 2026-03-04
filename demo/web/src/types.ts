/** Progress when status is "processing" (worker heartbeat + message). */
export interface DocumentProgress {
  message?: string
  page?: number
  total?: number
  /** 0–100 when page/total available. */
  percent?: number
  /** ISO timestamp of last progress update; recent = worker is alive. */
  heartbeat?: string
}

export interface DocumentSummary {
  id: string
  filename: string
  /** AI-generated or user-friendly title; when set, UI shows this instead of filename. */
  display_name?: string | null
  status: string
  pages?: number
  project_id?: string | null
  /** Processing/extraction failure message (when status is failed). */
  error?: string
  /** Set when extraction succeeded but RAG indexing failed (e.g. embedding API error). */
  index_error?: string
  /** When status is processing: message and heartbeat so UI can show "Extracting…" and live indicator. */
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

