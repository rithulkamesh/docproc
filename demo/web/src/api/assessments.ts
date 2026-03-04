import { apiClient } from './client'

export type ConfidenceLevel = 'low' | 'medium' | 'high'

export interface AssessmentQuestion {
  id: string
  assessment_id: string
  type: 'short_answer' | 'long_answer' | 'single_select' | 'mcq' | 'multi'
  prompt: string
  correct_answer: string | null
  options: string[] | null
  position: number
  /** From question metadata */
  difficulty?: 'easy' | 'medium' | 'hard' | null
  estimated_time?: number | null
  topic_tag?: string | null
  mark_weight?: number | null
}

export interface Assessment {
  id: string
  project_id: string
  title: string
  ai_generation_enabled: boolean
  ai_config: Record<string, unknown> | null
  created_at: string
  updated_at: string
  marking_scheme?: PaperPattern | null
  time_limit_minutes?: number | null
  /** Omitted in list response; present in getAssessment */
  questions?: AssessmentQuestion[]
}

export interface SubmitResponse {
  submission_id: string
  score_pct: number | null
  ai_status: 'pending_ai_evaluation' | 'completed'
  graded_count: number
  pending_ai_count: number
  adjusted_total_score?: number | null
}

export interface QuestionResult {
  score: number | null
  feedback: string
  justification?: string
  /** Student-safe: strengths/missing_concepts/confidence only; no reference answers */
  strengths?: string[]
  missing_concepts?: string[]
  confidence?: number
}

export interface TutorFeedbackItem {
  conceptual_gaps: string[]
  misunderstood_topics: string[]
  targeted_revision_plan: string[]
  recommended_practice_type: string
  encouragement: string
  difficulty_adjustment_advice: string
}

export interface Submission {
  id: string
  assessment_id: string
  answers: Record<string, string | string[]>
  status: string
  ai_status: string | null
  score_pct: number | null
  graded_at: string | null
  created_at: string
  /** Per-question score and feedback when available (from GET submission). Student-safe. */
  question_results?: Record<string, QuestionResult>
  grading_model_version?: string
  re_evaluation_used?: boolean
  question_times?: Record<string, number>
  total_time_spent_seconds?: number | null
  ended_at?: string | null
  adjusted_total_score?: number | null
  ability_score?: number | null
  tutor_feedback?: {
    per_question?: Array<{ question_id: string; feedback: TutorFeedbackItem }>
    summary_encouragement?: string
  } | null
  /** Soft note only; never an accusation */
  integrity_note?: string | null
}

export interface IntegritySignals {
  per_question?: Array<{
    question_id: string
    time_spent?: number
    paste_detected?: boolean
    tab_switch_count?: number
    answer_length_spike?: boolean
  }>
  session?: {
    average_time_per_question?: number
    variance_response_length?: number
    editing_bursts?: number
    paste_after_long_tab_switch?: boolean
  }
}

export interface PaperPatternSection {
  name: string
  question_count: number
  marks_each: number
  type: 'short' | 'long'
}

export interface PaperPattern {
  total_marks: number
  sections: PaperPatternSection[]
}

export interface PaperUploadResponse {
  pattern: PaperPattern
  extracted_preview: string
}

export interface CreateAssessmentParams {
  title: string
  project_id?: string
  document_id: string
  ai_generation_enabled?: boolean
  ai_config?: {
    subject?: string
    topics?: string[]
    difficulty?: 'easy' | 'medium' | 'hard' | 'mixed'
    question_count?: number
    time_limit_minutes?: number
    include_long_answers?: boolean
    [key: string]: unknown
  }
  marking_scheme?: PaperPattern
  time_limit_minutes?: number
}

export async function createAssessment(
  params: CreateAssessmentParams
): Promise<{ id: string; title: string; project_id: string }> {
  const body = {
    title: params.title,
    project_id: params.project_id ?? 'default',
    document_id: params.document_id,
    ai_generation_enabled: params.ai_generation_enabled ?? true,
    ai_config: params.ai_config ?? undefined,
    marking_scheme: params.marking_scheme ?? undefined,
    time_limit_minutes: params.time_limit_minutes ?? undefined,
  }
  return apiClient.post<Assessment>('/assessments', body, {
    cache: 'no-store',
  })
}

export async function getAssessment(assessmentId: string): Promise<Assessment> {
  return apiClient.get<Assessment>(`/assessments/${assessmentId}`)
}

export interface SubmitAssessmentOptions {
  question_times?: Record<string, number>
  total_time_spent_seconds?: number
  confidence_levels?: Record<string, ConfidenceLevel>
  integrity_signals?: IntegritySignals
}

export async function submitAssessment(
  assessmentId: string,
  answers: Record<string, string | string[]>,
  options?: SubmitAssessmentOptions
): Promise<SubmitResponse> {
  return apiClient.post<SubmitResponse>(`/assessments/${assessmentId}/submit`, {
    answers,
    question_times: options?.question_times,
    total_time_spent_seconds: options?.total_time_spent_seconds,
    confidence_levels: options?.confidence_levels,
    integrity_signals: options?.integrity_signals,
  })
}

/** Upload question paper (PDF/DOCX/image); returns detected pattern for confirmation. */
export async function uploadPaper(file: File): Promise<PaperUploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  return apiClient.postForm<PaperUploadResponse>('/assessments/paper-upload', formData)
}

export async function getSubmission(
  assessmentId: string,
  submissionId: string
): Promise<Submission> {
  return apiClient.get<Submission>(`/assessments/${assessmentId}/submissions/${submissionId}`)
}

/** Get submission by id only (assessment_id returned in body). */
export async function getSubmissionById(submissionId: string): Promise<Submission> {
  return apiClient.get<Submission>(`/assessments/submissions/${submissionId}`)
}

/** List assessments (optionally by project_id). */
export async function listAssessments(projectId?: string): Promise<Assessment[]> {
  const qs = projectId != null ? `?project_id=${encodeURIComponent(projectId)}` : ''
  return apiClient.get<Assessment[]>(`/assessments${qs}`)
}

/** List submissions for an assessment. */
export async function listSubmissions(assessmentId: string): Promise<Submission[]> {
  return apiClient.get<Submission[]>(`/assessments/${assessmentId}/submissions`)
}

/** Generate and store AI tutor feedback for a submission. Returns the feedback payload. */
export async function generateTutorFeedback(
  assessmentId: string,
  submissionId: string
): Promise<Submission['tutor_feedback']> {
  return apiClient.post<Submission['tutor_feedback']>(
    `/assessments/${assessmentId}/submissions/${submissionId}/tutor-feedback`,
    {}
  )
}

/** Re-evaluate a submission once. Returns new score; sets re_evaluation_used. */
export async function reEvaluateSubmission(
  assessmentId: string,
  submissionId: string
): Promise<{ submission_id: string; score_pct: number | null; re_evaluation_used: boolean; previous_score_pct?: number | null }> {
  return apiClient.post(`/assessments/${assessmentId}/submissions/${submissionId}/re-evaluate`, {})
}

/** Submission status for polling. Backend returns ai_status on GET submission. */
export type SubmissionStatus =
  | 'pending_ai_evaluation'
  | 'completed'
  | 'failed'
  | null

/** Map backend ai_status to unified status for UI. */
export function submissionStatusFromApi(aiStatus: string | null): SubmissionStatus {
  if (!aiStatus) return null
  if (aiStatus === 'pending_ai_evaluation') return 'pending_ai_evaluation'
  if (aiStatus === 'completed') return 'completed'
  if (aiStatus === 'failed') return 'failed'
  return null
}

export const POLL_INTERVAL_MS = 3000
export const POLL_TIMEOUT_MS = 5 * 60 * 1000 // 5 minutes
