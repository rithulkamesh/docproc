import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { LatexText } from '@/components/LatexText'
import { RichTextEditor } from '@/components/RichTextEditor'
import { Loader2 } from 'lucide-react'
import { getAssessment, submitAssessment, POLL_TIMEOUT_MS } from '@/api/assessments'
import type { Assessment, AssessmentQuestion, ConfidenceLevel, IntegritySignals } from '@/api/assessments'
import { useSubmissionPoll } from '@/hooks/useSubmissionPoll'

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

function QuestionBlock({
  question,
  value,
  onChange,
  disabled,
}: {
  question: AssessmentQuestion
  value: string | string[]
  onChange: (v: string | string[]) => void
  disabled: boolean
}) {
  const type = question.type
  const options = question.options ?? []
  // Single-select only: mcq, single_select; legacy "multi" rendered as radio for backward compat
  if (type === 'mcq' || type === 'single_select' || type === 'multi') {
    return (
      <div className="flex flex-col gap-3">
        {options.map((opt, idx) => (
          <label key={idx} className="flex cursor-pointer items-start gap-3 text-sm">
            <input type="radio" name={question.id} value={opt} checked={(value as string) === opt} onChange={() => onChange(opt)} disabled={disabled} className="mt-1" />
            <LatexText text={opt} className="flex-1" />
          </label>
        ))}
      </div>
    )
  }

  if (type === 'long_answer') {
    return (
      <RichTextEditor
        value={typeof value === 'string' ? value : ''}
        onChange={(html) => onChange(html)}
        placeholder="Write your answer. You can use bold, lists, and code."
        disabled={disabled}
        minHeight={200}
        showEquationPreview
      />
    )
  }

  if (type === 'short_answer') {
    return (
      <RichTextEditor
        value={typeof value === 'string' ? value : ''}
        onChange={(html) => onChange(html)}
        placeholder="Write your answer. Use the toolbar for equations ($...$ or ∑ button)."
        disabled={disabled}
        minHeight={140}
        showEquationPreview
      />
    )
  }

  return (
    <input
      type="text"
      className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm"
      value={typeof value === 'string' ? value : ''}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
    />
  )
}

const AUTO_SAVE_INTERVAL_MS = 30_000

export function TakeAssessmentView() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const assessmentId = id ?? ''

  const { data: assessment, error: fetchError } = useSWR<Assessment>(
    assessmentId ? ['assessment', assessmentId] : null,
    () => getAssessment(assessmentId)
  )

  const [answers, setAnswers] = useState<Record<string, string | string[]>>({})
  const [currentIndex, setCurrentIndex] = useState(0)
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submissionId, setSubmissionId] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [pollTimedOut, setPollTimedOut] = useState(false)
  const [questionTimes, setQuestionTimes] = useState<Record<string, number>>({})
  const [confidenceLevels, setConfidenceLevels] = useState<Record<string, ConfidenceLevel>>({})
  const integrityPerQuestionRef = useRef<Record<string, { paste_detected?: boolean; tab_switch_count?: number }>>({})
  const tabSwitchCountRef = useRef(0)
  const questionStartRef = useRef<number>(Date.now())
  const totalStartRef = useRef<number>(Date.now())
  const timeLimitMinutes = assessment?.time_limit_minutes ?? 30
  const timeLimitSeconds = timeLimitMinutes * 60

  const { status, isPolling } = useSubmissionPoll({
    assessmentId,
    submissionId,
    enabled: submitted && !!submissionId,
  })

  const questions = assessment?.questions ?? []
  const currentQuestion = questions[currentIndex] ?? null

  // Per-question timer: when switching question, accumulate time for the previous question
  useEffect(() => {
    if (submitted || !currentQuestion) return
    questionStartRef.current = Date.now()
    return () => {
      const prevId = questions[currentIndex]?.id
      if (prevId) {
        const elapsed = Math.round((Date.now() - questionStartRef.current) / 1000)
        setQuestionTimes((prev) => ({ ...prev, [prevId]: (prev[prevId] ?? 0) + elapsed }))
      }
    }
  }, [currentIndex, submitted, currentQuestion?.id, questions])

  // Tab visibility: count as integrity signal for current question (demo only; no accusation)
  useEffect(() => {
    if (submitted) return
    const handleVisibility = () => {
      if (document.visibilityState === 'hidden') {
        tabSwitchCountRef.current += 1
        const qid = questions[currentIndex]?.id
        if (qid) {
          const prev = integrityPerQuestionRef.current[qid] ?? {}
          integrityPerQuestionRef.current[qid] = {
            ...prev,
            tab_switch_count: (prev.tab_switch_count ?? 0) + 1,
          }
        }
      }
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => document.removeEventListener('visibilitychange', handleVisibility)
  }, [submitted, currentIndex, questions])

  // Global elapsed timer (for display)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  useEffect(() => {
    if (submitted) return
    const t = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - totalStartRef.current) / 1000))
    }, 1000)
    return () => clearInterval(t)
  }, [submitted])

  const handleChangeAnswer = useCallback((questionId: string, value: string | string[]) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }))
  }, [])

  useEffect(() => {
    if (!submitted || !submissionId) return
    if (status === 'completed' || status === 'failed') {
      navigate(`/assessments/${assessmentId}/result/${submissionId}`, { replace: true })
    }
  }, [submitted, submissionId, status, navigate, assessmentId])

  useEffect(() => {
    if (!submitted || !submissionId || !isPolling) return
    const t = setTimeout(() => setPollTimedOut(true), POLL_TIMEOUT_MS)
    return () => clearTimeout(t)
  }, [submitted, submissionId, isPolling])

  useEffect(() => {
    const t = setInterval(() => {
      if (submitted || !Object.keys(answers).length) return
      const stored = JSON.stringify(answers)
      try {
        window.localStorage.setItem(`assessment-${assessmentId}-draft`, stored)
      } catch {
        // ignore
      }
    }, AUTO_SAVE_INTERVAL_MS)
    return () => clearInterval(t)
  }, [answers, submitted, assessmentId])

  useEffect(() => {
    if (!assessmentId) return
    try {
      const raw = window.localStorage.getItem(`assessment-${assessmentId}-draft`)
      if (raw) {
        const parsed = JSON.parse(raw) as Record<string, string | string[]>
        setAnswers(parsed)
      }
    } catch {
      // ignore
    }
  }, [assessmentId])

  const isAnswered = useCallback(
    (q: AssessmentQuestion) => {
      const v = answers[q.id]
      if (v == null) return false
      if (Array.isArray(v)) return v.length > 0
      const s = (v as string).trim()
      if (!s) return false
      return s !== '<p></p>' && s !== '<p><br></p>'
    },
    [answers]
  )

  const handleSubmit = async () => {
    if (submitted) return
    setSubmitError(null)
    setSubmitting(true)
    const finalQuestionTimes = { ...questionTimes }
    if (currentQuestion) {
      const elapsed = Math.round((Date.now() - questionStartRef.current) / 1000)
      finalQuestionTimes[currentQuestion.id] = (finalQuestionTimes[currentQuestion.id] ?? 0) + elapsed
    }
    const totalSpent = Math.round((Date.now() - totalStartRef.current) / 1000)
    const per_question = questions.map((q) => ({
      question_id: q.id,
      time_spent: finalQuestionTimes[q.id],
      paste_detected: integrityPerQuestionRef.current[q.id]?.paste_detected ?? false,
      tab_switch_count: integrityPerQuestionRef.current[q.id]?.tab_switch_count ?? 0,
    }))
    const integrity_signals: IntegritySignals = {
      per_question,
      session: {
        average_time_per_question:
          questions.length > 0 ? Math.round(totalSpent / questions.length) : undefined,
      },
    }
    const confidencePayload: Record<string, ConfidenceLevel> = {}
    questions.forEach((q) => {
      if (answers[q.id] != null && (typeof answers[q.id] !== 'string' || (answers[q.id] as string).trim() !== '' && (answers[q.id] as string) !== '<p></p>' && (answers[q.id] as string) !== '<p><br></p>'))
        confidencePayload[q.id] = confidenceLevels[q.id] ?? 'medium'
    })
    try {
      const res = await submitAssessment(assessmentId, answers, {
        question_times: finalQuestionTimes,
        total_time_spent_seconds: totalSpent,
        confidence_levels: Object.keys(confidencePayload).length ? confidencePayload : undefined,
        integrity_signals: per_question.some((p) => p.paste_detected || (p.tab_switch_count ?? 0) > 0)
          ? integrity_signals
          : undefined,
      })
      setSubmissionId(res.submission_id)
      setSubmitted(true)
      try {
        window.localStorage.removeItem(`assessment-${assessmentId}-draft`)
      } catch {
        // ignore
      }
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Submit failed')
    } finally {
      setSubmitting(false)
    }
  }

  if (fetchError || !assessmentId) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-sm text-destructive">{fetchError?.message ?? 'Invalid assessment ID'}</p>
      </div>
    )
  }

  if (!assessment) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center p-6">
        <p className="text-sm text-muted-foreground">Loading assessment…</p>
      </div>
    )
  }

  if (!questions.length) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-muted-foreground">No questions in this assessment.</p>
      </div>
    )
  }

  if (submitted) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center p-6">
        <Card className="flex flex-col items-center gap-4 p-8">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <div className="text-center">
            <h2 className="text-xl font-semibold">AI evaluation in progress</h2>
            <p className="mt-2 text-sm text-muted-foreground">You will be redirected to your result when ready.</p>
          </div>
          {pollTimedOut && submissionId && (
            <p className="text-sm">
              Taking longer than expected?{' '}
              <button type="button" className="text-primary underline" onClick={() => navigate(`/assessments/${assessmentId}/result/${submissionId}`, { replace: true })}>View result</button>
            </p>
          )}
        </Card>
      </div>
    )
  }

  const remainingSeconds = Math.max(0, timeLimitSeconds - elapsedSeconds)
  const showWarning10 = remainingSeconds > 0 && remainingSeconds <= 600
  const showWarning1 = remainingSeconds > 0 && remainingSeconds <= 60

  return (
    <div className="flex min-h-0 flex-1 gap-6 p-6">
      <div
        className={`shrink-0 text-sm font-medium ${showWarning1 ? 'text-destructive' : showWarning10 ? 'text-amber-600' : 'text-muted-foreground'}`}
        title="Time spent"
      >
        {formatTime(elapsedSeconds)}
        {timeLimitMinutes > 0 && <span className="ml-2 text-muted-foreground">/ {timeLimitMinutes} min</span>}
      </div>
      <nav className="flex w-48 shrink-0 flex-col gap-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Questions</div>
        <div className="grid grid-cols-5 gap-1">
          {questions.map((q, idx) => {
            const active = idx === currentIndex
            const answered = isAnswered(q)
            return (
              <Button
                key={q.id}
                type="button"
                variant={active ? 'default' : 'outline'}
                size="icon"
                className={`h-8 w-8 shrink-0 ${answered && !active ? 'ring-1 ring-primary' : ''}`}
                onClick={() => !submitted && setCurrentIndex(idx)}
                disabled={submitted}
                title={`Question ${idx + 1}${answered ? ' (answered)' : ''}`}
              >
                {idx + 1}
              </Button>
            )
          })}
        </div>
      </nav>

      <div className="min-w-0 flex-1">
        <h1 className="mb-2 text-xl font-semibold">{assessment.title}</h1>
        <p className="mb-6 text-sm text-muted-foreground">Question {currentIndex + 1} of {questions.length}</p>

        {submitError && <p className="mb-4 text-sm text-destructive">{submitError}</p>}

        {currentQuestion && (
          <Card
            className="mb-6"
            onPaste={() => {
              const qid = currentQuestion.id
              const prev = integrityPerQuestionRef.current[qid] ?? {}
              integrityPerQuestionRef.current[qid] = { ...prev, paste_detected: true }
            }}
          >
            <div className="gap-md mb-md" style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
              {currentQuestion.difficulty && (
                <span className="badge">
                  Difficulty: {currentQuestion.difficulty.charAt(0).toUpperCase() + currentQuestion.difficulty.slice(1)}
                </span>
              )}
            </div>
            <div className="heading-lg mb-lg" style={{ lineHeight: 1.4 }}>
              <LatexText text={currentQuestion.prompt} />
            </div>
            <QuestionBlock
              question={currentQuestion}
              value={answers[currentQuestion.id] ?? ''}
              onChange={(v) => handleChangeAnswer(currentQuestion.id, v)}
              disabled={submitted}
            />
            {!submitted && (
              <div className="mt-lg pt-md" style={{ borderTop: 'var(--border-subtle)' }}>
                <div className="text-xs text-muted mb-sm" style={{ fontWeight: 600 }}>How confident are you?</div>
                <div className="gap-lg body-sm" style={{ display: 'flex', flexWrap: 'wrap' }}>
                  {(['low', 'medium', 'high'] as const).map((level) => (
                    <label key={level} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', cursor: 'pointer' }}>
                      <input
                        type="radio"
                        name={`confidence-${currentQuestion.id}`}
                        checked={(confidenceLevels[currentQuestion.id] ?? 'medium') === level}
                        onChange={() => setConfidenceLevels((prev) => ({ ...prev, [currentQuestion.id]: level }))}
                        style={{ margin: 0 }}
                      />
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </label>
                  ))}
                </div>
              </div>
            )}
          </Card>
        )}

        <div className="mt-6 flex flex-wrap items-center justify-between gap-4 border-t border-border pt-4">
          <div className="flex gap-2">
            <Button variant="secondary" onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))} disabled={currentIndex === 0 || submitted}>
              ← Previous
            </Button>
            <Button variant="secondary" onClick={() => setCurrentIndex((i) => Math.min(questions.length - 1, i + 1))} disabled={currentIndex === questions.length - 1 || submitted}>
              Next →
            </Button>
          </div>
          {!submitted && (
            <Button type="button" onClick={() => void handleSubmit()} disabled={submitting}>
              {submitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Submitting…
                </>
              ) : (
                'Submit assessment'
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
