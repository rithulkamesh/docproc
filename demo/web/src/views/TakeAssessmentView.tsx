import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import useSWR from 'swr'
import { SoftButton } from '../components/SoftButton'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { LatexText } from '../components/LatexText'
import { RichTextEditor } from '../components/RichTextEditor'
import { Spinner } from '../components/Spinner'
import { getAssessment, submitAssessment } from '../api/assessments'
import { useSubmissionPoll } from '../hooks/useSubmissionPoll'
import { POLL_TIMEOUT_MS } from '../api/assessments'
import type { Assessment, AssessmentQuestion, ConfidenceLevel, IntegritySignals } from '../api/assessments'

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
      <div className="form-card" style={{ gap: 'var(--space-md)' }}>
        {options.map((opt, idx) => (
          <label key={idx} className="body-base" style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-md)', cursor: disabled ? 'default' : 'pointer' }}>
            <input type="radio" name={question.id} value={opt} checked={(value as string) === opt} onChange={() => onChange(opt)} disabled={disabled} style={{ marginTop: 4 }} />
            <LatexText text={opt} style={{ flex: 1 }} />
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
      className="input"
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
      <div className="content-max">
        <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{fetchError?.message ?? 'Invalid assessment ID'}</p>
      </div>
    )
  }

  if (!assessment) {
    return (
      <div className="content-max loading-state">
        <p className="text-muted">Loading assessment…</p>
      </div>
    )
  }

  if (!questions.length) {
    return (
      <div className="content-max">
        <p className="text-muted">No questions in this assessment.</p>
      </div>
    )
  }

  if (submitted) {
    return (
      <div className="submit-pending-card">
        <Card>
          <Spinner size="md" />
          <div>
            <h2 className="heading-xl" style={{ marginBottom: 'var(--space-md)', marginTop: 0 }}>AI evaluation in progress</h2>
            <p className="text-muted" style={{ margin: 0 }}>You will be redirected to your result when ready.</p>
          </div>
          {pollTimedOut && submissionId && (
            <p className="body-sm" style={{ margin: 0 }}>
              Taking longer than expected?{' '}
              <button type="button" className="link-button" onClick={() => navigate(`/assessments/${assessmentId}/result/${submissionId}`, { replace: true })}>View result</button>
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
    <div className="take-layout">
      <div
        className={`take-timer ${showWarning1 ? 'take-timer--danger' : showWarning10 ? 'take-timer--warn' : ''}`}
        title="Time spent"
      >
        {formatTime(elapsedSeconds)}
        {timeLimitMinutes > 0 && (
          <span className="body-sm text-muted" style={{ marginLeft: 'var(--space-sm)' }}>/ {timeLimitMinutes} min</span>
        )}
      </div>
      <nav className="take-nav">
        <div className="section-label mb-sm">Questions</div>
        <div className="take-question-grid">
          {questions.map((q, idx) => {
            const active = idx === currentIndex
            const answered = isAnswered(q)
            return (
              <button
                key={q.id}
                type="button"
                className={`take-question-dot ${active ? 'take-question-dot--active' : ''} ${answered ? 'take-question-dot--answered' : ''}`}
                onClick={() => !submitted && setCurrentIndex(idx)}
                disabled={submitted}
                title={`Question ${idx + 1}${answered ? ' (answered)' : ''}`}
              >
                {idx + 1}
              </button>
            )
          })}
        </div>
      </nav>

      <div style={{ flex: 1, minWidth: 0 }}>
        <h1 className="heading-xl mb-md">{assessment.title}</h1>
        <p className="body-sm text-muted mb-xl">Question {currentIndex + 1} of {questions.length}</p>

        {submitError && (
          <p className="body-sm mb-md" style={{ color: 'var(--color-danger)' }}>{submitError}</p>
        )}

        {currentQuestion && (
          <section
            className="take-question-card"
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
          </section>
        )}

        <div className="flex-between gap-md flex-wrap">
          <div className="gap-md" style={{ display: 'flex' }}>
            <SoftButton onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))} disabled={currentIndex === 0 || submitted}>
              ← Previous
            </SoftButton>
            <SoftButton onClick={() => setCurrentIndex((i) => Math.min(questions.length - 1, i + 1))} disabled={currentIndex === questions.length - 1 || submitted}>
              Next →
            </SoftButton>
          </div>
          {!submitted && (
            <Button type="button" onClick={() => void handleSubmit()} loading={submitting} disabled={submitting}>
              Submit assessment
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
