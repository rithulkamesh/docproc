import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom'
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

  const editorClass = 'border border-border rounded-md overflow-hidden min-h-[160px] text-[15px]'
  const toolbarClass = 'bg-muted/50 px-3 py-2 gap-2 [&_button]:text-muted-foreground hover:[&_button]:text-foreground'

  if (type === 'long_answer') {
    return (
      <RichTextEditor
        value={typeof value === 'string' ? value : ''}
        onChange={(html) => onChange(html)}
        placeholder="Write your answer…"
        disabled={disabled}
        minHeight={200}
        className={editorClass}
        toolbarClassName={toolbarClass}
        mathInputMode="equationEditor"
      />
    )
  }

  if (type === 'short_answer') {
    return (
      <RichTextEditor
        value={typeof value === 'string' ? value : ''}
        onChange={(html) => onChange(html)}
        placeholder="Write your answer…"
        disabled={disabled}
        minHeight={160}
        className={editorClass}
        toolbarClassName={toolbarClass}
        mathInputMode="equationEditor"
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
  const location = useLocation()
  const assessmentId = id ?? ''

  const fromCreate = location.state?.assessment as Assessment | undefined
  const initialFromCreate = fromCreate?.id === assessmentId ? fromCreate : undefined

  const { data: assessment, error: fetchError } = useSWR<Assessment>(
    assessmentId ? ['assessment', assessmentId] : null,
    () => getAssessment(assessmentId),
    {
      fallbackData: initialFromCreate,
      revalidateOnMount: !initialFromCreate,
      revalidateOnFocus: !initialFromCreate,
      revalidateOnReconnect: !initialFromCreate,
    }
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

  // On question change, record time spent on the previous question
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

  // Track tab switches as integrity signal (demo; not used for accusations)
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
      } catch {}
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
    } catch {}
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
      } catch {}
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Submit failed')
    } finally {
      setSubmitting(false)
    }
  }

  if (fetchError || !assessmentId) {
    return (
      <div className="mx-auto max-w-2xl space-y-4 p-6">
        <p className="text-sm text-destructive">{fetchError?.message ?? 'Invalid assessment ID'}</p>
        <p className="text-sm text-muted-foreground">
          The test may not exist yet or the server may be unavailable. Try generating a new test.
        </p>
        <Button variant="outline" asChild>
          <Link to="/assessments/create">Create a new test</Link>
        </Button>
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
    <div className="flex min-h-0 flex-1 flex-col bg-background">
      <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center justify-between border-b border-border bg-background px-6">
        <h1 className="truncate text-lg font-semibold text-foreground">{assessment.title}</h1>
        <div className="flex items-center gap-4">
          {timeLimitMinutes > 0 && (
            <span
              className={`tabular-nums text-sm ${showWarning1 ? 'text-destructive font-medium' : showWarning10 ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground'}`}
              title="Time remaining"
            >
              {formatTime(remainingSeconds)} left
            </span>
          )}
          <Button type="button" onClick={() => void handleSubmit()} disabled={submitting}>
            {submitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Submitting…
              </>
            ) : (
              'Submit'
            )}
          </Button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 justify-center overflow-auto py-8">
        <div className="w-full max-w-[820px] px-6">
          <div className="space-y-6">
            <div className="mb-6">
              <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Questions
              </p>
              <div className="flex flex-wrap gap-2">
                {questions.map((q, idx) => {
                  const active = idx === currentIndex
                  const answered = isAnswered(q)
                  return (
                    <button
                      key={q.id}
                      type="button"
                      onClick={() => !submitted && setCurrentIndex(idx)}
                      disabled={submitted}
                      title={`Question ${idx + 1}${answered ? ' (answered)' : ''}`}
                      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-medium transition-colors disabled:pointer-events-none ${
                        active
                          ? 'bg-primary text-primary-foreground'
                          : answered
                            ? 'bg-muted text-muted-foreground'
                            : 'border border-border bg-transparent text-muted-foreground hover:bg-muted/50'
                      }`}
                    >
                      {answered && !active ? '✓' : idx + 1}
                    </button>
                  )
                })}
              </div>
            </div>

            {submitError && (
              <p className="text-sm text-destructive">{submitError}</p>
            )}

            {currentQuestion && (
              <Card
                className="rounded-xl border-border bg-card p-8 shadow-none"
                onPaste={() => {
                  const qid = currentQuestion.id
                  const prev = integrityPerQuestionRef.current[qid] ?? {}
                  integrityPerQuestionRef.current[qid] = { ...prev, paste_detected: true }
                }}
              >
                <div className="space-y-6">
                  <p className="text-sm text-muted-foreground">
                    Question {currentIndex + 1} of {questions.length}
                  </p>
                  <div className="text-[18px] leading-relaxed text-foreground">
                    <LatexText text={currentQuestion.prompt} />
                  </div>
                  <div className="pt-2">
                    <QuestionBlock
                      question={currentQuestion}
                      value={answers[currentQuestion.id] ?? ''}
                      onChange={(v) => handleChangeAnswer(currentQuestion.id, v)}
                      disabled={submitted}
                    />
                  </div>

                  {!submitted && (
                    <div className="space-y-3 border-t border-border pt-6">
                      <p className="text-xs font-medium text-muted-foreground">Confidence</p>
                      <div className="flex rounded-md border border-border p-0.5">
                        {(['low', 'medium', 'high'] as const).map((level) => {
                          const isActive = (confidenceLevels[currentQuestion.id] ?? 'medium') === level
                          return (
                            <button
                              key={level}
                              type="button"
                              onClick={() => setConfidenceLevels((prev) => ({ ...prev, [currentQuestion.id]: level }))}
                              className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                                isActive
                                  ? 'bg-primary/10 text-primary'
                                  : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
                              }`}
                            >
                              {level.charAt(0).toUpperCase() + level.slice(1)}
                            </button>
                          )
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}

            <div className="flex justify-between pt-6">
              <Button
                variant="outline"
                className="rounded-md px-4 py-2"
                onClick={() => setCurrentIndex((i) => Math.max(0, i - 1))}
                disabled={currentIndex === 0 || submitted}
              >
                ← Previous
              </Button>
              <Button
                variant="outline"
                className="rounded-md px-4 py-2"
                onClick={() => setCurrentIndex((i) => Math.min(questions.length - 1, i + 1))}
                disabled={currentIndex === questions.length - 1 || submitted}
              >
                Next →
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
