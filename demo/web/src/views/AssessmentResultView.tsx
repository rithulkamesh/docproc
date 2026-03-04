import { useParams, Link } from 'react-router-dom'
import { useState, useCallback } from 'react'
import useSWR from 'swr'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { LatexText } from '@/components/LatexText'
import { getSubmission, getAssessment, reEvaluateSubmission, generateTutorFeedback } from '@/api/assessments'
import type { Submission, Assessment, AssessmentQuestion, QuestionResult, TutorFeedbackItem } from '@/api/assessments'
import { sanitizeHtmlForDisplay } from '@/lib/sanitize'
import { Loader2 } from 'lucide-react'

function formatAnswer(value: string | string[] | undefined): string {
  if (value == null) return '—'
  if (Array.isArray(value)) return value.join(', ')
  return String(value).trim() || '—'
}

function AnswerDisplay({ content, isHtml }: { content: string; isHtml: boolean }) {
  if (!content || content === '—') return <span className="text-muted-foreground">No answer</span>
  if (isHtml) {
    return (
      <div
        className="text-sm leading-relaxed [&_p]:my-1"
        dangerouslySetInnerHTML={{ __html: sanitizeHtmlForDisplay(content) }}
      />
    )
  }
  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed">
      <LatexText text={content} />
    </p>
  )
}

export function AssessmentResultView() {
  const { id: assessmentId, submissionId } = useParams<{ id: string; submissionId: string }>()

  const { data: submission, error: subError, isLoading: submissionLoading, mutate: mutateSubmission } = useSWR<Submission>(
    assessmentId && submissionId ? ['submission', assessmentId, submissionId] : null,
    () => getSubmission(assessmentId!, submissionId!)
  )

  const [reEvaluating, setReEvaluating] = useState(false)
  const [reEvaluateError, setReEvaluateError] = useState<string | null>(null)
  const [tutorLoading, setTutorLoading] = useState(false)
  const [tutorError, setTutorError] = useState<string | null>(null)

  const handleReEvaluate = useCallback(async () => {
    if (!assessmentId || !submissionId || submission?.re_evaluation_used) return
    setReEvaluateError(null)
    setReEvaluating(true)
    try {
      await reEvaluateSubmission(assessmentId, submissionId)
      await mutateSubmission()
    } catch (err) {
      setReEvaluateError(err instanceof Error ? err.message : 'Re-evaluation failed')
    } finally {
      setReEvaluating(false)
    }
  }, [assessmentId, submissionId, submission?.re_evaluation_used, mutateSubmission])

  const handleGenerateTutorFeedback = useCallback(async () => {
    if (!assessmentId || !submissionId) return
    setTutorError(null)
    setTutorLoading(true)
    try {
      await generateTutorFeedback(assessmentId, submissionId)
      await mutateSubmission()
    } catch (err) {
      setTutorError(err instanceof Error ? err.message : 'Could not generate insights')
    } finally {
      setTutorLoading(false)
    }
  }, [assessmentId, submissionId, mutateSubmission])

  const { data: assessment, isLoading: assessmentLoading } = useSWR<Assessment>(
    assessmentId ? ['assessment', assessmentId] : null,
    () => getAssessment(assessmentId!),
    { revalidateOnFocus: false }
  )

  const isLoading = submissionLoading || (assessmentLoading && !assessment)

  if (isLoading) {
    return (
      <div className="flex min-h-[40vh] flex-col items-center justify-center gap-2 p-6">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-sm text-muted-foreground">Loading result…</p>
      </div>
    )
  }

  if (subError || !submission) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-sm text-destructive">{subError?.message ?? 'Result not found'}</p>
        <Button variant="ghost" asChild>
          <Link to="/assessments/create">Back to create</Link>
        </Button>
      </div>
    )
  }

  const scorePct = submission.adjusted_total_score ?? submission.score_pct ?? 0
  const rawScorePct = submission.score_pct ?? 0
  const questions = assessment?.questions ?? []
  const questionResults = submission.question_results ?? {}
  const totalTimeSpent = submission.total_time_spent_seconds
  const questionTimes = submission.question_times ?? {}
  const tutorFeedback = submission.tutor_feedback
  const integrityNote = submission.integrity_note

  function formatTime(sec: number): string {
    const m = Math.floor(sec / 60)
    const s = sec % 60
    return s > 0 ? `${m}m ${s}s` : `${m}m`
  }

  const chartData = questions.map((q, idx) => {
    const result = questionResults[q.id]
    const score = result?.score ?? 0
    return { name: `Q${idx + 1}`, score, fullMark: 100 }
  })

  const weakQuestions = questions.filter((q) => {
    const result = questionResults[q.id]
    const score = result?.score ?? 0
    return score < 70
  })

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6">
      <h1 className="text-2xl font-semibold tracking-tight">Assessment result</h1>
      {assessment && <p className="text-sm text-muted-foreground">{assessment.title}</p>}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Total score</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-3xl font-bold">
            {scorePct}% <span className="text-lg font-normal text-muted-foreground">/ 100%</span>
            {submission.adjusted_total_score != null && submission.adjusted_total_score !== rawScorePct && (
              <span className="ml-2 text-sm font-normal text-muted-foreground">(confidence-weighted)</span>
            )}
          </p>
          {questions.length > 0 && (
            <p className="text-sm text-muted-foreground">{questions.length} question{questions.length !== 1 ? 's' : ''}</p>
          )}
          {totalTimeSpent != null && totalTimeSpent > 0 && (
            <p className="text-sm text-muted-foreground">Total time: {formatTime(totalTimeSpent)}</p>
          )}
        </CardContent>
      </Card>

      {weakQuestions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Review weak topics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              These questions had lower scores. Use Chat to ask follow-ups or generate flashcards to practice.
            </p>
            <ul className="list-inside list-disc space-y-1 text-sm">
              {weakQuestions.map((q) => {
                const result = questionResults[q.id]
                const score = result?.score ?? 0
                const qIndex = questions.findIndex((qu) => qu.id === q.id) + 1
                const promptPreview = q.prompt.slice(0, 80) + (q.prompt.length > 80 ? '…' : '')
                return (
                  <li key={q.id}>
                    <span className="font-medium">Question {qIndex}:</span>{' '}
                    {promptPreview} <span className="text-muted-foreground">({score}%)</span>
                  </li>
                )
              })}
            </ul>
            <div className="flex flex-wrap gap-2 pt-2">
              <Button variant="secondary" size="sm" asChild>
                <Link to="/">Chat about these topics</Link>
              </Button>
              <Button variant="secondary" size="sm" asChild>
                <Link to="/">Generate flashcards</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {chartData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Score by question</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(value: number) => [`${value}%`, 'Score']} />
                  <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={entry.score >= 70 ? 'hsl(var(--success))' : entry.score >= 40 ? 'hsl(var(--primary))' : 'hsl(var(--destructive))'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {submission.ai_status === 'failed' && (
        <p className="text-sm text-destructive">AI evaluation could not be completed for some answers. Your score may be partial.</p>
      )}
      {reEvaluateError && <p className="text-sm text-destructive">{reEvaluateError}</p>}
      {integrityNote && <p className="text-sm italic text-muted-foreground">{integrityNote}</p>}
      <div className="flex flex-wrap items-center gap-2">
        {submission.grading_model_version && (
          <span className="text-xs text-muted-foreground">Grading model: {submission.grading_model_version}</span>
        )}
        {submission.re_evaluation_used && <Badge variant="secondary">Re-evaluation used</Badge>}
      </div>

      {questions.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-lg font-semibold">Review your answers</h2>
          <div className="space-y-4">
            {questions.map((q: AssessmentQuestion, idx: number) => {
              const result: QuestionResult | undefined = questionResults[q.id]
              const userAnswer = submission.answers[q.id]
              const displayAnswer = formatAnswer(userAnswer as string | string[] | undefined)
              const isHtml = typeof userAnswer === 'string' && (userAnswer.startsWith('<') || userAnswer.includes('<p>'))
              const score = result?.score ?? null
              const feedback = result?.feedback ?? ''
              const justification = result?.justification ?? ''
              const isCorrect = score !== null && score >= 100
              const strengths = (result as { strengths?: string[] })?.strengths
              const missingConcepts = (result as { missing_concepts?: string[] })?.missing_concepts
              const confidence = (result as { confidence?: number })?.confidence

              return (
                <Card key={q.id}>
                  <CardContent className="space-y-3 pt-6">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span className="font-semibold uppercase">Question {idx + 1}</span>
                      {questionTimes[q.id] != null && questionTimes[q.id] > 0 && (
                        <span>Time: {formatTime(questionTimes[q.id])}</span>
                      )}
                    </div>
                    <div className="font-medium leading-snug">
                      <LatexText text={q.prompt} />
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-semibold text-muted-foreground">Your answer</div>
                      <AnswerDisplay content={displayAnswer} isHtml={isHtml} />
                    </div>
                    {score != null && (
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant={isCorrect ? 'success' : 'destructive'}>{score}%</Badge>
                        {feedback && <span className="text-sm"><LatexText text={feedback} /></span>}
                      </div>
                    )}
                    {justification && (
                      <p className="text-sm leading-relaxed text-muted-foreground"><LatexText text={justification} /></p>
                    )}
                    {Array.isArray(strengths) && strengths.length > 0 && (
                      <div>
                        <div className="mb-1 text-xs font-semibold text-muted-foreground">Strengths</div>
                        <ul className="list-inside list-disc space-y-0.5 text-sm">
                          {strengths.map((s, i) => (
                            <li key={i}><LatexText text={s} /></li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {Array.isArray(missingConcepts) && missingConcepts.length > 0 && (
                      <div>
                        <div className="mb-1 text-xs font-semibold text-muted-foreground">Missing concepts</div>
                        <ul className="list-inside list-disc space-y-0.5 text-sm">
                          {missingConcepts.map((c, i) => (
                            <li key={i}><LatexText text={c} /></li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {confidence != null && (
                      <p className="text-xs text-muted-foreground">Confidence: {Math.round(confidence * 100)}%</p>
                    )}
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </section>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Performance Insights</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {tutorFeedback?.per_question && tutorFeedback.per_question.length > 0 ? (
            <>
              {tutorFeedback.summary_encouragement && (
                <p className="text-sm leading-relaxed">{tutorFeedback.summary_encouragement}</p>
              )}
              {tutorFeedback.per_question.map(({ question_id, feedback }: { question_id: string; feedback: TutorFeedbackItem }) => (
                <div key={question_id} className="space-y-2 rounded-lg border border-border p-3">
                  {feedback.conceptual_gaps?.length > 0 && (
                    <div>
                      <div className="text-xs font-semibold text-muted-foreground">Conceptual gaps</div>
                      <ul className="list-inside list-disc text-sm">{feedback.conceptual_gaps.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    </div>
                  )}
                  {feedback.misunderstood_topics?.length > 0 && (
                    <div>
                      <div className="text-xs font-semibold text-muted-foreground">Topics to revisit</div>
                      <ul className="list-inside list-disc text-sm">{feedback.misunderstood_topics.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    </div>
                  )}
                  {feedback.targeted_revision_plan?.length > 0 && (
                    <div>
                      <div className="text-xs font-semibold text-muted-foreground">Revision plan</div>
                      <ul className="list-inside list-disc text-sm">{feedback.targeted_revision_plan.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    </div>
                  )}
                  {feedback.recommended_practice_type && (
                    <p className="text-sm"><strong>Practice:</strong> {feedback.recommended_practice_type}</p>
                  )}
                  {feedback.encouragement && <p className="text-sm italic text-muted-foreground">{feedback.encouragement}</p>}
                </div>
              ))}
            </>
          ) : (
            <>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Get AI-generated feedback on your answers: conceptual gaps, revision plan, and encouragement (without revealing correct answers).
              </p>
              {tutorError && <p className="text-sm text-destructive">{tutorError}</p>}
              <Button variant="ghost" onClick={() => void handleGenerateTutorFeedback()} disabled={tutorLoading}>
                {tutorLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Generate performance insights
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">How to improve</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm leading-relaxed text-muted-foreground">
            Review your sources, add notes from the material, and use the chat to ask follow-up questions on topics you missed.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button variant="ghost" size="sm" asChild><Link to="/">Open workspace (chat)</Link></Button>
            <Button variant="ghost" size="sm" asChild><Link to="/">Notes</Link></Button>
            <Button variant="ghost" size="sm" asChild><Link to="/">Sources</Link></Button>
            <Button variant="ghost" size="sm" asChild><Link to="/">Flashcards</Link></Button>
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-wrap items-center gap-2">
        <Button asChild><Link to="/assessments/create">Create another assessment</Link></Button>
        {assessmentId && (
          <>
            <Button variant="ghost" asChild><Link to={`/assessments/${assessmentId}/submissions`}>Submissions</Link></Button>
            <Button variant="ghost" asChild><Link to={`/assessments/${assessmentId}/take`}>Retake</Link></Button>
          </>
        )}
        {assessmentId && submissionId && !submission.re_evaluation_used && (
          <Button variant="ghost" onClick={() => void handleReEvaluate()} disabled={reEvaluating} className="ml-auto">
            {reEvaluating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Re-evaluate (once)
          </Button>
        )}
      </div>
    </div>
  )
}
