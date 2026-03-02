import { useParams, Link } from 'react-router-dom'
import { useState, useCallback } from 'react'
import useSWR from 'swr'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { LatexText } from '../components/LatexText'
import { Spinner } from '../components/Spinner'
import { getSubmission, getAssessment, reEvaluateSubmission, generateTutorFeedback } from '../api/assessments'
import type { Submission, Assessment, AssessmentQuestion, QuestionResult, TutorFeedbackItem } from '../api/assessments'
import { sanitizeHtmlForDisplay } from '../lib/sanitize'

function formatAnswer(value: string | string[] | undefined): string {
  if (value == null) return '—'
  if (Array.isArray(value)) return value.join(', ')
  return String(value).trim() || '—'
}

function AnswerDisplay({ content, isHtml }: { content: string; isHtml: boolean }) {
  if (!content || content === '—') return <span className="text-muted">No answer</span>
  if (isHtml) {
    return (
      <div
        className="body-base"
        dangerouslySetInnerHTML={{ __html: sanitizeHtmlForDisplay(content) }}
        style={{ lineHeight: 1.5, marginTop: 'var(--space-sm)' }}
      />
    )
  }
  return (
    <p className="body-base" style={{ lineHeight: 1.5, marginTop: 'var(--space-sm)', whiteSpace: 'pre-wrap' }}>
      <LatexText text={content} />
    </p>
  )
}

export function AssessmentResultView() {
  const { id: assessmentId, submissionId } = useParams<{ id: string; submissionId: string }>()

  const { data: submission, error: subError, isLoading: submissionLoading, mutate: mutateSubmission } = useSWR<Submission>(
    assessmentId && submissionId
      ? ['submission', assessmentId, submissionId]
      : null,
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
      <div className="loading-state p-content">
        <Spinner size="md" />
        <p className="text-muted">Loading result…</p>
      </div>
    )
  }

  if (subError || !submission) {
    return (
      <div className="content-max">
        <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{subError?.message ?? 'Result not found'}</p>
        <Link to="/assessments/create">
          <Button type="button" variant="ghost">Back to create</Button>
        </Link>
      </div>
    )
  }

  const scorePct = submission.adjusted_total_score ?? submission.score_pct ?? 0
  const rawScorePct = submission.score_pct ?? 0
  const questions = assessment?.questions ?? []
  const questionResults = submission.question_results ?? {}
  const maxScore = 100
  const totalTimeSpent = submission.total_time_spent_seconds
  const questionTimes = submission.question_times ?? {}
  const tutorFeedback = submission.tutor_feedback
  const integrityNote = submission.integrity_note

  function formatTime(sec: number): string {
    const m = Math.floor(sec / 60)
    const s = sec % 60
    return s > 0 ? `${m}m ${s}s` : `${m}m`
  }

  return (
    <div className="content-max">
      <h1 className="heading-2xl mb-md">Assessment result</h1>
      {assessment && (
        <p className="body-sm text-muted mb-xl">{assessment.title}</p>
      )}

      <Card className="mb-xl">
        <div className="section-label mb-md">Total score</div>
        <p className="hero-score">
          <span style={{ display: 'inline-block' }}>{scorePct}%</span>{' '}
          <span className="heading-lg text-muted">/ {maxScore}%</span>
          {submission.adjusted_total_score != null && submission.adjusted_total_score !== rawScorePct && (
            <span className="body-sm text-muted ml-md">(confidence-weighted)</span>
          )}
        </p>
        {questions.length > 0 && (
          <p className="body-sm text-muted mt-sm" style={{ marginBottom: 0 }}>{questions.length} question{questions.length !== 1 ? 's' : ''}</p>
        )}
        {totalTimeSpent != null && totalTimeSpent > 0 && (
          <p className="body-sm text-muted mt-sm" style={{ marginBottom: 0 }}>Total time: {formatTime(totalTimeSpent)}</p>
        )}
      </Card>

      {submission.ai_status === 'failed' && (
        <p className="body-sm mb-md" style={{ color: 'var(--color-danger)' }}>AI evaluation could not be completed for some answers. Your score may be partial.</p>
      )}

      {reEvaluateError && (
        <p className="body-sm mb-md" style={{ color: 'var(--color-danger)' }}>{reEvaluateError}</p>
      )}

      {integrityNote && (
        <p className="body-sm text-muted mb-md" style={{ fontStyle: 'italic' }}>{integrityNote}</p>
      )}

      <div className="gap-md mb-md" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center' }}>
        {submission.grading_model_version && (
          <span className="text-xs text-muted">Grading model: {submission.grading_model_version}</span>
        )}
        {submission.re_evaluation_used && (
          <span className="badge" style={{ backgroundColor: 'var(--color-accent-soft)', color: 'var(--color-accent)', border: 'none' }}>Re-evaluation used</span>
        )}
      </div>

      {questions.length > 0 && (
        <section className="mb-xl">
          <h2 className="heading-lg mb-lg">Review your answers</h2>
          <div className="grid-cards" style={{ display: 'flex', flexDirection: 'column' }}>
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
                  <div className="card-actions mb-md">
                    <span className="section-label" style={{ marginBottom: 0 }}>Question {idx + 1}</span>
                    {questionTimes[q.id] != null && questionTimes[q.id] > 0 && (
                      <span className="text-xs text-muted">Time: {formatTime(questionTimes[q.id])}</span>
                    )}
                  </div>
                  <div className="body-base" style={{ fontWeight: 600, marginBottom: 'var(--space-md)', lineHeight: 1.4 }}>
                    <LatexText text={q.prompt} />
                  </div>
                  <div className="mb-md">
                    <div className="text-xs text-muted mb-xs" style={{ fontWeight: 600 }}>Your answer</div>
                    <AnswerDisplay content={displayAnswer} isHtml={isHtml} />
                  </div>
                  {score != null && (
                    <div className={justification ? 'mb-sm' : ''}>
                      <span className={`score-badge ${isCorrect ? 'score-badge--success' : 'score-badge--danger'}`}>{score}%</span>
                      {feedback && (
                        <span className="body-sm text-primary ml-md"><LatexText text={feedback} /></span>
                      )}
                    </div>
                  )}
                  {justification && (
                    <p className="body-sm text-muted" style={{ margin: 0, lineHeight: 1.5 }}><LatexText text={justification} /></p>
                  )}
                  {Array.isArray(strengths) && strengths.length > 0 && (
                    <div className="mt-md">
                      <div className="text-xs text-muted mb-xs" style={{ fontWeight: 600 }}>Strengths</div>
                      <ul className="body-sm" style={{ margin: 0, paddingLeft: 'var(--space-xl)' }}>
                        {strengths.map((s, i) => (
                          <li key={i}><LatexText text={s} /></li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {Array.isArray(missingConcepts) && missingConcepts.length > 0 && (
                    <div className="mt-md">
                      <div className="text-xs text-muted mb-xs" style={{ fontWeight: 600 }}>Missing concepts</div>
                      <ul className="body-sm" style={{ margin: 0, paddingLeft: 'var(--space-xl)' }}>
                        {missingConcepts.map((c, i) => (
                          <li key={i}><LatexText text={c} /></li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {confidence != null && (
                    <p className="text-xs text-muted mt-sm" style={{ margin: 0 }}>Confidence: {Math.round(confidence * 100)}%</p>
                  )}
                </Card>
              )
            })}
          </div>
        </section>
      )}

      <Card className="mb-xl">
        <h2 className="heading-lg mb-md">Performance Insights</h2>
        {tutorFeedback?.per_question && tutorFeedback.per_question.length > 0 ? (
          <div className="form-card">
            {tutorFeedback.summary_encouragement && (
              <p className="body-base text-primary" style={{ lineHeight: 1.5, margin: 0 }}>{tutorFeedback.summary_encouragement}</p>
            )}
            {tutorFeedback.per_question.map(({ question_id, feedback }: { question_id: string; feedback: TutorFeedbackItem }) => (
              <div key={question_id} className="insight-block">
                {feedback.conceptual_gaps?.length > 0 && (
                  <div className="mb-md">
                    <div className="text-xs text-muted mb-xs" style={{ fontWeight: 600 }}>Conceptual gaps</div>
                    <ul className="body-sm" style={{ margin: 0, paddingLeft: 'var(--space-xl)' }}>
                      {feedback.conceptual_gaps.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {feedback.misunderstood_topics?.length > 0 && (
                  <div className="mb-md">
                    <div className="text-xs text-muted mb-xs" style={{ fontWeight: 600 }}>Topics to revisit</div>
                    <ul className="body-sm" style={{ margin: 0, paddingLeft: 'var(--space-xl)' }}>
                      {feedback.misunderstood_topics.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {feedback.targeted_revision_plan?.length > 0 && (
                  <div className="mb-md">
                    <div className="text-xs text-muted mb-xs" style={{ fontWeight: 600 }}>Revision plan</div>
                    <ul className="body-sm" style={{ margin: 0, paddingLeft: 'var(--space-xl)' }}>
                      {feedback.targeted_revision_plan.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {feedback.recommended_practice_type && (
                  <p className="body-sm mb-sm" style={{ marginTop: 0 }}><strong>Practice:</strong> {feedback.recommended_practice_type}</p>
                )}
                {feedback.encouragement && (
                  <p className="body-sm text-muted" style={{ fontStyle: 'italic', margin: 0 }}>{feedback.encouragement}</p>
                )}
                {feedback.difficulty_adjustment_advice && (
                  <p className="body-sm mt-sm" style={{ marginBottom: 0 }}>{feedback.difficulty_adjustment_advice}</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <>
            <p className="body-base text-muted mb-md" style={{ lineHeight: 1.5 }}>Get AI-generated feedback on your answers: conceptual gaps, revision plan, and encouragement (without revealing correct answers).</p>
            {tutorError && (
              <p className="body-sm mb-md" style={{ color: 'var(--color-danger)' }}>{tutorError}</p>
            )}
            <Button type="button" variant="ghost" onClick={() => void handleGenerateTutorFeedback()} disabled={tutorLoading}>
              {tutorLoading ? 'Generating insights…' : 'Generate performance insights'}
            </Button>
          </>
        )}
      </Card>

      <Card className="mb-xl">
        <h2 className="heading-lg mb-md">How to improve</h2>
        <p className="body-base text-muted mb-lg" style={{ lineHeight: 1.5 }}>Review your sources, add notes from the material, and use the chat to ask follow-up questions on topics you missed.</p>
        <div className="gap-md" style={{ display: 'flex', flexWrap: 'wrap' }}>
          <Link to="/"><Button type="button" variant="ghost">Open workspace (chat)</Button></Link>
          <Link to="/" state={{ openPanel: 'notes' }}><Button type="button" variant="ghost">Notes</Button></Link>
          <Link to="/" state={{ openPanel: 'sources' }}><Button type="button" variant="ghost">Sources</Button></Link>
          <Link to="/" state={{ openPanel: 'flashcards' }}><Button type="button" variant="ghost">Flashcards</Button></Link>
        </div>
      </Card>

      <div className="gap-md" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center' }}>
        <Link to="/assessments/create">
          <Button type="button">Create another assessment</Button>
        </Link>
        {assessmentId && (
          <>
            <Link to={`/assessments/${assessmentId}/submissions`}><Button type="button" variant="ghost">Submissions</Button></Link>
            <Link to={`/assessments/${assessmentId}/take`}><Button type="button" variant="ghost">Retake</Button></Link>
          </>
        )}
        {assessmentId && submissionId && !submission.re_evaluation_used && (
          <Button type="button" variant="ghost" onClick={() => void handleReEvaluate()} disabled={reEvaluating} style={{ marginLeft: 'auto' }}>
            {reEvaluating ? 'Re-evaluating…' : 'Re-evaluate (once)'}
          </Button>
        )}
      </div>
    </div>
  )
}
