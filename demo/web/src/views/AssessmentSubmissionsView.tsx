import { useParams, Link } from 'react-router-dom'
import useSWR from 'swr'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { listSubmissions, getAssessment } from '../api/assessments'
import type { Submission, Assessment } from '../api/assessments'

function formatDate(iso: string | undefined): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { dateStyle: 'short' }) + ' ' + d.toLocaleTimeString(undefined, { timeStyle: 'short' })
  } catch {
    return iso
  }
}

export function AssessmentSubmissionsView() {
  const { id: assessmentId } = useParams<{ id: string }>()

  const { data: assessment, error: assessmentError } = useSWR<Assessment>(
    assessmentId ? ['assessment', assessmentId] : null,
    () => getAssessment(assessmentId!)
  )

  const { data: submissions, error: submissionsError, isLoading } = useSWR<Submission[]>(
    assessmentId ? ['submissions', assessmentId] : null,
    () => listSubmissions(assessmentId!)
  )

  if (!assessmentId) {
    return (
      <div className="content-max">
        <p className="body-sm" style={{ color: 'var(--color-danger)' }}>Missing assessment id</p>
        <Link to="/assessments">
          <Button type="button" variant="ghost">Back to assessments</Button>
        </Link>
      </div>
    )
  }

  if (assessmentError) {
    return (
      <div className="content-max">
        <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{assessmentError.message}</p>
        <Link to="/assessments">
          <Button type="button" variant="ghost">Back to assessments</Button>
        </Link>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="loading-state p-content">
        <Spinner size="md" />
        <p className="text-muted">Loading submissions…</p>
      </div>
    )
  }

  if (submissionsError) {
    return (
      <div className="content-max">
        <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{submissionsError.message}</p>
        <Link to="/assessments">
          <Button type="button" variant="ghost">Back to assessments</Button>
        </Link>
      </div>
    )
  }

  const list = submissions ?? []

  return (
    <div className="content-max">
      <div className="mb-xl">
        <Link to="/assessments" className="link-back body-sm">← Assessments</Link>
        <h1 className="heading-2xl mt-sm" style={{ marginBottom: 0 }}>{assessment?.title ?? 'Submissions'}</h1>
        <p className="body-sm text-muted mt-sm" style={{ margin: 0 }}>Past attempts and scores</p>
      </div>

      <div className="gap-md mb-lg" style={{ display: 'flex', flexWrap: 'wrap' }}>
        <Link to={`/assessments/${assessmentId}/take`}>
          <Button type="button">Take assessment</Button>
        </Link>
      </div>

      {list.length === 0 ? (
        <Card>
          <p className="text-muted" style={{ margin: 0 }}>No submissions yet. Take the assessment to see results here.</p>
        </Card>
      ) : (
        <ul className="list-reset grid-cards" style={{ display: 'flex', flexDirection: 'column' }}>
          {list.map((s) => (
            <li key={s.id}>
              <Card>
                <div className="card-actions" style={{ justifyContent: 'space-between' }}>
                  <div>
                    <p className="body-sm text-muted" style={{ margin: 0 }}>{formatDate(s.created_at)}</p>
                    <p className="body-base" style={{ fontWeight: 600, margin: 'var(--space-xs) 0 0 0' }}>Score: {s.score_pct != null ? `${s.score_pct}%` : '—'}</p>
                    {s.ai_status && (
                      <span className="text-xs text-muted mt-xs" style={{ display: 'inline-block' }}>
                        {s.ai_status === 'completed' ? 'Graded' : s.ai_status === 'pending_ai_evaluation' ? 'Grading…' : s.ai_status}
                        {s.re_evaluation_used && ' · Re-evaluated'}
                      </span>
                    )}
                  </div>
                  <Link to={`/assessments/${assessmentId}/result/${s.id}`}>
                    <Button type="button" variant="ghost">View result</Button>
                  </Link>
                </div>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
