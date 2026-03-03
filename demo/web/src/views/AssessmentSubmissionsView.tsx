import { useParams, Link } from 'react-router-dom'
import useSWR from 'swr'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { listSubmissions, getAssessment } from '@/api/assessments'
import type { Submission, Assessment } from '@/api/assessments'

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
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-sm text-destructive">Missing assessment id</p>
        <Button variant="ghost" asChild>
          <Link to="/assessments">Back to assessments</Link>
        </Button>
      </div>
    )
  }

  if (assessmentError) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-sm text-destructive">{assessmentError.message}</p>
        <Button variant="ghost" asChild>
          <Link to="/assessments">Back to assessments</Link>
        </Button>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="mx-auto max-w-2xl space-y-4 p-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-24 w-full" />
      </div>
    )
  }

  if (submissionsError) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-sm text-destructive">{submissionsError.message}</p>
        <Button variant="ghost" asChild>
          <Link to="/assessments">Back to assessments</Link>
        </Button>
      </div>
    )
  }

  const list = submissions ?? []

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <Button variant="ghost" size="sm" asChild>
        <Link to="/assessments">← Assessments</Link>
      </Button>
      <h1 className="text-2xl font-semibold tracking-tight">{assessment?.title ?? 'Submissions'}</h1>

      {list.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No submissions yet.
          </CardContent>
        </Card>
      ) : (
        <ul className="space-y-2">
          {list.map((s) => (
            <li key={s.id}>
              <Card>
                <CardContent className="flex flex-wrap items-center justify-between gap-4 py-4">
                  <div>
                    <p className="text-sm font-medium">
                      {s.score_pct != null ? `${s.score_pct}%` : '—'} · {formatDate(s.created_at)}
                    </p>
                    <p className="text-xs text-muted-foreground">Status: {s.status}</p>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link to={`/assessments/${assessmentId}/result/${s.id}`}>View result</Link>
                  </Button>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
