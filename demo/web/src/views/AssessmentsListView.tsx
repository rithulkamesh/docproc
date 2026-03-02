import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { listAssessments } from '../api/assessments'
import type { Assessment } from '../api/assessments'

function formatDate(iso: string | undefined): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { dateStyle: 'short' }) + ' ' + d.toLocaleTimeString(undefined, { timeStyle: 'short' })
  } catch {
    return iso
  }
}

export function AssessmentsListView() {
  const { data: assessments, error, isLoading } = useSWR<Assessment[]>(
    'assessments-list',
    () => listAssessments()
  )

  if (isLoading) {
    return (
      <div className="loading-state p-content">
        <Spinner size="md" />
        <p className="text-muted">Loading assessments…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="content-max">
        <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{error.message}</p>
        <Link to="/assessments/create">
          <Button type="button" variant="ghost">Create assessment</Button>
        </Link>
      </div>
    )
  }

  const list = assessments ?? []

  return (
    <div className="content-max">
      <div className="page-title-row">
        <h1 className="heading-2xl" style={{ margin: 0 }}>Assessments</h1>
        <Link to="/assessments/create">
          <Button type="button">Create assessment</Button>
        </Link>
      </div>

      {list.length === 0 ? (
        <Card>
          <p className="text-muted" style={{ margin: 0 }}>
            No assessments yet. Create one to get started.
          </p>
          <Link to="/assessments/create">
            <Button type="button" className="mt-md">Create assessment</Button>
          </Link>
        </Card>
      ) : (
        <ul className="list-reset grid-cards" style={{ display: 'flex', flexDirection: 'column' }}>
          {list.map((a) => (
            <li key={a.id}>
              <Card>
                <div className="card-actions" style={{ justifyContent: 'space-between' }}>
                  <div>
                    <h2 className="card-title">{a.title}</h2>
                    <p className="card-meta">Updated {formatDate(a.updated_at)}</p>
                  </div>
                  <div className="gap-md" style={{ display: 'flex', flexWrap: 'wrap' }}>
                    <Link to={`/assessments/${a.id}/take`}>
                      <Button type="button" variant="ghost">Take</Button>
                    </Link>
                    <Link to={`/assessments/${a.id}/submissions`}>
                      <Button type="button" variant="ghost">Submissions</Button>
                    </Link>
                  </div>
                </div>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
