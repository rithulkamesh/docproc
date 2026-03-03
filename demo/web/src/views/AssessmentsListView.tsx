import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { listAssessments } from '@/api/assessments'
import type { Assessment } from '@/api/assessments'
import { Plus, ClipboardList } from 'lucide-react'

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
      <div className="mx-auto max-w-2xl space-y-4 p-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="mx-auto max-w-2xl p-6">
        <p className="text-sm text-destructive">{error.message}</p>
        <Button variant="ghost" asChild className="mt-2">
          <Link to="/assessments/create">Create assessment</Link>
        </Button>
      </div>
    )
  }

  const list = assessments ?? []

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Assessments</h1>
        <Button asChild>
          <Link to="/assessments/create">
            <Plus className="mr-2 h-4 w-4" />
            Create assessment
          </Link>
        </Button>
      </div>

      {list.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center gap-4 py-12">
            <ClipboardList className="h-12 w-12 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              No assessments yet. Create one to get started.
            </p>
            <Button asChild>
              <Link to="/assessments/create">Create assessment</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <ul className="space-y-2">
          {list.map((a) => (
            <li key={a.id}>
              <Card>
                <CardContent className="flex flex-wrap items-center justify-between gap-4 py-4">
                  <div>
                    <h2 className="font-medium">{a.title}</h2>
                    <p className="text-xs text-muted-foreground">
                      Updated {formatDate(a.updated_at)}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="ghost" size="sm" asChild>
                      <Link to={`/assessments/${a.id}/take`}>Take</Link>
                    </Button>
                    <Button variant="ghost" size="sm" asChild>
                      <Link to={`/assessments/${a.id}/submissions`}>Submissions</Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
