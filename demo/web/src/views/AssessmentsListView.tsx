import { useState } from 'react'
import { Link } from 'react-router-dom'
import useSWR from 'swr'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { ConfirmModal } from '@/components/ConfirmModal'
import { useWorkspace } from '@/context/WorkspaceContext'
import { listAssessments, deleteAssessment } from '@/api/assessments'
import type { Assessment } from '@/api/assessments'
import { Plus, ClipboardList, Trash2, Loader2 } from 'lucide-react'

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
  const { currentProjectId } = useWorkspace()
  const { data: assessments, error, isLoading, mutate } = useSWR<Assessment[]>(
    ['assessments-list', currentProjectId],
    () => listAssessments(currentProjectId),
    { revalidateOnMount: true, revalidateOnFocus: true }
  )
  const [assessmentToDelete, setAssessmentToDelete] = useState<Assessment | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const handleConfirmDelete = async () => {
    if (!assessmentToDelete) return
    setDeletingId(assessmentToDelete.id)
    try {
      await deleteAssessment(assessmentToDelete.id)
      await mutate()
      setAssessmentToDelete(null)
    } finally {
      setDeletingId(null)
    }
  }

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
      <ConfirmModal
        open={assessmentToDelete !== null}
        onOpenChange={(open) => !open && setAssessmentToDelete(null)}
        title="Delete test"
        description={
          assessmentToDelete
            ? `"${assessmentToDelete.title}" will be permanently deleted, including all submissions. This cannot be undone.`
            : ''
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={handleConfirmDelete}
        loading={deletingId !== null}
      />
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Tests</h1>
        <Button asChild>
          <Link to="/assessments/create">
            <Plus className="mr-2 h-4 w-4" />
            Create test
          </Link>
        </Button>
      </div>

      {list.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center gap-4 py-12">
            <ClipboardList className="h-12 w-12 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              No tests yet. Create one from a document to get started.
            </p>
            <Button asChild>
              <Link to="/assessments/create">Create test</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <ul className="space-y-2">
          {list.map((a) => (
            <li key={a.id}>
              <Card>
                <CardContent className="flex flex-wrap items-center justify-between gap-4 py-4">
                  <div className="min-w-0 flex-1">
                    <h2 className="font-medium truncate" title={a.title}>{a.title}</h2>
                    <p className="text-xs text-muted-foreground">
                      Updated {formatDate(a.updated_at)}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="secondary" size="sm" asChild>
                      <Link to={`/assessments/${a.id}/take`}>Take</Link>
                    </Button>
                    <Button variant="ghost" size="sm" asChild>
                      <Link to={`/assessments/${a.id}/submissions`}>Submissions</Link>
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={(e) => { e.stopPropagation(); setAssessmentToDelete(a) }}
                      disabled={deletingId === a.id}
                      title="Delete test"
                    >
                      {deletingId === a.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
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
