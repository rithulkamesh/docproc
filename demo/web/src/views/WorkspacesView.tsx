import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { ConfirmModal } from '@/components/ConfirmModal'
import { NewProjectModal } from '@/components/NewProjectModal'
import { useWorkspace } from '@/context/WorkspaceContext'
import { deleteProject, type Project } from '@/api/projects'
import { fireConfetti } from '@/lib/confetti'
import { Plus, Trash2, Loader2, FolderOpen } from 'lucide-react'

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleDateString(undefined, { dateStyle: 'short' })
  } catch {
    return iso
  }
}

export function WorkspacesView() {
  const navigate = useNavigate()
  const { projects, loadProjects, currentProjectId, setCurrentProjectId } = useWorkspace()
  const [loading, setLoading] = useState(true)
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [newProjectOpen, setNewProjectOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    loadProjects().finally(() => {
      if (!cancelled) setLoading(false)
    })
    return () => { cancelled = true }
  }, [loadProjects])

  const handleConfirmDelete = async () => {
    if (!projectToDelete) return
    setDeletingId(projectToDelete.id)
    try {
      await deleteProject(projectToDelete.id)
      await loadProjects()
      setProjectToDelete(null)
      navigate('/', { replace: true })
    } finally {
      setDeletingId(null)
    }
  }

  const handleNewProjectCreated = async (projectId: string, projectName: string) => {
    setCurrentProjectId(projectId)
    setNewProjectOpen(false)
    fireConfetti()
    toast.success(`Created "${projectName}"`)
    await loadProjects() // so RequireProject sees projects.length >= 1 and doesn't redirect back to /workspaces
    navigate('/', { replace: true, state: { justCreatedProject: true, projectName } })
  }

  const handleSwitchTo = (project: Project) => {
    setCurrentProjectId(project.id)
    toast.success(`Switched to ${project.name}`)
    navigate('/', { replace: true })
  }

  if (loading) {
    return (
      <div className="mx-auto flex max-w-2xl items-center justify-center p-12">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <ConfirmModal
        open={projectToDelete !== null}
        onOpenChange={(open) => !open && setProjectToDelete(null)}
        title="Delete workspace"
        description={
          projectToDelete
            ? `"${projectToDelete.name}" will be permanently deleted, including all documents, notes, and assessments. This cannot be undone.`
            : ''
        }
        confirmLabel="Delete"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={handleConfirmDelete}
        loading={deletingId !== null}
      />

      <NewProjectModal
        open={newProjectOpen}
        onOpenChange={setNewProjectOpen}
        onCreated={handleNewProjectCreated}
      />

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Workspaces</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Switch between workspaces or create and delete them.
          </p>
        </div>
        <Button onClick={() => setNewProjectOpen(true)} className="shrink-0">
          <Plus className="mr-2 h-4 w-4" />
          New workspace
        </Button>
      </div>

      {projects.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center gap-5 py-14">
            <p className="text-center text-sm text-muted-foreground">
              Create your first workspace to start adding documents, chatting, and building study material.
            </p>
            <Button onClick={() => setNewProjectOpen(true)} size="lg">
              <Plus className="mr-2 h-4 w-4" />
              Create your first workspace
            </Button>
          </CardContent>
        </Card>
      ) : (
        <ul className="space-y-3">
          {projects.map((project) => (
            <li key={project.id}>
              <Card>
                <CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
                  <button
                    type="button"
                    onClick={() => handleSwitchTo(project)}
                    className="min-w-0 flex-1 cursor-pointer rounded-md text-left transition-colors hover:bg-muted/50 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 -m-2 p-2"
                  >
                    <p className="font-medium truncate">{project.name}</p>
                    <p className="text-xs text-muted-foreground">
                      Created {formatDate(project.created_at)}
                      {project.id === currentProjectId && (
                        <span className="ml-2 font-medium text-foreground">· Current</span>
                      )}
                    </p>
                  </button>
                  <div className="flex shrink-0 items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleSwitchTo(project)}
                      disabled={project.id === currentProjectId}
                    >
                      <FolderOpen className="mr-2 h-4 w-4" />
                      {project.id === currentProjectId ? 'Current' : 'Switch to'}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                      onClick={() => setProjectToDelete(project)}
                      disabled={deletingId !== null}
                      title="Delete workspace"
                    >
                      {deletingId === project.id ? (
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

      <Button variant="ghost" asChild>
        <Link to="/">← Back to workspace</Link>
      </Button>
    </div>
  )
}
