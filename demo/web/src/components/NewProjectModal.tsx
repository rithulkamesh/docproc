import { useState } from 'react'
import { createProject } from '@/api/projects'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

const PROJECT_DESCRIPTION_KEY = 'docproc-project-description'

export function getProjectDescription(projectId: string): string {
  if (typeof window === 'undefined') return ''
  try {
    return window.localStorage.getItem(`${PROJECT_DESCRIPTION_KEY}-${projectId}`) ?? ''
  } catch {
    return ''
  }
}

export function setProjectDescription(projectId: string, description: string): void {
  try {
    if (description.trim()) {
      window.localStorage.setItem(`${PROJECT_DESCRIPTION_KEY}-${projectId}`, description.trim())
    } else {
      window.localStorage.removeItem(`${PROJECT_DESCRIPTION_KEY}-${projectId}`)
    }
  } catch {}
}

interface NewProjectModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated: (projectId: string, projectName: string) => void
}

export function NewProjectModal({ open, onOpenChange, onCreated }: NewProjectModalProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reset = () => {
    setName('')
    setDescription('')
    setError(null)
  }

  const handleOpenChange = (next: boolean) => {
    if (!next) reset()
    onOpenChange(next)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) {
      setError('Project name is required')
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const created = await createProject({ name: trimmed })
      if (description.trim()) {
        setProjectDescription(created.id, description.trim())
      }
      onCreated(created.id, trimmed)
      handleOpenChange(false)
    } catch {
      setError('Failed to create project. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md" showClose={true}>
        <DialogHeader>
          <DialogTitle>New project</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="grid gap-2">
            <Label htmlFor="new-project-name">Project name</Label>
            <Input
              id="new-project-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Quantum Optics"
              autoFocus
              disabled={submitting}
              aria-required
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="new-project-description">Description (optional)</Label>
            <Input
              id="new-project-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this project"
              disabled={submitting}
            />
          </div>
          {error && (
            <p className="text-sm text-destructive" role="alert">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={submitting || !name.trim()}>
              {submitting ? 'Creating…' : 'Create project'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
