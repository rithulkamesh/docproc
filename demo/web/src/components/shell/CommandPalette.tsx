import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Command } from 'cmdk'
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog'
import { useWorkspace } from '@/context/WorkspaceContext'
import {
  MessageSquare,
  FileText,
  Layers,
  ClipboardList,
  FolderOpen,
  Settings,
  Focus,
  Upload,
} from 'lucide-react'

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()
  const {
    setCanvasMode,
    setFocusMode,
    focusMode,
    projects,
    setCurrentProjectId,
  } = useWorkspace()

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      }
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const run = (fn: () => void) => {
    fn()
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent showClose={false} className="max-w-xl gap-0 overflow-hidden p-0">
        <DialogTitle className="sr-only">Command palette</DialogTitle>
        <Command className="rounded-lg border-0 shadow-none" shouldFilter={true}>
          <div className="flex items-center border-b border-border px-3">
            <span className="text-muted-foreground">⌘K</span>
            <Command.Input
              placeholder="Search commands…"
              className="flex h-12 w-full rounded-none border-0 bg-transparent px-3 text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>
          <Command.List className="max-h-[min(50vh,20rem)] overflow-y-auto p-2">
            <Command.Group heading="Canvas" className="mb-2">
              <Command.Item
                value="converse chat"
                onSelect={() => run(() => setCanvasMode('converse'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <MessageSquare className="h-4 w-4" />
                Switch to Converse
              </Command.Item>
              <Command.Item
                value="notes"
                onSelect={() => run(() => setCanvasMode('notes'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <FileText className="h-4 w-4" />
                Switch to Notes
              </Command.Item>
              <Command.Item
                value="flashcards"
                onSelect={() => run(() => setCanvasMode('flashcards'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <Layers className="h-4 w-4" />
                Switch to Flashcards
              </Command.Item>
              <Command.Item
                value="tests"
                onSelect={() => run(() => setCanvasMode('tests'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <ClipboardList className="h-4 w-4" />
                Switch to Tests
              </Command.Item>
              <Command.Item
                value="sources documents"
                onSelect={() => run(() => setCanvasMode('sources'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <FolderOpen className="h-4 w-4" />
                Switch to Sources
              </Command.Item>
            </Command.Group>
            <Command.Group heading="Actions" className="mb-2">
              <Command.Item
                value="focus mode"
                onSelect={() => run(() => setFocusMode(!focusMode))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <Focus className="h-4 w-4" />
                {focusMode ? 'Exit Focus Mode' : 'Enter Focus Mode'}
              </Command.Item>
              <Command.Item
                value="add document upload"
                onSelect={() => run(() => setCanvasMode('sources'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <Upload className="h-4 w-4" />
                Add document
              </Command.Item>
              <Command.Item
                value="settings"
                onSelect={() => run(() => navigate('/settings'))}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
              >
                <Settings className="h-4 w-4" />
                Open Settings
              </Command.Item>
            </Command.Group>
            {projects.length > 1 && (
              <Command.Group heading="Projects">
                {projects.map((p) => (
                  <Command.Item
                    key={p.id}
                    value={`switch project ${p.name}`}
                    onSelect={() => run(() => setCurrentProjectId(p.id))}
                    className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 aria-selected:bg-accent"
                  >
                    Switch to {p.name}
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>
        </Command>
      </DialogContent>
    </Dialog>
  )
}
