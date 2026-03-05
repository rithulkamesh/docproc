import { useState, useRef, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { useWorkspace } from '@/context/WorkspaceContext'
import { NewProjectModal } from '@/components/NewProjectModal'
import { fireConfetti } from '@/lib/confetti'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import {
  loadUserPreferences,
  dicebearAvatarUrl,
  initialsFromDisplayName,
  onUserPreferencesChange,
} from '@/lib/userPreferences'
import { Maximize2, Minimize2, Settings, ChevronDown, Plus, FolderOpen } from 'lucide-react'
import { cn } from '@/lib/utils'

export function TopBar() {
  const navigate = useNavigate()
  const {
    projects,
    currentProject,
    currentProjectId,
    setCurrentProjectId,
    setCurrentProjectName,
    loadProjects,
    documents,
    focusMode,
    setFocusMode,
  } = useWorkspace()
  const [editingName, setEditingName] = useState(false)
  const [newProjectOpen, setNewProjectOpen] = useState(false)
  const [editValue, setEditValue] = useState(currentProject?.name ?? '')
  const [userPrefs, setUserPrefs] = useState(loadUserPreferences)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    return onUserPreferencesChange(() => setUserPrefs(loadUserPreferences()))
  }, [])

  useEffect(() => {
    setEditValue(currentProject?.name ?? '')
  }, [currentProject?.name])

  useEffect(() => {
    if (editingName && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editingName])

  const handleSaveName = async () => {
    setEditingName(false)
    const trimmed = editValue.trim()
    if (trimmed && trimmed !== currentProject?.name) {
      await setCurrentProjectName(trimmed)
    } else {
      setEditValue(currentProject?.name ?? '')
    }
  }

  const handleNewProjectCreated = async (projectId: string, projectName: string) => {
    await loadProjects()
    setCurrentProjectId(projectId)
    setNewProjectOpen(false)
    fireConfetti()
    toast.success(`Created "${projectName}"`)
    navigate('/', { replace: true, state: { justCreatedProject: true, projectName } })
  }

  const processingCount = documents.filter((d) => d.status === 'processing').length

  return (
    <header
      className={cn(
        'flex h-[clamp(3.5rem,4vw,4.5rem)] shrink-0 items-center justify-between gap-4 border-b border-border bg-card px-[clamp(1rem,2vw,2rem)]',
        focusMode && 'h-10'
      )}
      aria-label="App header"
    >
      <div className="flex min-w-0 items-center gap-4">
        <Link
          to="/"
          className="shrink-0 text-sm font-medium tracking-wide text-muted-foreground hover:text-foreground"
        >
          docproc // edu
        </Link>
        <span className="text-muted-foreground">/</span>
        <span className="text-muted-foreground text-sm">Project:</span>
        {editingName ? (
          <Input
            ref={inputRef}
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={() => void handleSaveName()}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void handleSaveName()
              if (e.key === 'Escape') {
                setEditingName(false)
                setEditValue(currentProject?.name ?? '')
              }
            }}
            aria-label="Project name"
            className="h-8 max-w-[20ch] border-0 bg-transparent text-lg font-semibold shadow-none focus-visible:ring-0"
          />
        ) : (
          <div className="flex min-w-0 items-center gap-0.5">
            <button
              type="button"
              onClick={() => setEditingName(true)}
              className="truncate max-w-[20ch] text-left text-lg font-semibold hover:text-muted-foreground"
              title="Rename project"
            >
              {currentProject?.name ?? '—'}
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" aria-label="Switch project">
                  <ChevronDown className="h-4 w-4 opacity-70" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-[60vh] min-w-[14rem] overflow-y-auto">
                <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">Switch project</div>
                {projects.map((p) => (
                  <DropdownMenuItem
                    key={p.id}
                    onClick={() => {
                      setCurrentProjectId(p.id)
                      toast.success(`Switched to ${p.name}`)
                    }}
                    className={cn(currentProjectId === p.id && 'bg-accent')}
                  >
                    <FolderOpen className="mr-2 h-4 w-4 shrink-0 opacity-70" />
                    {p.name}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => setNewProjectOpen(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  New project
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link to="/workspaces">
                    <FolderOpen className="mr-2 h-4 w-4" />
                    Manage workspaces
                  </Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>
      <NewProjectModal
        open={newProjectOpen}
        onOpenChange={setNewProjectOpen}
        onCreated={handleNewProjectCreated}
      />

      <div className="flex shrink-0 items-center gap-2">
        <span className="hidden text-xs text-muted-foreground sm:inline">
          Docs: {documents.length}
          {processingCount > 0 && ` · ${processingCount} processing`}
        </span>
        <Button
          variant="ghost"
          size="icon"
          aria-label={focusMode ? 'Exit focus mode' : 'Focus mode'}
          onClick={() => setFocusMode(!focusMode)}
        >
          {focusMode ? (
            <Minimize2 className="h-4 w-4" />
          ) : (
            <Maximize2 className="h-4 w-4" />
          )}
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="rounded-full">
              <Avatar className="h-8 w-8">
                <AvatarImage
                  src={dicebearAvatarUrl(userPrefs.displayName || userPrefs.avatarSeed || 'user', userPrefs.avatarStyle)}
                  alt=""
                />
                <AvatarFallback className="text-xs">
                  {initialsFromDisplayName(userPrefs.displayName)}
                </AvatarFallback>
              </Avatar>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem asChild>
              <Link to="/workspaces">
                <FolderOpen className="mr-2 h-4 w-4" />
                Workspaces
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <Link to="/settings">
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
