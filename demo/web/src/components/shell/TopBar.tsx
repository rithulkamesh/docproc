import { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useWorkspace } from '@/context/WorkspaceContext'
import { createProject } from '@/api/projects'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Sun, Moon, Maximize2, Minimize2, Settings, ChevronDown, Plus } from 'lucide-react'
import { cn } from '@/lib/utils'

export function TopBar() {
  const {
    projects,
    currentProject,
    currentProjectId,
    setCurrentProjectId,
    setCurrentProjectName,
    loadProjects,
    documents,
    lastIndexedLabel,
    themeMode,
    setThemeMode,
    focusMode,
    setFocusMode,
  } = useWorkspace()
  const [editingName, setEditingName] = useState(false)
  const [editValue, setEditValue] = useState(currentProject?.name ?? '')
  const inputRef = useRef<HTMLInputElement>(null)

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

  const handleNewProject = async () => {
    const name = window.prompt('New project name')
    if (!name?.trim()) return
    try {
      const created = await createProject({ name: name.trim() })
      await loadProjects()
      setCurrentProjectId(created.id)
    } catch {
      // ignore
    }
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
              className="truncate text-left text-lg font-semibold hover:text-muted-foreground"
            >
              {currentProject?.name ?? '—'}
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0" aria-label="Switch project">
                  <ChevronDown className="h-4 w-4 opacity-70" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-[60vh] min-w-[12rem] overflow-y-auto">
                {projects.map((p) => (
                  <DropdownMenuItem
                    key={p.id}
                    onClick={() => setCurrentProjectId(p.id)}
                    className={cn(currentProjectId === p.id && 'bg-accent')}
                  >
                    {p.name}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => void handleNewProject()}>
                  <Plus className="mr-2 h-4 w-4" />
                  New project
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>

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
        <Button
          variant="ghost"
          size="icon"
          aria-label={themeMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          onClick={() => setThemeMode(themeMode === 'light' ? 'dark' : 'light')}
        >
          {themeMode === 'light' ? (
            <Moon className="h-4 w-4" />
          ) : (
            <Sun className="h-4 w-4" />
          )}
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="rounded-full">
              <Avatar className="h-8 w-8">
                <AvatarFallback className="text-xs">U</AvatarFallback>
              </Avatar>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
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
