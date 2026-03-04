import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useWorkspace } from '@/context/WorkspaceContext'
import type { CanvasMode } from '@/context/WorkspaceContext'
import {
  Home,
  MessageSquare,
  FileText,
  ClipboardList,
  FolderOpen,
  PanelLeftClose,
  PanelLeft,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { motion as motionTokens } from '@/design/tokens'
import { Button } from '@/components/ui/button'

const sidebarItems: { mode: CanvasMode; label: string; icon: typeof Home }[] = [
  { mode: 'home', label: 'Home', icon: Home },
  { mode: 'converse', label: 'Chat', icon: MessageSquare },
  { mode: 'notes', label: 'Notes', icon: FileText },
  { mode: 'tests', label: 'Tests', icon: ClipboardList },
  { mode: 'sources', label: 'Sources', icon: FolderOpen },
]

interface StudySidebarProps {
  collapsed: boolean
  onToggleCollapse: () => void
}

export function StudySidebar({ collapsed, onToggleCollapse }: StudySidebarProps) {
  const { canvasMode, setCanvasMode, focusMode } = useWorkspace()

  if (focusMode) return null

  return (
    <motion.nav
      initial={{ x: -24, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{
        duration: motionTokens.durationPanel / 1000,
        ease: motionTokens.easingFramer,
      }}
      className={cn(
        'flex shrink-0 flex-col border-r border-border bg-card py-2 transition-[width] duration-200 ease-out',
        collapsed ? 'w-[4.5rem]' : 'w-48'
      )}
      aria-label="Study navigation"
    >
      <div className="flex flex-col gap-0.5">
        {sidebarItems.map(({ mode, label, icon: Icon }) => {
          const isActive = canvasMode === mode
          return (
            <Link
              key={mode}
              to="/"
              onClick={(e) => {
                e.preventDefault()
                setCanvasMode(mode)
              }}
              className={cn(
                'relative flex w-full min-w-0 items-center gap-3 rounded-none py-2 pr-2 pl-2 text-muted-foreground transition-colors duration-200 hover:bg-secondary hover:text-foreground',
                isActive && 'bg-secondary/80 text-foreground'
              )}
              aria-current={isActive ? 'page' : undefined}
              title={collapsed ? label : undefined}
            >
              {isActive && (
                <motion.span
                  layoutId="study-sidebar-active-indicator"
                  className="absolute left-0 top-1 bottom-1 w-0.5 rounded-r bg-primary"
                  aria-hidden
                />
              )}
              <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                <Icon className="h-5 w-5" />
              </span>
              {!collapsed && <span className="min-w-0 truncate text-sm font-medium">{label}</span>}
            </Link>
          )
        })}
      </div>
      <div className="mt-auto pt-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9"
          onClick={onToggleCollapse}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? (
            <PanelLeft className="h-4 w-4" />
          ) : (
            <PanelLeftClose className="h-4 w-4" />
          )}
        </Button>
      </div>
    </motion.nav>
  )
}
