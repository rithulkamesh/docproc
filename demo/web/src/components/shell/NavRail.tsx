import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useWorkspace } from '@/context/WorkspaceContext'
import type { CanvasMode } from '@/context/WorkspaceContext'
import {
  MessageSquare,
  FileText,
  ClipboardList,
  FolderOpen,
  Settings,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { motion as motionTokens } from '@/design/tokens'

const railItems: { mode: CanvasMode; label: string; icon: typeof MessageSquare }[] = [
  { mode: 'converse', label: 'Chat', icon: MessageSquare },
  { mode: 'notes', label: 'Notes', icon: FileText },
  { mode: 'tests', label: 'Tests', icon: ClipboardList },
  { mode: 'sources', label: 'Sources', icon: FolderOpen },
]

export function NavRail() {
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
      className="flex w-[clamp(4rem,5vw,5rem)] shrink-0 flex-col border-r border-border bg-card px-1 py-2"
      aria-label="Main navigation"
    >
      <div className="flex flex-col gap-1">
        {railItems.map(({ mode, label, icon: Icon }) => {
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
                'relative flex flex-col items-center gap-1 rounded-md py-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground',
                isActive && 'bg-secondary/80 text-foreground'
              )}
              aria-current={isActive ? 'page' : undefined}
            >
              {isActive && (
                <motion.span
                  layoutId="nav-rail-active-indicator"
                  className="absolute left-0 top-0 bottom-0 w-0.5 rounded-r bg-primary"
                  aria-hidden
                />
              )}
              <Icon className="h-5 w-5" />
              <span className="text-[10px] font-medium">{label}</span>
            </Link>
          )
        })}
      </div>
      <div className="mt-auto">
        <Link
          to="/settings"
          className="flex flex-col items-center gap-1 rounded-md py-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        >
          <Settings className="h-5 w-5" />
          <span className="text-[10px] font-medium">Settings</span>
        </Link>
      </div>
    </motion.nav>
  )
}
