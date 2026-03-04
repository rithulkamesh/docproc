import type { ReactNode } from 'react'
import { TopBar } from './TopBar'
import { StudyWorkspaceLayout } from '@/components/layout/StudyWorkspaceLayout'

interface AppShellProps {
  children: ReactNode
  /** When true, render without framed container (e.g. full-bleed settings or assessment pages). */
  fullBleed?: boolean
}

export function AppShell({ children, fullBleed = false }: AppShellProps) {
  if (fullBleed) {
    return (
      <div className="flex min-h-screen flex-col bg-background">
        <TopBar />
        <main className="flex-1 overflow-auto">{children}</main>
      </div>
    )
  }

  return <StudyWorkspaceLayout>{children}</StudyWorkspaceLayout>
}
