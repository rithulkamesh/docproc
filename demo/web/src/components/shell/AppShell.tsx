import type { ReactNode } from 'react'
import { TopBar } from './TopBar'
import { NavRail } from './NavRail'

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

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <NavRail />
        <main className="min-w-0 flex-1 overflow-auto">
          <div className="mx-auto w-full max-w-[min(80ch,65vw)] px-4 py-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
