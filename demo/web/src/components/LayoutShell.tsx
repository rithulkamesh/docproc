import type { ReactNode } from 'react'
import { ProjectHeaderBar } from './ProjectHeaderBar'

interface LayoutShellProps {
  children: ReactNode
}

export function LayoutShell({ children }: LayoutShellProps) {
  return (
    <div className="layout-shell">
      <ProjectHeaderBar />
      <div className="main-content">
        {children}
      </div>
    </div>
  )
}
