import type { ReactNode } from 'react'
import { useState } from 'react'
import { TopBar } from '@/components/shell/TopBar'
import { StudySidebar } from '@/components/shell/StudySidebar'
import { ContextPanel } from '@/components/context/ContextPanel'

interface StudyWorkspaceLayoutProps {
  children: ReactNode
}

export function StudyWorkspaceLayout({ children }: StudyWorkspaceLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [contextPanelCollapsed, setContextPanelCollapsed] = useState(true)

  return (
    <div className="flex h-screen flex-col bg-background">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <StudySidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((c) => !c)}
        />
        <main className="min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[880px] px-4 py-8">{children}</div>
        </main>
        <ContextPanel
          collapsed={contextPanelCollapsed}
          onToggleCollapse={() => setContextPanelCollapsed((c) => !c)}
        />
      </div>
    </div>
  )
}
