import type { ReactNode } from 'react'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ContextPanelProps {
  collapsed: boolean
  onToggleCollapse: () => void
  children?: ReactNode
  title?: string
}

const EXPANDED_WIDTH = 'min(18rem, 20vw)'

function ContextPanelContent() {
  const {
    canvasMode,
    contextPanelSources,
    documents,
    selectedDocumentId,
  } = useWorkspace()

  const selectedDoc = documents.find((d) => d.id === selectedDocumentId)

  switch (canvasMode) {
    case 'converse':
      if (contextPanelSources && contextPanelSources.length > 0) {
        return (
          <ul className="space-y-2 text-sm">
            {contextPanelSources.map((s, idx) => (
              <li key={s.document_id ?? idx} className="rounded border border-border p-2">
                <p className="font-medium text-foreground">
                  {s.display_name ?? s.filename ?? 'Document'}
                </p>
                {s.content && (
                  <p className="mt-1 line-clamp-3 text-xs text-muted-foreground">
                    {s.content}
                  </p>
                )}
              </li>
            ))}
          </ul>
        )
      }
      return (
        <p className="text-xs text-muted-foreground">
          Sources from the latest message appear here.
        </p>
      )
    case 'notes':
      return (
        <p className="text-xs text-muted-foreground">
          {selectedDoc
            ? `From document: ${selectedDoc.display_name ?? selectedDoc.filename}`
            : 'Sections linked to your documents.'}
        </p>
      )
    case 'sources':
      if (selectedDoc) {
        return (
          <div className="space-y-2 text-sm">
            <p className="font-medium text-foreground">
              {selectedDoc.display_name ?? selectedDoc.filename}
            </p>
            <p className="text-xs text-muted-foreground">
              {selectedDoc.status === 'completed'
                ? `Ready · ${selectedDoc.pages ?? '?'} pages`
                : selectedDoc.status}
            </p>
          </div>
        )
      }
      return (
        <p className="text-xs text-muted-foreground">
          Select a document to see details.
        </p>
      )
    case 'home':
      return (
        <p className="text-xs text-muted-foreground">
          Recent or suggested document.
        </p>
      )
    case 'tests':
      return (
        <p className="text-xs text-muted-foreground">
          Assessment info and weak topics.
        </p>
      )
    default:
      return (
        <p className="text-xs text-muted-foreground">
          Context for this view appears here.
        </p>
      )
  }
}

const MODE_TITLES: Record<string, string> = {
  home: 'Recent',
  converse: 'From this message',
  notes: 'Linked documents',
  sources: 'Document',
  tests: 'Assessment',
}

export function ContextPanel({ collapsed, onToggleCollapse, children, title: titleProp }: ContextPanelProps) {
  const { canvasMode } = useWorkspace()
  const title = titleProp ?? MODE_TITLES[canvasMode] ?? 'Context'
  return (
    <aside
      className={cn(
        'flex shrink-0 flex-col border-l border-border/60 bg-muted/30 transition-[width] duration-200 ease-out',
        collapsed ? 'w-10' : `w-[${EXPANDED_WIDTH}]`
      )}
      style={{ width: collapsed ? '2.5rem' : EXPANDED_WIDTH }}
      aria-label="Context panel"
    >
      <div className="flex h-10 shrink-0 items-center justify-between gap-1 border-b border-border/60 px-2">
        {!collapsed && (
          <span className="min-w-0 truncate text-xs font-medium text-muted-foreground">
            {title}
          </span>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0 text-muted-foreground"
          onClick={onToggleCollapse}
          aria-label={collapsed ? 'Expand context panel' : 'Collapse context panel'}
        >
          {collapsed ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      </div>
      {!collapsed && (
        <div className="min-h-0 flex-1 overflow-y-auto p-3 text-sm">
          {children ?? <ContextPanelContent />}
        </div>
      )}
    </aside>
  )
}
