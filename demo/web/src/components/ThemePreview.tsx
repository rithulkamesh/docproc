import { cn } from '@/lib/utils'
import type { ThemeId } from '@/lib/themeStorage'

interface ThemePreviewProps {
  themeId: ThemeId
  label: string
  selected?: boolean
  onClick?: () => void
  className?: string
}

/**
 * Renders a small palette preview by scoping the theme to this element.
 * Uses CSS variables from [data-theme] so the preview matches the actual theme.
 */
export function ThemePreview({ themeId, label, selected, onClick, className }: ThemePreviewProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      aria-label={`Theme: ${label}`}
      className={cn(
        'flex flex-col overflow-hidden rounded-xl border-2 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
        selected
          ? 'border-primary ring-2 ring-primary/20'
          : 'border-border hover:border-accent hover:bg-accent/30',
        className
      )}
    >
      <div
        data-theme={themeId}
        className="flex h-[72px] w-full flex-col rounded-t-lg bg-[hsl(var(--viewport))]"
      >
        <div className="h-2 flex-shrink-0 bg-[hsl(var(--viewport))]" />
        <div className="mx-1.5 mt-1 flex flex-1 flex-col rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-1.5 shadow-sm">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-4 rounded-sm bg-[hsl(var(--primary))]" />
            <span
              className="text-[10px] font-medium leading-tight truncate"
              style={{ color: 'hsl(var(--foreground))' }}
            >
              Aa
            </span>
          </div>
          <div className="mt-1 flex gap-0.5">
            <div className="h-1 w-2 rounded-[2px] bg-[hsl(var(--muted))]" />
            <div className="h-1 w-2 rounded-[2px] bg-[hsl(var(--accent))]" />
            <div className="h-1 w-3 rounded-[2px] bg-[hsl(var(--primary))]" />
          </div>
        </div>
      </div>
      <span className="px-2 py-1.5 text-xs font-medium text-foreground">{label}</span>
    </button>
  )
}
