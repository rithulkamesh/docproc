
export interface SlashCommandItem {
  id: string
  label: string
  description?: string
}

interface SlashCommandMenuProps {
  open: boolean
  items: SlashCommandItem[]
  selectedIndex: number
  onSelect: (id: string) => void
  anchorStyle?: React.CSSProperties
}

/** Minimal slash command menu: list of commands, keyboard select. */
export function SlashCommandMenu({
  open,
  items,
  selectedIndex,
  onSelect,
  anchorStyle,
}: SlashCommandMenuProps) {
  if (!open || items.length === 0) return null

  return (
    <div
      role="listbox"
      aria-label="Commands"
      style={{
        position: 'absolute',
        ...anchorStyle,
        minWidth: '12rem',
        maxWidth: '20rem',
        padding: 'var(--space-sm)',
        background: 'var(--color-bg-alt)',
        border: 'var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
        zIndex: 100,
      }}
    >
      {items.map((item, i) => (
        <button
          key={item.id}
          type="button"
          role="option"
          aria-selected={i === selectedIndex}
          style={{
            display: 'block',
            width: '100%',
            textAlign: 'left',
            padding: `${'var(--space-sm)'} ${'var(--space-sm)'}`,
            fontFamily: 'var(--font-family)',
            fontSize: 'var(--text-sm)',
            color: 'var(--color-text)',
            background: i === selectedIndex ? 'var(--color-bg-hover)' : 'transparent',
            border: 'none',
            borderRadius: 'var(--radius-sm)',
            cursor: 'pointer',
            transition: 'background-color 120ms ease',
          }}
          onClick={() => onSelect(item.id)}
        >
          {item.label}
          {item.description && (
            <span style={{ marginLeft: 'var(--space-sm)', color: 'var(--color-text-muted)' }}>
              {item.description}
            </span>
          )}
        </button>
      ))}
    </div>
  )
}
