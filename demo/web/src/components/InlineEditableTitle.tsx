import { useState, useRef, useEffect } from 'react'

interface InlineEditableTitleProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  level?: 1 | 2 | 3
  id?: string
}

export function InlineEditableTitle({
  value,
  onChange,
  placeholder = 'Untitled',
  level = 2,
  id,
}: InlineEditableTitleProps) {
  const [editing, setEditing] = useState(false)
  const [editValue, setEditValue] = useState(value)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setEditValue(value)
  }, [value])

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing])

  const save = () => {
    setEditing(false)
    const trimmed = editValue.trim()
    if (trimmed !== value) onChange(trimmed)
    else setEditValue(value)
  }

  const fontSize =
    level === 1 ? 'var(--text-xl)' : level === 2 ? 'var(--text-lg)' : 'var(--text-base)'
  const Tag = level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3'

  if (editing) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={editValue}
        onChange={(e) => setEditValue(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === 'Enter') save()
          if (e.key === 'Escape') {
            setEditValue(value)
            setEditing(false)
          }
        }}
        aria-label="Edit title"
        style={{
          width: '100%',
          fontFamily: 'var(--font-family)',
          fontSize,
          fontWeight: 600,
          lineHeight: 'var(--line-height-heading)',
          color: 'var(--color-text)',
          background: 'var(--color-bg-interactive)',
          border: 'none',
          borderRadius: 'var(--radius-sm)',
          padding: `${'var(--space-xs)'} ${'var(--space-sm)'}`,
          outline: 'none',
        }}
      />
    )
  }

  return (
    <Tag
      id={id}
      role="button"
      tabIndex={0}
      onClick={() => setEditing(true)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          setEditing(true)
        }
      }}
      style={{
        margin: 0,
        fontFamily: 'var(--font-family)',
        fontSize,
        fontWeight: 600,
        lineHeight: 'var(--line-height-heading)',
        color: 'var(--color-text)',
        cursor: 'text',
        borderRadius: 'var(--radius-sm)',
        padding: `${'var(--space-xs)'} ${'var(--space-sm)'}`,
        marginLeft: `-${'var(--space-sm)'}`,
        transition: 'background-color 120ms ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--color-bg-hover)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.backgroundColor = 'transparent'
      }}
    >
      {value.trim() || placeholder}
    </Tag>
  )
}
