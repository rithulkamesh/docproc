import { useEditor, EditorContent, type Editor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'
import Placeholder from '@tiptap/extension-placeholder'
import Subscript from '@tiptap/extension-subscript'
import Superscript from '@tiptap/extension-superscript'
import { useEffect, useState, useCallback, useRef } from 'react'
import { theme } from '../design/theme'
import { sanitizeHtml } from '../lib/sanitize'
import { SlashCommandMenu, type SlashCommandItem } from './SlashCommandMenu'

const ToolbarButton = ({
  active,
  onClick,
  children,
  title,
}: {
  active?: boolean
  onClick: () => void
  children: React.ReactNode
  title: string
}) => (
  <button
    type="button"
    title={title}
    onClick={onClick}
    style={{
      padding: '6px 10px',
      border: '1px solid var(--color-border-light)',
      borderRadius: theme.radius.badge,
      background: active ? 'var(--color-accent-soft)' : 'var(--color-bg)',
      color: 'var(--color-text)',
      cursor: 'pointer',
      fontSize: 'var(--text-sm)',
    }}
  >
    {children}
  </button>
)

function Toolbar({ editor }: { editor: Editor | null }) {
  if (!editor) return null
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 'var(--space-sm)',
        padding: 'var(--space-sm)',
        borderBottom: '1px solid var(--color-border-light)',
        backgroundColor: 'var(--color-bg-alt)',
        borderTopLeftRadius: 'var(--radius-input)',
        borderTopRightRadius: 'var(--radius-input)',
      }}
    >
      <ToolbarButton
        title="Bold"
        active={editor.isActive('bold')}
        onClick={() => editor.chain().focus().toggleBold().run()}
      >
        <strong>B</strong>
      </ToolbarButton>
      <ToolbarButton
        title="Italic"
        active={editor.isActive('italic')}
        onClick={() => editor.chain().focus().toggleItalic().run()}
      >
        <em>I</em>
      </ToolbarButton>
      <ToolbarButton
        title="Underline"
        active={editor.isActive('underline')}
        onClick={() => editor.chain().focus().toggleUnderline().run()}
      >
        <u>U</u>
      </ToolbarButton>
      <ToolbarButton
        title="Bullet list"
        active={editor.isActive('bulletList')}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
      >
        • List
      </ToolbarButton>
      <ToolbarButton
        title="Numbered list"
        active={editor.isActive('orderedList')}
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
      >
        1. List
      </ToolbarButton>
      <ToolbarButton
        title="Code block"
        active={editor.isActive('codeBlock')}
        onClick={() => editor.chain().focus().toggleCodeBlock().run()}
      >
        {'</>'}
      </ToolbarButton>
      <ToolbarButton
        title="Inline code"
        active={editor.isActive('code')}
        onClick={() => editor.chain().focus().toggleCode().run()}
      >
        code
      </ToolbarButton>
      <ToolbarButton
        title="Superscript"
        active={editor.isActive('superscript')}
        onClick={() => editor.chain().focus().toggleSuperscript().run()}
      >
        x²
      </ToolbarButton>
      <ToolbarButton
        title="Subscript"
        active={editor.isActive('subscript')}
        onClick={() => editor.chain().focus().toggleSubscript().run()}
      >
        x₂
      </ToolbarButton>
      <ToolbarButton
        title="Insert equation (LaTeX: type $...$ or $$...$$)"
        onClick={() => editor.chain().focus().insertContent(' $ ').run()}
      >
        ∑
      </ToolbarButton>
    </div>
  )
}

const SLASH_ITEMS: SlashCommandItem[] = [
  { id: 'equation', label: 'Equation', description: 'Inline LaTeX $...$' },
  { id: 'blockEquation', label: 'Block equation', description: 'Display $$...$$' },
  { id: 'code', label: 'Code block', description: 'Fenced code' },
  { id: 'bulletList', label: 'Bullet list', description: '• List' },
  { id: 'orderedList', label: 'Numbered list', description: '1. List' },
]

interface RichTextEditorProps {
  value: string
  onChange: (html: string) => void
  placeholder?: string
  disabled?: boolean
  minHeight?: number
  /** Legacy: no longer used; single surface, no split preview. Use $...$ and $$...$$ in the editor. */
  showEquationPreview?: boolean
}

export function RichTextEditor({
  value,
  onChange,
  placeholder = 'Write your answer… Use $...$ for inline math, $$...$$ for block math. Type / for commands.',
  disabled = false,
  minHeight = 160,
}: RichTextEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [slashOpen, setSlashOpen] = useState(false)
  const [slashAnchor, setSlashAnchor] = useState<React.CSSProperties>({})
  const [slashSelected, setSlashSelected] = useState(0)

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ heading: false }),
      Underline,
      Placeholder.configure({ placeholder }),
      Subscript,
      Superscript,
    ],
    content: value || '',
    editable: !disabled,
    onUpdate: ({ editor }) => {
      const html = editor.getHTML()
      onChange(sanitizeHtml(html))
    },
    editorProps: {
      attributes: {
        style: `min-height: ${minHeight}px; padding: ${'var(--space-md)'}; font-family: ${'var(--font-family)'}; font-size: ${'var(--text-base)'};`,
      },
      handleKeyDown: (view, event) => {
        if (slashOpen) {
          if (event.key === 'Escape') {
            setSlashOpen(false)
            return true
          }
          if (event.key === 'ArrowDown') {
            setSlashSelected((i) => Math.min(i + 1, SLASH_ITEMS.length - 1))
            return true
          }
          if (event.key === 'ArrowUp') {
            setSlashSelected((i) => Math.max(i - 1, 0))
            return true
          }
          if (event.key === 'Enter') {
            event.preventDefault()
            return true
          }
          return false
        }
        if (event.key === '/') {
          event.preventDefault()
          const { from } = view.state.selection
          const coords = view.coordsAtPos(from)
          setSlashAnchor({
            position: 'absolute',
            left: coords.left,
            top: coords.bottom + 4,
            zIndex: 100,
          })
          setSlashOpen(true)
          setSlashSelected(0)
          return true
        }
        return false
      },
    },
  })

  const handleSlashSelect = useCallback(
    (id: string) => {
      if (!editor) return
      editor.chain().focus()
      switch (id) {
        case 'equation':
          editor.chain().focus().insertContent(' $ ').run()
          break
        case 'blockEquation':
          editor.chain().focus().insertContent('\n\n$$ \n\n').run()
          break
        case 'code':
          editor.chain().focus().toggleCodeBlock().run()
          break
        case 'bulletList':
          editor.chain().focus().toggleBulletList().run()
          break
        case 'orderedList':
          editor.chain().focus().toggleOrderedList().run()
          break
        default:
          break
      }
      setSlashOpen(false)
    },
    [editor]
  )

  useEffect(() => {
    if (!editor) return
    const current = editor.getHTML()
    const sanitized = sanitizeHtml(value || '')
    if (sanitized !== current) {
      editor.commands.setContent(sanitized, false)
    }
  }, [value, editor])

  useEffect(() => {
    if (editor) {
      editor.setEditable(!disabled)
    }
  }, [disabled, editor])

  useEffect(() => {
    if (!slashOpen) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        handleSlashSelect(SLASH_ITEMS[slashSelected]?.id ?? '')
      }
    }
    const onClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setSlashOpen(false)
    }
    window.addEventListener('keydown', onKeyDown, true)
    setTimeout(() => window.addEventListener('mousedown', onClickOutside), 0)
    return () => {
      window.removeEventListener('keydown', onKeyDown, true)
      window.removeEventListener('mousedown', onClickOutside)
    }
  }, [slashOpen, slashSelected, handleSlashSelect])

  return (
    <div
      ref={containerRef}
      style={{
        border: `${'1px'} solid ${'var(--color-border-strong)'}`,
        borderRadius: 'var(--radius-input)',
        boxShadow: 'var(--shadow-card)',
        backgroundColor: 'var(--color-bg)',
        color: 'var(--color-text)',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <Toolbar editor={editor} />
      <div style={{ minHeight: minHeight, position: 'relative' }}>
        <EditorContent editor={editor} />
      </div>
      <SlashCommandMenu
        open={slashOpen}
        items={SLASH_ITEMS}
        selectedIndex={slashSelected}
        onSelect={handleSlashSelect}
        anchorStyle={slashAnchor}
      />
    </div>
  )
}
