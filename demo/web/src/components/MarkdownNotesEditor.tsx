import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useEditor, EditorContent } from '@tiptap/react'
import { Extension } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import Placeholder from '@tiptap/extension-placeholder'
import { Markdown } from 'tiptap-markdown'
import type { Editor } from '@tiptap/core'
import { Bold, Italic, List, ListOrdered, Code, Heading1, Heading2, Heading3, Sigma } from 'lucide-react'
import { InlineMath, BlockMath } from './editor/mathExtensions'
import { notesMathConversionPlugin } from './editor/notesMathPlugin'
import {
  EquationEditorContext,
  type EquationEditorContextValue,
} from './EquationEditorContext'
import { EquationEditorModal } from './EquationEditorModal'
import 'katex/dist/katex.min.css'

const ICON_SIZE = 16

const NotesMathPluginExtension = Extension.create({
  name: 'notesMathPlugin',
  addProseMirrorPlugins() {
    return [notesMathConversionPlugin(this.editor.schema)]
  },
})

function ToolbarButton({
  active,
  onClick,
  children,
  title,
}: {
  active?: boolean
  onClick: () => void
  children: React.ReactNode
  title: string
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      style={{
        padding: '4px 6px',
        border: 'none',
        borderRadius: 'var(--radius-sm)',
        background: active ? 'var(--color-bg-hover)' : 'transparent',
        color: 'var(--color-text)',
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}

function Toolbar({ editor }: { editor: Editor | null }) {
  if (!editor) return null
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        padding: 'var(--space-xs) 0',
        borderBottom: '1px solid var(--color-border-light)',
        marginBottom: 'var(--space-sm)',
        flexWrap: 'wrap',
      }}
    >
      <ToolbarButton
        title="Heading 1"
        active={editor.isActive('heading', { level: 1 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
      >
        <Heading1 size={ICON_SIZE} strokeWidth={2} />
      </ToolbarButton>
      <ToolbarButton
        title="Heading 2"
        active={editor.isActive('heading', { level: 2 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
      >
        <Heading2 size={ICON_SIZE} strokeWidth={2} />
      </ToolbarButton>
      <ToolbarButton
        title="Heading 3"
        active={editor.isActive('heading', { level: 3 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
      >
        <Heading3 size={ICON_SIZE} strokeWidth={2} />
      </ToolbarButton>
      <span style={{ width: 1, height: 16, background: 'var(--color-border-light)', margin: '0 4px' }} />
      <ToolbarButton
        title="Bold"
        active={editor.isActive('bold')}
        onClick={() => editor.chain().focus().toggleBold().run()}
      >
        <Bold size={ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Italic"
        active={editor.isActive('italic')}
        onClick={() => editor.chain().focus().toggleItalic().run()}
      >
        <Italic size={ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Bullet list"
        active={editor.isActive('bulletList')}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
      >
        <List size={ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Numbered list"
        active={editor.isActive('orderedList')}
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
      >
        <ListOrdered size={ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Code block"
        active={editor.isActive('codeBlock')}
        onClick={() => editor.chain().focus().toggleCodeBlock().run()}
      >
        <Code size={ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <span style={{ width: 1, height: 16, background: 'var(--color-border-light)', margin: '0 4px' }} />
      <ToolbarButton
        title="Inline math ($...$)"
        onClick={() => editor.chain().focus().insertContent(' $ ').run()}
      >
        <Sigma size={ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
    </div>
  )
}

export interface MarkdownNotesEditorProps {
  value: string
  onChange: (markdown: string) => void
  placeholder?: string
  readOnly?: boolean
  debounceMs?: number
  showToolbar?: boolean
  className?: string
  contentClassName?: string
  style?: React.CSSProperties
  onEditorReady?: (editor: Editor) => void
}

export function MarkdownNotesEditor({
  value,
  onChange,
  placeholder = 'Write your note… (Markdown and LaTeX $...$ / $$...$$ supported)',
  readOnly = false,
  debounceMs = 500,
  showToolbar = true,
  className,
  contentClassName,
  style,
  onEditorReady,
}: MarkdownNotesEditorProps) {
  const debounceRef = useRef<number | null>(null)
  const lastValueRef = useRef(value)

  const flushOnChange = useCallback(
    (editor: Editor) => {
      try {
        const md = (editor.storage.markdown as { getMarkdown?: () => string } | undefined)?.getMarkdown?.()
        if (typeof md === 'string' && md !== lastValueRef.current) {
          lastValueRef.current = md
          onChange(md)
        }
      } catch {}
    },
    [onChange]
  )

  const [equationModalOpen, setEquationModalOpen] = useState(false)
  const [equationModalState, setEquationModalState] = useState<{
    initialLatex: string
    onSave?: (latex: string) => void
    onInsert?: (latex: string, type: 'inline' | 'block') => void
  }>({ initialLatex: '' })

  const openEquationModal = useCallback(
    (opts: {
      initialLatex?: string
      onSave?: (latex: string) => void
      onInsert?: (latex: string, type: 'inline' | 'block') => void
    }) => {
      setEquationModalState({
        initialLatex: opts.initialLatex ?? '',
        onSave: opts.onSave,
        onInsert: opts.onInsert,
      })
      setEquationModalOpen(true)
    },
    []
  )

  const equationContextValue = useMemo<EquationEditorContextValue>(
    () => ({ openEquationModal }),
    [openEquationModal]
  )

  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: { levels: [1, 2, 3] },
      }),
      Placeholder.configure({ placeholder }),
      InlineMath,
      BlockMath,
      NotesMathPluginExtension,
      Markdown.configure({ transformPastedText: true, transformCopiedText: true }),
    ],
    content: value || '',
    editable: !readOnly,
    editorProps: {
      attributes: {
        class: ['markdown-notes-editor-content', contentClassName].filter(Boolean).join(' '),
        style: 'min-height: 120px; outline: none; font-family: var(--font-family); font-size: var(--text-base); line-height: 1.6;',
      },
    },
    onUpdate: ({ editor: ed }) => {
      if (debounceRef.current !== null) window.clearTimeout(debounceRef.current)
      debounceRef.current = window.setTimeout(() => flushOnChange(ed), debounceMs)
    },
  })

  useEffect(() => {
    if (editor && onEditorReady) onEditorReady(editor)
  }, [editor, onEditorReady])

  // Conversion plugin (notesMath) only runs in appendTransaction, which is not
  // invoked when the editor is created with initial content. Dispatch a no-op
  // after mount so appendTransaction runs and raw $$...$$ become blockMath/inlineMath.
  useEffect(() => {
    if (!editor) return
    const id = requestAnimationFrame(() => {
      if (editor.view) {
        editor.view.dispatch(editor.state.tr)
      }
    })
    return () => cancelAnimationFrame(id)
  }, [editor])

  // Sync when value prop changes from parent (e.g. after load from API)
  useEffect(() => {
    if (!editor) return
    const currentMd = (editor.storage.markdown as { getMarkdown?: () => string } | undefined)?.getMarkdown?.()
    const normalized = (value || '').trim()
    if (normalized !== (currentMd ?? '').trim()) {
      lastValueRef.current = value
      editor.commands.setContent(value || '', false)
    }
  }, [value, editor])

  useEffect(() => {
    if (editor) editor.setEditable(!readOnly)
  }, [readOnly, editor])

  useEffect(() => {
    return () => {
      if (debounceRef.current !== null) window.clearTimeout(debounceRef.current)
    }
  }, [])

  if (!editor) return null

  return (
    <EquationEditorContext.Provider value={equationContextValue}>
      <div
        className={className}
        style={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: 120,
          ...style,
        }}
      >
        {showToolbar && !readOnly && <Toolbar editor={editor} />}
        <EditorContent editor={editor} />
      </div>
      <EquationEditorModal
        open={equationModalOpen}
        onClose={() => setEquationModalOpen(false)}
        initialLatex={equationModalState.initialLatex}
        onSave={equationModalState.onSave}
        onInsert={equationModalState.onInsert}
      />
    </EquationEditorContext.Provider>
  )
}
