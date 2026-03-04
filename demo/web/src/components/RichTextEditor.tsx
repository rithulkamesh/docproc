import { useCallback, useMemo } from 'react'
import { useEditor, EditorContent, type Editor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'
import Placeholder from '@tiptap/extension-placeholder'
import Subscript from '@tiptap/extension-subscript'
import Superscript from '@tiptap/extension-superscript'
import { useEffect, useState, useRef } from 'react'
import {
  Bold,
  Italic,
  Underline as UnderlineIcon,
  List,
  ListOrdered,
  Code,
  CodeXml,
  Superscript as SuperscriptIcon,
  Subscript as SubscriptIcon,
  Sigma,
  Braces,
} from 'lucide-react'
import { sanitizeHtml } from '../lib/sanitize'
import { SlashCommandMenu, type SlashCommandItem } from './SlashCommandMenu'
import { InlineMath, BlockMath } from './editor/mathExtensions'
import { EquationEditorModal } from './EquationEditorModal'

const TOOLBAR_ICON_SIZE = 16

import {
  EquationEditorContext,
  type EquationEditorContextValue,
} from './EquationEditorContext'

export type MathInputMode = 'equationEditor' | 'latex'
export type { EquationEditorContextValue }
export { useEquationEditorContext } from './EquationEditorContext'

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
    className={`rich-editor-toolbar-button ${active ? 'is-active' : ''}`}
  >
    {children}
  </button>
)

function Toolbar({
  editor,
  className,
  mathInputMode,
  openEquationModal,
}: {
  editor: Editor | null
  className?: string
  mathInputMode: MathInputMode
  openEquationModal: EquationEditorContextValue['openEquationModal']
}) {
  if (!editor) return null
  const useEquationEditor = mathInputMode === 'equationEditor'
  return (
    <div className={`rich-editor-toolbar ${className ?? ''}`.trim()}>
      <ToolbarButton
        title="Bold"
        active={editor.isActive('bold')}
        onClick={() => editor.chain().focus().toggleBold().run()}
      >
        <Bold size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Italic"
        active={editor.isActive('italic')}
        onClick={() => editor.chain().focus().toggleItalic().run()}
      >
        <Italic size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Underline"
        active={editor.isActive('underline')}
        onClick={() => editor.chain().focus().toggleUnderline().run()}
      >
        <UnderlineIcon size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Bullet list"
        active={editor.isActive('bulletList')}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
      >
        <List size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Numbered list"
        active={editor.isActive('orderedList')}
        onClick={() => editor.chain().focus().toggleOrderedList().run()}
      >
        <ListOrdered size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Code block"
        active={editor.isActive('codeBlock')}
        onClick={() => editor.chain().focus().toggleCodeBlock().run()}
      >
        <CodeXml size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Inline code"
        active={editor.isActive('code')}
        onClick={() => editor.chain().focus().toggleCode().run()}
      >
        <Code size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Superscript"
        active={editor.isActive('superscript')}
        onClick={() => editor.chain().focus().toggleSuperscript().run()}
      >
        <SuperscriptIcon size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      <ToolbarButton
        title="Subscript"
        active={editor.isActive('subscript')}
        onClick={() => editor.chain().focus().toggleSubscript().run()}
      >
        <SubscriptIcon size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
      </ToolbarButton>
      {useEquationEditor ? (
        <ToolbarButton
          title="Equation"
          onClick={() =>
            openEquationModal({
              initialLatex: '',
              onInsert: (latex, type) => {
                editor.chain().focus().insertContent({
                  type: type === 'inline' ? 'inlineMath' : 'blockMath',
                  attrs: { latex },
                }).run()
              },
            })
          }
        >
          <Sigma size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
        </ToolbarButton>
      ) : (
        <>
          <ToolbarButton
            title="Inline math ($...$)"
            onClick={() => editor.chain().focus().insertContent(' $ ').run()}
          >
            <Sigma size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
          </ToolbarButton>
          <ToolbarButton
            title="Block math ($$...$$)"
            onClick={() => editor.chain().focus().insertContent('\n\n$$ \n\n').run()}
          >
            <Braces size={TOOLBAR_ICON_SIZE} strokeWidth={2.25} />
          </ToolbarButton>
        </>
      )}
    </div>
  )
}

const SLASH_ITEMS_LATEX: SlashCommandItem[] = [
  { id: 'equation', label: 'Equation', description: 'Inline LaTeX $...$' },
  { id: 'blockEquation', label: 'Block equation', description: 'Display $$...$$' },
  { id: 'code', label: 'Code block', description: 'Fenced code' },
  { id: 'bulletList', label: 'Bullet list', description: '• List' },
  { id: 'orderedList', label: 'Numbered list', description: '1. List' },
]
const SLASH_ITEMS_EQUATION_EDITOR: SlashCommandItem[] = [
  { id: 'equation', label: 'Equation', description: 'Insert equation' },
  { id: 'blockEquation', label: 'Block equation', description: 'Insert display equation' },
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
  className?: string
  toolbarClassName?: string
  mathInputMode?: MathInputMode
  showEquationPreview?: boolean
}

export function RichTextEditor({
  value,
  onChange,
  placeholder = 'Write your answer…',
  disabled = false,
  minHeight = 160,
  className,
  toolbarClassName,
  mathInputMode = 'equationEditor',
}: RichTextEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [slashOpen, setSlashOpen] = useState(false)
  const [slashAnchor, setSlashAnchor] = useState<React.CSSProperties>({})
  const [slashSelected, setSlashSelected] = useState(0)
  const [equationModalOpen, setEquationModalOpen] = useState(false)
  const [equationModalState, setEquationModalState] = useState<{
    initialLatex: string
    onSave?: (latex: string) => void
    onInsert?: (latex: string, type: 'inline' | 'block') => void
  }>({ initialLatex: '' })

  const useEquationEditor = mathInputMode === 'equationEditor'
  const extensions = useMemo(
    () => [
      StarterKit.configure({ heading: false }),
      Underline,
      Placeholder.configure({ placeholder }),
      Subscript,
      Superscript,
      useEquationEditor
        ? InlineMath.configure({ allowDollarInputRules: false })
        : InlineMath,
      useEquationEditor
        ? BlockMath.configure({ allowDollarInputRules: false })
        : BlockMath,
    ],
    [placeholder, useEquationEditor]
  )

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

  const equationEditorContextValue = useMemo<EquationEditorContextValue>(
    () => ({ openEquationModal }),
    [openEquationModal]
  )

  const editor = useEditor({
    extensions,
    content: value || '',
    editable: !disabled,
    onUpdate: ({ editor }) => {
      const html = editor.getHTML()
      onChange(sanitizeHtml(html))
    },
    editorProps: {
      attributes: {
        class: 'editor-content-area',
        style: `min-height: ${minHeight}px; padding: 16px; font-size: 16px; line-height: 1.7;`,
      },
      handleKeyDown: (view, event) => {
        if (slashOpen) {
          if (event.key === 'Escape') {
            setSlashOpen(false)
            return true
          }
          if (event.key === 'ArrowDown') {
            setSlashSelected((i) => Math.min(i + 1, 4))
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

  const slashItems = useEquationEditor ? SLASH_ITEMS_EQUATION_EDITOR : SLASH_ITEMS_LATEX

  const handleSlashSelect = useCallback(
    (id: string) => {
      if (!editor) return
      editor.chain().focus()
      if (useEquationEditor && (id === 'equation' || id === 'blockEquation')) {
        openEquationModal({
          initialLatex: '',
          onInsert: (latex, type) => {
            editor.chain().focus().insertContent({
              type: type === 'inline' ? 'inlineMath' : 'blockMath',
              attrs: { latex },
            }).run()
          },
        })
        setSlashOpen(false)
        return
      }
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
    [editor, useEquationEditor, openEquationModal]
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
        handleSlashSelect(slashItems[slashSelected]?.id ?? '')
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
  }, [slashOpen, slashSelected, handleSlashSelect, slashItems])

  return (
    <EquationEditorContext.Provider value={equationEditorContextValue}>
      <div
        ref={containerRef}
        className={`rich-editor-wrapper ${className ?? ''}`.trim()}
        style={{ position: 'relative' }}
      >
        <Toolbar
          editor={editor}
          className={toolbarClassName}
          mathInputMode={mathInputMode}
          openEquationModal={openEquationModal}
        />
        <div className="editor-content-area" style={{ minHeight: minHeight }}>
          <EditorContent editor={editor} />
        </div>
        <SlashCommandMenu
          open={slashOpen}
          items={slashItems}
          selectedIndex={slashSelected}
          onSelect={handleSlashSelect}
          anchorStyle={slashAnchor}
        />
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
