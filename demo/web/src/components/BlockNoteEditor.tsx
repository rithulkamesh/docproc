import { useRef, useEffect, useCallback } from 'react'
import { useCreateBlockNote, useEditorChange } from '@blocknote/react'
import { BlockNoteView } from '@blocknote/mantine'
import type { Block, PartialBlock } from '@blocknote/core'
import '@blocknote/mantine/style.css'
import '@blocknote/core/fonts/inter.css'
import { theme } from '../design/theme'
import { useWorkspace } from '../context/WorkspaceContext'

/** Our simple block shape (backend / fallback). */
export interface SimpleContentBlock {
  id: string
  type: string
  data?: { text?: string; [key: string]: unknown }
  children?: SimpleContentBlock[]
}

/** Convert our simple blocks to BlockNote PartialBlock[]. */
function simpleBlocksToPartialBlocks(blocks: SimpleContentBlock[]): PartialBlock[] {
  return blocks.map((b) => {
    const text = (b.data?.text as string) ?? ''
    const content: { type: 'text'; text: string; styles: Record<string, string> }[] | undefined = text
      ? [{ type: 'text', text, styles: {} }]
      : undefined
    const children = b.children?.length ? simpleBlocksToPartialBlocks(b.children) : undefined
    return {
      id: b.id,
      type: (b.type as 'paragraph') || 'paragraph',
      content,
      children,
    }
  })
}

/** Heuristic: does this look like BlockNote native format? (has content array with inline items) */
function isBlockNoteFormat(blocks: unknown): blocks is Block[] {
  if (!Array.isArray(blocks) || blocks.length === 0) return false
  const first = blocks[0] as Record<string, unknown>
  const data = first?.data as { text?: string } | undefined
  return (
    typeof first === 'object' &&
    first !== null &&
    ('content' in first || 'props' in first) &&
    !(data && typeof data.text === 'string')
  )
}

interface BlockNoteEditorProps {
  /** Initial content: our simple blocks or BlockNote Block[] (stored as-is). */
  initialBlocks?: SimpleContentBlock[] | Block[] | null
  /** Legacy markdown string (used when no blocks). */
  initialMarkdown?: string
  /** Callback with current blocks (BlockNote document) for save. */
  onSave: (blocks: Block[]) => void
  /** Debounce ms for onSave. */
  saveDebounceMs?: number
  /** Optional title (displayed above editor). */
  title?: string | null
  /** Disabled state. */
  readOnly?: boolean
}

export function BlockNoteEditor({
  initialBlocks,
  initialMarkdown,
  onSave,
  saveDebounceMs = 800,
  title,
  readOnly = false,
}: BlockNoteEditorProps) {
  const initialContentRef = useRef<PartialBlock[] | undefined>(undefined)
  const saveTimerRef = useRef<number | null>(null)

  if (initialContentRef.current === undefined) {
    if (initialBlocks?.length) {
      if (isBlockNoteFormat(initialBlocks)) {
        initialContentRef.current = initialBlocks as PartialBlock[]
      } else {
        initialContentRef.current = simpleBlocksToPartialBlocks(initialBlocks as SimpleContentBlock[])
      }
    } else if (initialMarkdown?.trim()) {
      initialContentRef.current = [
        {
          type: 'paragraph',
          content: [{ type: 'text' as const, text: initialMarkdown.trim(), styles: {} }],
        },
      ]
    } else {
      initialContentRef.current = [{ type: 'paragraph', content: [] }]
    }
  }

  const editor = useCreateBlockNote({
    initialContent: initialContentRef.current,
  })

  const flushSave = useCallback(() => {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    if (editor?.document) {
      onSave(editor.document)
    }
  }, [editor, onSave])

  useEditorChange(() => {
    if (readOnly || !editor?.document) return
    if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current)
    saveTimerRef.current = window.setTimeout(flushSave, saveDebounceMs)
  }, editor)

  useEffect(() => () => { if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current) }, [])

  const { themeMode } = useWorkspace()

  if (!editor) return null

  return (
    <div style={{ fontFamily: 'var(--font-family)', color: 'var(--color-text)' }}>
      {title && (
        <h1 style={{ fontSize: theme.fontSizes['2xl'], fontWeight: 700, marginBottom: 'var(--space-md)', lineHeight: theme.lineHeight.heading }}>
          {title}
        </h1>
      )}
      <BlockNoteView editor={editor} editable={!readOnly} data-theme={themeMode} />
    </div>
  )
}
