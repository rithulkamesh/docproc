import { useRef, useEffect, useCallback, useState } from 'react'

/** Our simple block shape (backend / fallback). */
export interface SimpleContentBlock {
  id: string
  type: string
  data?: { text?: string; [key: string]: unknown }
  children?: SimpleContentBlock[]
}

/** Minimal block shape for onSave (replaces BlockNote Block). */
export interface Block {
  id: string
  type: string
  content?: { type: 'text'; text: string; styles?: Record<string, string> }[]
  children?: Block[]
}

function blocksToText(blocks: SimpleContentBlock[] | Block[]): string {
  return blocks
    .map((b) => {
      const simple = b as SimpleContentBlock
      if (simple.data?.text != null) return String(simple.data.text)
      const withContent = b as Block
      return withContent.content?.map((c) => c.text ?? '').join('') ?? ''
    })
    .join('\n\n')
}

function textToBlocks(text: string): Block[] {
  if (!text.trim()) return [{ id: crypto.randomUUID(), type: 'paragraph', content: [] }]
  return text.split(/\n\n+/).map((para) => ({
    id: crypto.randomUUID(),
    type: 'paragraph',
    content: [{ type: 'text' as const, text: para.trim(), styles: {} }],
  }))
}

interface BlockNoteEditorProps {
  initialBlocks?: SimpleContentBlock[] | Block[] | null
  initialMarkdown?: string
  onSave: (blocks: Block[]) => void
  saveDebounceMs?: number
  title?: string | null
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
  const [value, setValue] = useState(() => {
    if (initialBlocks?.length) return blocksToText(initialBlocks as SimpleContentBlock[] | Block[])
    return initialMarkdown?.trim() ?? ''
  })
  const saveTimerRef = useRef<number | null>(null)

  const flushSave = useCallback(() => {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
    onSave(textToBlocks(value))
  }, [value, onSave])

  useEffect(() => {
    if (readOnly) return
    if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current)
    saveTimerRef.current = window.setTimeout(flushSave, saveDebounceMs)
    return () => {
      if (saveTimerRef.current !== null) window.clearTimeout(saveTimerRef.current)
    }
  }, [value, readOnly, saveDebounceMs, flushSave])

  return (
    <div className="font-sans text-foreground">
      {title && (
        <h1 className="mb-4 text-2xl font-bold leading-tight">{title}</h1>
      )}
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        readOnly={readOnly}
        className="min-h-[200px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm"
        placeholder="Write your note…"
      />
    </div>
  )
}
