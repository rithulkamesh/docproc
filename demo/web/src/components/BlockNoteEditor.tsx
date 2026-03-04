import { MarkdownNotesEditor } from './MarkdownNotesEditor'

export interface SimpleContentBlock {
  id: string
  type: string
  data?: { text?: string; [key: string]: unknown }
  children?: SimpleContentBlock[]
}

export interface Block {
  id: string
  type: string
  content?: { type: 'text'; text: string; styles?: Record<string, string> }[]
  children?: Block[]
}

function blocksToMarkdown(blocks: SimpleContentBlock[] | Block[]): string {
  return blocks
    .map((b) => {
      const simple = b as SimpleContentBlock
      if (simple.data?.text != null) return String(simple.data.text)
      const withContent = b as Block
      return withContent.content?.map((c) => c.text ?? '').join('') ?? ''
    })
    .filter(Boolean)
    .join('\n\n')
}

export interface BlockNoteEditorProps {
  initialBlocks?: SimpleContentBlock[] | Block[] | null
  initialMarkdown?: string
  onSave: (markdown: string) => void
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
  const initialValue =
    (initialMarkdown?.trim() && initialMarkdown) ||
    (initialBlocks?.length ? blocksToMarkdown(initialBlocks) : '') ||
    ''

  return (
    <div className="font-sans text-foreground" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {title && (
        <h1 style={{ marginBottom: 'var(--space-md)', fontSize: 'var(--text-2xl)', fontWeight: 700 }}>
          {title}
        </h1>
      )}
      <MarkdownNotesEditor
        value={initialValue}
        onChange={onSave}
        readOnly={readOnly}
        debounceMs={saveDebounceMs}
        showToolbar={!readOnly}
        placeholder="Write your note… (Markdown and LaTeX $...$ / $$...$$ supported)"
      />
    </div>
  )
}
