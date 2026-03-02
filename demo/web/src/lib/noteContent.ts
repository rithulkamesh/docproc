/**
 * Strip surrounding markdown code fence(s) (```...``` or ```markdown\n...\n```)
 * so note content is never stored or displayed as a raw code block.
 */
export function stripNoteCodeFence(content: string): string {
  if (typeof content !== 'string') return ''
  let s = content.trim()
  const open = /^```(?:markdown|md)?\s*\n?/i
  const close = /\n?```\s*$/
  while (open.test(s) && close.test(s)) {
    s = s.replace(open, '').replace(close, '').trim()
  }
  return s
}

/**
 * Normalize display-math syntax so common AI/LaTeX patterns render.
 * Converts [ \frac{...} ... ] style (line with backslash) to \[ ... \] for remark-math/rehype-katex.
 */
export function normalizeDisplayMath(content: string): string {
  if (typeof content !== 'string') return content
  // Line that looks like " [ \frac... ] " or " [ \partial... ] " -> " \[ ... \] "
  return content.replace(/^(\s*)\[\s*\\([^\n]*?)\]\s*$/gm, '$1\\[ $2 \\]')
}

/** Strip code fence and normalize math; use before rendering. */
export function prepareNoteContentForRender(content: string): string {
  return normalizeDisplayMath(stripNoteCodeFence(content || ''))
}

/** Content block shape (matches api/notes ContentBlock). */
export interface ContentBlock {
  id: string
  type: string
  data?: { text?: string; [key: string]: unknown }
  children?: ContentBlock[]
}

/** Get plain text from a single block (e.g. for snippet). */
function blockToPlain(b: ContentBlock): string {
  const t = b.data?.text
  if (typeof t === 'string') return t
  if (b.children?.length) return b.children.map(blockToPlain).join(' ')
  return ''
}

/** Get a short snippet from note (title or first block text), max length. */
export function noteSnippet(
  title: string | null | undefined,
  content: string | undefined,
  contentBlocks: ContentBlock[] | null | undefined,
  maxLen: number = 120
): string {
  if (title?.trim()) {
    const s = title.trim()
    return s.length > maxLen ? s.slice(0, maxLen) + '…' : s
  }
  if (contentBlocks?.length) {
    const first = contentBlocks[0]
    const text = first ? blockToPlain(first) : ''
    if (text) return text.length > maxLen ? text.slice(0, maxLen) + '…' : text
  }
  if (content?.trim()) {
    const s = content.trim().replace(/\n+/g, ' ')
    return s.length > maxLen ? s.slice(0, maxLen) + '…' : s
  }
  return 'Untitled'
}
