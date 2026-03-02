/**
 * Parse text for LaTeX: $$...$$ (display), $...$ (inline), \[...\], \(...\).
 * Returns segments of { type: 'text' | 'inline' | 'block', content: string }.
 */

export type LatexSegment =
  | { type: 'text'; content: string }
  | { type: 'inline'; content: string }
  | { type: 'block'; content: string }

/**
 * Normalize LaTeX string before parsing/rendering:
 * - Convert \\ to \ (unescape backslashes from DB/JSON)
 * - Trim whitespace
 */
export function normalizeLatexContent(input: string): string {
  if (typeof input !== 'string') return ''
  return input.replace(/\\\\/g, '\\').trim()
}

/**
 * Strip HTML tags so we never render arbitrary HTML from AI/content.
 * Keeps only plain text and LaTeX delimiters. Use before parsing.
 */
export function sanitizeForMath(input: string): string {
  if (typeof input !== 'string') return ''
  return input.replace(/<[^>]*>/g, '')
}

/**
 * Prepare content for math rendering: sanitize then normalize.
 */
export function prepareForMathRender(input: string): string {
  return normalizeLatexContent(sanitizeForMath(input))
}

/**
 * Parse in order: $$ ... $$, \[ ... \], \( ... \), $ ... $
 * Dollar amounts like $5 are left as text when content is only digits.
 */
export function parseLatexSegments(input: string): LatexSegment[] {
  const segments: LatexSegment[] = []
  let s = input

  while (s.length > 0) {
    const iBlock2 = s.indexOf('$$')
    const iBlockB = s.indexOf('\\[')
    const iInlineP = s.indexOf('\\(')
    const iInline = s.indexOf('$')

    const next = [
      { i: iBlock2, len: 2, type: 'block2' as const },
      { i: iBlockB, len: 2, type: 'blockB' as const },
      { i: iInlineP, len: 2, type: 'inlineP' as const },
      { i: iInline, len: 1, type: 'inline' as const },
    ]
      .filter((x) => x.i >= 0)
      .sort((a, b) => a.i - b.i)[0]

    if (!next) {
      if (s) segments.push({ type: 'text', content: s })
      break
    }

    if (next.i > 0) {
      segments.push({ type: 'text', content: s.slice(0, next.i) })
    }
    s = s.slice(next.i + next.len)

    if (next.type === 'block2') {
      const end = s.indexOf('$$')
      if (end === -1) {
        segments.push({ type: 'text', content: '$$' + s })
        break
      }
      segments.push({ type: 'block', content: s.slice(0, end).trim() })
      s = s.slice(end + 2)
      continue
    }

    if (next.type === 'blockB') {
      const end = s.indexOf('\\]')
      if (end === -1) {
        segments.push({ type: 'text', content: '\\[' + s })
        break
      }
      segments.push({ type: 'block', content: s.slice(0, end).trim() })
      s = s.slice(end + 2)
      continue
    }

    if (next.type === 'inlineP') {
      const end = s.indexOf('\\)')
      if (end === -1) {
        segments.push({ type: 'text', content: '\\(' + s })
        break
      }
      segments.push({ type: 'inline', content: s.slice(0, end).trim() })
      s = s.slice(end + 2)
      continue
    }

    // next.type === 'inline' ($ ... $)
    const end = s.indexOf('$')
    if (end === -1) {
      segments.push({ type: 'text', content: '$' + s })
      break
    }
    const content = s.slice(0, end).trim()
    if (content.length > 0 && !/^\d+\.?\d*$/.test(content)) {
      segments.push({ type: 'inline', content })
    } else {
      segments.push({ type: 'text', content: '$' + content + '$' })
    }
    s = s.slice(end + 1)
  }

  return segments
}
