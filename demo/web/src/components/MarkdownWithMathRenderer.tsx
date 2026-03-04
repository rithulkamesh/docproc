import { useMemo } from 'react'
import MarkdownIt from 'markdown-it'
// @ts-expect-error no types for markdown-it-katex
import markdownItKatex from 'markdown-it-katex'
import DOMPurify from 'dompurify'

let md: MarkdownIt | null = null

function getMarkdownRenderer(): MarkdownIt {
  if (md) return md
  md = new MarkdownIt({ html: true, linkify: false, breaks: true })
  md.use(markdownItKatex, { throwOnError: false, errorColor: '#cc0000' })
  return md
}

export interface MarkdownWithMathRendererProps {
  content: string
  className?: string
  style?: React.CSSProperties
}

/**
 * Renders Markdown with LaTeX math using markdown-it and markdown-it-katex.
 * Inline: $...$  Block: $$...$$
 * Headings (# ## ###), lists, bold, code blocks are rendered properly.
 */
export function MarkdownWithMathRenderer({
  content,
  className,
  style,
}: MarkdownWithMathRendererProps) {
  const html = useMemo(() => {
    const raw = (content || '').trim()
    if (!raw) return ''
    const renderer = getMarkdownRenderer()
    const out = renderer.render(raw)
    return DOMPurify.sanitize(out, {
      ADD_TAGS: ['span'],
      ADD_ATTR: ['class', 'style', 'aria-hidden'],
    })
  }, [content])

  if (!html) {
    return (
      <div
        className={className}
        style={{ color: 'var(--color-text-muted)', ...style }}
      >
        No content
      </div>
    )
  }

  return (
    <div
      className={['markdown-with-math-rendered', className].filter(Boolean).join(' ')}
      style={{
        fontFamily: 'var(--font-family)',
        fontSize: 'var(--text-base)',
        lineHeight: 1.6,
        color: 'var(--color-text)',
        ...style,
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
