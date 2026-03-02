import katex from 'katex'
import { parseLatexSegments, prepareForMathRender } from '../lib/latex'
import 'katex/dist/katex.min.css'

export interface LatexTextProps {
  text: string
  /** Optional className for the root span (e.g. for font size). */
  className?: string
  /** Inline style for the root span. */
  style?: React.CSSProperties
}

/**
 * Renders text with LaTeX: $$...$$ and \[...\] for display math, $...$ and \(...\) for inline math.
 * Content is sanitized (no arbitrary HTML) and normalized (\\ → \). Only KaTeX output is used for math.
 * Use for question prompts, answers, feedback, strengths, missing concepts, etc.
 */
export function LatexText({ text, className, style }: LatexTextProps) {
  if (!text) return null
  const prepared = prepareForMathRender(text)
  const segments = parseLatexSegments(prepared)
  const nodes = segments.map((seg, i) => {
    if (seg.type === 'text') {
      return <span key={i}>{seg.content}</span>
    }
    try {
      const html = katex.renderToString(seg.content, {
        displayMode: seg.type === 'block',
        throwOnError: false,
        output: 'html',
      })
      return (
        <span
          key={i}
          className={seg.type === 'block' ? 'latex-block' : 'latex-inline'}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )
    } catch {
      return (
        <span key={i} className="latex-inline" style={{ color: 'var(--color-text-muted)' }}>
          {seg.type === 'block' ? '$$' : '$'}{seg.content}{seg.type === 'block' ? '$$' : '$'}
        </span>
      )
    }
  })
  return (
    <span className={className} style={style}>
      {nodes}
    </span>
  )
}

/** Alias for use as global math render wrapper. */
export const MathRenderer = LatexText
