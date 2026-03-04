import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { prepareNoteContentForRender } from '../lib/noteContent'

export interface NoteContentProps {
  content: string
  className?: string
  style?: React.CSSProperties
}

export function NoteContent({ content, className, style }: NoteContentProps) {
  const cleaned = prepareNoteContentForRender(content || '')
  if (!cleaned) return null

  return (
    <div
      className={['note-content-document', className].filter(Boolean).join(' ')}
      style={{
        fontFamily: 'var(--font-family)',
        fontSize: 'var(--text-base)',
        lineHeight: 'var(--line-height-body)',
        color: 'var(--color-text)',
        ...style,
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h1: ({ children }) => (
            <h1 style={{ marginTop: '1em', marginBottom: '0.5em', fontWeight: 700, fontSize: 'var(--text-xl)' }}>
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 style={{ marginTop: '1em', marginBottom: '0.5em', fontWeight: 600, fontSize: 'var(--text-lg)' }}>
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 style={{ marginTop: '0.75em', marginBottom: '0.35em', fontWeight: 600, fontSize: 'var(--text-base)' }}>
              {children}
            </h3>
          ),
          p: ({ children }) => <p style={{ marginTop: 0, marginBottom: '0.75em' }}>{children}</p>,
          ul: ({ children }) => <ul style={{ marginTop: 0, marginBottom: '0.75em', paddingLeft: '1.5em' }}>{children}</ul>,
          ol: ({ children }) => <ol style={{ marginTop: 0, marginBottom: '0.75em', paddingLeft: '1.5em' }}>{children}</ol>,
          li: ({ children }) => <li style={{ marginBottom: '0.25em' }}>{children}</li>,
          code: ({ className, children, ...rest }) => {
            const isMath = className?.includes('math')
            if (isMath) {
              return <code {...rest}>{children}</code>
            }
            return (
              <code
                {...rest}
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.9em',
                  padding: '0.1em 0.3em',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--color-bg-alt)',
                }}
              >
                {children}
              </code>
            )
          },
          pre: ({ children }) => (
            <pre
              style={{
                marginTop: 0,
                marginBottom: '0.75em',
                padding: 'var(--space-md)',
                borderRadius: 'var(--radius-md)',
                background: 'var(--color-bg-alt)',
                overflow: 'auto',
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--text-sm)',
              }}
            >
              {children}
            </pre>
          ),
        }}
      >
        {cleaned}
      </ReactMarkdown>
    </div>
  )
}
