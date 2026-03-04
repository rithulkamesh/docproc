import { useState, useEffect, useRef } from 'react'
import { NodeViewWrapper } from '@tiptap/react'
import katex from 'katex'
import { useEquationEditorContext } from '@/components/EquationEditorContext'

interface BlockMathViewProps {
  node: { attrs: { latex?: string } }
  updateAttributes: (attrs: { latex: string }) => void
}

export function BlockMathView({ node, updateAttributes }: BlockMathViewProps) {
  const latex = node.attrs.latex ?? ''
  const [input, setInput] = useState(latex)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const equationContext = useEquationEditorContext()
  const useEquationModal = Boolean(equationContext?.openEquationModal)

  useEffect(() => {
    setInput(latex)
  }, [latex])

  useEffect(() => {
    if (!useEquationModal) {
      const trimmed = input.trim()
      if (trimmed !== latex) {
        const t = setTimeout(() => updateAttributes({ latex: trimmed }), 300)
        return () => clearTimeout(t)
      }
    }
  }, [input, latex, updateAttributes, useEquationModal])

  const openModalToEdit = () => {
    equationContext?.openEquationModal({
      initialLatex: latex,
      onSave: (newLatex) => updateAttributes({ latex: newLatex }),
    })
  }

  let rendered: string | null = null
  if (latex) {
    try {
      rendered = katex.renderToString(latex, {
        displayMode: true,
        throwOnError: false,
        output: 'html',
      })
    } catch {
      rendered = null
    }
  }

  if (useEquationModal) {
    return (
      <NodeViewWrapper as="div" className="math-block-node">
        <div
          className="math-block-preview math-block cursor-pointer rounded-md border border-dashed border-border p-4 min-h-[60px] flex items-center justify-center"
          onClick={openModalToEdit}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') openModalToEdit()
          }}
          title="Click to edit"
        >
          {rendered ? (
            <span dangerouslySetInnerHTML={{ __html: rendered }} />
          ) : (
            <span className="text-muted-foreground text-sm">Equation — click to edit</span>
          )}
        </div>
      </NodeViewWrapper>
    )
  }

  return (
    <NodeViewWrapper as="div" className="math-block-node">
      <div className="math-block-editor">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="LaTeX (e.g. \frac{1}{2})"
          className="math-textarea"
          rows={3}
        />
        <div className="math-block-preview math-block">
          {rendered ? (
            <span dangerouslySetInnerHTML={{ __html: rendered }} />
          ) : (
            <span className="text-muted-foreground text-sm">Preview appears when LaTeX is valid</span>
          )}
        </div>
      </div>
    </NodeViewWrapper>
  )
}
