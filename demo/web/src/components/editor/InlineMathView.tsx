import { useState, useEffect, useRef } from 'react'
import { NodeViewWrapper } from '@tiptap/react'
import katex from 'katex'
import { useEquationEditorContext } from '@/components/EquationEditorContext'

interface InlineMathViewProps {
  node: { attrs: { latex?: string } }
  updateAttributes: (attrs: { latex: string }) => void
  selected?: boolean
}

export function InlineMathView({ node, updateAttributes, selected }: InlineMathViewProps) {
  const latex = node.attrs.latex ?? ''
  const [editing, setEditing] = useState(false)
  const [input, setInput] = useState(latex)
  const inputRef = useRef<HTMLInputElement>(null)
  const equationContext = useEquationEditorContext()
  const useEquationModal = Boolean(equationContext?.openEquationModal)

  useEffect(() => {
    setInput(latex)
  }, [latex])

  useEffect(() => {
    if (editing && !useEquationModal && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing, useEquationModal])

  const handleBlur = () => {
    setEditing(false)
    const trimmed = input.trim()
    if (trimmed !== latex) {
      updateAttributes({ latex: trimmed })
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === 'Escape') {
      e.preventDefault()
      inputRef.current?.blur()
    }
  }

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
        displayMode: false,
        throwOnError: false,
        output: 'html',
      })
    } catch {
      rendered = null
    }
  }

  if (useEquationModal) {
    if (rendered) {
      return (
        <NodeViewWrapper as="span" className="inline-math-node">
          <span
            className="katex-inline"
            dangerouslySetInnerHTML={{ __html: rendered }}
            onClick={openModalToEdit}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') openModalToEdit()
            }}
            title="Click to edit"
          />
        </NodeViewWrapper>
      )
    }
    return (
      <NodeViewWrapper as="span" className="inline-math-node">
        <span
          className="math-placeholder"
          onClick={openModalToEdit}
          role="button"
          tabIndex={0}
          title="Click to edit"
        >
          Equation
        </span>
      </NodeViewWrapper>
    )
  }

  if (editing || selected) {
    return (
      <NodeViewWrapper as="span" className="inline-math-node">
        <span className="math-delim">$</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          className="math-input-inline"
          placeholder="LaTeX..."
        />
        <span className="math-delim">$</span>
      </NodeViewWrapper>
    )
  }

  if (rendered) {
    return (
      <NodeViewWrapper as="span" className="inline-math-node">
        <span
          className="katex-inline"
          dangerouslySetInnerHTML={{ __html: rendered }}
          onClick={() => setEditing(true)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') setEditing(true)
          }}
          title="Click to edit"
        />
      </NodeViewWrapper>
    )
  }

  return (
    <NodeViewWrapper as="span" className="inline-math-node">
      <span
        className="math-placeholder"
        onClick={() => setEditing(true)}
        role="button"
        tabIndex={0}
      >
        ${latex || '…'}$
      </span>
    </NodeViewWrapper>
  )
}
