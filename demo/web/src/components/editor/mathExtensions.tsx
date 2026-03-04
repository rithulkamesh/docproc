import { Node, mergeAttributes, nodeInputRule } from '@tiptap/core'
import { ReactNodeViewRenderer } from '@tiptap/react'
import { InlineMathView } from './InlineMathView'
import { BlockMathView } from './BlockMathView'

const INLINE_MATH_INPUT_REGEX = /\$([^$]+)\$/
const BLOCK_MATH_INPUT_REGEX = /\$\$([\s\S]*?)\$\$/

export const InlineMath = Node.create({
  name: 'inlineMath',
  group: 'inline',
  inline: true,
  atom: true,
  addOptions() {
    return { allowDollarInputRules: true }
  },
  addAttributes() {
    return {
      latex: {
        default: '',
        parseHTML: (el) => (el as HTMLElement).getAttribute('data-latex') ?? '',
        renderHTML: (attrs) => ({ 'data-latex': attrs.latex }),
      },
    }
  },
  parseHTML() {
    return [{ tag: 'span[data-type="inline-math"]' }]
  },
  renderHTML({ node, HTMLAttributes }) {
    return [
      'span',
      mergeAttributes(HTMLAttributes, { 'data-type': 'inline-math' }),
      `$${node.attrs.latex}$`,
    ]
  },
  addNodeView() {
    return ReactNodeViewRenderer(InlineMathView)
  },
  addInputRules() {
    if (this.options.allowDollarInputRules === false) return []
    return [
      nodeInputRule({
        find: INLINE_MATH_INPUT_REGEX,
        type: this.type,
        getAttributes: (match) => ({ latex: (match[1] ?? '').trim() }),
      }),
    ]
  },
  addStorage() {
    return {
      markdown: {
        serialize(_state: unknown, node: { attrs: { latex?: string } }, _parent: unknown, _index: number) {
          const state = _state as { write: (s: string) => void }
          state.write('$' + (node.attrs.latex ?? '') + '$')
        },
      },
    }
  },
})

export const BlockMath = Node.create({
  name: 'blockMath',
  group: 'block',
  atom: true,
  addOptions() {
    return { allowDollarInputRules: true }
  },
  addAttributes() {
    return {
      latex: {
        default: '',
        parseHTML: (el) => (el as HTMLElement).getAttribute('data-latex') ?? '',
        renderHTML: (attrs) => ({ 'data-latex': attrs.latex }),
      },
    }
  },
  parseHTML() {
    return [{ tag: 'div[data-type="block-math"]' }]
  },
  renderHTML({ node, HTMLAttributes }) {
    return [
      'div',
      mergeAttributes(HTMLAttributes, { 'data-type': 'block-math' }),
      `$$${node.attrs.latex}$$`,
    ]
  },
  addNodeView() {
    return ReactNodeViewRenderer(BlockMathView)
  },
  addInputRules() {
    if (this.options.allowDollarInputRules === false) return []
    return [
      nodeInputRule({
        find: BLOCK_MATH_INPUT_REGEX,
        type: this.type,
        getAttributes: (match) => ({ latex: (match[1] ?? '').trim() }),
      }),
    ]
  },
  addStorage() {
    return {
      markdown: {
        serialize(_state: unknown, node: { attrs: { latex?: string } }) {
          const state = _state as { write: (s: string) => void }
          state.write('$$\n' + (node.attrs.latex ?? '') + '\n$$')
        },
      },
    }
  },
})
