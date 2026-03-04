import { Plugin, PluginKey } from '@tiptap/pm/state'
import type { Transaction, EditorState } from '@tiptap/pm/state'
import type { Node as PMNode } from '@tiptap/pm/model'

const NOTES_MATH_PLUGIN_KEY = new PluginKey('notesMath')

/** Find non-overlapping inline $...$ in text (skip those inside $$...$$). */
function findInlineMathRanges(text: string): { start: number; end: number; latex: string }[] {
  const result: { start: number; end: number; latex: string }[] = []
  const blockRe = /\$\$[\s\S]*?\$\$/g
  const inlineRe = /\$([^$\n]+)\$/g
  let m: RegExpExecArray | null
  const blockRanges: { start: number; end: number }[] = []
  while ((m = blockRe.exec(text)) !== null) {
    blockRanges.push({ start: m.index, end: m.index + m[0].length })
  }
  while ((m = inlineRe.exec(text)) !== null) {
    const start = m.index
    const end = m.index + m[0].length
    const insideBlock = blockRanges.some((b) => start >= b.start && end <= b.end)
    if (!insideBlock) {
      result.push({ start, end, latex: (m[1] ?? '').trim() })
    }
  }
  return result
}

/** Find block $$...$$ (whole-node only for replacement). */
function findBlockMathRanges(text: string): { start: number; end: number; latex: string }[] {
  const result: { start: number; end: number; latex: string }[] = []
  const blockRe = /\$\$([\s\S]*?)\$\$/g
  let m: RegExpExecArray | null
  while ((m = blockRe.exec(text)) !== null) {
    result.push({
      start: m.index,
      end: m.index + m[0].length,
      latex: (m[1] ?? '').trim(),
    })
  }
  return result
}

/**
 * ProseMirror plugin: replace raw $...$ and $$...$$ in text nodes with inlineMath/blockMath
 * so they render in the editor like the test portal.
 */
export function notesMathConversionPlugin(schema: {
  nodes: {
    inlineMath?: { create: (attrs: { latex: string }) => unknown }
    blockMath?: { create: (attrs: { latex: string }) => unknown }
  }
}): Plugin {
  const inlineMath = schema.nodes.inlineMath
  const blockMath = schema.nodes.blockMath
  if (!inlineMath && !blockMath) return new Plugin({ key: NOTES_MATH_PLUGIN_KEY })

  return new Plugin({
    key: NOTES_MATH_PLUGIN_KEY,
    appendTransaction(_transactions: unknown, _oldState: EditorState, state: EditorState) {
      const tr = state.tr
      let modified = false
      const replacements: { from: number; to: number; node: unknown }[] = []

      state.doc.descendants((node: PMNode, pos: number) => {
        if (!node.isText || !node.text) return
        const text = node.text
        const base = pos + 1

        if (blockMath) {
          const blocks = findBlockMathRanges(text)
          if (blocks.length === 1 && blocks[0].start === 0 && blocks[0].end === text.length) {
            const res = state.doc.resolve(base)
            const paraFrom = res.before(res.depth)
            const paraTo = res.after(res.depth)
            replacements.push({
              from: paraFrom,
              to: paraTo,
              node: blockMath.create({ latex: blocks[0].latex }),
            })
          }
        }

        if (inlineMath) {
          const inlines = findInlineMathRanges(text)
          for (const r of inlines) {
            const from = base + r.start
            const to = base + r.end
            replacements.push({ from, to, node: inlineMath.create({ latex: r.latex }) })
          }
        }
      })

      replacements.sort((a, b) => b.from - a.from)
      const seen = new Set<string>()
      for (const { from, to, node } of replacements) {
        const key = `${from}-${to}`
        if (seen.has(key)) continue
        seen.add(key)
        try {
          tr.replaceWith(from, to, node as Parameters<Transaction['replaceWith']>[2])
          modified = true
        } catch {
          // skip
        }
      }
      return modified ? tr : null
    },
  })
}
