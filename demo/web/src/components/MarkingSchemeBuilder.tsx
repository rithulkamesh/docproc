import { useState, useCallback } from 'react'
import type { PaperPattern, PaperPatternSection } from '../api/assessments'

const DEFAULT_SECTIONS: PaperPatternSection[] = [
  { name: 'Part A', question_count: 20, marks_each: 1, type: 'short' },
  { name: 'Part B', question_count: 4, marks_each: 10, type: 'long' },
]

function totalFromSections(sections: PaperPatternSection[]): number {
  return sections.reduce((sum, s) => sum + s.question_count * s.marks_each, 0)
}

interface MarkingSchemeBuilderProps {
  value: PaperPattern | null
  onChange: (pattern: PaperPattern | null) => void
  /** When true, show as editable builder; when false, show read-only summary */
  editable?: boolean
}

export function MarkingSchemeBuilder({ value, onChange, editable = true }: MarkingSchemeBuilderProps) {
  const sections = value?.sections ?? DEFAULT_SECTIONS
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null)
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)

  const setSections = useCallback(
    (next: PaperPatternSection[]) => {
      onChange({ total_marks: totalFromSections(next), sections: next })
    },
    [onChange]
  )

  const updateSection = useCallback(
    (index: number, patch: Partial<PaperPatternSection>) => {
      const next = sections.map((s, i) => (i === index ? { ...s, ...patch } : s))
      setSections(next)
    },
    [sections, setSections]
  )

  const addSection = useCallback(() => {
    setSections([...sections, { name: 'New section', question_count: 1, marks_each: 1, type: 'short' }])
  }, [sections, setSections])

  const removeSection = useCallback(
    (index: number) => {
      if (sections.length <= 1) return
      setSections(sections.filter((_, i) => i !== index))
    },
    [sections, setSections]
  )

  const handleDragStart = (index: number) => (e: React.DragEvent) => {
    setDraggedIndex(index)
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.setData('text/plain', String(index))
    e.dataTransfer.setData('application/json', JSON.stringify(sections[index]))
  }

  const handleDragOver = (index: number) => (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDragOverIndex(index)
  }

  const handleDragLeave = () => {
    setDragOverIndex(null)
  }

  const handleDrop = (dropIndex: number) => (e: React.DragEvent) => {
    e.preventDefault()
    setDragOverIndex(null)
    setDraggedIndex(null)
    const fromIndex = draggedIndex
    if (fromIndex == null || fromIndex === dropIndex) return
    const next = [...sections]
    const [removed] = next.splice(fromIndex, 1)
    next.splice(dropIndex, 0, removed)
    setSections(next)
  }

  const handleDragEnd = () => {
    setDraggedIndex(null)
    setDragOverIndex(null)
  }

  if (!editable) {
    return (
      <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
        Total: {totalFromSections(sections)} marks
        <ul style={{ margin: `${'var(--space-sm)'} 0 0`, paddingLeft: 'var(--space-xl)' }}>
          {sections.map((s, i) => (
            <li key={i}>
              {s.name}: {s.question_count} × {s.marks_each} ({s.type})
            </li>
          ))}
        </ul>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
        <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>Marking scheme</span>
        <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
          Total: <strong style={{ color: 'var(--color-text)' }}>{totalFromSections(sections)}</strong> marks
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '24px 1fr 80px 80px 80px auto', gap: 'var(--space-md)', alignItems: 'center', padding: `0 ${'var(--space-md)'}`, fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text-muted)' }}>
        <span />
        <span>Section</span>
        <span>Count</span>
        <span>Marks each</span>
        <span>Type</span>
        <span />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        {sections.map((section, index) => (
          <div
            key={index}
            draggable
            onDragStart={handleDragStart(index)}
            onDragOver={handleDragOver(index)}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop(index)}
            onDragEnd={handleDragEnd}
            style={{
              display: 'grid',
              gridTemplateColumns: '24px 1fr 80px 80px 80px auto',
              gap: 'var(--space-md)',
              alignItems: 'center',
              padding: 'var(--space-md)',
              border: 'var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              backgroundColor: dragOverIndex === index ? 'var(--color-bg-hover)' : 'var(--color-bg)',
              opacity: draggedIndex === index ? 0.6 : 1,
              cursor: editable ? 'grab' : 'default',
            }}
          >
            <span
              style={{
                cursor: 'grab',
                color: 'var(--color-text-muted)',
                fontSize: 'var(--text-sm)',
              }}
              title="Drag to reorder"
            >
              ⋮⋮
            </span>
            <input
              type="text"
              value={section.name}
              onChange={(e) => updateSection(index, { name: e.target.value })}
              placeholder="Section name"
              style={{
                padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
                border: 'var(--border-subtle)',
                borderRadius: 'var(--radius-input)',
                fontFamily: 'var(--font-family)',
                fontSize: 'var(--text-sm)',
                backgroundColor: 'var(--color-bg)',
                color: 'var(--color-text)',
              }}
            />
            <input
              type="number"
              min={0}
              step={1}
              value={section.question_count}
              onChange={(e) => updateSection(index, { question_count: Math.max(0, parseInt(e.target.value, 10) || 0) })}
              style={{
                padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
                border: 'var(--border-subtle)',
                borderRadius: 'var(--radius-input)',
                fontFamily: 'var(--font-family)',
                fontSize: 'var(--text-sm)',
                backgroundColor: 'var(--color-bg)',
                color: 'var(--color-text)',
              }}
            />
            <input
              type="number"
              min={0}
              step={0.5}
              value={section.marks_each}
              onChange={(e) => updateSection(index, { marks_each: Math.max(0, parseFloat(e.target.value) || 0) })}
              title="Marks per question (fractional allowed)"
              style={{
                padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
                border: 'var(--border-subtle)',
                borderRadius: 'var(--radius-input)',
                fontFamily: 'var(--font-family)',
                fontSize: 'var(--text-sm)',
                backgroundColor: 'var(--color-bg)',
                color: 'var(--color-text)',
              }}
            />
            <select
              value={section.type}
              onChange={(e) => updateSection(index, { type: e.target.value as 'short' | 'long' })}
              style={{
                padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
                border: 'var(--border-subtle)',
                borderRadius: 'var(--radius-input)',
                fontFamily: 'var(--font-family)',
                fontSize: 'var(--text-sm)',
                backgroundColor: 'var(--color-bg)',
                color: 'var(--color-text)',
              }}
            >
              <option value="short">Short</option>
              <option value="long">Long</option>
            </select>
            <button
              type="button"
              onClick={() => removeSection(index)}
              disabled={sections.length <= 1}
              title="Remove section"
              style={{
                padding: 'var(--space-sm)',
                border: 'none',
                background: 'none',
                color: 'var(--color-text-muted)',
                cursor: sections.length <= 1 ? 'not-allowed' : 'pointer',
                fontSize: 'var(--text-lg)',
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={addSection}
        style={{
          alignSelf: 'flex-start',
          padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
          border: '1px dashed var(--color-border-light)',
          borderRadius: 'var(--radius-md)',
          background: 'none',
          color: 'var(--color-text-muted)',
          fontFamily: 'var(--font-family)',
          fontSize: 'var(--text-sm)',
          cursor: 'pointer',
        }}
      >
        + Add section
      </button>
    </div>
  )
}
