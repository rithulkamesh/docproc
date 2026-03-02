import { useWorkspace } from '../context/WorkspaceContext'
import { ConverseCanvas } from './ConverseCanvas'
import { SourcesCanvas } from './SourcesCanvas'
import { NotesCanvas } from './NotesCanvas'
import { FlashcardsCanvas } from './FlashcardsCanvas'
import { TestsCanvas } from './TestsCanvas'

export function KnowledgeCanvas() {
  const { canvasMode, focusMode } = useWorkspace()

  const maxWidth = focusMode ? 'var(--canvas-max-width-focus)' : 'var(--canvas-max-width)'

  return (
    <div
      style={{
        width: '100%',
        maxWidth,
        marginInline: 'auto',
        padding: 'var(--content-gap)',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100%',
        gap: 'var(--content-gap)',
        transition: 'max-width 150ms ease',
      }}
    >
      {canvasMode === 'converse' && <ConverseCanvas />}
      {canvasMode === 'sources' && <SourcesCanvas />}
      {canvasMode === 'notes' && <NotesCanvas />}
      {canvasMode === 'flashcards' && <FlashcardsCanvas />}
      {canvasMode === 'tests' && <TestsCanvas />}
    </div>
  )
}
