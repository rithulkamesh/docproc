import { motion } from 'framer-motion'
import { useWorkspace } from '../context/WorkspaceContext'
import { ConverseCanvas } from './ConverseCanvas'
import { SourcesCanvas } from './SourcesCanvas'
import { NotesCanvas } from './NotesCanvas'
import { TestsCanvas } from './TestsCanvas'
import { AdaptivePanel } from './AdaptivePanel'
import { AIActionBar } from './AIActionBar'

const DOCK_OFFSET_LEFT = 'clamp(1.5rem, 3vw, 3rem)'
const DOCK_WIDTH = '4.5rem'

export function WorkspaceLayout() {
  const { canvasMode, focusMode } = useWorkspace()

  const panelContent =
    canvasMode === 'sources' ? (
      <SourcesCanvas />
    ) : canvasMode === 'notes' ? (
      <NotesCanvas />
    ) : canvasMode === 'tests' ? (
      <TestsCanvas />
    ) : null

  return (
    <div
      style={{
        position: 'relative',
        minHeight: '100%',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        paddingLeft: focusMode ? 0 : `calc(${DOCK_OFFSET_LEFT} + ${DOCK_WIDTH} + ${'var(--space-md)'})`,
        paddingBottom: 'var(--space-md)',
        paddingRight: 'var(--space-md)',
        paddingTop: 'var(--space-md)',
      }}
      className="workspace-layout"
    >
      <AIActionBar />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr minmax(280px, 0.5fr)',
          gap: 'var(--content-gap)',
          flex: 1,
          minHeight: 0,
        }}
        className="workspace-layout-grid"
      >
        <motion.div
          layout
          style={{
            flex: 1,
            minWidth: 0,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
            maxWidth: focusMode ? 'var(--canvas-max-width-focus)' : 'var(--canvas-max-width)',
            marginRight: focusMode ? 'auto' : 0,
            marginLeft: focusMode ? 'auto' : 0,
            overflow: 'hidden',
          }}
        >
          <ConverseCanvas />
        </motion.div>
        <AdaptivePanel panelKey={canvasMode} showPlaceholder>
          {panelContent}
        </AdaptivePanel>
      </div>
    </div>
  )
}
