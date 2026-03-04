import { AnimatePresence, motion } from 'framer-motion'
import { useWorkspace } from '@/context/WorkspaceContext'
import { HomeDashboard } from '@/components/dashboard/HomeDashboard'
import { ConverseCanvas } from './ConverseCanvas'
import { SourcesCanvas } from './SourcesCanvas'
import { NotesCanvas } from './NotesCanvas'
import { TestsCanvas } from './TestsCanvas'
import { motion as motionTokens } from '@/design/tokens'

export function KnowledgeCanvas() {
  const { canvasMode } = useWorkspace()

  const Canvas = (() => {
    switch (canvasMode) {
      case 'home':
        return <HomeDashboard />
      case 'converse':
        return <ConverseCanvas />
      case 'sources':
        return <SourcesCanvas />
      case 'notes':
        return <NotesCanvas />
      case 'tests':
        return <TestsCanvas />
      default:
        return null
    }
  })()

  return (
    <div className="flex flex-col gap-8">
      <AnimatePresence mode="wait">
        <motion.div
          key={canvasMode}
          initial={{ opacity: 0, y: 16, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -12, scale: 0.98 }}
          transition={{
            duration: motionTokens.durationPanel / 1000,
            ease: motionTokens.easingFramer,
          }}
        >
          {Canvas}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
