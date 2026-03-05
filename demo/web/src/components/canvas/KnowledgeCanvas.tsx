import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { useWorkspace } from '@/context/WorkspaceContext'
import { HomeDashboard } from '@/components/dashboard/HomeDashboard'
import { ConverseCanvas } from './ConverseCanvas'
import { SourcesCanvas } from './SourcesCanvas'
import { NotesCanvas } from './NotesCanvas'
import { TestsCanvas } from './TestsCanvas'
import { motion as motionTokens } from '@/design/tokens'

export function KnowledgeCanvas() {
  const location = useLocation()
  const navigate = useNavigate()
  const { canvasMode, setCanvasMode, documents } = useWorkspace()
  const [welcomeProjectName, setWelcomeProjectName] = useState<string | null>(null)

  useEffect(() => {
    const s = location.state as { justCreatedProject?: boolean; projectName?: string } | undefined
    if (s?.justCreatedProject && s?.projectName) {
      setCanvasMode('sources')
      setWelcomeProjectName(s.projectName)
      navigate(location.pathname, { replace: true, state: {} })
      return
    }
    const storedName =
      typeof window !== 'undefined' ? window.sessionStorage.getItem('docproc-welcome-project-name') : null
    if (storedName) {
      try {
        window.sessionStorage.removeItem('docproc-welcome-project-name')
      } catch {}
      setCanvasMode('sources')
      setWelcomeProjectName(storedName)
    }
  }, [location.state, location.pathname, navigate, setCanvasMode])

  useEffect(() => {
    if (canvasMode !== 'sources') setWelcomeProjectName(null)
  }, [canvasMode])

  useEffect(() => {
    if (documents.length > 0) setWelcomeProjectName(null)
  }, [documents.length])

  const Canvas = (() => {
    switch (canvasMode) {
      case 'home':
        return <HomeDashboard />
      case 'converse':
        return <ConverseCanvas />
      case 'sources':
        return <SourcesCanvas welcomeProjectName={welcomeProjectName} />
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
