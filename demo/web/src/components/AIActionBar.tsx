import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { theme } from '../design/theme'
import { useWorkspace, type CanvasMode } from '../context/WorkspaceContext'
import { Button } from './Button'

interface ActionItem {
  id: string
  label: string
  onClick: () => void
  primary?: boolean
}

export function AIActionBar() {
  const { canvasMode, setCanvasMode } = useWorkspace()
  const navigate = useNavigate()

  const actionsByMode: Record<CanvasMode, ActionItem[]> = {
    home: [],
    converse: [],
    sources: [
      {
        id: 'add-doc',
        label: 'Add document',
        primary: true,
        onClick: () => {
          const input = document.getElementById('canvas-doc-upload') as HTMLInputElement | null
          input?.click()
        },
      },
    ],
    notes: [
      { id: 'add-section', label: 'Add section', onClick: () => setCanvasMode('notes') },
      { id: 'generate-summary', label: 'Generate summary', primary: true, onClick: () => setCanvasMode('notes') },
    ],
    tests: [
      {
        id: 'create-assessment',
        label: 'Create assessment',
        primary: true,
        onClick: () => navigate('/assessments/create'),
      },
    ],
  }

  const actions = actionsByMode[canvasMode]
  if (actions.length === 0) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: theme.motion.durationMicro / 1000, ease: theme.motion.easeFramer }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--space-md)',
        padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
        borderRadius: 'var(--radius-panel)',
        boxShadow: 'var(--shadow-card)',
        border: '1px solid var(--color-border-strong)',
        backgroundColor: 'var(--color-bg-alt)',
        marginBottom: 'var(--space-md)',
      }}
      role="toolbar"
      aria-label="Contextual actions"
    >
      {actions.map((action) => (
        <Button
          key={action.id}
          type="button"
          variant={action.primary ? 'primary' : 'ghost'}
          onClick={action.onClick}
        >
          {action.label}
        </Button>
      ))}
    </motion.div>
  )
}
