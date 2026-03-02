import { motion } from 'framer-motion'
import { theme } from '../design/theme'
import { useWorkspace } from '../context/WorkspaceContext'
import { SidebarNavigator } from './SidebarNavigator'
import { UtilityPanel } from './UtilityPanel'
import { ConverseCanvas } from './ConverseCanvas'
import { FlashcardsCanvas } from './FlashcardsCanvas'
import { TestsCanvas } from './TestsCanvas'

interface NotebookLayoutProps {
  /** When true, do not render SidebarNavigator (e.g. when used inside a route layout that already has it). */
  embedInLayout?: boolean
}

export function NotebookLayout({ embedInLayout }: NotebookLayoutProps = {}) {
  const { activePanel, setActivePanel } = useWorkspace()

  const mainContent = <ConverseCanvas />

  const panelContent =
    activePanel === 'flashcards' ? (
      <FlashcardsCanvas />
    ) : activePanel === 'tests' ? (
      <TestsCanvas />
    ) : null

  const utilityOpen = activePanel !== null

  return (
    <div
      style={{
        display: 'flex',
        flex: 1,
        minHeight: 0,
        overflow: 'hidden',
      }}
      className="notebook-layout"
    >
      {!embedInLayout && <SidebarNavigator />}

      <motion.main
        layout
        style={{
          flex: 1,
          minWidth: 0,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          maxWidth: theme.layout.mainMaxWidth,
          marginInline: 'auto',
          paddingInline: 'var(--space-lg)',
          paddingTop: '3rem',
          paddingBottom: '3rem',
          overflow: 'hidden',
        }}
        className="main-notebook-column"
      >
        {mainContent}
      </motion.main>

      <UtilityPanel
        isOpen={utilityOpen}
        onClose={() => setActivePanel(null)}
        title={
          activePanel === 'flashcards'
            ? 'Flashcards'
            : activePanel === 'tests'
              ? 'Tests'
              : undefined
        }
      >
        {panelContent}
      </UtilityPanel>
    </div>
  )
}
