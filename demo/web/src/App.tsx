import { Routes, Route, useLocation } from 'react-router-dom'
import './App.css'
import { LayoutShell } from './components/LayoutShell'
import { NotebookLayout } from './components/NotebookLayout'
import { CommandPalette } from './components/CommandPalette'
import { SidebarNavigator } from './components/SidebarNavigator'
import { NotesShell } from './views/NotesShell'
import { SettingsView } from './views/SettingsView'
import { AssessmentsListView } from './views/AssessmentsListView'
import { CreateAssessmentView } from './views/CreateAssessmentView'
import { TakeAssessmentView } from './views/TakeAssessmentView'
import { AssessmentResultView } from './views/AssessmentResultView'
import { AssessmentSubmissionsView } from './views/AssessmentSubmissionsView'
import { WorkspaceProvider } from './context/WorkspaceContext'

function WorkspaceLayout() {
  const { pathname } = useLocation()
  const isNotes = pathname === '/notes' || pathname.startsWith('/notes/')
  return (
    <div style={{ display: 'flex', flex: 1, minHeight: 0, position: 'relative' }}>
      <SidebarNavigator />
      {isNotes ? (
        <NotesShell />
      ) : (
        <>
          <NotebookLayout embedInLayout />
          <CommandPalette />
        </>
      )}
    </div>
  )
}

function App() {
  return (
    <WorkspaceProvider>
      <LayoutShell>
        <Routes>
          <Route path="/" element={<WorkspaceLayout />} />
          <Route path="/notes/*" element={<WorkspaceLayout />} />
          <Route path="/settings" element={<SettingsView />} />
          <Route path="/assessments/create" element={<CreateAssessmentView />} />
          <Route path="/assessments/:id/submissions" element={<AssessmentSubmissionsView />} />
          <Route path="/assessments/:id/take" element={<TakeAssessmentView />} />
          <Route path="/assessments/:id/result/:submissionId" element={<AssessmentResultView />} />
          <Route path="/assessments" element={<AssessmentsListView />} />
        </Routes>
      </LayoutShell>
    </WorkspaceProvider>
  )
}

export default App
