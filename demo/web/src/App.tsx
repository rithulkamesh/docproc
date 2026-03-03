import { Routes, Route } from 'react-router-dom'
import './App.css'
import { WorkspaceProvider } from '@/context/WorkspaceContext'
import { AppShell } from '@/components/shell/AppShell'
import { CommandPalette } from '@/components/shell/CommandPalette'
import { KnowledgeCanvas } from '@/components/canvas/KnowledgeCanvas'
import { SettingsView } from '@/views/SettingsView'
import { AssessmentsListView } from '@/views/AssessmentsListView'
import { CreateAssessmentView } from '@/views/CreateAssessmentView'
import { TakeAssessmentView } from '@/views/TakeAssessmentView'
import { AssessmentResultView } from '@/views/AssessmentResultView'
import { AssessmentSubmissionsView } from '@/views/AssessmentSubmissionsView'

function App() {
  return (
    <WorkspaceProvider>
      <Routes>
        <Route
          path="/"
          element={
            <>
              <AppShell>
                <KnowledgeCanvas />
              </AppShell>
              <CommandPalette />
            </>
          }
        />
        <Route
          path="/settings"
          element={
            <AppShell fullBleed>
              <SettingsView />
            </AppShell>
          }
        />
        <Route
          path="/assessments/create"
          element={
            <AppShell fullBleed>
              <CreateAssessmentView />
            </AppShell>
          }
        />
        <Route
          path="/assessments/:id/submissions"
          element={
            <AppShell fullBleed>
              <AssessmentSubmissionsView />
            </AppShell>
          }
        />
        <Route
          path="/assessments/:id/take"
          element={
            <AppShell fullBleed>
              <TakeAssessmentView />
            </AppShell>
          }
        />
        <Route
          path="/assessments/:id/result/:submissionId"
          element={
            <AppShell fullBleed>
              <AssessmentResultView />
            </AppShell>
          }
        />
        <Route
          path="/assessments"
          element={
            <AppShell fullBleed>
              <AssessmentsListView />
            </AppShell>
          }
        />
      </Routes>
    </WorkspaceProvider>
  )
}

export default App
