import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { Toaster } from 'sonner'
import './App.css'
import { WorkspaceProvider, useWorkspace } from '@/context/WorkspaceContext'
import { AIProviderProvider } from '@/context/AIProviderContext'
import { AppShell } from '@/components/shell/AppShell'
import { CommandPalette } from '@/components/shell/CommandPalette'
import { KnowledgeCanvas } from '@/components/canvas/KnowledgeCanvas'
import { SettingsView } from '@/views/SettingsView'
import { OnboardingView, isOnboardingDone } from '@/views/OnboardingView'
import { AssessmentsListView } from '@/views/AssessmentsListView'
import { CreateAssessmentView } from '@/views/CreateAssessmentView'
import { TakeAssessmentView } from '@/views/TakeAssessmentView'
import { AssessmentResultView } from '@/views/AssessmentResultView'
import { AssessmentSubmissionsView } from '@/views/AssessmentSubmissionsView'
import { WorkspacesView } from '@/views/WorkspacesView'

function RequireOnboardingDone({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  if (!isOnboardingDone()) {
    return <Navigate to="/onboarding" state={{ from: location }} replace />
  }
  return <>{children}</>
}

function RequireProject({ children }: { children: React.ReactNode }) {
  const { projects } = useWorkspace()
  if (projects.length === 0) {
    return <Navigate to="/workspaces" replace />
  }
  return <>{children}</>
}

function App() {
  return (
    <WorkspaceProvider>
      <AIProviderProvider>
        <Routes>
        <Route path="/onboarding" element={<OnboardingView />} />
        <Route
          path="/"
          element={
            <RequireOnboardingDone>
              <RequireProject>
                <AppShell>
                  <KnowledgeCanvas />
                </AppShell>
                <CommandPalette />
              </RequireProject>
            </RequireOnboardingDone>
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
        <Route
          path="/workspaces"
          element={
            <AppShell fullBleed>
              <WorkspacesView />
            </AppShell>
          }
        />
      </Routes>
      <Toaster position="bottom-right" richColors closeButton />
      </AIProviderProvider>
    </WorkspaceProvider>
  )
}

export default App
