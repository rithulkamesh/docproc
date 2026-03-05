import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { fetchStatus } from '@/api/status'
import { useAIProvider } from '@/context/AIProviderContext'
import {
  providerIdFromBackend,
  getDefaultModelForProvider,
  PROVIDER_LABELS,
} from '@/lib/aiProviderConfig'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const ONBOARDING_DONE_KEY = 'docproc-onboarding-done'

export function isOnboardingDone(): boolean {
  if (typeof window === 'undefined') return true
  return window.localStorage.getItem(ONBOARDING_DONE_KEY) === 'true'
}

export function setOnboardingDone(): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(ONBOARDING_DONE_KEY, 'true')
}

export function OnboardingView() {
  const navigate = useNavigate()
  const { updateConfig } = useAIProvider()
  const [step, setStep] = useState(1)
  const [status, setStatus] = useState<Awaited<ReturnType<typeof fetchStatus>> | null>(null)

  useEffect(() => {
    fetchStatus()
      .then(setStatus)
      .catch(() => setStatus(null))
  }, [])

  const backendProvider = status ? providerIdFromBackend(status.primary_ai ?? undefined) : null
  const backendModel = status?.default_rag_model?.trim() ?? null
  const canUseBackendDefault = status && (backendProvider || backendModel)

  const handleUseBackendDefault = () => {
    if (!backendProvider) return
    const model = backendModel || getDefaultModelForProvider(backendProvider)
    updateConfig({ provider: backendProvider, model })
  }

  const handleFinish = () => {
    setOnboardingDone()
    navigate('/', { replace: true })
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background p-4">
      <div className="w-full max-w-lg space-y-6">
        {step === 1 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Welcome to docproc // edu</CardTitle>
              <CardDescription>
                Study with AI-powered chat over your documents. Upload PDFs, ask questions, take
                notes, and run practice tests. Configure your AI provider next so you can start
                chatting.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={() => setStep(2)} className="w-full">
                Get started
              </Button>
            </CardContent>
          </Card>
        )}

        {step === 2 && (
          <Card>
            <CardHeader>
              <CardTitle>Set up AI</CardTitle>
              <CardDescription>
                Choose a provider and API key for study chat. You can change this anytime in
                Settings.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {canUseBackendDefault && (
                <div className="rounded-md border border-border bg-muted/40 p-3 text-sm">
                  <p className="font-medium">Use server default</p>
                  <p className="text-muted-foreground text-xs mt-1">
                    {PROVIDER_LABELS[backendProvider ?? 'openai']}
                    {backendModel && ` · ${backendModel}`}
                  </p>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="mt-2"
                    onClick={handleUseBackendDefault}
                  >
                    Use backend default
                  </Button>
                </div>
              )}
              <p className="text-sm text-muted-foreground">
                To use your own key (OpenAI, Azure, Anthropic, etc.), go to Settings and set your
                provider and API key.
              </p>
              <div className="flex gap-2">
                <Button variant="outline" asChild className="flex-1">
                  <Link to="/settings">Open Settings</Link>
                </Button>
                <Button onClick={handleFinish} className="flex-1">
                  Continue
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
