import { useEffect, useState } from 'react'
import { fetchStatus, type ApiStatus } from '@/api/status'
import { apiClient } from '@/api/client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useWorkspace } from '@/context/WorkspaceContext'
import { useAIProvider } from '@/context/AIProviderContext'
import {
  type AIProviderId,
  PROVIDER_LABELS,
  PROVIDER_MODELS,
  providerIdFromBackend,
} from '@/lib/aiProviderConfig'
import { THEME_LABELS, themeIdsByVariant } from '@/lib/themeStorage'
import { ThemePreview } from '@/components/ThemePreview'

export function SettingsView() {
  const { themeId, setThemeId } = useWorkspace()
  const { config, setProvider, setApiKey, setModel, updateConfig } = useAIProvider()
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiKeyMasked, setApiKeyMasked] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchStatus()
        setStatus(data)
        setError(null)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to fetch status')
      }
    }
    void load()
  }, [])

  const models = PROVIDER_MODELS[config.provider]
  const showCustomModel = config.model && !models.includes(config.model)
  const backendProvider = status ? providerIdFromBackend(status.primary_ai ?? undefined) : null
  const backendModel = status?.default_rag_model ?? null
  const canLoadFromBackend = status && (backendProvider || backendModel)

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      {/* AI Provider */}
      <Card>
        <CardHeader>
          <CardTitle>AI provider</CardTitle>
          <CardDescription>
            Configure the AI provider and model for study chat. Stored in your browser and sent with each request so the backend can use your key and model for chat; backend env is used for RAG indexing and when no key is set. Defaults are loaded from the backend when nothing is saved.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {canLoadFromBackend && (
            <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-sm">
              <span className="text-muted-foreground">Backend default:</span>
              <span className="font-medium">
                {PROVIDER_LABELS[backendProvider ?? 'openai']}
                {backendModel && ` · ${backendModel}`}
              </span>
              <button
                type="button"
                onClick={() => {
                  const provider = backendProvider ?? config.provider
                  const model = backendModel?.trim() || PROVIDER_MODELS[provider][0]
                  updateConfig({ provider, model })
                }}
                className="ml-auto rounded bg-primary px-2 py-1 text-xs font-medium text-primary-foreground hover:opacity-90"
              >
                Use backend default
              </button>
            </div>
          )}
          <div className="space-y-2">
            <Label htmlFor="settings-provider">Provider</Label>
            <Select
              value={config.provider}
              onValueChange={(v) => setProvider(v as AIProviderId)}
            >
              <SelectTrigger id="settings-provider">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(PROVIDER_LABELS) as AIProviderId[]).map((id) => (
                  <SelectItem key={id} value={id}>
                    {PROVIDER_LABELS[id]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="settings-apikey">API key</Label>
            <div className="flex gap-2">
              <Input
                id="settings-apikey"
                type={apiKeyMasked ? 'password' : 'text'}
                placeholder="sk-…"
                value={config.apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="font-mono text-sm"
                autoComplete="off"
              />
              <button
                type="button"
                onClick={() => setApiKeyMasked((m) => !m)}
                className="rounded-md border border-input bg-transparent px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              >
                {apiKeyMasked ? 'Show' : 'Hide'}
              </button>
            </div>
            <p className="text-xs text-muted-foreground">
              When set, the backend uses this key for chat on each request (with the model above). RAG indexing and retrieval use backend env (OPENAI_API_KEY or AZURE_OPENAI_*). Leave empty to use server defaults only.
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="settings-model">Model</Label>
            <Select
              value={config.model || models[0]}
              onValueChange={setModel}
            >
              <SelectTrigger id="settings-model">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
                {showCustomModel && (
                  <SelectItem value={config.model}>{config.model} (custom)</SelectItem>
                )}
              </SelectContent>
            </Select>
            {showCustomModel && (
              <Input
                className="mt-2 font-mono text-sm"
                placeholder="Custom model ID"
                value={config.model}
                onChange={(e) => setModel(e.target.value)}
              />
            )}
          </div>
        </CardContent>
      </Card>

      {/* Theme */}
      <Card>
        <CardHeader>
          <CardTitle>Appearance</CardTitle>
          <CardDescription>
            Choose a theme for background, text, and accent colors. Previews show the palette. Saved automatically.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {(() => {
            const { light, dark } = themeIdsByVariant()
            return (
              <>
                <div>
                  <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Light</h3>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
                    {light.map((id) => (
                      <ThemePreview
                        key={id}
                        themeId={id}
                        label={THEME_LABELS[id]}
                        selected={themeId === id}
                        onClick={() => setThemeId(id)}
                        className="min-w-0"
                      />
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Dark</h3>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
                    {dark.map((id) => (
                      <ThemePreview
                        key={id}
                        themeId={id}
                        label={THEME_LABELS[id]}
                        selected={themeId === id}
                        onClick={() => setThemeId(id)}
                        className="min-w-0"
                      />
                    ))}
                  </div>
                </div>
              </>
            )
          })()}
        </CardContent>
      </Card>

      {/* API status (existing) */}
      <Card>
        <CardHeader>
          <CardTitle>API status</CardTitle>
          <CardDescription>
            Connection to the docproc backend
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            API base URL:{' '}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
              {apiClient.baseUrl}
            </code>
          </p>
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
          {status && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="font-medium text-muted-foreground">Status</dt>
              <dd>
                <Badge variant={status.ok ? 'success' : 'destructive'}>
                  {status.ok ? 'Connected' : 'Error'}
                </Badge>
              </dd>
              <dt className="font-medium text-muted-foreground">RAG backend</dt>
              <dd>{status.rag_backend ?? '—'}</dd>
              <dt className="font-medium text-muted-foreground">RAG configured</dt>
              <dd>{status.rag_configured ? 'Yes' : 'No'}</dd>
              <dt className="font-medium text-muted-foreground">Database</dt>
              <dd>{status.database_provider ?? '—'}</dd>
              <dt className="font-medium text-muted-foreground">Primary AI</dt>
              <dd>{status.primary_ai ?? '—'}</dd>
              <dt className="font-medium text-muted-foreground">Namespace</dt>
              <dd>{status.namespace ?? '—'}</dd>
            </dl>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
