import { useEffect, useState } from 'react'
import { fetchStatus, type ApiStatus } from '@/api/status'
import { apiClient } from '@/api/client'
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { useWorkspace } from '@/context/WorkspaceContext'
import { useAIProvider } from '@/context/AIProviderContext'
import {
  type AIProviderId,
  PROVIDER_LABELS,
  PROVIDER_MODELS,
  providerIdFromBackend,
} from '@/lib/aiProviderConfig'
import { fetchAIConfig, saveAIConfig, type AIConfigFromServer } from '@/api/aiConfig'
import { THEME_LABELS, themeIdsByVariant } from '@/lib/themeStorage'
import { ThemePreview } from '@/components/ThemePreview'
import {
  loadUserPreferences,
  saveUserPreferences,
  dicebearAvatarUrl,
  initialsFromDisplayName,
  type UserPreferences,
  type AvatarStyle,
} from '@/lib/userPreferences'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'

const AVATAR_STYLE_LABELS: Record<AvatarStyle, string> = {
  avataaars: 'Avataaars',
  initials: 'Initials',
  lorelei: 'Lorelei',
}

function SettingsRow({
  id,
  label,
  description,
  children,
  className,
}: {
  id: string
  label: string
  description?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={className ?? 'py-4 first:pt-0 last:pb-0'}>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
        <div className="min-w-0 flex-1">
          <Label htmlFor={id} className="text-sm font-medium">
            {label}
          </Label>
          {description != null && description !== '' && (
            <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
          )}
        </div>
        <div className="mt-2 w-full sm:mt-0 sm:w-[280px] sm:shrink-0">{children}</div>
      </div>
    </div>
  )
}

export function SettingsView() {
  const { themeId, setThemeId } = useWorkspace()
  const { updateConfig } = useAIProvider()
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiKeyMasked, setApiKeyMasked] = useState(true)
  const [userPrefs, setUserPrefs] = useState<UserPreferences>(loadUserPreferences)

  // Server-stored AI config (keys never returned; source of truth for chat)
  const [serverAiConfig, setServerAiConfig] = useState<AIConfigFromServer | null>(null)
  const [serverAiConfigLoading, setServerAiConfigLoading] = useState(false)
  const [serverAiConfigError, setServerAiConfigError] = useState<string | null>(null)
  const [aiForm, setAiForm] = useState({
    provider: 'openai' as AIProviderId,
    model: 'gpt-4o-mini',
    baseUrl: '',
    endpoint: '',
    deployment: '',
    embeddingDeployment: '',
    apiKeyInput: '',
    apiKeyDirty: false,
  })
  const [aiSavePending, setAiSavePending] = useState(false)

  const updateUserPrefs = (partial: Partial<UserPreferences>) => {
    saveUserPreferences(partial)
    setUserPrefs(loadUserPreferences())
  }

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

  // Load server-stored AI config (used for chat; keys stored encrypted in DB)
  useEffect(() => {
    let cancelled = false
    setServerAiConfigLoading(true)
    setServerAiConfigError(null)
    fetchAIConfig()
      .then((data) => {
        if (!cancelled) {
          setServerAiConfig(data)
          setAiForm((prev) => ({
            ...prev,
            provider: (data.provider as AIProviderId) || 'openai',
            model: data.model || 'gpt-4o-mini',
            baseUrl: data.base_url ?? '',
            endpoint: data.endpoint ?? '',
            deployment: data.deployment ?? '',
            embeddingDeployment: data.embedding_deployment ?? '',
            apiKeyInput: '',
            apiKeyDirty: false,
          }))
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setServerAiConfigError(e instanceof Error ? e.message : 'Failed to load AI config')
        }
      })
      .finally(() => {
        if (!cancelled) setServerAiConfigLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const handleSaveAIConfig = async () => {
    setServerAiConfigError(null)
    setAiSavePending(true)
    try {
      const payload: Parameters<typeof saveAIConfig>[0] = {
        provider: aiForm.provider,
        model: aiForm.model,
        base_url: aiForm.baseUrl || undefined,
        endpoint: aiForm.endpoint || undefined,
        deployment: aiForm.deployment || undefined,
        embedding_deployment: aiForm.embeddingDeployment || undefined,
      }
      if (aiForm.apiKeyDirty) {
        payload.api_key = aiForm.apiKeyInput
      }
      const saved = await saveAIConfig(payload)
      setServerAiConfig(saved)
      setAiForm((prev) => ({ ...prev, apiKeyInput: '', apiKeyDirty: false }))
      updateConfig({
        provider: aiForm.provider,
        model: aiForm.model,
        baseUrl: aiForm.baseUrl || undefined,
        endpoint: aiForm.endpoint || undefined,
        deployment: aiForm.deployment || undefined,
        embeddingDeployment: aiForm.embeddingDeployment || undefined,
        apiKey: '', // so frontend never sends key; backend uses DB
      })
    } catch (e) {
      setServerAiConfigError(e instanceof Error ? e.message : 'Failed to save')
    } finally {
      setAiSavePending(false)
    }
  }

  const models = PROVIDER_MODELS[aiForm.provider]
  const showCustomModel = aiForm.model && !models.includes(aiForm.model)
  const backendProvider = status ? providerIdFromBackend(status.primary_ai ?? undefined) : null
  const backendModel = status?.default_rag_model?.trim() ?? null
  const backendEmbeddingDeployment = status?.embedding_deployment?.trim() ?? null
  const backendHasAiFromEnv = Boolean(status?.rag_configured && backendProvider)

  return (
    <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Manage your profile, AI provider, appearance, and connection to the backend.
        </p>
      </div>

      <Tabs defaultValue="profile" className="w-full">
        <TabsList className="mb-6 h-auto w-full justify-start gap-0 rounded-none border-b border-border bg-transparent p-0">
          <TabsTrigger
            value="profile"
            className="rounded-none border-b-2 border-transparent px-4 py-3 text-sm font-medium data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none"
          >
            Profile
          </TabsTrigger>
          <TabsTrigger
            value="ai"
            className="rounded-none border-b-2 border-transparent px-4 py-3 text-sm font-medium data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none"
          >
            AI provider
          </TabsTrigger>
          <TabsTrigger
            value="appearance"
            className="rounded-none border-b-2 border-transparent px-4 py-3 text-sm font-medium data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none"
          >
            Appearance
          </TabsTrigger>
          <TabsTrigger
            value="api"
            className="rounded-none border-b-2 border-transparent px-4 py-3 text-sm font-medium data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:shadow-none"
          >
            API & data
          </TabsTrigger>
        </TabsList>

        <TabsContent value="profile" className="mt-0">
          <section>
            <h2 className="text-lg font-medium">Profile</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Display name and avatar shown in the app bar.
            </p>
            <Separator className="my-4" />
            <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
              <Avatar className="h-20 w-20 shrink-0">
                <AvatarImage
                  src={dicebearAvatarUrl(userPrefs.displayName || userPrefs.avatarSeed || 'user', userPrefs.avatarStyle)}
                  alt=""
                />
                <AvatarFallback className="text-xl">
                  {initialsFromDisplayName(userPrefs.displayName)}
                </AvatarFallback>
              </Avatar>
              <div className="min-w-0 flex-1 space-y-0">
                <SettingsRow
                  id="settings-display-name"
                  label="Display name"
                  description="Shown in the app bar and as avatar fallback initials."
                >
                  <Input
                    id="settings-display-name"
                    placeholder="Your name"
                    value={userPrefs.displayName}
                    onChange={(e) => updateUserPrefs({ displayName: e.target.value })}
                  />
                </SettingsRow>
                <Separator />
                <SettingsRow id="settings-avatar-style" label="Avatar style">
                  <Select
                    value={userPrefs.avatarStyle}
                    onValueChange={(v) => updateUserPrefs({ avatarStyle: v as AvatarStyle })}
                  >
                    <SelectTrigger id="settings-avatar-style">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(Object.keys(AVATAR_STYLE_LABELS) as AvatarStyle[]).map((id) => (
                        <SelectItem key={id} value={id}>
                          {AVATAR_STYLE_LABELS[id]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </SettingsRow>
              </div>
            </div>
          </section>
        </TabsContent>

        <TabsContent value="ai" className="mt-0">
          <section>
            <h2 className="text-lg font-medium">AI provider</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Configure the AI provider and model for study chat. Stored securely on the server
              (encrypted in the database). API keys are never sent back to the browser.
            </p>
            <Separator className="my-4" />
            {serverAiConfigError && (
              <p className="mb-4 text-sm text-destructive">{serverAiConfigError}</p>
            )}
            {serverAiConfigLoading && (
              <p className="mb-4 text-sm text-muted-foreground">Loading saved config…</p>
            )}
            <div className="space-y-0">
              {backendHasAiFromEnv && !serverAiConfigLoading && (
                <>
                  <div className="rounded-lg border border-border bg-muted/30 px-4 py-3">
                    <h3 className="text-sm font-medium">Fallback from server .env</h3>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      When no key is saved below, the server can use these env values. Save your own
                      config to use the database (recommended).
                    </p>
                    <dl className="mt-3 grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
                      <div>
                        <dt className="text-xs font-medium text-muted-foreground">Provider</dt>
                        <dd className="font-medium">
                          {PROVIDER_LABELS[backendProvider ?? 'openai']}
                        </dd>
                      </div>
                      {backendModel && (
                        <div>
                          <dt className="text-xs font-medium text-muted-foreground">Chat model</dt>
                          <dd className="font-mono text-xs">{backendModel}</dd>
                        </div>
                      )}
                      {backendEmbeddingDeployment && backendProvider === 'azure' && (
                        <div>
                          <dt className="text-xs font-medium text-muted-foreground">
                            Embedding deployment
                          </dt>
                          <dd className="font-mono text-xs">{backendEmbeddingDeployment}</dd>
                        </div>
                      )}
                    </dl>
                    <div className="mt-3">
                      <button
                        type="button"
                        onClick={() => {
                          const provider = (backendProvider ?? aiForm.provider) as AIProviderId
                          const model =
                            backendModel?.trim() || PROVIDER_MODELS[provider][0]
                          setAiForm((prev) => ({
                            ...prev,
                            provider,
                            model,
                            embeddingDeployment:
                              provider === 'azure' && backendEmbeddingDeployment
                                ? backendEmbeddingDeployment
                                : prev.embeddingDeployment,
                          }))
                        }}
                        className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:opacity-90"
                      >
                        Copy to form
                      </button>
                    </div>
                  </div>
                  <Separator className="my-4" />
                </>
              )}
              <SettingsRow id="settings-provider" label="Provider">
                <Select
                  value={aiForm.provider}
                  onValueChange={(v) =>
                    setAiForm((prev) => ({
                      ...prev,
                      provider: v as AIProviderId,
                      model: PROVIDER_MODELS[v as AIProviderId]?.[0] ?? prev.model,
                    }))
                  }
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
              </SettingsRow>
              <Separator />

              {aiForm.provider === 'azure' && (
                <>
                  <SettingsRow
                    id="settings-azure-endpoint"
                    label="Azure endpoint"
                    description="Your Azure OpenAI resource URL. Required when using your own key."
                  >
                    <Input
                      id="settings-azure-endpoint"
                      placeholder="https://your-resource.openai.azure.com"
                      value={aiForm.endpoint}
                      onChange={(e) =>
                        setAiForm((prev) => ({ ...prev, endpoint: e.target.value.trim() }))
                      }
                      className="font-mono text-sm"
                    />
                  </SettingsRow>
                  <Separator />
                  <SettingsRow id="settings-azure-deployment" label="Chat deployment">
                    <Input
                      id="settings-azure-deployment"
                      placeholder="gpt-4o"
                      value={aiForm.deployment}
                      onChange={(e) =>
                        setAiForm((prev) => ({ ...prev, deployment: e.target.value.trim() }))
                      }
                      className="font-mono text-sm"
                    />
                  </SettingsRow>
                  <Separator />
                  <SettingsRow
                    id="settings-azure-embedding"
                    label="Embedding deployment"
                    description="Used for RAG indexing when using Azure. Backend may use env if unset."
                  >
                    <Input
                      id="settings-azure-embedding"
                      placeholder="text-embedding-ada-002"
                      value={aiForm.embeddingDeployment}
                      onChange={(e) =>
                        setAiForm((prev) => ({
                          ...prev,
                          embeddingDeployment: e.target.value.trim(),
                        }))
                      }
                      className="font-mono text-sm"
                    />
                  </SettingsRow>
                  <Separator />
                </>
              )}

              {(aiForm.provider === 'openai' || aiForm.provider === 'anthropic') && (
                <>
                  <SettingsRow
                    id="settings-base-url"
                    label="Base URL (optional)"
                    description="Override the API base URL (e.g. proxy or custom endpoint). Leave empty for default."
                  >
                    <Input
                      id="settings-base-url"
                      placeholder={
                        aiForm.provider === 'openai' ? 'https://api.openai.com/v1' : ''
                      }
                      value={aiForm.baseUrl}
                      onChange={(e) =>
                        setAiForm((prev) => ({ ...prev, baseUrl: e.target.value.trim() }))
                      }
                      className="font-mono text-sm"
                    />
                  </SettingsRow>
                  <Separator />
                </>
              )}

              {(aiForm.provider === 'ollama' || aiForm.provider === 'litellm') && (
                <>
                  <SettingsRow
                    id="settings-base-url-ollama"
                    label={
                      aiForm.provider === 'ollama'
                        ? 'Base URL (e.g. http://localhost:11434)'
                        : 'Base URL'
                    }
                    description={
                      aiForm.provider === 'ollama'
                        ? 'Ollama API URL. Default is http://localhost:11434.'
                        : 'LiteLLM proxy or server URL.'
                    }
                  >
                    <Input
                      id="settings-base-url-ollama"
                      placeholder={
                        aiForm.provider === 'ollama'
                          ? 'http://localhost:11434'
                          : 'http://localhost:4000'
                      }
                      value={aiForm.baseUrl}
                      onChange={(e) =>
                        setAiForm((prev) => ({ ...prev, baseUrl: e.target.value.trim() }))
                      }
                      className="font-mono text-sm"
                    />
                  </SettingsRow>
                  <Separator />
                </>
              )}

              <SettingsRow
                id="settings-apikey"
                label="API key"
                description={
                  serverAiConfig?.api_key_configured
                    ? 'Stored on server. Enter a new key to change, or leave blank to keep current.'
                    : 'Stored encrypted on the server. Required for chat unless using server .env fallback.'
                }
              >
                <div className="flex gap-2">
                  <Input
                    id="settings-apikey"
                    type={apiKeyMasked ? 'password' : 'text'}
                    placeholder={
                      serverAiConfig?.api_key_configured && !aiForm.apiKeyDirty
                        ? '•••••••• (leave blank to keep)'
                        : 'sk-…'
                    }
                    value={aiForm.apiKeyInput}
                    onChange={(e) =>
                      setAiForm((prev) => ({
                        ...prev,
                        apiKeyInput: e.target.value,
                        apiKeyDirty: true,
                      }))
                    }
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
              </SettingsRow>
              <Separator />
              <SettingsRow id="settings-model" label="Model">
                <div className="space-y-2">
                  <Select
                    value={aiForm.model || models[0]}
                    onValueChange={(v) => setAiForm((prev) => ({ ...prev, model: v }))}
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
                        <SelectItem value={aiForm.model}>{aiForm.model} (custom)</SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                  {showCustomModel && (
                    <Input
                      className="font-mono text-sm"
                      placeholder="Custom model ID"
                      value={aiForm.model}
                      onChange={(e) =>
                        setAiForm((prev) => ({ ...prev, model: e.target.value }))
                      }
                    />
                  )}
                </div>
              </SettingsRow>
              <div className="pt-4">
                <button
                  type="button"
                  onClick={() => void handleSaveAIConfig()}
                  disabled={aiSavePending}
                  className="rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
                >
                  {aiSavePending ? 'Saving…' : 'Save AI config'}
                </button>
              </div>
            </div>
          </section>
        </TabsContent>

        <TabsContent value="appearance" className="mt-0">
          <section>
            <h2 className="text-lg font-medium">Appearance</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Choose a theme for background, text, and accent colors. Saved automatically.
            </p>
            <Separator className="my-4" />
            <div className="space-y-6">
              <div>
                <h3 className="mb-3 text-sm font-medium text-muted-foreground">Light</h3>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
                  {themeIdsByVariant().light.map((id) => (
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
              <Separator />
              <div>
                <h3 className="mb-3 text-sm font-medium text-muted-foreground">Dark</h3>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
                  {themeIdsByVariant().dark.map((id) => (
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
            </div>
          </section>
        </TabsContent>

        <TabsContent value="api" className="mt-0">
          <section>
            <h2 className="text-lg font-medium">API & data</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Connection to the docproc backend and current configuration.
            </p>
            <Separator className="my-4" />
            <div className="space-y-4">
              <div>
                <Label className="text-sm font-medium text-muted-foreground">API base URL</Label>
                <p className="mt-1 font-mono text-sm">
                  <code className="rounded bg-muted px-1.5 py-0.5">{apiClient.baseUrl}</code>
                </p>
              </div>
              {error && (
                <p className="text-sm text-destructive">{error}</p>
              )}
              {status && (
                <>
                  <Separator />
                  <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div className="flex flex-col gap-0.5">
                      <dt className="text-xs font-medium text-muted-foreground">Status</dt>
                      <dd>
                        <Badge variant={status.ok ? 'success' : 'destructive'}>
                          {status.ok ? 'Connected' : 'Error'}
                        </Badge>
                      </dd>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      <dt className="text-xs font-medium text-muted-foreground">RAG backend</dt>
                      <dd className="text-sm">{status.rag_backend ?? '—'}</dd>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      <dt className="text-xs font-medium text-muted-foreground">RAG configured</dt>
                      <dd className="text-sm">{status.rag_configured ? 'Yes' : 'No'}</dd>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      <dt className="text-xs font-medium text-muted-foreground">Database</dt>
                      <dd className="text-sm">{status.database_provider ?? '—'}</dd>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      <dt className="text-xs font-medium text-muted-foreground">Primary AI</dt>
                      <dd className="text-sm">{status.primary_ai ?? '—'}</dd>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      <dt className="text-xs font-medium text-muted-foreground">Namespace</dt>
                      <dd className="text-sm">{status.namespace ?? '—'}</dd>
                    </div>
                  </dl>
                </>
              )}
            </div>
          </section>
        </TabsContent>
      </Tabs>
    </div>
  )
}
