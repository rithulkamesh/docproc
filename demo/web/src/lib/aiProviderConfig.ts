/**
 * AI provider configuration persisted in localStorage.
 * Provider set matches docproc backend: openai, azure, anthropic, ollama, litellm.
 */

const STORAGE_KEY = 'docproc-ai-provider-config'

export type AIProviderId = 'openai' | 'azure' | 'anthropic' | 'ollama' | 'litellm'

export interface AIProviderConfig {
  provider: AIProviderId
  apiKey: string
  model: string
  /** OpenAI, Anthropic, Ollama, LiteLLM: optional custom base URL */
  baseUrl?: string
  /** Azure: endpoint URL (e.g. https://your-resource.openai.azure.com) */
  endpoint?: string
  /** Azure: chat deployment name */
  deployment?: string
  /** Azure: embedding deployment name */
  embeddingDeployment?: string
}

const DEFAULT_CONFIG: AIProviderConfig = {
  provider: 'openai',
  apiKey: '',
  model: 'gpt-4o-mini',
}

export const PROVIDER_LABELS: Record<AIProviderId, string> = {
  openai: 'OpenAI',
  azure: 'Azure OpenAI',
  anthropic: 'Anthropic',
  ollama: 'Ollama',
  litellm: 'LiteLLM',
}

/** Models offered per provider (subset; user can type custom model id). */
export const PROVIDER_MODELS: Record<AIProviderId, string[]> = {
  openai: [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-4',
    'gpt-3.5-turbo',
  ],
  azure: [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4-turbo',
    'gpt-4',
    'gpt-35-turbo',
  ],
  anthropic: [
    'claude-sonnet-4-20250514',
    'claude-3-5-sonnet-20241022',
    'claude-3-opus-20240229',
    'claude-3-haiku-20240307',
  ],
  ollama: [
    'llama3.1',
    'llama3.2',
    'mistral',
    'codellama',
    'phi',
  ],
  litellm: [
    'openai/gpt-4o',
    'anthropic/claude-3.5-sonnet',
    'ollama/llama3.1',
    'azure/gpt-4o',
  ],
}

const VALID_PROVIDER_IDS = new Set<string>(Object.keys(PROVIDER_LABELS))

function str(v: unknown): string {
  return typeof v === 'string' ? v : ''
}

function parseStored(raw: string | null): AIProviderConfig {
  if (!raw?.trim()) return { ...DEFAULT_CONFIG }
  try {
    const parsed = JSON.parse(raw) as Partial<AIProviderConfig>
    const provider = parsed.provider && VALID_PROVIDER_IDS.has(parsed.provider)
      ? (parsed.provider as AIProviderId)
      : DEFAULT_CONFIG.provider
    const models = PROVIDER_MODELS[provider]
    const model =
      typeof parsed.model === 'string' && parsed.model.trim()
        ? parsed.model.trim()
        : (models[0] ?? DEFAULT_CONFIG.model)
    return {
      provider,
      apiKey: str(parsed.apiKey) || DEFAULT_CONFIG.apiKey,
      model: models.includes(model) ? model : model,
      baseUrl: str(parsed.baseUrl) || undefined,
      endpoint: str(parsed.endpoint) || undefined,
      deployment: str(parsed.deployment) || undefined,
      embeddingDeployment: str(parsed.embeddingDeployment) || undefined,
    }
  } catch {
    return { ...DEFAULT_CONFIG }
  }
}

export function loadAIProviderConfig(): AIProviderConfig {
  if (typeof window === 'undefined') return { ...DEFAULT_CONFIG }
  return parseStored(window.localStorage.getItem(STORAGE_KEY))
}

/** Returns true if key exists in localStorage (user or backend has written config). */
export function hasStoredAIProviderConfig(): boolean {
  if (typeof window === 'undefined') return false
  return window.localStorage.getItem(STORAGE_KEY) != null
}

export function saveAIProviderConfig(config: Partial<AIProviderConfig>): void {
  if (typeof window === 'undefined') return
  const current = loadAIProviderConfig()
  const next: AIProviderConfig = {
    provider: (config.provider ?? current.provider) as AIProviderId,
    apiKey: config.apiKey !== undefined ? config.apiKey : current.apiKey,
    model: config.model !== undefined ? config.model : current.model,
    baseUrl: config.baseUrl !== undefined ? config.baseUrl : current.baseUrl,
    endpoint: config.endpoint !== undefined ? config.endpoint : current.endpoint,
    deployment: config.deployment !== undefined ? config.deployment : current.deployment,
    embeddingDeployment:
      config.embeddingDeployment !== undefined ? config.embeddingDeployment : current.embeddingDeployment,
  }
  const models = PROVIDER_MODELS[next.provider]
  // When switching provider without an explicit model, use first in list; otherwise keep (e.g. backend default)
  if (config.model === undefined && models.length && !models.includes(next.model)) {
    next.model = models[0]
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
}

export function getDefaultModelForProvider(provider: AIProviderId): string {
  const list = PROVIDER_MODELS[provider]
  return list[0] ?? 'default'
}

/**
 * Map backend primary_ai (from /status) to our provider id.
 * Backend uses openai, azure; ai_options include anthropic, ollama, litellm.
 */
export function providerIdFromBackend(primaryAi: string | null | undefined): AIProviderId {
  if (!primaryAi || typeof primaryAi !== 'string') return DEFAULT_CONFIG.provider
  const lower = primaryAi.toLowerCase()
  if (VALID_PROVIDER_IDS.has(lower)) return lower as AIProviderId
  return DEFAULT_CONFIG.provider
}
