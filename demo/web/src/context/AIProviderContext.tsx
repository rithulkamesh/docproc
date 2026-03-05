import type { ReactNode } from 'react'
import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import {
  loadAIProviderConfig,
  saveAIProviderConfig,
  hasStoredAIProviderConfig,
  providerIdFromBackend,
  type AIProviderConfig,
  type AIProviderId,
  getDefaultModelForProvider,
} from '@/lib/aiProviderConfig'
import { fetchStatus } from '@/api/status'

interface AIProviderContextValue {
  config: AIProviderConfig
  setProvider: (provider: AIProviderId) => void
  setApiKey: (apiKey: string) => void
  setModel: (model: string) => void
  /** Persist partial update (e.g. after editing in Settings). */
  updateConfig: (partial: Partial<AIProviderConfig>) => void
}

const AIProviderContext = createContext<AIProviderContextValue | null>(null)

export function useAIProvider() {
  const ctx = useContext(AIProviderContext)
  if (!ctx) throw new Error('useAIProvider must be used within AIProviderProvider')
  return ctx
}

interface AIProviderProviderProps {
  children: ReactNode
}

export function AIProviderProvider({ children }: AIProviderProviderProps) {
  const [config, setConfigState] = useState<AIProviderConfig>(loadAIProviderConfig)

  const refresh = useCallback(() => {
    setConfigState(loadAIProviderConfig())
  }, [])

  // When no config is stored, load backend defaults from /status (non-secret values only; keys stay on server)
  useEffect(() => {
    if (hasStoredAIProviderConfig()) return
    let cancelled = false
    fetchStatus()
      .then((status) => {
        if (cancelled) return
        const provider = providerIdFromBackend(status.primary_ai ?? undefined)
        const model =
          typeof status.default_rag_model === 'string' && status.default_rag_model.trim()
            ? status.default_rag_model.trim()
            : getDefaultModelForProvider(provider)
        const embeddingDeployment =
          provider === 'azure' && status.embedding_deployment?.trim()
            ? status.embedding_deployment.trim()
            : undefined
        saveAIProviderConfig({ provider, model, ...(embeddingDeployment && { embeddingDeployment }) })
        setConfigState(loadAIProviderConfig())
      })
      .catch(() => { /* keep app default when backend unreachable */ })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'docproc-ai-provider-config') refresh()
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [refresh])

  const updateConfig = useCallback((partial: Partial<AIProviderConfig>) => {
    saveAIProviderConfig(partial)
    setConfigState(loadAIProviderConfig())
  }, [])

  const setProvider = useCallback((provider: AIProviderId) => {
    const nextModel = getDefaultModelForProvider(provider)
    saveAIProviderConfig({ provider, model: nextModel })
    setConfigState(loadAIProviderConfig())
  }, [])

  const setApiKey = useCallback((apiKey: string) => {
    saveAIProviderConfig({ apiKey })
    setConfigState(loadAIProviderConfig())
  }, [])

  const setModel = useCallback((model: string) => {
    saveAIProviderConfig({ model })
    setConfigState(loadAIProviderConfig())
  }, [])

  const value: AIProviderContextValue = {
    config,
    setProvider,
    setApiKey,
    setModel,
    updateConfig,
  }

  return (
    <AIProviderContext.Provider value={value}>
      {children}
    </AIProviderContext.Provider>
  )
}
