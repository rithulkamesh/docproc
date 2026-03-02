import { useEffect, useState } from 'react'
import { fetchStatus, type ApiStatus } from '../api/status'
import { apiClient } from '../api/client'
import { Card } from '../components/Card'

export function SettingsView() {
  const [status, setStatus] = useState<ApiStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

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

  return (
    <div className="content-max">
      <Card>
        <div className="section-label mb-md">Settings</div>
        <p className="body-sm" style={{ marginTop: 0 }}>
          API base URL:{' '}
          <code className="code-snippet">{apiClient.baseUrl}</code>
        </p>
        {error && (
          <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{error}</p>
        )}
        {status && (
          <dl className="settings-dl">
            <dt>Status</dt>
            <dd>{status.ok ? 'Connected' : 'Error'}</dd>
            <dt>RAG backend</dt>
            <dd>{status.rag_backend ?? '—'}</dd>
            <dt>RAG configured</dt>
            <dd>{status.rag_configured ? 'Yes' : 'No'}</dd>
            <dt>Database</dt>
            <dd>{status.database_provider ?? '—'}</dd>
            <dt>Primary AI</dt>
            <dd>{status.primary_ai ?? '—'}</dd>
            <dt>Namespace</dt>
            <dd>{status.namespace ?? '—'}</dd>
          </dl>
        )}
      </Card>
    </div>
  )
}
