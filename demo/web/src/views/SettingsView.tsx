import { useEffect, useState } from 'react'
import { fetchStatus, type ApiStatus } from '@/api/status'
import { apiClient } from '@/api/client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

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
    <div className="mx-auto max-w-2xl p-6">
      <Card>
        <CardHeader>
          <CardTitle>Settings</CardTitle>
          <CardDescription>
            API connection and service status
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
