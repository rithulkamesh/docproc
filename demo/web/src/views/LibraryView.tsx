import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import type { DocumentDetail } from '../types'
import { getDocument } from '../api/documents'
import { Button } from '../components/Button'
import { Card } from '../components/Card'

interface LibraryViewProps {
  selectedDocumentId: string | null
}

export function LibraryView({ selectedDocumentId }: LibraryViewProps) {
  const [detail, setDetail] = useState<DocumentDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!selectedDocumentId) {
      setDetail(null)
      return
    }
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const doc = await getDocument(selectedDocumentId)
        setDetail(doc)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load document')
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [selectedDocumentId])

  if (!selectedDocumentId) {
    return (
      <Card>
        <div className="section-label mb-sm">LIBRARY</div>
        <p className="body-sm" style={{ marginTop: 0 }}>Select a document in the sidebar to view its full content and generate quizzes.</p>
      </Card>
    )
  }

  return (
    <div className="form-stack" style={{ gap: 'var(--space-lg)' }}>
      <Card>
        <div className="section-label mb-sm">LIBRARY</div>
        {loading && <p className="body-sm">Loading document…</p>}
        {error && <p className="body-sm" style={{ color: 'var(--color-danger)' }}>{error}</p>}
        {detail && !loading && !error && (
          <>
            <h2 className="heading-lg mb-xs">{detail.filename ?? 'Document'}</h2>
            <div className="text-xs text-muted mb-md">{detail.pages ?? 0} pages · ID: {detail.id}</div>
            <div
              className="code-snippet body-sm"
              style={{ display: 'block', padding: 'var(--space-md)', maxHeight: 360, overflowY: 'auto', whiteSpace: 'pre-wrap' }}
            >
              {detail.full_text || 'No full text available.'}
            </div>
          </>
        )}
      </Card>

      {detail && (
        <Card>
          <div className="section-label mb-sm">ASSESSMENT</div>
          <p className="body-sm mt-0 mb-md">Create an assessment from this document. Questions are AI-generated and graded on the server.</p>
          <Link to="/assessments/create" state={{ documentId: selectedDocumentId }}>
            <Button type="button">Create assessment</Button>
          </Link>
        </Card>
      )}
    </div>
  )
}
