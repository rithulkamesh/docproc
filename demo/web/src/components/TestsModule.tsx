import { Link } from 'react-router-dom'
import type { DocumentSummary } from '../types'
import { Button } from './Button'

interface TestsModuleProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
}

export function TestsModule(_props: TestsModuleProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      <div
        style={{
          fontSize: 'var(--text-xs)',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: 'var(--color-text-muted)',
        }}
      >
        Exam-style tests over your project.
      </div>
      <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', margin: 0 }}>
        Create an assessment; questions are AI-generated and graded on the server.
      </p>
      <section
        style={{
          border: '1px solid var(--color-border-strong)',
          backgroundColor: 'var(--color-bg)',
          padding: 'var(--space-md)',
        }}
      >
        <Link to="/assessments/create">
          <Button type="button">Create assessment</Button>
        </Link>
      </section>
    </div>
  )
}
