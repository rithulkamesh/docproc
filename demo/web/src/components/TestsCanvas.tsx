import { Link } from 'react-router-dom'
import { theme } from '../design/theme'
import { SoftButton } from './SoftButton'

export function TestsCanvas() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
      <p style={{ fontSize: 'var(--text-sm)', lineHeight: theme.lineHeight.body, color: 'var(--color-text-muted)', margin: 0 }}>
        Create an assessment from your documents. Questions are AI-generated; submit for grading.
      </p>
      <section
        style={{
          border: 'var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          padding: 'var(--space-lg)',
          backgroundColor: 'var(--color-bg-alt)',
          maxWidth: '60ch',
          marginInline: 'auto',
        }}
      >
        <Link to="/assessments/create" style={{ textDecoration: 'none' }}>
          <SoftButton>Create assessment</SoftButton>
        </Link>
      </section>
    </div>
  )
}
