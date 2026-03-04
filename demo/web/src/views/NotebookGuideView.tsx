import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { runQuery } from '../api/query'
import type { DocumentSummary, RagSource } from '../types'
import { Button } from '../components/Button'
import { Card } from '../components/Card'

interface NotebookGuideViewProps {
  documents: DocumentSummary[]
}

interface QuickReport {
  id: string
  label: string
  description: string
  prompt: string
}

const QUICK_REPORTS: QuickReport[] = [
  {
    id: 'faq',
    label: 'FAQ',
    description: 'Key questions and answers over this corpus.',
    prompt:
      'You are an expert tutor. Create an FAQ (8–12 Q&A pairs) that covers the most important concepts in this corpus of documents. Use clear section headings and concise answers.',
  },
  {
    id: 'study',
    label: 'Study Guide',
    description: 'Hierarchical outline for revision.',
    prompt:
      'Produce a structured study guide for this corpus. Use markdown headings and bullet points. Start with 3–5 big sections, each with key ideas and definitions a student should remember.',
  },
  {
    id: 'briefing',
    label: 'Briefing Doc',
    description: 'One-page executive summary.',
    prompt:
      'Write a concise briefing document summarizing this corpus. Assume the reader is a busy professor. Use short sections: Context, Main Findings, Implications, Open Questions.',
  },
]

export function NotebookGuideView({ documents }: NotebookGuideViewProps) {
  const [summary, setSummary] = useState<string>('')
  const [summarySources, setSummarySources] = useState<RagSource[]>([])
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [activeReportId, setActiveReportId] = useState<string | null>(null)
  const [reportContent, setReportContent] = useState<string | null>(null)
  const [reportSources, setReportSources] = useState<RagSource[]>([])
  const [reportError, setReportError] = useState<string | null>(null)

  const navigate = useNavigate()

  useEffect(() => {
    if (!documents.some((d) => d.status === 'completed')) {
      setSummary('')
      setSummarySources([])
      return
    }
    if (summary) return

    const load = async () => {
      try {
        setLoadingSummary(true)
        setError(null)
        const res = await runQuery(
          'You are summarizing a notebook of documents for a student. Provide 3–6 bullet points capturing the main themes, topics, and takeaways across all documents. Focus on what a learner should remember.',
          5,
        )
        setSummary(res.answer)
        setSummarySources(res.sources)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load summary')
      } finally {
        setLoadingSummary(false)
      }
    }
    void load()
  }, [documents, summary])

  const suggestedQuestions = [
    'What are the main ideas across these documents?',
    'List key definitions and terms I should memorize.',
    'Explain the overall story these readings are telling.',
    'What are the main arguments and how do they relate?',
    'What should I review the day before an exam?',
  ]

  const handleRunReport = async (report: QuickReport) => {
    setActiveReportId(report.id)
    setReportError(null)
    setReportContent(null)
    try {
      const res = await runQuery(report.prompt, 8)
      setReportContent(res.answer)
      setReportSources(res.sources)
    } catch (e) {
      setReportError(e instanceof Error ? e.message : 'Failed to generate report')
    }
  }

  if (documents.length === 0) {
    return (
      <Card>
        <div
          style={{
            fontFamily: 'var(--font-family)',
            fontSize: 'var(--text-lg)',
            fontWeight: 600,
            letterSpacing: '0.06em',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-md)',
          }}
        >
          Empty notebook
        </div>
        <p style={{ fontSize: 'var(--text-sm)', maxWidth: 520, marginTop: 0 }}>
          Attach lecture slides, papers, or textbooks in the sidebar. We will index them into one corpus so chat, notes, and
          flashcards are all grounded in your sources.
        </p>
      </Card>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--card-margin)' }}>
      <Card>
        <div
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 600,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: 'var(--color-text-muted)',
            marginBottom: 'var(--space-sm)',
          }}
        >
          Notebook guide
        </div>
        <h2
          style={{
            fontFamily: 'var(--font-family)',
            fontSize: 'var(--text-lg)',
            fontWeight: 600,
            margin: '0 0 ' + 'var(--space-sm)',
            lineHeight: 'var(--line-height-heading)',
          }}
        >
          Overview
        </h2>
        {loadingSummary && <p style={{ fontSize: 'var(--text-sm)' }}>Summarizing your corpus…</p>}
        {error && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-danger)' }}>
            {error} (check that the API is running and RAG is configured).
          </p>
        )}
        {!loadingSummary && !error && summary && (
          <div style={{ fontSize: 'var(--text-sm)', lineHeight: 'var(--line-height-body)' }}>
            <div dangerouslySetInnerHTML={{ __html: summary.replace(/\n/g, '<br />') }} />
          </div>
        )}
        {summarySources.length > 0 && (
          <details style={{ marginTop: 'var(--space-md)', fontSize: 'var(--text-xs)' }}>
            <summary style={{ cursor: 'pointer' }}>Sources</summary>
            <ul>
              {summarySources.map((s, idx) => (
                <li key={`${s.document_id ?? idx}`}>
                  <strong>{s.display_name ?? s.filename ?? 'Document'}</strong>
                </li>
              ))}
            </ul>
          </details>
        )}
      </Card>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.5fr) minmax(0, 1fr)',
          gap: 'var(--card-margin)',
        }}
      >
        <Card>
          <div
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
              marginBottom: 'var(--space-sm)',
            }}
          >
            Suggested questions
          </div>
          <p style={{ fontSize: 'var(--text-sm)', marginTop: 0, marginBottom: 'var(--space-md)' }}>
            Jump into chat with questions tuned to this corpus.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
            {suggestedQuestions.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => navigate('/chat', { state: { initialPrompt: q } })}
                style={{
                  textAlign: 'left',
                  fontSize: 13,
                  padding: '6px 8px',
                  border: '1px solid var(--color-border-strong)',
                  backgroundColor: 'var(--color-bg)',
                  cursor: 'pointer',
                }}
              >
                {q}
              </button>
            ))}
          </div>
        </Card>

        <Card>
          <div
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
              marginBottom: 'var(--space-sm)',
            }}
          >
            Quick reports
          </div>
          <p style={{ fontSize: 13, marginTop: 0, marginBottom: 'var(--space-md)' }}>
            One-click study artifacts over your notebook sources.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
            {QUICK_REPORTS.map((report) => (
              <Button key={report.id} type="button" variant="ghost" onClick={() => handleRunReport(report)}>
                {report.label}
              </Button>
            ))}
          </div>
        </Card>
      </div>

      {activeReportId && (
        <Card>
          <div
            style={{
              fontSize: 'var(--text-xs)',
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--color-text-muted)',
              marginBottom: 'var(--space-sm)',
            }}
          >
            Report
          </div>
          {reportError && (
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-danger)' }}>
              {reportError}
            </p>
          )}
          {!reportError && !reportContent && <p style={{ fontSize: 'var(--text-sm)' }}>Generating report…</p>}
          {reportContent && (
            <div style={{ fontSize: 'var(--text-sm)', lineHeight: 'var(--line-height-body)' }}>
              <div dangerouslySetInnerHTML={{ __html: reportContent.replace(/\n/g, '<br />') }} />
            </div>
          )}
          {reportSources.length > 0 && (
            <details style={{ marginTop: 'var(--space-md)', fontSize: 'var(--text-xs)' }}>
              <summary style={{ cursor: 'pointer' }}>Sources</summary>
              <ul>
                {reportSources.map((s, idx) => (
                  <li key={`${s.document_id ?? idx}`}>
                    <strong>{s.display_name ?? s.filename ?? 'Document'}</strong>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </Card>
      )}
    </div>
  )
}

