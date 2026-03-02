import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { Spinner } from '../components/Spinner'
import { MarkingSchemeBuilder } from '../components/MarkingSchemeBuilder'
import { createAssessment, uploadPaper } from '../api/assessments'
import type { PaperPattern, PaperUploadResponse } from '../api/assessments'
import { useWorkspace } from '../context/WorkspaceContext'

const DEFAULT_MARKING_SCHEME: PaperPattern = {
  total_marks: 60,
  sections: [
    { name: 'Part A', question_count: 20, marks_each: 1, type: 'short' },
    { name: 'Part B', question_count: 4, marks_each: 10, type: 'long' },
  ],
}

const CREATION_STEPS = [
  'Uploading document...',
  'Extracting text...',
  'Detecting question pattern...',
  'Generating assessment...',
  'Applying marking scheme...',
  'Finalizing...',
]

export function CreateAssessmentView() {
  const navigate = useNavigate()
  const location = useLocation()
  const { documents, selectedDocumentId, setSelectedDocumentId } = useWorkspace()
  const [subject, setSubject] = useState('')

  useEffect(() => {
    const state = location.state as { documentId?: string } | null
    if (state?.documentId) {
      setSelectedDocumentId(state.documentId)
    }
  }, [location.state, setSelectedDocumentId])
  const [topics, setTopics] = useState('')
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard' | 'mixed'>('mixed')
  const [questionCount, setQuestionCount] = useState(8)
  const [timeLimitMinutes, setTimeLimitMinutes] = useState(30)
  const [includeLongAnswers, setIncludeLongAnswers] = useState(false)
  const [aiEnabled, setAiEnabled] = useState(true)
  const [loading, setLoading] = useState(false)
  const [creationStep, setCreationStep] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [markingScheme, setMarkingScheme] = useState<PaperPattern>(DEFAULT_MARKING_SCHEME)
  const [paperUploading, setPaperUploading] = useState(false)
  const [paperError, setPaperError] = useState<string | null>(null)

  const documentId = selectedDocumentId || (documents.length ? documents[0]?.id : null)
  const readyDocuments = documents.filter((d) => d.status === 'completed')

  const handlePaperUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setPaperError(null)
    setPaperUploading(true)
    try {
      const data: PaperUploadResponse = await uploadPaper(file)
      setMarkingScheme(data.pattern)
    } catch (err) {
      setPaperError(err instanceof Error ? err.message : 'Paper upload failed')
    } finally {
      setPaperUploading(false)
      e.target.value = ''
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!documentId) {
      setError('Select a document (with status completed) as the source.')
      return
    }
    setError(null)
    setLoading(true)
    setCreationStep(0)
    const stepInterval = setInterval(() => {
      setCreationStep((s) => Math.min(s + 1, CREATION_STEPS.length - 1))
    }, 800)
    try {
      const res = await createAssessment({
        title: subject.trim() || 'Assessment',
        document_id: documentId,
        ai_generation_enabled: aiEnabled,
        ai_config: {
          subject: subject.trim() || undefined,
          topics: topics.split(/[,;]/).map((t) => t.trim()).filter(Boolean),
          difficulty,
          question_count: questionCount,
          time_limit_minutes: timeLimitMinutes,
          include_long_answers: includeLongAnswers,
        },
        marking_scheme: markingScheme,
        time_limit_minutes: timeLimitMinutes,
      })
      clearInterval(stepInterval)
      setCreationStep(CREATION_STEPS.length - 1)
      navigate(`/assessments/${res.id}/take`, { replace: true })
    } catch (err) {
      clearInterval(stepInterval)
      setError(err instanceof Error ? err.message : 'Failed to create assessment')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="content-max-narrow">
      <Link to="/assessments" className="link-back body-sm">
        ← Assessments
      </Link>
      <h1 className="heading-2xl mb-xl">Create assessment</h1>
      <form onSubmit={handleSubmit} className="form-stack">
        {error && (
          <p className="body-sm text-primary" style={{ color: 'var(--color-danger)', margin: 0 }}>{error}</p>
        )}

        <Card className="form-card">
          <div>
            <label className="label">Subject / Title</label>
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="e.g. Biology Chapter 5"
              className="input"
            />
          </div>

          <div>
            <label className="label">Source document (required)</label>
            <select
              value={documentId || ''}
              onChange={(e) => setSelectedDocumentId(e.target.value || null)}
              required
              className="input select-input"
            >
              <option value="">Select a document</option>
              {readyDocuments.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.filename} {d.status !== 'completed' ? `(${d.status})` : ''}
                </option>
              ))}
            </select>
            {readyDocuments.length === 0 && (
              <p className="text-xs text-muted mt-sm">Upload and process a document in Sources first.</p>
            )}
          </div>

          <div>
            <label className="label">Topics (comma-separated)</label>
            <input
              type="text"
              value={topics}
              onChange={(e) => setTopics(e.target.value)}
              placeholder="e.g. mitosis, cell cycle"
              className="input"
            />
          </div>

          <div>
            <label className="label">Difficulty</label>
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value as 'easy' | 'medium' | 'hard' | 'mixed')}
              className="input select-input"
            >
              <option value="mixed">Mixed</option>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          </div>

          <div>
            <label className="label">Question count</label>
            <input
              type="number"
              min={1}
              max={20}
              value={questionCount}
              onChange={(e) => setQuestionCount(Number(e.target.value) || 8)}
              className="input"
            />
          </div>

          <div>
            <label className="label">Time limit (minutes)</label>
            <input
              type="number"
              min={5}
              max={180}
              value={timeLimitMinutes}
              onChange={(e) => setTimeLimitMinutes(Number(e.target.value) || 30)}
              className="input"
            />
          </div>

          <label className="body-sm" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
            <input
              type="checkbox"
              checked={includeLongAnswers}
              onChange={(e) => setIncludeLongAnswers(e.target.checked)}
            />
            Include long-answer questions (AI-graded)
          </label>

          <label className="body-sm text-primary" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
            <input
              type="checkbox"
              checked={aiEnabled}
              onChange={(e) => setAiEnabled(e.target.checked)}
            />
            AI generation enabled
          </label>
        </Card>

        <Card>
          <div className="body-sm text-primary" style={{ fontWeight: 600, marginBottom: 'var(--space-md)' }}>
            Upload question paper (PDF/DOCX/image) to auto-fill marking scheme
          </div>
          <input type="file" accept=".pdf,.docx,.png,.jpg,.jpeg,.webp" onChange={handlePaperUpload} disabled={paperUploading} className="mb-md" />
          {paperUploading && <p className="body-sm text-muted" style={{ margin: 0 }}>Extracting text and detecting pattern…</p>}
          {paperError && <p className="body-sm" style={{ color: 'var(--color-danger)', margin: 0 }}>{paperError}</p>}
        </Card>

        <Card>
          <MarkingSchemeBuilder
            value={markingScheme}
            onChange={(pattern) => { if (pattern != null) setMarkingScheme(pattern) }}
            editable
          />
        </Card>

        {loading ? (
          <div className="loading-state p-content">
            <Spinner size="md" />
            <p className="text-muted body-sm">{CREATION_STEPS[creationStep]}</p>
          </div>
        ) : (
          <Button type="submit" loading={loading} disabled={loading || !documentId}>
            Create and take assessment
          </Button>
        )}
      </form>
    </div>
  )
}
