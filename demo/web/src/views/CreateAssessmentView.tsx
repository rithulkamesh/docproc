import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { MarkingSchemeBuilder } from '@/components/MarkingSchemeBuilder'
import { createAssessment, uploadPaper } from '@/api/assessments'
import type { PaperPattern, PaperUploadResponse } from '@/api/assessments'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Loader2 } from 'lucide-react'

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
    if (state?.documentId) setSelectedDocumentId(state.documentId)
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
      navigate(`/assessments/${res.id}/take`, { replace: true })
    } catch (err) {
      clearInterval(stepInterval)
      setError(err instanceof Error ? err.message : 'Failed to create assessment')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <Button variant="ghost" size="sm" asChild>
        <Link to="/assessments">← Assessments</Link>
      </Button>
      <h1 className="text-2xl font-semibold tracking-tight">Create assessment</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {error && <p className="text-sm text-destructive">{error}</p>}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="subject">Subject / Title</Label>
              <Input
                id="subject"
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g. Biology Chapter 5"
              />
            </div>
            <div className="space-y-2">
              <Label>Source document (required)</Label>
              <Select
                value={documentId || ''}
                onValueChange={(v) => setSelectedDocumentId(v || null)}
                required
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a document" />
                </SelectTrigger>
                <SelectContent>
                  {readyDocuments.map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      {d.filename} {d.status !== 'completed' ? `(${d.status})` : ''}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {readyDocuments.length === 0 && (
                <p className="text-xs text-muted-foreground">Upload and process a document in Sources first.</p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="topics">Topics (comma-separated)</Label>
              <Input
                id="topics"
                type="text"
                value={topics}
                onChange={(e) => setTopics(e.target.value)}
                placeholder="e.g. mitosis, cell cycle"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Difficulty</Label>
                <Select value={difficulty} onValueChange={(v) => setDifficulty(v as typeof difficulty)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mixed">Mixed</SelectItem>
                    <SelectItem value="easy">Easy</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="hard">Hard</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="questionCount">Question count</Label>
                <Input
                  id="questionCount"
                  type="number"
                  min={1}
                  max={20}
                  value={questionCount}
                  onChange={(e) => setQuestionCount(Number(e.target.value) || 8)}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="timeLimit">Time limit (minutes)</Label>
              <Input
                id="timeLimit"
                type="number"
                min={5}
                max={180}
                value={timeLimitMinutes}
                onChange={(e) => setTimeLimitMinutes(Number(e.target.value) || 30)}
              />
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={includeLongAnswers}
                onChange={(e) => setIncludeLongAnswers(e.target.checked)}
                className="rounded border-input"
              />
              Include long-answer questions (AI-graded)
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={aiEnabled}
                onChange={(e) => setAiEnabled(e.target.checked)}
                className="rounded border-input"
              />
              AI generation enabled
            </label>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Marking scheme (optional)</CardTitle>
            <p className="text-sm text-muted-foreground">
              Upload question paper (PDF/DOCX/image) to auto-fill marking scheme
            </p>
          </CardHeader>
          <CardContent className="space-y-2">
            <input
              type="file"
              accept=".pdf,.docx,.png,.jpg,.jpeg,.webp"
              onChange={handlePaperUpload}
              disabled={paperUploading}
              className="text-sm"
            />
            {paperUploading && (
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Extracting text and detecting pattern…
              </p>
            )}
            {paperError && <p className="text-sm text-destructive">{paperError}</p>}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <MarkingSchemeBuilder
              value={markingScheme}
              onChange={(pattern) => {
                if (pattern != null) setMarkingScheme(pattern)
              }}
              editable
            />
          </CardContent>
        </Card>

        {loading ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">{CREATION_STEPS[creationStep]}</p>
          </div>
        ) : (
          <Button type="submit" disabled={loading || !documentId}>
            Create and take assessment
          </Button>
        )}
      </form>
    </div>
  )
}
