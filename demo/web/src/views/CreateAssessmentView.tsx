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
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { createAssessment } from '@/api/assessments'
import type { PaperPattern } from '@/api/assessments'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Loader2, ChevronDown, ChevronRight } from 'lucide-react'

const LENGTH_OPTIONS = [
  { value: 5, label: 'Quick (5 questions)' },
  { value: 10, label: 'Standard (10 questions)' },
  { value: 20, label: 'Full (20 questions)' },
] as const

const SOURCE_ALL = '__all__'

function defaultMarkingScheme(questionCount: number, includeLongAnswers: boolean): PaperPattern {
  if (includeLongAnswers && questionCount >= 4) {
    const shortCount = questionCount - 2
    const longCount = 2
    return {
      total_marks: shortCount * 2 + longCount * 10,
      sections: [
        { name: 'Short answer', question_count: shortCount, marks_each: 2, type: 'short' },
        { name: 'Long answer', question_count: longCount, marks_each: 10, type: 'long' },
      ],
    }
  }
  return {
    total_marks: questionCount * 2,
    sections: [
      { name: 'Questions', question_count: questionCount, marks_each: 2, type: 'short' },
    ],
  }
}

const CREATION_STEPS = [
  'Preparing...',
  'Generating questions...',
  'Applying settings...',
  'Finalizing...',
]

export function CreateAssessmentView() {
  const navigate = useNavigate()
  const location = useLocation()
  const { documents, setSelectedDocumentId } = useWorkspace()
  const [title, setTitle] = useState('Practice test')

  useEffect(() => {
    const state = location.state as { documentId?: string } | null
    if (state?.documentId) {
      setSelectedDocumentId(state.documentId)
      setSource(state.documentId)
    }
  }, [location.state, setSelectedDocumentId])

  // Stage 1
  const [source, setSource] = useState(SOURCE_ALL)
  const [length, setLength] = useState<5 | 10 | 20>(10)
  const [difficulty, setDifficulty] = useState<'easy' | 'mixed' | 'hard'>('mixed')

  // Stage 2 (advanced)
  const [topics, setTopics] = useState('')
  const [timeLimitMinutes, setTimeLimitMinutes] = useState(30)
  const [includeLongAnswers, setIncludeLongAnswers] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  const [loading, setLoading] = useState(false)
  const [creationStep, setCreationStep] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const readyDocuments = documents.filter((d) => d.status === 'completed')
  const documentId =
    source === SOURCE_ALL
      ? readyDocuments.length ? readyDocuments[0]?.id ?? null : null
      : source

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!documentId) {
      setError('No documents available. Upload and process documents in Sources first.')
      return
    }
    setError(null)
    setCreationStep(0)
    setLoading(true)
    const stepInterval = setInterval(() => {
      setCreationStep((s) => Math.min(s + 1, CREATION_STEPS.length - 1))
    }, 600)
    try {
      const markingScheme = defaultMarkingScheme(length, includeLongAnswers)
      const res = await createAssessment({
        title: title.trim() || 'Practice test',
        document_id: documentId,
        ai_generation_enabled: true,
        ai_config: {
          subject: title.trim() || undefined,
          topics: topics.split(/[,;]/).map((t) => t.trim()).filter(Boolean),
          difficulty: difficulty === 'mixed' ? 'mixed' : difficulty,
          question_count: length,
          time_limit_minutes: timeLimitMinutes,
          include_long_answers: includeLongAnswers,
        },
        marking_scheme: markingScheme,
        time_limit_minutes: timeLimitMinutes,
      })
      clearInterval(stepInterval)
      if (!res?.id) {
        setError('Server did not return an assessment ID. Please try again.')
        return
      }
      navigate(`/assessments/${res.id}/take`, { replace: true, state: { assessment: res } })
    } catch (err) {
      clearInterval(stepInterval)
      setError(err instanceof Error ? err.message : 'Failed to generate test')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative mx-auto max-w-2xl space-y-6 p-6">
      {/* Full-screen loading overlay when generating */}
      {loading && (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-background/95 backdrop-blur-sm"
          aria-live="polite"
          aria-busy="true"
        >
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <div className="text-center space-y-1">
            <p className="text-lg font-medium">Generating your test</p>
            <p className="text-sm text-muted-foreground">{CREATION_STEPS[creationStep]}</p>
          </div>
        </div>
      )}

      <Button variant="ghost" size="sm" asChild>
        <Link to="/assessments">← Assessments</Link>
      </Button>
      <h1 className="text-2xl font-semibold tracking-tight">Generate practice test</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {error && <p className="text-sm text-destructive">{error}</p>}

        <fieldset disabled={loading} className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Test options</CardTitle>
            <p className="text-sm text-muted-foreground">
              Choose source, length, and difficulty. Use advanced settings for more control.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Source</Label>
              <Select
                value={source}
                onValueChange={(v) => setSource(v)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={SOURCE_ALL}>All documents</SelectItem>
                  {readyDocuments.map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      {d.display_name ?? d.filename}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {readyDocuments.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  Upload and process documents in Sources first.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label>Length</Label>
              <Select
                value={String(length)}
                onValueChange={(v) => setLength(Number(v) as 5 | 10 | 20)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LENGTH_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={String(opt.value)}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Difficulty</Label>
              <Select value={difficulty} onValueChange={(v) => setDifficulty(v as typeof difficulty)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="easy">Easy</SelectItem>
                  <SelectItem value="mixed">Mixed</SelectItem>
                  <SelectItem value="hard">Hard</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <Card>
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer select-none hover:bg-muted/50 rounded-t-lg transition-colors">
                <div className="flex items-center gap-2">
                  {advancedOpen ? (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  )}
                  <CardTitle className="text-base">Advanced settings</CardTitle>
                </div>
                <p className="text-sm text-muted-foreground pl-6">
                  Topics, time limit, long answers, title
                </p>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="space-y-4 pt-0">
                <div className="space-y-2">
                  <Label htmlFor="title">Title</Label>
                  <Input
                    id="title"
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g. Biology Chapter 5"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="topics">Topics (optional)</Label>
                  <Input
                    id="topics"
                    type="text"
                    value={topics}
                    onChange={(e) => setTopics(e.target.value)}
                    placeholder="e.g. mitosis, cell cycle"
                  />
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
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>

        <Button type="submit" disabled={loading || !documentId}>
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating…
            </>
          ) : (
            'Generate Test'
          )}
        </Button>
        </fieldset>
      </form>
    </div>
  )
}
