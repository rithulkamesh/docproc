import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { ClipboardList } from 'lucide-react'

export function TestsCanvas() {
  return (
    <div className="flex flex-col space-y-8">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Tests
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Create assessments from your documents with AI-generated questions. Take them and get graded feedback.
        </p>
      </div>

      <section className="flex flex-col items-center justify-center gap-4 rounded-md border border-border bg-muted/10 px-6 py-10 text-center">
        <ClipboardList className="h-10 w-10 text-muted-foreground" />
        <p className="max-w-md text-sm text-muted-foreground">
          Create an assessment from a document. Choose question count, difficulty, and time limit. AI grades your answers.
        </p>
        <Button asChild>
          <Link to="/assessments/create">Create assessment</Link>
        </Button>
      </section>
    </div>
  )
}
