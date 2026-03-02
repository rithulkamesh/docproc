# Assessments and grading (demo)

Assessments — create, take, submit, and grade — are implemented in the **demo** (Go app), not in the docproc CLI. docproc is document processing only.

## Where it lives

- **Backend:** `demo/go/` — API routes for assessments, questions, submissions; grading in `internal/grade/` (single-select, formula, conceptual, derivation).
- **Frontend:** `demo/web/` — Create assessments, take them, view results.
- **Database:** PostgreSQL (from `demo/docker-compose.yml`): `docproc_assessments`, `docproc_questions`, `docproc_submissions`.

See [demo/README.md](../demo/README.md) for how to run the demo.

## Grading modes (Go)

| Type | Mode | Behavior |
|------|------|----------|
| single_select / mcq | single_select | Deterministic: student choice matches correct choice → full marks, else 0. |
| short_answer (equation-like ref) | formula | LLM: "Are these two mathematical expressions equivalent?" → score 0–100. |
| short_answer (default) | conceptual | LLM: grade by coverage of key concepts; paraphrases get credit. |
| long_answer | derivation | LLM: rubric-based partial credit per step. |

Grading uses the demo’s OpenAI client (`OPENAI_API_KEY`). No docproc CLI involved.

## Submission flow

1. **POST /assessments/:id/submit** with `answers: { question_id: value, ... }`.
2. Go loads questions, infers mode per question, runs the grader (single_select in-process; formula/conceptual/derivation via LLM).
3. Results are stored in `docproc_submissions` (score_pct, question_results) and returned.

No background queue for grading; submission is synchronous.
