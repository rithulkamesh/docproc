# docproc Web App — Full Feature & UX Specification

This document describes **every feature** of the `demo/web/` frontend in detail, for use in prompts or by a UX engineer to refine the product. The app is a **document-centric study/education workspace**: users attach documents (PDF, DOCX, PPTX, XLSX), then use RAG-grounded chat, AI-generated notes, flashcards, and AI-generated/graded assessments.

---

## 1. Product Overview

- **Name / branding:** “docproc” (shown in header); product is “docproc / edu” in PDF exports.
- **Core value:** One workspace per **project**; each project has **documents** that are processed and indexed. All study features (chat, notes, flashcards, tests) are grounded in those documents.
- **Primary user flow:** Create/select project → Add documents → Wait for processing → Use Converse (chat), Notes, Flashcards, or create/take Assessments. Sources canvas manages documents.
- **No auth in app:** Assumes backend is configured (API base URL, optional namespace). Settings view shows API status only.

---

## 2. Tech Stack & Repo Structure

- **Stack:** React 19, TypeScript, Vite 7, React Router 7, SWR for data fetching.
- **Key deps:** TipTap (rich text), KaTeX (LaTeX in assessments), DOMPurify (sanitization), jsPDF (notes PDF export).
- **Important paths:**
  - `src/App.tsx` — Routes and workspace route wrapper.
  - `src/main.tsx` — React root + BrowserRouter.
  - `src/context/WorkspaceContext.tsx` — Global workspace state.
  - `src/design/theme.ts` — Design tokens (colors, fonts, spacing).
  - `src/index.css` — CSS variables (light/dark), layout, buttons, spinner.
  - `src/App.css` — ProseMirror and LaTeX display.
  - `src/api/*` — All backend calls.
  - `src/views/*` — Full-page views (Settings, Assessments).
  - `src/components/*` — Reusable UI and canvas components.

---

## 3. Routing

| Path | Description |
|------|-------------|
| `/` | **Workspace route:** KnowledgeCanvas + ToolRail + CommandPalette. No sidebar in current implementation; canvas mode switches content. |
| `/settings` | SettingsView — API base URL, status (RAG, DB, primary AI, namespace). |
| `/assessments/create` | CreateAssessmentView — Form to create an assessment from a document. |
| `/assessments/:id/take` | TakeAssessmentView — Take the assessment (questions, submit). |
| `/assessments/:id/result/:submissionId` | AssessmentResultView — Score and links to retake or create another. |

**Note:** Views such as `LibraryView`, `NotebookGuideView`, `FlashcardsView`, `NotesView`, and `ChatView` exist in code but are **not** mounted in the current router. The live UI uses the canvas components (ConverseCanvas, SourcesCanvas, NotesCanvas, FlashcardsCanvas, TestsCanvas) inside KnowledgeCanvas.

---

## 4. Global State (WorkspaceContext)

Provided by `WorkspaceProvider` and consumed via `useWorkspace()`.

| State | Type | Purpose |
|-------|------|---------|
| `projects` | `Project[]` | All projects from API. |
| `currentProjectId` | `string` | Active project (e.g. `'default'`). |
| `setCurrentProjectId` | `(id: string) => void` | Switch project; clears selected document. |
| `currentProject` | `Project \| null` | Full project object for current id. |
| `setCurrentProjectName` | `(name: string) => Promise<void>` | Rename current project. |
| `documents` | `DocumentSummary[]` | Documents for current project. |
| `setDocuments` | setter | Replace documents list. |
| `selectedDocumentId` | `string \| null` | Document selected for notes/flashcards/generation. |
| `setSelectedDocumentId` | `(id \| null) => void` | Select document. |
| `loadProjects` | `() => Promise<void>` | Refetch projects. |
| `loadDocuments` | `() => Promise<void>` | Refetch documents for current project. |
| `handleUploadFile` | `(file: File) => Promise<void>` | Upload to current project and refresh list. |
| `status` | `ApiStatus \| null` | From `/status` (ok, rag_backend, database_provider, etc.). |
| `apiStatusLabel` | `string` | Human-readable status or “API unreachable…”. |
| `themeMode` | `'light' \| 'dark'` | Persisted in localStorage (`docproc-theme`); respects `prefers-color-scheme` on first load. |
| `setThemeMode` | `(mode) => void` | Toggle theme. |
| `focusMode` | `boolean` | When true: header shrinks, tool rail hidden, canvas max-width widens. |
| `setFocusMode` | `(on: boolean) => void` | Toggle focus mode. |
| `canvasMode` | `CanvasMode` | One of: `'converse' \| 'notes' \| 'flashcards' \| 'tests' \| 'sources'`. |
| `setCanvasMode` | `(mode: CanvasMode) => void` | Switch main canvas. |
| `lastIndexedLabel` | `string` | Date/time of last completed document update, or “—”. |

**Side effects:**

- Documents are polled every 2s while any document has `status === 'processing'`.
- On project change, documents and project are refetched; selected document is cleared or set to first doc.
- If current project has no documents, `canvasMode` is set to `'sources'`.

---

## 5. Design System

### 5.1 Theme (`src/design/theme.ts`)

- **Colors:** All from CSS variables so light/dark swap (e.g. `--color-bg`, `--color-text`, `--color-accent`, `--color-danger`). Theme object references `var(--...)`.
- **Fonts:** Heading = “Space Grotesk” + system; body = system; mono = “JetBrains Mono” + system.
- **Font sizes:** xs (11px) → 3xl (24px) in named steps.
- **Borders:** `thin` (structural 1px), `strong`/container (2–3px). **Radius:** sm/md/lg all 0 (neo-brutalist).
- **Shadows:** Intentionally none in theme; dark mode uses subtle inset panel shadow in CSS.
- **Spacing:** 8px grid via `theme.spacing(factor)`.
- **Canvas:** `canvasMaxWidth` (75ch), `canvasMaxWidthFocus` (90ch), `contentGap` 2rem, `toolRailWidth` 4.5rem.

### 5.2 Light Theme (default, `index.css`)

- Backgrounds: light grays/white; borders black; accent blue; danger red.
- `--border-width-container: 3px` for panels.

### 5.3 Dark Theme (`data-theme="dark"`)

- Background hierarchy: app surface → primary panels → elevated (sidebars/dock) → interactive (inputs/hover).
- Structural borders use gray, not white; primary buttons: accent fill with dark text; ghost buttons: 1px accent border, hover fill.
- Panel containers get subtle inset shadow for depth.

### 5.4 Typography & Labels

- Section labels: uppercase, small font, letter-spacing (0.06em–0.14em), muted color.
- Headers use theme heading font and weight 600–700.

---

## 6. Layout & Shell

### 6.1 LayoutShell

- Full viewport height; flex column.
- **ProjectHeaderBar** at top (always visible unless focus mode shrinks it).
- **main-content:** flex 1, min-height 0, overflow-y auto; renders `children` (route content).

### 6.2 ProjectHeaderBar

- **Left:** “docproc” link (to `/`), then **editable project name** (click to edit inline; save on blur/Enter, revert on Escape).
- **Right:** Docs count (+ “· X processing” if any); “Last indexed: {date}”; **Focus** button (toggles focus mode); **Light/Dark** theme toggle.
- In focus mode: min-height reduced to 2.5rem.

---

## 7. Workspace Route (Home)

The `/` route renders:

1. **KnowledgeCanvas** — Central content; max-width from theme (wider in focus mode). Renders one of:
   - **ConverseCanvas**
   - **SourcesCanvas**
   - **NotesCanvas**
   - **FlashcardsCanvas**
   - **TestsCanvas**
2. **ToolRail** — Mode switcher (Converse, Notes, Flashcards, Tests, Sources). Hidden in focus mode.
3. **CommandPalette** — Modal command palette (⌘K / Ctrl+K).

### 7.1 ToolRail

- **Desktop:** Fixed left, vertical strip (4.5rem), 5 icon buttons; active mode has accent pill/indicator.
- **Narrow (≤48rem):** Fixed bottom bar, same 5 buttons in a row.
- Icons: Converse (speech bubble), Notes (doc), Flashcards (card), Tests (checklist), Sources (folder).
- Click sets `canvasMode`; focus mode hides the rail entirely.

### 7.2 Command Palette

- **Shortcut:** ⌘K / Ctrl+K toggles open/close; Escape closes.
- **UI:** Overlay + centered panel; search input; list of commands; arrow keys + Enter to run.
- **Commands:** Switch to Converse / Notes / Flashcards / Tests / Sources; Enter/Exit Focus Mode; Add document (switches to Sources and closes); Open Settings; one “Switch to {project name}” per project.
- Filtering by label and keywords; selection follows keyboard.

---

## 8. Canvas Features (by mode)

### 8.1 Converse (Chat)

- **Empty state (no documents):** Message “No documents yet”, “Add your first document…”, **Add document** button, hint “Or press ⌘K…”.
- **With documents:** Renders **ChatConsole** with `documents`, `selectedDocumentId`, `projectId`.

**ChatConsole:**

- Header line: “Grounded in **X document(s)** in this project. Ask questions…”
- Scrollable message list: user vs “Workspace” messages; assistant messages show **Save as note** and expandable **Sources** (filename + content snippet).
- Input: single line, placeholder “Ask a question or request a study artifact…”, **Send** button (disabled when empty or sending).
- Errors shown below (e.g. deployment/API key messages normalized for user).
- **Save as note** calls `createNote(message.content, selectedDocumentId, projectId)`.

**ChatView** (not in router): Similar chat UI with “Notebook” label and optional `location.state.initialPrompt` to prefill input; used by NotebookGuideView’s suggested questions (which navigate to `/chat` — route not present in current App).

---

### 8.2 Sources

**SourcesCanvas:**

- Title “Sources”; short description that documents ground chat, notes, flashcards, tests.
- **Add document** button + hidden file input (accept `.pdf,.docx,.pptx,.xlsx`). On upload, `handleUploadFile`; errors shown via alert.
- **Document list:** If none, empty state “No documents yet. Add a PDF or document…”.
- If list exists: each doc is a button (filename, status: Processing… with Spinner / Ready · N pages / Failed). Selected doc has accent background.
- **Status block:** “X document(s)”, “X processing” if any, and `apiStatusLabel`.

No sidebar document list in current layout; document selection is only within this canvas (or in Create Assessment form).

---

### 8.3 Notes

**NotesCanvas:**

- Section **Notes**; copy: “Notes live at the project level. Sections can be linked to specific documents (choose in Sources).”
- **AI-generated summary:**
  - Collapsible “AI-generated summary” (Show/Hide).
  - Two modes: **From document** (uses `selectedDocumentId`) or **From text** (textarea). “Generate summary” button; result shown with bullet styling; **Save as section** and **Clear**.
- **Sections:**
  - “+ Add section”, “Download PDF”.
  - List of note sections: each is textarea (auto-save debounced 600ms via `updateNote`); metadata: source filename, updated time; Saving…/Saved indicator.
  - “Add section” creates note with optional “Section for: {filename}” if a doc is selected.
- **Download PDF:** jsPDF; header “docproc / edu”, “Project Notes”, project id and date; generated summary (if any) then each section with optional “Section N — {filename}”; filename `docproc-notes-{projectId}-{timestamp}.pdf`.

**NotesModule** (used in StudyDock): Same concepts in a more compact layout for the dock; sections in a scrollable area with max-height.

---

### 8.4 Flashcards

**FlashcardsCanvas:**

- **Your decks:** List of decks (name, card count, created date); **Study** and **Delete** per deck.
- **Study mode (full-screen):** After “Study”, deck cards load; progress bar (percent); stack illusion (next 2 cards behind); current card shows front/back (click or Space to flip). Buttons: Easy, Medium, Hard (rate), Skip; **Exit Study**. Keyboard: ← → navigate, Space flip, 1/2/3 rate. No timed/reverse here.
- **Generate deck:** Toggle “From document” / “From text”; if document, uses selected doc; if text, textarea. Number input (3–20) for count, optional deck name, **Generate**.
- Errors and loading states shown.

**FlashcardsModule** (StudyDock): Decks in compact list; select deck to load cards in same panel. Study modes: **Classic**, **Timed** (countdown), **Reverse** (show back first). Progress bar, New/Review/Mastered counts; card stack with pointer drag to prev/next; Prev, Show answer, Next; Hard (1), Medium (2), Easy (3) rating buttons. Keyboard same as canvas.

---

### 8.5 Tests

**TestsCanvas:**

- Title “Tests”; short description: create assessment from documents, AI-generated questions, submit for grading.
- Single CTA: **Create assessment** linking to `/assessments/create`.

**TestsModule** (StudyDock): Same idea; “Exam-style tests…”; **Create assessment** link.

---

## 9. Assessment Flow (Views)

### 9.1 CreateAssessmentView (`/assessments/create`)

- **Form fields:** Subject/Title; **Source document** (required select of completed docs); Topics (comma-separated); Difficulty (Mixed / Easy / Medium / Hard); Question count (1–20, default 8); Time limit (minutes, 5–180, default 30); “Include long-answer questions (AI-graded)” checkbox; “AI generation enabled” checkbox.
- **Submit:** “Create and take assessment” → `createAssessment` with document_id, ai_config, etc. → navigate to `/assessments/:id/take`.
- Location state can pass `documentId` to pre-select document (e.g. from LibraryView “Create assessment” with state).

### 9.2 TakeAssessmentView (`/assessments/:id/take`)

- **Data:** Assessment by id (SWR); questions array.
- **State:** `answers` (per question id); `currentIndex`; `submitted`; `submissionId`; `submitError`; draft auto-saved to localStorage every 30s; on load, draft restored.
- **Layout:** Sticky left nav “Questions” with grid of numbered buttons (answered state styled); main area: assessment title, “Question X of N”, current question block, prev/next, **Submit assessment**.
- **Question types and inputs:**
  - **mcq:** Radio options; prompt and options rendered with **LatexText**.
  - **multi:** Checkboxes; **LatexText** for options.
  - **long_answer / short_answer:** **RichTextEditor** (TipTap: bold, italic, underline, lists, code, sub/superscript, equation placeholder).
  - Fallback: plain text input.
- **Submit:** `submitAssessment(id, answers)` → receive `submission_id`; then **poll** submission status (useSubmissionPoll). When status is `completed` or `failed`, redirect to `/assessments/:id/result/:submissionId`.
- **Post-submit screen:** “AI evaluation in progress”, spinner, “You will be redirected…”. If poll times out (5 min), link “View result” to result URL.

### 9.3 AssessmentResultView (`/assessments/:id/result/:submissionId`)

- Load submission and assessment (SWR). **Total score:** `score_pct`% / 100%; question count. If `ai_status === 'failed'`, message that evaluation could not complete for some answers.
- **Actions:** “Create another assessment” (→ create), “Retake” (→ take).

---

## 10. Settings View

- **Settings** heading.
- Displays **API base URL** (from `apiClient.baseUrl` — env `VITE_DOCPROC_API_URL` or `http://localhost:8000`).
- Fetches `/status`; shows: Status (Connected/Error), RAG backend, RAG configured, Database, Primary AI, Namespace.
- Error state if fetch fails.

---

## 11. Components (summary)

| Component | Purpose |
|----------|---------|
| **LayoutShell** | App shell: header + scrollable main. |
| **ProjectHeaderBar** | Brand, editable project name, doc count, last indexed, Focus, theme toggle. |
| **ProjectSidebar** | PROJECT + PROJECT DATA sections, document list, Add document, corpus status. Not rendered in current route tree; available for alternate layouts. |
| **KnowledgeCanvas** | Container that switches Converse / Sources / Notes / Flashcards / Tests by `canvasMode`. |
| **ConverseCanvas** | Empty state or ChatConsole. |
| **ChatConsole** | Chat UI: messages, sources, Save as note, input, send. |
| **SourcesCanvas** | Document list, upload, selection, status. |
| **NotesCanvas** | AI summary (from doc/text), sections (textarea + auto-save), PDF export. |
| **FlashcardsCanvas** | Decks list, study mode (full-screen), generate deck (doc/text). |
| **TestsCanvas** | CTA to create assessment. |
| **StudyDock** | Collapsible panels: Notes (NotesModule), Flashcards (FlashcardsModule), Tests (TestsModule). Used inside **ProjectWorkspace** (chat + dock layout); ProjectWorkspace is not in current routes. |
| **NotesModule** | Notes summary + sections in compact form for dock. |
| **FlashcardsModule** | Decks + card study (classic/timed/reverse) in dock. |
| **TestsModule** | “Create assessment” link. |
| **ToolRail** | 5-mode switcher (vertical or bottom on narrow). |
| **CommandPalette** | ⌘K modal: switch mode, focus, add doc, settings, switch project. |
| **Button** | primary / ghost / danger; fullWidth; loading (spinner + “Loading…”). |
| **Spinner** | sm/md; CSS rotation. |
| **RichTextEditor** | TipTap + toolbar (B, I, U, lists, code, sub/superscript, ∑ equation placeholder); sanitized HTML output. |
| **LatexText** | Parses `$$...$$` and `$...$`, renders with KaTeX. |
| **ProjectWorkspace** | Grid: ChatConsole | resizer | StudyDock. Not in routes. |

---

## 12. API Layer

- **Base URL:** `VITE_DOCPROC_API_URL` or `http://localhost:8000`.
- **client:** `get`, `post`, `patch`, `delete`; JSON; errors parsed from `detail`.
- **Endpoints used:**
  - **Status:** `GET /status` → ApiStatus.
  - **Projects:** `GET /projects/`, `GET /projects/:id`, `PATCH /projects/:id`, `POST /projects/`.
  - **Documents:** `GET /documents/?project_id=`, `GET /documents/:id`, `POST /documents/upload` (FormData, project_id query).
  - **Query (RAG):** `POST /query` { prompt, top_k } → { answer, sources }.
  - **Notes:** `GET /notes/?document_id=&project_id=`, `POST /notes`, `PATCH /notes/:id`, `POST /notes/generate` (source_type: document|text).
  - **Flashcards:** `GET /flashcards/decks?...`, `GET /flashcards/decks/:id/cards`, `DELETE /flashcards/decks/:id`, `POST /flashcards/generate`, `POST /flashcards/decks` (manual pairs).
  - **Assessments:** `POST /assessments`, `GET /assessments/:id`, `POST /assessments/:id/submit`, `GET /assessments/:id/submissions/:submissionId`.
  - **Quiz:** `POST /quiz/generate` (document_id, count) — returns pairs; used for legacy/alternate flows.

---

## 13. Data Types (high level)

- **DocumentSummary:** id, filename, status, pages?, project_id?.
- **DocumentDetail:** extends Summary; full_text?.
- **RagSource:** document_id?, filename?, content?.
- **Project:** id, name, is_default, created_at, updated_at.
- **Note:** id, content, document_id?, project_id?, filename?, created_at?, updated_at?.
- **FlashcardDeck:** id, name, document_id?, project_id?, created_at?, card_count?.
- **FlashcardCard:** id, deck_id, front, back, source_document_id?.
- **Assessment:** id, project_id, title, ai_generation_enabled, ai_config, created_at, updated_at, questions[].
- **AssessmentQuestion:** id, type (short_answer|long_answer|mcq|multi), prompt, correct_answer?, options?, position.
- **Submission:** id, assessment_id, answers, status, ai_status, score_pct, graded_at, created_at.

---

## 14. Unused / Alternate Views (for context)

- **LibraryView:** Shows selected document detail (full_text) and “Create assessment” with documentId in state. Not mounted.
- **NotebookGuideView:** Corpus summary (RAG query), suggested questions (navigate to `/chat` with initialPrompt), “Quick reports” (FAQ, Study Guide, Briefing Doc) — each runs a RAG prompt and shows result + sources. Not mounted.
- **ChatView:** Chat UI with optional initial prompt from location state. Not mounted.
- **FlashcardsView:** Full-page flashcards (generate + decks + review panel). Not mounted.
- **NotesView:** Full-page notes (generate, manual add, list). Not mounted.

These can be wired into routes or reused for a different layout (e.g. sidebar + main content).

---

## 15. Accessibility & Keyboard

- **Focus:** Visible focus ring (2px accent, outline-offset) on inputs, buttons, tabbable elements.
- **Command palette:** ⌘K/Ctrl+K; Escape; Arrow keys + Enter.
- **Assessments:** Sticky nav buttons for question jump; RichTextEditor and inputs focusable.
- **Flashcards (study):** ← → navigate, Space flip, 1/2/3 rate; buttons and labels for same.
- **ARIA:** Tool rail “Tool navigation” / “Tool rail”; command palette “Command palette”; dialogs and sections labeled where implemented. No sidebar in current layout, but ProjectSidebar uses semantic sections.

---

## 16. Responsive Behavior

- **ToolRail:** At `max-width: 48rem` it moves from left vertical strip to bottom horizontal bar.
- **Focus mode:** Hides tool rail and shrinks header; canvas max-width increases.
- **Flashcards deck/review grid:** In CSS, at 48rem the two-column layout collapses to one column (class `flashcards-decks-review`).
- **Take assessment:** Layout is flexible (flex/grid) with sticky question nav; usable on smaller screens.

---

## 17. File Upload & Processing

- **Accepted types:** `.pdf`, `.docx`, `.pptx`, `.xlsx`.
- **Flow:** User clicks “Add document” (in Sources or sidebar); file input; `handleUploadFile(file)` → POST to `/documents/upload` with project_id → context refetches documents. Documents with `status === 'processing'` show spinner; context polls every 2s until none processing.
- **States:** processing → completed (with pages) or failed.

---

## 18. Sanitization & Security

- **Rich text:** Output sanitized with DOMPurify (allowed tags: p, br, strong, em, u, code, pre, lists, blockquote, sub, sup, span, div; no attributes). `sanitizeHtmlForDisplay` for display-only.
- **LaTeX:** KaTeX renders server or client-provided math; no arbitrary HTML from user in LaTeX segments.

---

This spec covers every route, canvas, view, component, API, and major UX behavior in the current `demo/web/` app. Use it to refine flows, add features, or hand off to a UX engineer for improvements.
