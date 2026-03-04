# docproc // edu — Product Redesign Spec

**Version:** 1.0  
**Audience:** Students preparing for exams with dense material  
**Core idea:** A **thinking workspace** and **learning engine**, not a CRUD dashboard.

---

## 1. Redesigned Information Architecture

### 1.1 Experience model

The product is organized around a single cycle:

**DOCUMENT → UNDERSTAND → PRACTICE → TEST**

- **Document:** Upload and index; documents are first-class objects.
- **Understand:** Chat, summaries, notes — exploration and comprehension.
- **Practice:** Flashcards and spaced repetition.
- **Test:** AI-graded assessments and feedback.

Navigation and layout should make this cycle visible and guide users through it.

### 1.2 Primary navigation (sidebar)

| Item | Purpose | Route / mode |
|------|--------|---------------|
| **Home** | Study dashboard: overview, progress, next actions | `/` (default view) |
| **Chat** | Perplexity-style Q&A over documents | `/` + mode `converse` |
| **Notes** | Summaries and sections linked to docs | `/` + mode `notes` |
| **Flashcards** | Decks and study session | `/` + mode `flashcards` |
| **Tests** | Create / take / results | `/assessments/*` + list entry in sidebar |
| **Sources** | Documents; upload and manage | `/` + mode `sources` |

**Settings** remains in top bar or user menu; not in primary study flow.

- **Lighter, faster nav:** Notion/Linear style — icon + short label, subtle active state, no heavy cards. Optional: collapsible to icons-only.
- **Context over chrome:** The right panel (context) changes by workspace; the sidebar stays minimal.

### 1.3 Route structure

- **`/`** — Main workspace. Query param or context drives **center** content: Home | Chat | Notes | Flashcards | Tests (list) | Sources.
- **`/assessments`** — List of assessments (create, take, view results); can be reached from sidebar “Tests” or from Home.
- **`/assessments/create`** — Create assessment (unchanged conceptually).
- **`/assessments/:id/take`** — Take assessment (full-bleed, focused).
- **`/assessments/:id/result/:submissionId`** — Result + feedback (full-bleed).
- **`/assessments/:id/submissions`** — Submissions list.
- **`/settings`** — Settings (unchanged).

Home is the default view at `/` when no mode is selected (or mode=home).

---

## 2. New Layout System

### 2.1 Three-column layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TopBar (project, theme, user)                                                │
├──────────┬────────────────────────────────────────────┬─────────────────────┤
│ Sidebar  │  Study Workspace (center)                  │  Context Panel      │
│ (nav)    │  — Home / Chat / Notes / Flashcards /      │  (right, dynamic)   │
│          │    Tests list / Sources                    │  — Sources refs     │
│          │                                            │  — Referenced text  │
│          │  Max-width content area, not full-bleed    │  — Notes / cards    │
│          │  within this column                        │  — Collapsible      │
└──────────┴────────────────────────────────────────────┴─────────────────────┘
```

- **Sidebar:** Fixed width (e.g. `clamp(4.5rem, 6vw, 6rem)` expanded, or ~12rem with labels). Background: subtle surface (e.g. `--color-bg-alt` or card).
- **Workspace:** Flexible width, `min-width: 0`, scrollable. Content inside uses a readable max-width (e.g. `min(80ch, 65vw)` or slightly wider for chat).
- **Context panel:** Fixed or min/max width (e.g. `min(24rem, 28vw)`), collapsible. Content depends on center view (see Section 5).

### 2.2 Layout rules

- **Full-bleed only** for: Take Assessment, Assessment Result, Create Assessment, Settings. These hide sidebar and context panel (or show minimal chrome).
- **Home default:** On `/`, if no mode, show Home (study dashboard). First-time or empty state can nudge “Add a document” or “Start chatting.”
- **Reduce empty space:** No single giant card in the center. Use dense but breathable sections; compact headers; inline actions on documents and messages.

### 2.3 Responsive behavior

- **Desktop:** Sidebar + Workspace + Context (context collapsible).
- **Tablet / narrow:** Sidebar can collapse to icons; context becomes overlay or bottom sheet.
- **Mobile:** Single column; sidebar → drawer or bottom nav; context → slide-over or inline.

---

## 3. Wireframe Descriptions by Screen

### 3.1 Home / Overview (Study dashboard)

**Purpose:** One place to see state and “what’s next.” Feels like the cockpit of the learning engine.

- **Header:** “Study workspace” or project name; optional short greeting.
- **Documents block:** Row or grid of document chips (icon, name, status). Each chip has inline actions: “Summary,” “Flashcards,” “Test,” “Chat.” Clicking a chip can open Chat with that doc in context or scroll to it in Sources.
- **Flashcard decks:** Compact list: deck name, card count, “Continue” or “Start.” Progress indicator (e.g. X/Y reviewed today or “last studied 2 days ago”).
- **Tests / assessments:** Recent or available assessments; “Take” or “View result” or “Create new.”
- **Progress / momentum:** Small “streak” or “studied X days” and “Recommended: Continue Chapter 3” / “Review weak topics” / “Generate flashcards from [doc].” These are tappable and drive the user into Chat, Flashcards, or Tests.
- **Empty state:** No docs → prominent “Add document” and short explanation. No decks/tests → prompts to generate from documents.

**Wireframe (text):**

```
[ Study workspace — Project Name                    ]

[ Documents (3)                    ]  [ + Add document ]
  [PDF] Chapter 3.pdf    Ready · 12p   [Summary] [Cards] [Test] [Chat]
  [PDF] Notes week 2    Processing 67%
  [DOC] Essay draft     Ready · 4p    [Summary] [Cards] [Test] [Chat]

[ Flashcard decks ]
  Biology Ch3 · 8 cards    [Continue]   last studied yesterday
  History dates · 12       [Start]

[ Tests ]
  Midterm practice · 10 Q    [Take]    [View last result]
  [ Create assessment ]

[ Recommended ]
  → Continue studying Chapter 3
  → Review weak topics from last test
  → Generate flashcards from "Notes week 2" when ready
```

### 3.2 Chat (Perplexity for your documents)

**Purpose:** Ask questions and get answers grounded in uploaded docs; turn answers into notes or flashcards.

- **Header:** “Chat” + short line: “Ask anything from your N documents.”
- **Suggested prompts:** 3–5 chips or buttons (e.g. “Main ideas of Ch3,” “Explain concept X,” “Summarize in 3 bullet points,” “Make 5 practice questions”). Click fills input or sends.
- **Message list:** User and assistant messages; assistant messages show **sources** (document + snippet) in a compact strip or expandable section below the answer. No huge empty card; messages are the content.
- **Quick actions on assistant messages:** “Save as note,” “Turn into flashcards” (and optionally “Copy”).
- **Input:** Large textarea (e.g. 3–4 lines min), placeholder: “Ask a question or request study material…” Send button; optional attachment or “from document” selector.
- **Empty state:** Suggested prompts only + input; no “No documents” in this view if user came from Home (redirect empty users to Home or Sources).

**Wireframe (text):**

```
Chat — Ask anything from your 3 documents

[ Main ideas Ch3 ] [ Explain key terms ] [ 3 bullet summary ] [ Practice questions ]

—————————————————————————————————————————
You: What is the main argument in Chapter 3?
—————————————————————————————————————————
Workspace: [Answer text…]

  Sources: Chapter 3.pdf (p.12), Notes week 2 (excerpt)
  [ Save as note ]  [ Turn into flashcards ]
—————————————————————————————————————————

[ Ask a question or request study material…                    ] [ Send ]
```

### 3.3 Documents / Sources (core objects)

**Purpose:** Documents are the center; every doc offers the same next steps.

- **Upload zone:** Prominent but not dominant — drag-and-drop or “Add document” button at top.
- **Document list/cards:** Each row/card: icon, name, status (Ready / Processing / Failed), page count. **Inline actions on each row:** “Summary,” “Flashcards,” “Test,” “Chat” (or “Ask”). So the flow is: see doc → act on doc without switching views.
- **Selection:** Selecting a doc can set “current document” for Chat/Notes/Flashcards (e.g. “from this document”).
- **Processing:** Clear progress (e.g. “Extracting… 67%”) and “Ready” when done.

**Wireframe (text):**

```
Sources — Documents ground chat, notes, flashcards, and tests

[ + Add document ]  or drop files here (PDF, DOCX, PPTX, XLSX)

  Chapter 3.pdf       Ready · 12 pages   [ Summary ] [ Flashcards ] [ Test ] [ Chat ]
  Notes week 2.pdf    Extracting… 67%
  Essay draft.docx    Ready · 4 pages   [ Summary ] [ Flashcards ] [ Test ] [ Chat ]
```

### 3.4 Notes

**Purpose:** Summaries and sections linked to documents; part of “Understand.”

- **Header:** “Notes” + “Sections linked to your documents.”
- **AI summary:** One collapsible “Generate summary” (from document or pasted text). Result: preview + “Save as section.”
- **Sections list:** Dense list of notes (title or first line, linked doc, last edited). Click to expand/edit. Inline “Export PDF” or “Download” where useful.
- **Add section:** Button to add blank or “from document” section. Reduce empty space by keeping sections in one scrollable list, not one card per section with large padding.

### 3.5 Flashcards (learning game)

**Purpose:** Study session feels like a game: progress, streak, clear feedback.

- **Deck list (when not in session):** Deck name, card count, “Study” button. Optional “Last studied,” “Mastered / Review” counts if backend supports.
- **Study session:**
  - **Top:** Progress bar (X of Y cards) + optional streak or “Session goal” (e.g. “5 more to complete”).
  - **Center:** One card; **flip animation** (existing flip is good; keep it smooth and obvious).
  - **After flip:** Buttons: “Again” / “Hard” / “Good” / “Easy” (or simplified: “Review again” / “Got it”). Spaced-repetition feedback: e.g. “See you in 2 days” or “Next review: tomorrow.”
  - **End of session:** Completion state: “You reviewed 8 cards,” “3 to review again,” optional confetti or checkmark. “Study again” or “Back to decks.”
- **Generate deck:** Same as now (from document or text, count, deck name) but visually grouped under “New deck” so the main view is “Your decks” and “Study.”

**Wireframe (text) — study mode:**

```
  [=========>                    ]  5 / 12  ·  Streak: 2

        ┌─────────────────────────────────────┐
        │  What is the capital of France?     │
        │         [ tap to flip ]              │
        └─────────────────────────────────────┘

  [ Again ]  [ Hard ]  [ Good ]  [ Easy ]     ← → nav   Space flip
```

### 3.6 Tests (assessment experience)

**Purpose:** Feels like a real assessment: clear timing, question nav, and results that teach.

- **List (sidebar or Home):** Assessment name, question count, “Take” / “View result” / “Submissions.”
- **Take assessment:**
  - **Sticky header:** Assessment title, **timer** (visible, e.g. “12:45 left”), “Submit” button.
  - **Question nav:** Numbers (1 2 3 …) or a compact strip; current question highlighted; answered state (e.g. dot or check).
  - **Body:** One question at a time (or scrollable list); clear question text and answer inputs (MCQ, short/long). Progress indicator (e.g. “Question 3 of 10”).
- **Results:**
  - **Score and summary:** Overall score, time taken, maybe a simple bar or list by question (correct/incorrect).
  - **Per-question:** Your answer, model answer, and feedback. “Weak topics” or “Review these” derived from wrong answers.
  - **Next steps:** “Generate flashcards from wrong answers,” “Chat about this topic,” “Retake.”

---

## 4. Component Redesign Suggestions

### 4.1 Sidebar (replacing NavRail)

- **Component:** `StudySidebar` or keep `NavRail` with new structure.
- **Content:** Icon + label per item (Home, Chat, Notes, Flashcards, Tests, Sources). Optional collapse to icons-only with tooltip.
- **Active state:** Subtle background + left border or dot, not a heavy pill.
- **Spacing:** Tight vertical rhythm (e.g. 6–8px gap). Optional section divider above “Sources” or “Tests.”
- **Tests:** Can open a sub-list (recent assessments) or link to `/assessments` with a list there.

### 4.2 Top bar

- **Keep:** Project switcher, theme toggle, user/settings.
- **Optional:** Global “Command palette” (⌘K) for “Go to Chat,” “Add document,” “Start flashcards,” etc.
- **Remove or soften:** “Docs: N” and “Last indexed” from primary prominence; move to Settings or footer of Sources.

### 4.3 Document chip / row

- **New or refactor:** `DocumentRow` or `DocumentChip`: icon (file type), name, status, page count.
- **Inline actions:** Small buttons or icon buttons: Summary, Flashcards, Test, Chat. Tooltips for clarity.
- **Selection:** Clicking row selects doc (for context); actions trigger the right flow (e.g. “Chat” opens Chat with doc context).

### 4.4 Chat message block

- **Assistant message:** Clear “Workspace” or “AI” label; body (markdown); **sources** always visible in a compact block (document name + snippet or “View in context panel”). Actions: “Save as note,” “Turn into flashcards.”
- **User message:** Simple bubble or left-aligned block.
- **Layout:** No single large card wrapping the whole thread; messages are the unit. Slightly larger input area (textarea min-height ~4em).

### 4.5 Suggested prompts

- **Component:** Horizontal strip of chips or buttons, above or below input. Copy: short, actionable (“Main ideas,” “Explain X,” “Summarize,” “Practice questions”).
- **Interaction:** Click inserts into input or sends immediately (configurable).

### 4.6 Flashcard card (study mode)

- **Keep:** 3D flip animation.
- **Add:** Progress bar at top; streak or “Session progress” text; rating buttons with clear labels (Again / Hard / Good / Easy or simplified).
- **Completion:** Dedicated “Session complete” state with summary and CTA.

### 4.7 Context panel

- **Container:** Right column, collapsible, scrollable. Title: “Sources,” “From this message,” “Notes,” etc.
- **Content components:** 
  - **Source ref:** Document name, optional snippet; link or “Show in Sources.”
  - **Note preview:** Title or first line, link to Notes.
  - **Flashcards from chat:** “N cards created” + link to deck or study.

### 4.8 Home dashboard blocks

- **Reusable:** `DashboardSection` with title and optional “See all” link.
- **Document block:** Uses `DocumentRow`/chips + inline actions.
- **Recommendation block:** List of links/buttons (“Continue Chapter 3,” “Review weak topics,” “Generate flashcards from …”).

---

## 5. Right Context Panel (dynamic content)

- **Chat open:** Show sources for the **selected or latest message** (referenced paragraphs/documents). Optional: “Notes from this chat” or “Flashcards created.”
- **Notes open:** Show “From document: [name]” or list of linked docs for current note.
- **Flashcards open:** Show “From deck: [name]” or “Cards in this deck” summary.
- **Sources open:** Optional: “Selected document” detail (pages, status, quick actions again) or leave panel empty/collapsed.
- **Home open:** Optional: “Recent” or “Suggested document” in panel.
- **Tests list or Take/Result:** Can hide context panel (full-bleed) or show “Assessment info” / “Weak topics.”

Implementation: one `ContextPanel` component that reads current route/mode and selected message/deck/doc and renders the right subview.

---

## 6. Interaction Improvements

- **Keyboard:** Keep and document: ⌘K command palette; in flashcards: Space (flip), ← → (prev/next), 1–3 (rate). In chat: Enter to send (Shift+Enter newline).
- **Document → action:** From Sources or Home, “Summary” / “Flashcards” / “Test” / “Chat” open the right view with that document in context (e.g. Chat pre-scoped to doc; Flashcards “from this document”).
- **Chat → notes/flashcards:** One click from assistant message to “Save as note” or “Turn into flashcards”; success toast and optional “Open in Notes” / “Study now.”
- **Progress and momentum:** On Home, show “Studied X days” or “Y cards reviewed this week” if data exists; “Recommended next” actions are clickable.
- **Empty states:** Every view has a clear empty state and CTA (e.g. “Add document,” “Start a chat,” “Generate a deck”). No dead ends.
- **Loading:** Skeleton or inline “Generating…” for AI actions; progress for document processing. No full-page spinners where a small inline state suffices.

---

## 7. Visual Design Direction

### 7.1 Traits (Notion / Perplexity / Linear / Arc)

- **Calm:** Neutral backgrounds, low saturation accents, no busy patterns.
- **Modern:** Generous whitespace in content (not chrome), clear typography, subtle shadows or borders.
- **Academic:** Readable fonts, good contrast for long reading; optional serif for headings if it fits.
- **Minimal but alive:** Subtle motion (page transitions, card flip, progress bars); optional micro-interactions (hover states, button feedback). No dashboard clutter.

### 7.2 What to avoid

- Admin/dashboard: no dense tables, no “Cards” for every single item, no gray-on-gray grids.
- Toy-like: no loud colors or playful illustrations unless deliberately chosen for onboarding.
- Empty and static: avoid one big empty card in the center; use the dashboard and inline actions to keep the screen purposeful.

### 7.3 Concrete suggestions

- **Typography:** Keep a clear hierarchy (e.g. one heading font, one body). Slightly larger body for chat and notes (e.g. 16px) for readability.
- **Color:** Accent for primary actions (e.g. “Send,” “Study,” “Generate”); success for “Saved,” “Complete”; muted for secondary. Use existing `--color-*` variables; tune for calm + academic.
- **Surfaces:** Differentiate sidebar (slightly elevated), workspace (base), context panel (elevated or same as sidebar). Cards only where needed (e.g. message blocks, document rows), not as the only container.
- **Motion:** Page/mode transitions (existing Framer); progress bars animate; flashcard flip; optional “Saved” checkmark or short toast. Keep durations short (200–300ms).
- **Icons:** Consistent set (e.g. Lucide); use for nav and inline actions. Optional: simple illustration only on empty states (e.g. “Add your first document”).

---

## 8. Making the Experience More Engaging for Students

- **Progress and momentum:** Home dashboard and optional streak or “studied X days” / “Y cards this week” so effort feels visible.
- **Recommended next actions:** “Continue Chapter 3,” “Review weak topics,” “Generate flashcards from [doc]” turn the app into a guide, not just tools.
- **Documents at the center:** Inline “Summary / Flashcards / Test / Chat” on every document so the path from reading to practice is one click.
- **Chat as the assistant:** Suggested prompts and “Save as note” / “Turn into flashcards” make the assistant feel helpful and actionable.
- **Flashcards as a game:** Progress bar, streak, flip animation, and completion feedback make sessions feel like a clear, achievable task.
- **Tests that teach:** Results with per-question feedback and “weak topics” or “Review these” plus links to Chat or Flashcards close the loop (Test → Understand → Practice).
- **Reduced friction:** Fewer clicks to start studying (e.g. “Study” from Home or from document row); command palette for power users.
- **Consistency:** Same three-column layout and same “document actions” everywhere so the mental model is simple: documents → understand → practice → test.

---

## 9. Implementation Order (suggested)

1. **Layout:** Implement three-column shell (sidebar, workspace, context panel); route/mode drives workspace content; Home as default view.
2. **Home:** Build study dashboard (documents, decks, tests, recommended actions) using existing APIs.
3. **Sidebar:** Replace or extend NavRail with new nav items (Home, etc.) and lighter styling.
4. **Chat:** Suggested prompts, message layout with visible sources, quick actions (save as note, turn into flashcards), larger input.
5. **Sources:** Document list with inline actions (Summary, Flashcards, Test, Chat); keep upload and processing UX.
6. **Context panel:** Dynamic content by mode (sources for chat, etc.); collapsible.
7. **Flashcards:** Progress bar, streak (if data exists), rating labels, completion state; keep flip animation.
8. **Notes:** Denser layout, same functionality; optional link from context panel.
9. **Tests:** Sticky timer, question nav, result visualization and weak topics; optional “Create flashcards from wrong answers.”
10. **Visual pass:** Typography, spacing, motion, and empty states aligned to this spec.

---

*End of design spec.*
