import type { ComponentPropsWithoutRef, FormEvent, KeyboardEvent, ReactNode } from 'react'
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { runQuery, runQueryStream } from '@/api/query'
import { createNote } from '@/api/notes'
import { generateFlashcardsFromText } from '@/api/flashcards'
import type { RagSource } from '@/types'
import { useWorkspace } from '@/context/WorkspaceContext'
import { useAIProvider } from '@/context/AIProviderContext'
import { loadDocumentMessages, saveDocumentMessages } from '@/lib/chatSessions'
import { Button } from '@/components/ui/button'
import { FileText, Layers } from 'lucide-react'

const STUDY_ACTIONS = [
  {
    title: 'Explain key concepts',
    description: 'Understand difficult topics in the document',
    prompt: 'Explain the key concepts in this document in simple terms.',
  },
  {
    title: 'Summarize document',
    description: 'Get a quick overview',
    prompt: 'Summarize this document in a few paragraphs.',
  },
  {
    title: 'Generate flashcards',
    description: 'Create review cards from the content',
    prompt: 'Generate flashcards to help me review the main points of this document.',
  },
  {
    title: 'Generate practice questions',
    description: 'Prepare for exams',
    prompt: 'Generate practice questions to test my understanding of this document.',
  },
]

/** Normalize LaTeX delimiters for remark-math: unescape \\, then \[ \] → $$ $$, \( \) → $ $, (( )) → $ $ */
function normalizeChatMath(content: string): string {
  if (!content || typeof content !== 'string') return content
  let out = content
  // Unescape double backslashes from API/JSON so \frac etc. work
  out = out.replace(/\\\\/g, '\\')
  // Display math: \[ ... \] → $$ ... $$ (multiline-safe)
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => `$$${math.trim()}$$`)
  // Inline math: \( ... \) → $ ... $
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => `$${math.trim()}$`)
  // Legacy (( ... )) → $ ... $
  out = out.replace(/\(\(([\s\S]*?)\)\)/g, (_, math) => `$${math.trim()}$`)
  return out
}

const SIGN_OFF_PATTERNS = [
  /\n\s*Let me know if[^.!]*[.!]\s*$/i,
  /\n\s*Feel free to (?:ask|reach out)[^.!]*[.!]\s*$/i,
  /\n\s*I(?:'m)? (?:hope|glad)[^.!]*[.!]\s*$/i,
  /\n\s*If you have (?:any )?more questions[^.!]*[.!]\s*$/i,
  /\n\s*Happy to (?:help|clarify)[^.!]*[.!]\s*$/i,
  /\n\s*Don't hesitate to ask[^.!]*[.!]\s*$/i,
]

function trimAssistantSignOff(text: string): string {
  if (!text || typeof text !== 'string') return text
  let out = text.trimEnd()
  let changed = true
  while (changed) {
    changed = false
    for (const re of SIGN_OFF_PATTERNS) {
      if (re.test(out)) {
        out = out.replace(re, '').trimEnd()
        changed = true
        break
      }
    }
  }
  return out
}

const STREAM_CHUNK_INTERVAL_MS = 28

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RagSource[]
}

function normalizeQueryError(msg: string): string {
  if (msg.includes('DeploymentNotFound') || msg.includes('404') || msg.includes('deployment')) {
    return "AI provider isn't set up correctly. Check your config."
  }
  if (msg.includes('api_key') || msg.includes('401') || msg.includes('403')) {
    return 'API key missing or invalid.'
  }
  return msg
}

export function ConverseCanvas() {
  const { documents, selectedDocumentId, currentProjectId, setContextPanelSources } = useWorkspace()
  const { config: aiConfig } = useAIProvider()
  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    loadDocumentMessages(currentProjectId, selectedDocumentId)
  )

  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)
  const lastMessageRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const streamBufferRef = useRef<string>('')
  const streamFlushTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const streamingAssistantIdRef = useRef<string | null>(null)

  useEffect(() => {
    setMessages(loadDocumentMessages(currentProjectId, selectedDocumentId))
  }, [currentProjectId, selectedDocumentId])

  useEffect(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')
    setContextPanelSources(lastAssistant?.sources ?? null)
    return () => setContextPanelSources(null)
  }, [messages, setContextPanelSources])

  useEffect(() => {
    saveDocumentMessages(currentProjectId, selectedDocumentId, messages)
  }, [currentProjectId, selectedDocumentId, messages])

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 3000)
    return () => clearTimeout(t)
  }, [toast])

  const handleSubmit = async (e?: FormEvent) => {
    e?.preventDefault()
    if (!input.trim() || sending) return
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: input.trim() }
    const assistantId = crypto.randomUUID()
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
    }

    setMessages((prev) => [...prev, userMessage])
    setMessages((prev) => [...prev, assistantPlaceholder])

    setInput('')
    setSending(true)
    setError(null)
    streamingAssistantIdRef.current = assistantId
    streamBufferRef.current = ''

    const flushStreamBuffer = () => {
      if (!streamBufferRef.current || !streamingAssistantIdRef.current) return
      const id = streamingAssistantIdRef.current
      const chunk = streamBufferRef.current
      streamBufferRef.current = ''
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, content: m.content + chunk } : m))
      )
    }

    const stopStreamFlush = () => {
      if (streamFlushTimerRef.current) {
        clearInterval(streamFlushTimerRef.current)
        streamFlushTimerRef.current = null
      }
      flushStreamBuffer()
      streamingAssistantIdRef.current = null
    }

    streamFlushTimerRef.current = setInterval(flushStreamBuffer, STREAM_CHUNK_INTERVAL_MS)

    const applyError = (message: string) => {
      stopStreamFlush()
      setError(normalizeQueryError(message))
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || 'Sorry, the request failed.' }
            : m
        )
      )
      setSending(false)
    }
    const aiRequestConfig = {
      model: aiConfig.model || undefined,
      api_key: aiConfig.apiKey || undefined,
      provider: aiConfig.provider || undefined,
    }
    try {
      const streamUsed = await runQueryStream(userMessage.content, {
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
          )
        },
        onDelta: (delta) => {
          streamBufferRef.current += delta
        },
        onDone: () => {
          stopStreamFlush()
          setSending(false)
        },
        onError: applyError,
      }, aiRequestConfig)
      if (!streamUsed) {
        stopStreamFlush()
        const res = await runQuery(userMessage.content, 5, aiRequestConfig)
        if (res.answer.startsWith('Query failed:')) {
          const raw = res.answer.replace(/^Query failed:\s*/, '').trim()
          applyError(raw)
        } else {
          // Simulate streaming for non-streaming API so UX is consistent
          const full = res.answer
          const chunkSize = Math.max(1, Math.floor(full.length / 40))
          let i = 0
          const tick = () => {
            const end = Math.min(i + chunkSize, full.length)
            const chunk = full.slice(i, end)
            i = end
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: m.content + chunk } : m
              )
            )
            if (i < full.length) {
              setTimeout(tick, STREAM_CHUNK_INTERVAL_MS)
            } else {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, sources: res.sources ?? [] } : m
                )
              )
              setSending(false)
            }
          }
          setTimeout(tick, STREAM_CHUNK_INTERVAL_MS)
        }
      }
    } catch (e) {
      stopStreamFlush()
      applyError(e instanceof Error ? e.message : 'Query failed')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSubmit()
    }
  }

  const handleSaveAsNote = async (msg: ChatMessage) => {
    try {
      await createNote({
        content: msg.content,
        documentId: selectedDocumentId ?? undefined,
        projectId: currentProjectId,
      })
      setToast('Saved as note')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
    }
  }

  const handleTurnIntoFlashcards = async (msg: ChatMessage) => {
    try {
      await generateFlashcardsFromText({
        text: msg.content.slice(0, 8000),
        projectId: currentProjectId,
      })
      setToast('Flashcards created')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create flashcards')
    }
  }

  if (documents.length === 0) {
    return (
      <div className="flex min-h-[40vh] flex-col items-center justify-center gap-8 text-center">
        <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          No documents yet
        </p>
        <p className="max-w-[36ch] text-base leading-relaxed text-foreground">
          Add a document in Sources to start chatting and generating study material.
        </p>
        <p className="text-sm text-muted-foreground">
          Or press <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-xs">⌘K</kbd> for commands.
        </p>
      </div>
    )
  }

  const selectedDocument = documents.find((d) => d.id === selectedDocumentId)
  const documentLabel = selectedDocument?.display_name ?? selectedDocument?.filename ?? 'Document'

  return (
    <div className="flex flex-col flex-1 min-h-0 gap-6">
      <div className="flex flex-col flex-1 min-w-0 min-h-0 rounded-xl border border-border/80 bg-background/50 shadow-sm overflow-hidden">
        {/* Header: document context */}
        <header className="shrink-0 px-5 pt-5 pb-1">
          <p className="text-sm text-muted-foreground">
            Document: <span className="font-medium text-foreground">{documentLabel}</span>
          </p>
        </header>

        {/* Conversation history: above the input */}
        <div
          ref={listRef}
          className="flex-1 overflow-y-auto min-h-0 px-5 flex flex-col gap-5"
          style={{ paddingTop: 24 }}
        >
          {messages.length > 0 &&
            messages.map((msg, index) => {
              const isLast = index === messages.length - 1
              const isUser = msg.role === 'user'
              const isStreaming = sending && isLast && msg.role === 'assistant'
              return (
                <div
                  key={msg.id}
                  ref={isLast ? lastMessageRef : undefined}
                  className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <MessageBlock
                    message={msg}
                    index={index}
                    isStreaming={isStreaming}
                    onSaveAsNote={() => handleSaveAsNote(msg)}
                    onTurnIntoFlashcards={() => handleTurnIntoFlashcards(msg)}
                  />
                </div>
              )
            })}
          {messages.length === 0 && (
            <p className="text-sm text-muted-foreground">
              Your conversation with this document will appear here.
            </p>
          )}
        </div>

        {/* Chat input: below conversation */}
        <div className="shrink-0 px-5 pb-5" style={{ paddingTop: 24 }}>
          <form
            onSubmit={(e) => { e.preventDefault(); void handleSubmit(e) }}
            className="flex flex-col gap-2"
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your document…"
              aria-label="Message input"
              rows={3}
              className="min-h-[80px] max-h-[200px] w-full resize-y rounded-xl border-2 border-border bg-background px-4 py-3 text-base placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-shadow hover:border-border focus:border-primary/50"
              disabled={sending}
            />
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-muted-foreground">Enter to send, Shift+Enter for new line</span>
              <Button type="submit" disabled={sending || !input.trim()} className="shrink-0">
                {sending ? '…' : 'Send'}
              </Button>
            </div>
          </form>
          {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
          {toast && <p className="mt-2 text-sm text-muted-foreground rounded-lg bg-muted px-3 py-2">{toast}</p>}
        </div>

        {/* Study actions: only when no messages yet */}
        {messages.length === 0 && (
          <div className="shrink-0 px-5 pb-5" style={{ paddingTop: 24 }}>
            <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3">
              Study actions
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {STUDY_ACTIONS.map((action) => (
                <button
                  key={action.title}
                  type="button"
                  onClick={() => {
                    setInput(action.prompt)
                    textareaRef.current?.focus()
                  }}
                  className="text-left rounded-xl border border-border/80 bg-muted/20 hover:bg-muted/40 px-4 py-3 transition-colors"
                >
                  <p className="font-medium text-foreground text-sm">{action.title}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{action.description}</p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const markdownChatComponents = {
  p: ({ children }: { children?: ReactNode }) => (
    <p className="mb-3 last:mb-0 leading-relaxed text-foreground">{children}</p>
  ),
  h1: ({ children }: { children?: ReactNode }) => (
    <h1 className="mt-5 mb-2 text-lg font-semibold text-foreground first:mt-0">{children}</h1>
  ),
  h2: ({ children }: { children?: ReactNode }) => (
    <h2 className="mt-4 mb-1.5 border-b border-border/60 pb-1 text-base font-semibold text-foreground first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }: { children?: ReactNode }) => (
    <h3 className="mt-3 mb-1 text-sm font-semibold text-foreground first:mt-0">{children}</h3>
  ),
  h4: ({ children }: { children?: ReactNode }) => (
    <h4 className="mt-2.5 mb-1 text-sm font-medium text-foreground first:mt-0">{children}</h4>
  ),
  h5: ({ children }: { children?: ReactNode }) => (
    <h5 className="mt-2 mb-0.5 text-sm font-medium text-muted-foreground first:mt-0">{children}</h5>
  ),
  h6: ({ children }: { children?: ReactNode }) => (
    <h6 className="mt-2 mb-0.5 text-xs font-medium text-muted-foreground uppercase tracking-wide first:mt-0">
      {children}
    </h6>
  ),
  ul: ({ children }: { children?: ReactNode }) => (
    <ul className="my-2 list-disc space-y-0.5 pl-5">{children}</ul>
  ),
  ol: ({ children }: { children?: ReactNode }) => (
    <ol className="my-2 list-decimal space-y-0.5 pl-5">{children}</ol>
  ),
  li: ({ children }: { children?: ReactNode }) => (
    <li className="leading-relaxed">{children}</li>
  ),
  strong: ({ children }: { children?: ReactNode }) => (
    <strong className="font-semibold text-foreground">{children}</strong>
  ),
  blockquote: ({ children }: { children?: ReactNode }) => (
    <blockquote className="my-2 border-l-2 border-primary/50 pl-3 text-muted-foreground">
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...rest }: ComponentPropsWithoutRef<'code'>) =>
    className?.includes('math') ? (
      <code {...rest}>{children}</code>
    ) : (
      <code
        className="rounded bg-muted/80 px-1.5 py-0.5 font-mono text-[0.85em] text-foreground"
        {...rest}
      >
        {children}
      </code>
    ),
  pre: ({ children }: { children?: ReactNode }) => (
    <pre className="my-2 overflow-auto rounded-lg border border-border/50 bg-muted/50 px-3 py-2.5 text-[0.8rem] leading-relaxed">
      {children}
    </pre>
  ),
}

function ThinkingDots() {
  return (
    <span className="inline-flex gap-0.5" aria-hidden>
      <span
        className="h-1.5 w-1.5 rounded-full bg-muted-foreground/70 animate-thinking"
        style={{ animationDelay: '0ms' }}
      />
      <span
        className="h-1.5 w-1.5 rounded-full bg-muted-foreground/70 animate-thinking"
        style={{ animationDelay: '200ms' }}
      />
      <span
        className="h-1.5 w-1.5 rounded-full bg-muted-foreground/70 animate-thinking"
        style={{ animationDelay: '400ms' }}
      />
    </span>
  )
}

function StreamCursor() {
  return (
    <span
      className="inline-block h-4 w-0.5 -translate-y-0.5 align-middle bg-primary animate-stream-cursor"
      aria-hidden
    />
  )
}

/** Unwrap optional leading code fence only; preserve all newlines and rest of content. */
function unwrapSourceFence(text: string): string {
  const raw = text.trimStart()
  if (raw.startsWith('```markdown')) return raw.slice(11).trimStart()
  if (raw.startsWith('```')) {
    const after = raw.slice(3)
    const end = after.indexOf('```')
    if (end !== -1) return after.slice(0, end).trimEnd()
    return after.trimEnd()
  }
  return text.trim()
}

/** Truncate at a safe point: prefer last newline to avoid cutting mid-LaTeX; avoid cutting inside $$ or $. */
function truncateSourceSafe(content: string, maxLen: number): string {
  if (content.length <= maxLen) return content
  let cut = Math.min(maxLen, content.length)
  const segment = content.slice(0, cut)
  const lastNewline = segment.lastIndexOf('\n')
  if (lastNewline > cut * 0.4) cut = lastNewline + 1
  let out = content.slice(0, cut).trimEnd()
  const displayCount = (out.match(/\$\$/g) ?? []).length
  if (displayCount % 2 !== 0) {
    const lastDD = out.lastIndexOf('$$')
    if (lastDD >= 0) out = out.slice(0, lastDD).trimEnd()
  }
  const inlineCount = (out.replace(/\$\$/g, '').match(/\$/g) ?? []).length
  if (inlineCount % 2 !== 0) {
    const lastD = out.lastIndexOf('$')
    if (lastD >= 0) out = out.slice(0, lastD).trimEnd()
  }
  return (out || content.slice(0, maxLen).trimEnd()) + '…'
}

function normalizeSourceExcerpt(text: string): string {
  if (!text || typeof text !== 'string') return ''
  const unwrapped = unwrapSourceFence(text)
  return normalizeChatMath(unwrapped)
}

function SourceExcerpt({ content, maxLength = 520 }: { content: string; maxLength?: number }) {
  const excerpt = truncateSourceSafe(content, maxLength)
  const normalized = normalizeSourceExcerpt(excerpt)
  if (!normalized) return null
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none text-xs break-words">
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
        components={markdownChatComponents}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  )
}

function SourcesToggle({ sources, messageId }: { sources: RagSource[]; messageId: string }) {
  if (!sources.length) return null
  return (
    <details className="group mt-2 rounded-lg border border-border bg-muted/30">
      <summary className="cursor-pointer list-none px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span className="inline-flex items-center gap-1.5">
          <span>Sources</span>
          <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">
            {sources.length}
          </span>
        </span>
      </summary>
      <ul className="border-t border-border px-0 py-2">
        {sources.map((s, idx) => (
          <li
            key={`${messageId}-source-${idx}`}
            className="border-b border-border/50 px-3 py-2 last:border-b-0"
          >
            <p className="mb-1.5 text-xs font-semibold text-foreground">
              {s.display_name ?? s.filename ?? 'Document'}
            </p>
            {s.content ? (
              <div className="rounded bg-background/60 p-2 text-muted-foreground">
                <SourceExcerpt content={s.content} />
              </div>
            ) : null}
          </li>
        ))}
      </ul>
    </details>
  )
}

function MessageBlock({
  message,
  index: _index,
  isStreaming,
  onSaveAsNote,
  onTurnIntoFlashcards,
}: {
  message: ChatMessage
  index: number
  isStreaming: boolean
  onSaveAsNote: () => void
  onTurnIntoFlashcards: () => void
}) {
  const isAssistant = message.role === 'assistant'
  const isUser = message.role === 'user'
  const showThinking = isAssistant && isStreaming && !message.content
  const showCursor = isAssistant && isStreaming && message.content.length > 0
  const showSourcesAndActions = isAssistant && !isStreaming

  return (
    <article
      className={`w-full max-w-[85%] shrink-0 rounded-2xl px-4 py-3 shadow-sm ${
        isUser
          ? 'rounded-tr-md bg-primary text-primary-foreground'
          : 'rounded-tl-md border border-border bg-muted/60 text-foreground'
      }`}
    >
      <p className={`mb-1.5 text-xs font-semibold uppercase tracking-wide ${isUser ? 'text-primary-foreground/90' : 'text-muted-foreground'}`}>
        {message.role === 'user' ? 'You' : 'Workspace'}
      </p>
      <div className="text-base leading-relaxed">
        {isAssistant ? (
          <>
            {showThinking ? (
              <p className="text-muted-foreground">
                Searching your documents… <ThinkingDots />
              </p>
            ) : (
              <>
                <div className="chat-markdown-with-math [&_.katex]:text-inherit">
                  <ReactMarkdown
                    remarkPlugins={[remarkMath, remarkGfm]}
                    rehypePlugins={[[rehypeKatex, { throwOnError: false, errorColor: 'var(--color-muted-foreground)' }]]}
                    components={markdownChatComponents}
                  >
                    {normalizeChatMath(trimAssistantSignOff(message.content))}
                  </ReactMarkdown>
                </div>
                {showCursor && <StreamCursor />}
              </>
            )}
          </>
        ) : (
          <span className="whitespace-pre-wrap">{message.content}</span>
        )}
      </div>
      {showSourcesAndActions && (
        <>
          {message.sources && message.sources.length > 0 && (
            <SourcesToggle sources={message.sources} messageId={message.id} />
          )}
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Button variant="secondary" size="sm" onClick={onSaveAsNote}>
              <FileText className="mr-1 h-3.5 w-3.5" />
              Save as note
            </Button>
            <Button variant="secondary" size="sm" onClick={onTurnIntoFlashcards}>
              <Layers className="mr-1 h-3.5 w-3.5" />
              Turn into flashcards
            </Button>
          </div>
        </>
      )}
    </article>
  )
}
