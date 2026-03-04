import type { FormEvent, KeyboardEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
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
import { Button } from '@/components/ui/button'
import { FileText, Layers } from 'lucide-react'
import { motion as motionConfig } from '@/design/tokens'

const CONVERSE_HISTORY_KEY = 'docproc-converse-history'

const SUGGESTED_PROMPTS = [
  'Main ideas of the document',
  'Explain key terms',
  'Summarize in 3 bullet points',
  'Generate 5 practice questions',
]

/** Normalize LaTeX in chat: convert (( ... )) to $ ... $ for remark-math. */
function normalizeChatMath(content: string): string {
  if (!content || typeof content !== 'string') return content
  return content.replace(/\(\(([\s\S]*?)\)\)/g, (_, math) => `$${math.trim()}$`)
}

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

function loadHistory(projectId: string): ChatMessage[] {
  try {
    const raw = localStorage.getItem(`${CONVERSE_HISTORY_KEY}-${projectId}`)
    if (!raw) return []
    const parsed = JSON.parse(raw) as ChatMessage[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function saveHistory(projectId: string, messages: ChatMessage[]) {
  try {
    localStorage.setItem(`${CONVERSE_HISTORY_KEY}-${projectId}`, JSON.stringify(messages))
  } catch {
    // ignore
  }
}

export function ConverseCanvas() {
  const { documents, selectedDocumentId, currentProjectId, setContextPanelSources } = useWorkspace()
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadHistory(currentProjectId))

  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [toast, setToast] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setMessages(loadHistory(currentProjectId))
  }, [currentProjectId])

  useEffect(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')
    setContextPanelSources(lastAssistant?.sources ?? null)
    return () => setContextPanelSources(null)
  }, [messages, setContextPanelSources])

  useEffect(() => {
    saveHistory(currentProjectId, messages)
  }, [currentProjectId, messages])

  useEffect(() => {
    if (!listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
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
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setSending(true)
    setError(null)
    const assistantId = crypto.randomUUID()
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
    }
    setMessages((prev) => [...prev, assistantPlaceholder])
    const applyError = (message: string) => {
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
    try {
      const streamUsed = await runQueryStream(userMessage.content, {
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
          )
        },
        onDelta: (delta) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + delta } : m
            )
          )
        },
        onDone: () => setSending(false),
        onError: applyError,
      })
      if (!streamUsed) {
        const res = await runQuery(userMessage.content, 5)
        if (res.answer.startsWith('Query failed:')) {
          const raw = res.answer.replace(/^Query failed:\s*/, '').trim()
          applyError(raw)
        } else {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: res.answer, sources: res.sources ?? [] }
                : m
            )
          )
          setSending(false)
        }
      }
    } catch (e) {
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

  const completedCount = documents.filter((d) => d.status === 'completed').length

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

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
          Chat
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Ask anything from your <strong className="text-foreground">{completedCount} document{completedCount === 1 ? '' : 's'}</strong>.
        </p>
      </div>

      {/* Suggested prompts: only when chat is empty */}
      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2 py-2">
          {SUGGESTED_PROMPTS.map((prompt) => (
            <Button
              key={prompt}
              variant="secondary"
              size="sm"
              className="text-xs"
              onClick={() => setInput((prev) => (prev ? `${prev}\n\n${prompt}` : prompt))}
            >
              {prompt}
            </Button>
          ))}
        </div>
      )}

      {/* Message list: render when chatHistory.length > 0 */}
      <div
        ref={listRef}
        className="flex max-h-[50vh] flex-col gap-4 overflow-y-auto py-2"
      >
        {messages.length > 0 &&
          messages.map((msg, index) => (
            <MessageBlock
              key={msg.id}
              message={msg}
              index={index}
              onSaveAsNote={() => handleSaveAsNote(msg)}
              onTurnIntoFlashcards={() => handleTurnIntoFlashcards(msg)}
            />
          ))}
        {messages.length === 0 && (
          <p className="text-sm text-muted-foreground">
            Start by asking about main ideas, definitions, or arguments across your sources.
          </p>
        )}
      </div>

      {/* Input: textarea + send */}
      <form onSubmit={(e) => { e.preventDefault(); void handleSubmit(e) }} className="flex flex-col gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question or request study material…"
          aria-label="Message input"
          rows={3}
          className="min-h-[90px] w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={sending}
        />
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">Enter to send, Shift+Enter for new line</span>
          <Button type="submit" disabled={sending || !input.trim()} className="shrink-0">
            {sending ? '…' : 'Send'}
          </Button>
        </div>
      </form>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {toast && (
        <p className="text-sm text-muted-foreground rounded-md bg-muted px-3 py-2">{toast}</p>
      )}
    </div>
  )
}

function MessageBlock({
  message,
  index,
  onSaveAsNote,
  onTurnIntoFlashcards,
}: {
  message: ChatMessage
  index: number
  onSaveAsNote: () => void
  onTurnIntoFlashcards: () => void
}) {
  const isAssistant = message.role === 'assistant'

  return (
    <motion.article
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{
        duration: motionConfig.durationStandard / 1000,
        ease: motionConfig.easingFramer,
        delay: index * 0.03,
      }}
      className={`rounded-lg p-4 ${isAssistant ? 'border-l-2 border-primary bg-muted/50 pl-6' : ''}`}
    >
      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {message.role === 'user' ? 'You' : 'Workspace'}
      </p>
      <div className="text-base leading-relaxed text-foreground">
        {isAssistant ? (
          <ReactMarkdown
            remarkPlugins={[remarkMath, remarkGfm]}
            rehypePlugins={[rehypeKatex]}
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="my-2 list-disc pl-5">{children}</ul>,
              ol: ({ children }) => <ol className="my-2 list-decimal pl-5">{children}</ol>,
              li: ({ children }) => <li className="mb-0.5">{children}</li>,
              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
              code: ({ className, children, ...rest }) =>
                className?.includes('math') ? (
                  <code {...rest}>{children}</code>
                ) : (
                  <code
                    className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs"
                    {...rest}
                  >
                    {children}
                  </code>
                ),
              pre: ({ children }) => (
                <pre className="my-2 overflow-auto rounded-md bg-muted p-3 text-xs">{children}</pre>
              ),
            }}
          >
            {normalizeChatMath(message.content)}
          </ReactMarkdown>
        ) : (
          <span className="whitespace-pre-wrap">{message.content}</span>
        )}
      </div>
      {isAssistant && (
        <>
          {message.sources && message.sources.length > 0 && (
            <div className="mt-2 rounded border border-border bg-background/50 px-2 py-1.5 text-xs">
              <span className="font-medium text-muted-foreground">Sources: </span>
              {message.sources.map((s, idx) => (
                <span key={s.document_id ?? idx}>
                  {idx > 0 && ', '}
                  <span className="text-foreground">{s.display_name ?? s.filename ?? 'Document'}</span>
                  {s.content && (
                    <span className="text-muted-foreground">
                      {' '}({s.content.slice(0, 80)}{s.content.length > 80 ? '…' : ''})
                    </span>
                  )}
                </span>
              ))}
            </div>
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
    </motion.article>
  )
}
