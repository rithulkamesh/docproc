import type { FormEvent } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { runQuery } from '../api/query'
import type { DocumentSummary, RagSource } from '../types'
import { Button } from './Button'
import { createNote } from '../api/notes'
import { useWorkspace } from '../context/WorkspaceContext'
import {
  createNewSession,
  loadSessions,
  saveSessions,
  sessionTitleFromMessage,
  type ChatSession,
  type SessionsState,
} from '../lib/chatSessions'
const STREAM_CHUNK_INTERVAL_MS = 28

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

/** Normalize LaTeX delimiters for remark-math: \[ \] → $$ $$, \( \) → $ $, (( )) → $ $ */
function normalizeChatMath(content: string): string {
  if (!content || typeof content !== 'string') return content
  let out = content
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => `$$${math.trim()}$$`)
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => `$${math.trim()}$`)
  out = out.replace(/\(\(([\s\S]*?)\)\)/g, (_, math) => `$${math.trim()}$`)
  return out
}

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
  return normalizeChatMath(unwrapSourceFence(text))
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RagSource[]
}

interface ChatConsoleProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

function applyMessagesToSession(
  sessions: ChatSession[],
  activeId: string | null,
  updater: (messages: ChatMessage[]) => ChatMessage[]
): ChatSession[] {
  if (activeId == null) return sessions
  return sessions.map((s) =>
    s.id === activeId
      ? { ...s, messages: updater(s.messages as ChatMessage[]), updatedAt: Date.now() }
      : s
  )
}

export function ChatConsole({ documents, selectedDocumentId, projectId }: ChatConsoleProps) {
  const { status } = useWorkspace()
  const defaultModel = status?.default_rag_model ?? undefined
  const [sessionsState, setSessionsState] = useState<SessionsState>(() =>
    loadSessions(projectId)
  )
  const { sessions, activeId: activeSessionId } = sessionsState
  const activeSession = activeSessionId != null ? sessions.find((s) => s.id === activeSessionId) ?? null : null
  const messages = (activeSession?.messages ?? []) as ChatMessage[]
  const isNewChatMode = activeSessionId === null

  const setMessages = useCallback(
    (updater: React.SetStateAction<ChatMessage[]>) => {
      setSessionsState((prev) => ({
        ...prev,
        sessions: applyMessagesToSession(
          prev.sessions,
          prev.activeId,
          typeof updater === 'function' ? updater : () => updater
        ),
      }))
    },
    []
  )

  const [input, setInput] = useState('')
  const [ragModel, setRagModel] = useState<string>(defaultModel ?? '')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)
  const lastMessageRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    setSessionsState(loadSessions(projectId))
  }, [projectId])

  useEffect(() => {
    saveSessions(projectId, sessionsState)
  }, [projectId, sessionsState])

  const handleNewChat = useCallback(() => {
    setSessionsState((prev) => ({ ...prev, activeId: null }))
    setInput('')
    setError(null)
  }, [])

  const handleSelectSession = useCallback((id: string | null) => {
    setSessionsState((prev) => ({ ...prev, activeId: id }))
    setError(null)
  }, [])

  useEffect(() => {
    if (defaultModel != null && defaultModel !== '' && !ragModel) setRagModel(defaultModel)
  }, [defaultModel, ragModel])

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!input.trim() || sending) return
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: input.trim() }
    const assistantId = crypto.randomUUID()
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
    }
    if (isNewChatMode) {
      const newSession = createNewSession()
      newSession.title = sessionTitleFromMessage(userMessage.content)
      newSession.messages = [userMessage, assistantPlaceholder]
      setSessionsState((prev) => ({
        sessions: [...prev.sessions, newSession],
        activeId: newSession.id,
      }))
    } else {
      const isFirstInSession = messages.length === 0
      setMessages((prev) => [...prev, userMessage])
      if (isFirstInSession && activeSession?.title === 'New chat') {
        setSessionsState((prev) => ({
          ...prev,
          sessions: prev.sessions.map((s) =>
            s.id === prev.activeId
              ? { ...s, title: sessionTitleFromMessage(userMessage.content), updatedAt: Date.now() }
              : s
          ),
        }))
      }
      setMessages((prev) => [...prev, assistantPlaceholder])
    }
    setInput('')
    setSending(true)
    setError(null)
    try {
      const res = await runQuery(userMessage.content, 5, ragModel.trim() || undefined)
      if (res.answer.startsWith('Query failed:')) {
        const raw = res.answer.replace(/^Query failed:\s*/, '').trim()
        setError(normalizeQueryError(raw))
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: 'Sorry, the request failed.' } : m
          )
        )
        setSending(false)
        return
      }
      const full = res.answer
      const chunkSize = Math.max(1, Math.floor(full.length / 40))
      let i = 0
      const tick = () => {
        const end = Math.min(i + chunkSize, full.length)
        const chunk = full.slice(i, end)
        i = end
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk } : m))
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
    } catch (e) {
      const raw = e instanceof Error ? e.message : 'Query failed'
      setError(normalizeQueryError(raw))
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: 'Sorry, the request failed.' } : m
        )
      )
      setSending(false)
    }
  }

  const handleSaveAsNote = async (message: ChatMessage) => {
    try {
      await createNote({ content: message.content, documentId: selectedDocumentId ?? undefined, projectId })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
    }
  }

  const completedCount = documents.filter((d) => d.status === 'completed').length

  function normalizeQueryError(msg: string): string {
    if (msg.includes('DeploymentNotFound') || msg.includes('404') || msg.includes('deployment')) {
      return "AI provider isn't set up correctly. Check your config: use a valid OpenAI model (e.g. gpt-4o-mini) or a valid Azure deployment name in .env."
    }
    if (msg.includes('api_key') || msg.includes('401') || msg.includes('403')) {
      return 'API key missing or invalid. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env.'
    }
    return msg
  }

  return (
    <div className="chat-console" style={{ display: 'flex', flexDirection: 'column', minHeight: 0, background: 'var(--color-bg-alt)' }}>
      <div
        style={{
          padding: 'var(--space-lg)',
          borderBottom: `1px solid ${'var(--color-border-light)'}`,
          fontSize: 'var(--text-sm)',
          color: 'var(--color-text-muted)',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: 'var(--space-md)',
        }}
      >
        <Button type="button" variant="secondary" onClick={handleNewChat} style={{ flexShrink: 0 }}>
          New chat
        </Button>
        <select
          value={activeSessionId ?? ''}
          onChange={(e) => handleSelectSession(e.target.value === '' ? null : e.target.value)}
          title="Switch chat session"
          style={{
            maxWidth: 220,
            padding: '6px 10px',
            borderRadius: 'var(--radius-sm)',
            border: `1px solid ${'var(--color-border-light)'}`,
            fontSize: 'var(--text-sm)',
            background: 'var(--color-bg)',
            color: 'var(--color-text)',
          }}
        >
          <option value="">New chat</option>
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.title}
            </option>
          ))}
        </select>
        <span style={{ marginLeft: 'auto' }}>
          Grounded in{' '}
          <strong>
            {completedCount} document{completedCount === 1 ? '' : 's'}
          </strong>{' '}
          in this project.
        </span>
        <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <span>RAG model:</span>
          <input
            type="text"
            value={ragModel}
            onChange={(e) => setRagModel(e.target.value)}
            placeholder={defaultModel ?? 'e.g. gpt-4o'}
            title="Override chat model for answers (e.g. gpt-4o or your Azure deployment name)"
            style={{
              width: 120,
              padding: '4px 8px',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${'var(--color-border-light)'}`,
              fontSize: 'var(--text-sm)',
            }}
          />
        </label>
      </div>

      <div
        ref={listRef}
        style={{
          padding: 'var(--space-lg)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-md)',
        }}
      >
        {messages.map((msg, index) => {
          const isStreaming =
            sending && index === messages.length - 1 && msg.role === 'assistant'
          const showThinking = msg.role === 'assistant' && isStreaming && !msg.content
          const showCursor = msg.role === 'assistant' && isStreaming && msg.content.length > 0
          const showSourcesAndActions = msg.role === 'assistant' && !isStreaming
          const isLast = index === messages.length - 1
          const isUser = msg.role === 'user'
          return (
            <div
              key={msg.id}
              ref={isLast ? lastMessageRef : undefined}
              className="chat-msg"
              style={{
                display: 'flex',
                width: '100%',
                justifyContent: isUser ? 'flex-end' : 'flex-start',
              }}
            >
              <div
                style={{
                  maxWidth: '85%',
                  borderRadius: 16,
                  padding: 'var(--space-md) var(--space-lg)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                  ...(isUser
                    ? {
                        background: 'var(--color-primary)',
                        color: 'var(--color-primary-foreground)',
                        borderTopRightRadius: 6,
                      }
                    : {
                        border: '1px solid var(--color-border-light)',
                        background: 'var(--color-bg-alt)',
                        color: 'var(--color-text)',
                        borderTopLeftRadius: 6,
                      }),
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    textTransform: 'uppercase',
                    letterSpacing: '0.1em',
                    opacity: isUser ? 0.9 : 1,
                    color: isUser ? 'inherit' : 'var(--color-text-muted)',
                  }}
                >
                  {msg.role === 'user' ? 'You' : 'Workspace'}
                </div>
                <div style={{ fontSize: 14, lineHeight: 1.6 }}>
                  {msg.role === 'assistant' ? (
                    <>
                      {showThinking ? (
                        <span style={{ color: 'var(--color-text-muted)' }}>
                          Searching your documents…
                          <span style={{ marginLeft: 4, display: 'inline-flex', gap: 3 }}>
                            <span className="thinking-dot" />
                            <span className="thinking-dot" />
                            <span className="thinking-dot" />
                          </span>
                        </span>
                      ) : (
                        <>
                          <div className="chat-console-markdown">
                            <ReactMarkdown
                              remarkPlugins={[remarkMath, remarkGfm]}
                              rehypePlugins={[rehypeKatex]}
                              components={{
                                p: ({ children }) => <p>{children}</p>,
                                ul: ({ children }) => <ul>{children}</ul>,
                                ol: ({ children }) => <ol>{children}</ol>,
                                li: ({ children }) => <li>{children}</li>,
                                strong: ({ children }) => <strong>{children}</strong>,
                                blockquote: ({ children }) => <blockquote>{children}</blockquote>,
                                code: ({ className, children, ...rest }) =>
                                  className?.includes('math') ? (
                                    <code {...rest}>{children}</code>
                                  ) : (
                                    <code {...rest}>{children}</code>
                                  ),
                                pre: ({ children }) => <pre>{children}</pre>,
                              }}
                            >
                              {trimAssistantSignOff(msg.content)}
                            </ReactMarkdown>
                          </div>
                          {showCursor && <span className="stream-cursor" aria-hidden />}
                        </>
                      )}
                    </>
                  ) : (
                    <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
                  )}
                </div>
                {showSourcesAndActions && (
                  <>
                    {msg.sources && msg.sources.length > 0 && (
                      <details
                        className="chat-console-sources"
                        style={{
                          marginTop: 'var(--space-sm)',
                          fontSize: 11,
                          border: '1px solid var(--color-border-light)',
                          borderRadius: 8,
                          overflow: 'hidden',
                          background: 'var(--color-bg)',
                        }}
                      >
                        <summary
                          style={{
                            cursor: 'pointer',
                            padding: '8px 12px',
                            listStyle: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 8,
                            fontWeight: 600,
                            color: 'var(--color-text-muted)',
                          }}
                        >
                          <span>Sources</span>
                          <span
                            style={{
                              borderRadius: 4,
                              padding: '2px 6px',
                              background: 'var(--color-bg-alt)',
                              fontFamily: 'var(--font-mono)',
                              fontSize: 10,
                            }}
                          >
                            {msg.sources.length}
                          </span>
                        </summary>
                        <ul style={{ padding: 0, margin: 0, listStyle: 'none', borderTop: '1px solid var(--color-border-light)' }}>
                          {msg.sources.map((s, idx) => (
                            <li
                              key={s.document_id ?? idx}
                              style={{
                                borderBottom: idx < msg.sources!.length - 1 ? '1px solid var(--color-border-light)' : 'none',
                                padding: 'var(--space-sm) var(--space-md)',
                              }}
                            >
                              <p style={{ margin: '0 0 6px', fontWeight: 600, fontSize: 12, color: 'var(--color-text)' }}>
                                {s.display_name ?? s.filename ?? 'Document'}
                              </p>
                              {s.content && (
                                <div className="chat-console-markdown chat-console-source-excerpt">
                                  <ReactMarkdown
                                    remarkPlugins={[remarkMath, remarkGfm]}
                                    rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
                                    components={{
                                      p: ({ children }) => <p style={{ margin: '0 0 4px', fontSize: 12 }}>{children}</p>,
                                      h1: ({ children }) => <h1 style={{ margin: '6px 0 4px', fontSize: 14, fontWeight: 600 }}>{children}</h1>,
                                      h2: ({ children }) => <h2 style={{ margin: '6px 0 4px', fontSize: 13, fontWeight: 600 }}>{children}</h2>,
                                      h3: ({ children }) => <h3 style={{ margin: '4px 0 2px', fontSize: 12, fontWeight: 600 }}>{children}</h3>,
                                      ul: ({ children }) => <ul style={{ margin: '4px 0', paddingLeft: 16 }}>{children}</ul>,
                                      ol: ({ children }) => <ol style={{ margin: '4px 0', paddingLeft: 16 }}>{children}</ol>,
                                      li: ({ children }) => <li style={{ marginBottom: 2 }}>{children}</li>,
                                      strong: ({ children }) => <strong>{children}</strong>,
                                      code: ({ className, children, ...rest }) =>
                                        className?.includes('math') ? <code {...rest}>{children}</code> : <code style={{ fontSize: 11, padding: '1px 4px', borderRadius: 4, background: 'var(--color-bg-alt)' }} {...rest}>{children}</code>,
                                      pre: ({ children }) => <pre style={{ margin: '4px 0', padding: 8, fontSize: 11, overflow: 'auto', borderRadius: 4, background: 'var(--color-bg-alt)' }}>{children}</pre>,
                                    }}
                                  >
                                    {normalizeSourceExcerpt(truncateSourceSafe(s.content, 520))}
                                  </ReactMarkdown>
                                </div>
                              )}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center', marginTop: 'var(--space-sm)' }}>
                      <Button type="button" variant="ghost" onClick={() => handleSaveAsNote(msg)}>
                        Save as note
                      </Button>
                    </div>
                  </>
                )}
              </div>
            </div>
          )
        })}
        {messages.length === 0 && (
          <div style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>
            Start by asking about the main ideas, definitions, or arguments across your sources.
          </div>
        )}
      </div>

      <form
        className="chat-input-area"
        onSubmit={handleSubmit}
        style={{
          borderTop: `1px solid ${'var(--color-border-light)'}`,
          padding: 'var(--space-lg)',
          display: 'flex',
          gap: 'var(--space-md)',
          alignItems: 'center',
        }}
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask a question or request a study artifact…"
          style={{
            flex: 1,
            padding: '12px 14px',
            borderRadius: 'var(--radius-md)',
            border: `1px solid ${'var(--color-border-light)'}`,
            fontSize: 'var(--text-base)',
            fontFamily: 'var(--font-family)',
            backgroundColor: 'var(--color-bg-interactive)',
            color: 'var(--color-text)',
          }}
        />
        <Button type="submit" disabled={sending || !input.trim()} loading={sending}>
          Send
        </Button>
      </form>
      {error && (
        <p
          style={{
            fontSize: 12,
            color: 'var(--color-danger)',
            margin: 0,
            padding: `0 ${'var(--space-lg)'} ${'var(--space-md)'}`,
          }}
        >
          {error}
        </p>
      )}
    </div>
  )
}

