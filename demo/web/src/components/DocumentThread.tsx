import type { FormEvent } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { runQuery } from '../api/query'
import { createNote } from '../api/notes'
import type { DocumentSummary, RagSource } from '../types'
import { theme } from '../design/theme'
import { SoftButton } from './SoftButton'
import {
  createNewSession,
  loadSessions,
  saveSessions,
  sessionTitleFromMessage,
  type ChatSession,
  type SessionsState,
} from '../lib/chatSessions'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RagSource[]
}

interface DocumentThreadProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
  projectId: string
}

function normalizeQueryError(msg: string): string {
  if (msg.includes('DeploymentNotFound') || msg.includes('404') || msg.includes('deployment')) {
    return "AI provider isn't set up correctly. Check your config: use a valid OpenAI model (e.g. gpt-4o-mini) or a valid Azure deployment name in .env."
  }
  if (msg.includes('api_key') || msg.includes('401') || msg.includes('403')) {
    return 'API key missing or invalid. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env.'
  }
  return msg
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

export function DocumentThread({
  documents,
  selectedDocumentId,
  projectId,
}: DocumentThreadProps) {
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
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

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
    if (!listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!input.trim() || sending) return
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: input.trim() }
    if (isNewChatMode) {
      const newSession = createNewSession()
      newSession.title = sessionTitleFromMessage(userMessage.content)
      newSession.messages = [userMessage]
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
    }
    setInput('')
    setSending(true)
    setError(null)
    try {
      const res = await runQuery(userMessage.content, 5)
      if (res.answer.startsWith('Query failed:')) {
        const raw = res.answer.replace(/^Query failed:\s*/, '').trim()
        setError(normalizeQueryError(raw))
        return
      }
      const assistant: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: res.answer,
        sources: res.sources ?? [],
      }
      setMessages((prev) => [...prev, assistant])
    } catch (e) {
      const raw = e instanceof Error ? e.message : 'Query failed'
      setError(normalizeQueryError(raw))
    } finally {
      setSending(false)
    }
  }

  const handleSaveAsNote = async (msg: ChatMessage) => {
    try {
      await createNote({ content: msg.content, documentId: selectedDocumentId ?? undefined, projectId })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
    }
  }

  const completedCount = documents.filter((d) => d.status === 'completed').length

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--color-bg)',
      }}
    >
      <div
        style={{
          flexShrink: 0,
          padding: 'var(--space-md)',
          paddingBottom: 'var(--space-sm)',
          borderBottom: 'var(--border-subtle)',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          gap: 'var(--space-md)',
        }}
      >
        <SoftButton onClick={handleNewChat}>New chat</SoftButton>
        <select
          value={activeSessionId ?? ''}
          onChange={(e) => handleSelectSession(e.target.value === '' ? null : e.target.value)}
          title="Switch chat session"
          style={{
            maxWidth: 200,
            padding: '6px 10px',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--border-subtle)',
            fontSize: 'var(--text-sm)',
            background: 'var(--color-bg-alt)',
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
        <p
          style={{
            margin: 0,
            fontSize: 'var(--text-sm)',
            lineHeight: theme.lineHeight.body,
            color: 'var(--color-text-muted)',
          }}
        >
          Grounded in{' '}
          <strong style={{ color: 'var(--color-text)' }}>
            {completedCount} document{completedCount === 1 ? '' : 's'}
          </strong>
          . Ask questions or request study material.
        </p>
      </div>

      <div
        ref={listRef}
        style={{
          flex: 1,
          minHeight: 0,
          padding: 'var(--space-lg)',
          paddingTop: 'var(--space-lg)',
          paddingBottom: 'var(--space-md)',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '3rem',
        }}
      >
        {messages.map((msg, index) => (
          <MessageBlock
            key={msg.id}
            message={msg}
            index={index}
            onSaveAsNote={() => handleSaveAsNote(msg)}
          />
        ))}
        {messages.length === 0 && (
          <p
            style={{
              margin: 0,
              fontSize: 'var(--text-base)',
              lineHeight: theme.lineHeight.body,
              color: 'var(--color-text-muted)',
            }}
          >
            Start by asking about main ideas, definitions, or arguments across your sources.
          </p>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        style={{
          flexShrink: 0,
          borderTop: 'var(--border-subtle)',
          padding: 'var(--space-lg)',
          paddingTop: 'var(--space-md)',
          paddingBottom: 'var(--space-md)',
          display: 'flex',
          gap: 'var(--space-md)',
          alignItems: 'center',
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or request a study artifact…"
          aria-label="Message input"
          style={{
            flex: 1,
            padding: `${'var(--space-sm)'} ${'var(--space-md)'}`,
            borderRadius: 'var(--radius-md)',
            border: 'var(--border-subtle)',
            fontSize: 'var(--text-base)',
            fontFamily: 'var(--font-family)',
            lineHeight: theme.lineHeight.body,
            background: 'var(--color-bg-alt)',
            color: 'var(--color-text)',
            transition: 'border-color 120ms ease',
          }}
        />
        <SoftButton type="submit" disabled={sending || !input.trim()}>
          {sending ? '…' : 'Send'}
        </SoftButton>
      </form>

      {error && (
        <p
          style={{
            margin: 0,
            padding: 'var(--space-lg)',
            paddingBottom: 'var(--space-md)',
            fontSize: 'var(--text-sm)',
            color: 'var(--color-danger)',
          }}
        >
          {error}
        </p>
      )}
    </div>
  )
}

function MessageBlock({
  message,
  index,
  onSaveAsNote,
}: {
  message: ChatMessage
  index: number
  onSaveAsNote: () => void
}) {
  const [hover, setHover] = useState(false)
  const isAssistant = message.role === 'assistant'

  return (
    <motion.article
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{
        duration: theme.motion.durationPanel / 1000,
        ease: [0.4, 0, 0.2, 1],
        delay: index * 0.03,
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        padding: 'var(--space-md)',
        paddingLeft: isAssistant ? theme.spacing(2.5) : 'var(--space-md)',
        borderLeft: isAssistant ? '2px solid var(--color-accent)' : 'none',
        borderRadius: 'var(--radius-md)',
        background: isAssistant ? 'var(--color-bg-interactive)' : 'transparent',
      }}
    >
      <div
        style={{
          fontSize: 'var(--text-xs)',
          fontWeight: 600,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          color: 'var(--color-text-muted)',
          marginBottom: 'var(--space-xs)',
        }}
      >
        {message.role === 'user' ? 'You' : 'Workspace'}
      </div>
      <div
        style={{
          fontSize: 'var(--text-base)',
          lineHeight: theme.lineHeight.body,
          whiteSpace: 'pre-wrap',
          color: 'var(--color-text)',
        }}
      >
        {message.content}
      </div>
      {isAssistant && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 'var(--space-sm)',
            alignItems: 'center',
            marginTop: 'var(--space-sm)',
            opacity: hover ? 1 : 0.7,
            transition: 'opacity 120ms ease',
          }}
        >
          <SoftButton onClick={onSaveAsNote}>Save as note</SoftButton>
          {message.sources && message.sources.length > 0 && (
            <details style={{ fontSize: 'var(--text-xs)' }}>
              <summary style={{ cursor: 'pointer', color: 'var(--color-text-muted)' }}>Sources</summary>
              <ul style={{ paddingLeft: '1.25rem', margin: '0.25rem 0 0', color: 'var(--color-text-muted)' }}>
                {message.sources.map((s, idx) => (
                  <li key={`${s.document_id ?? idx}`}>
                    <strong style={{ color: 'var(--color-text)' }}>{s.display_name ?? s.filename ?? 'Document'}</strong>
                    {s.content ? (
                      <span style={{ marginLeft: 'var(--space-xs)' }}>
                        — {s.content.slice(0, 120)}
                        {s.content.length > 120 ? '…' : ''}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </motion.article>
  )
}
