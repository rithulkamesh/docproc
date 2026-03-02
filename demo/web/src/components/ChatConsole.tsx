import type { FormEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { runQuery } from '../api/query'
import type { DocumentSummary, RagSource } from '../types'
import { Button } from './Button'
import { createNote } from '../api/notes'
import { useWorkspace } from '../context/WorkspaceContext'

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

export function ChatConsole({ documents, selectedDocumentId, projectId }: ChatConsoleProps) {
  const { status } = useWorkspace()
  const defaultModel = status?.default_rag_model ?? undefined
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [ragModel, setRagModel] = useState<string>(defaultModel ?? '')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (defaultModel != null && defaultModel !== '' && !ragModel) setRagModel(defaultModel)
  }, [defaultModel, ragModel])

  useEffect(() => {
    if (!listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages])

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    if (!input.trim() || sending) return
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: input.trim() }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setSending(true)
    setError(null)
    try {
      const res = await runQuery(userMessage.content, 5, ragModel.trim() || undefined)
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
    <div className="chat-console" style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--color-bg-alt)' }}>
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
        <span>
          Grounded in{' '}
          <strong>
            {completedCount} document{completedCount === 1 ? '' : 's'}
          </strong>{' '}
          in this project. Ask questions, derive structure, or generate study material.
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
          flex: 1,
          minHeight: 0,
          padding: 'var(--space-lg)',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-md)',
        }}
      >
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-msg chat-msg--${msg.role}`} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div
              style={{
                fontSize: 11,
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                color: 'var(--color-text-muted)',
              }}
            >
              {msg.role === 'user' ? 'You' : 'Workspace'}
            </div>
            <div
              style={{
                fontSize: 14,
                lineHeight: 1.6,
                whiteSpace: 'pre-wrap',
              }}
            >
              {msg.content}
            </div>
            {msg.role === 'assistant' && (
              <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center' }}>
                <Button type="button" variant="ghost" onClick={() => handleSaveAsNote(msg)}>
                  Save as note
                </Button>
                {msg.sources && msg.sources.length > 0 && (
                  <details style={{ fontSize: 11 }}>
                    <summary style={{ cursor: 'pointer' }}>Sources</summary>
                    <ul style={{ paddingLeft: 16 }}>
                      {msg.sources.map((s, idx) => (
                        <li key={`${s.document_id ?? idx}`}>
                          <strong>{s.filename || 'Document'}</strong>
                          {s.content ? (
                            <span style={{ marginLeft: 4, color: 'var(--color-text-muted)' }}>
                              — {s.content.slice(0, 160)}
                              {s.content.length > 160 ? '…' : ''}
                            </span>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}
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

