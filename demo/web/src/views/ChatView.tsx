import { useLocation } from 'react-router-dom'
import type { FormEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { runQuery } from '../api/query'
import type { RagSource } from '../types'
import { Button } from '../components/Button'
import { Card } from '../components/Card'
import { createNote } from '../api/notes'
import type { DocumentSummary } from '../types'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: RagSource[]
}

interface ChatViewProps {
  documents: DocumentSummary[]
  selectedDocumentId: string | null
}

export function ChatView({ documents, selectedDocumentId }: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const location = useLocation()
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const state = location.state as { initialPrompt?: string } | null
    if (state?.initialPrompt) {
      setInput(state.initialPrompt)
      window.history.replaceState({}, document.title)
    }
  }, [location.state])

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
      const res = await runQuery(userMessage.content, 5)
      const assistant: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: res.answer,
        sources: res.sources,
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
      await createNote({ content: message.content, documentId: selectedDocumentId ?? undefined })
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
      return "API key missing or invalid. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env."
    }
    return msg
  }

  return (
    <div className="form-stack" style={{ height: '100%', gap: 'var(--space-lg)' }}>
      <Card>
        <div className="section-label mb-sm">CHAT</div>
        <p className="body-sm" style={{ marginTop: 0, marginBottom: 0 }}>
          Grounded in <strong>{completedCount} document{completedCount === 1 ? '' : 's'}</strong> in this notebook. Ask questions, request explanations, or generate examples.
        </p>
      </Card>

      <Card className="chat-console" style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <div ref={listRef} className="p-content form-card" style={{ flex: 1, overflowY: 'auto' }}>
          {messages.map((msg) => (
            <div key={msg.id} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div className="text-xs text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                {msg.role === 'user' ? 'You' : 'Notebook'}
              </div>
              <div className="body-sm" style={{ lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{msg.content}</div>
              {msg.role === 'assistant' && (
                <div className="gap-sm" style={{ display: 'flex', alignItems: 'center' }}>
                  <Button type="button" variant="ghost" onClick={() => handleSaveAsNote(msg)}>Save as note</Button>
                  {msg.sources && msg.sources.length > 0 && (
                    <details className="text-xs">
                      <summary style={{ cursor: 'pointer' }}>Sources</summary>
                      <ul style={{ paddingLeft: 16 }}>
                        {msg.sources.map((s, idx) => (
                          <li key={`${s.document_id ?? idx}`}>
                            <strong>{s.filename || 'Document'}</strong>
                            {s.content ? (
                              <span className="text-muted" style={{ marginLeft: 4 }}>— {s.content.slice(0, 160)}{s.content.length > 160 ? '…' : ''}</span>
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
            <div className="text-muted" style={{ fontSize: 13 }}>Ask anything about your uploaded documents.</div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="chat-input-area" style={{ padding: 'var(--space-lg)', display: 'flex', gap: 'var(--space-md)', alignItems: 'center' }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question…"
            className="input"
            style={{ flex: 1 }}
          />
          <Button type="submit" disabled={sending || !input.trim()} loading={sending}>Send</Button>
        </form>
      </Card>
      {error && <p className="text-xs" style={{ color: 'var(--color-danger)' }}>{error}</p>}
    </div>
  )
}
