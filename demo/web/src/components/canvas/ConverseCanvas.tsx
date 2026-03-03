import type { FormEvent } from 'react'
import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { runQuery } from '@/api/query'
import { createNote } from '@/api/notes'
import type { RagSource } from '@/types'
import { useWorkspace } from '@/context/WorkspaceContext'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { FileText } from 'lucide-react'
import { motion as motionConfig } from '@/design/tokens'

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
  const { documents, selectedDocumentId, currentProjectId } = useWorkspace()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim() || sending) return
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: input.trim() }
    setMessages((prev) => [...prev, userMessage])
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
      await createNote({ content: msg.content, documentId: selectedDocumentId ?? undefined, projectId: currentProjectId })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note')
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
          Converse
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Grounded in <strong className="text-foreground">{completedCount} document{completedCount === 1 ? '' : 's'}</strong>. Ask questions or request study material.
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{
          duration: motionConfig.durationPanel / 1000,
          ease: motionConfig.easingFramer,
        }}
      >
        <Card className="flex flex-1 flex-col overflow-hidden border-border">
        <ScrollArea className="h-[50vh] flex-1 p-4">
          <div ref={listRef} className="flex flex-col gap-8">
            {messages.map((msg, index) => (
              <MessageBlock
                key={msg.id}
                message={msg}
                index={index}
                onSaveAsNote={() => handleSaveAsNote(msg)}
              />
            ))}
            {messages.length === 0 && (
              <p className="text-muted-foreground">
                Start by asking about main ideas, definitions, or arguments across your sources.
              </p>
            )}
          </div>
        </ScrollArea>

        <form
          onSubmit={handleSubmit}
          className="flex gap-2 border-t border-border p-4"
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question or request a study artifact…"
            aria-label="Message input"
            className="flex-1"
          />
          <Button type="submit" disabled={sending || !input.trim()}>
            {sending ? '…' : 'Send'}
          </Button>
        </form>
        </Card>
      </motion.div>

      {error && (
        <p className="text-sm text-destructive">{error}</p>
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
        duration: motionConfig.durationStandard / 1000,
        ease: motionConfig.easingFramer,
        delay: index * 0.03,
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className={`rounded-lg p-4 ${isAssistant ? 'border-l-2 border-primary bg-muted/50 pl-6' : ''}`}
    >
      <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {message.role === 'user' ? 'You' : 'Workspace'}
      </p>
      <div className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
        {message.content}
      </div>
      {isAssistant && (
        <div
          className="mt-2 flex flex-wrap items-center gap-2"
          style={{ opacity: hover ? 1 : 0.7 }}
        >
          <Button variant="secondary" size="sm" onClick={onSaveAsNote}>
            <FileText className="mr-1 h-3.5 w-3.5" />
            Save as note
          </Button>
          {message.sources && message.sources.length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground">Sources</summary>
              <ul className="mt-1 list-inside list-disc space-y-0.5 text-muted-foreground">
                {message.sources.map((s, idx) => (
                  <li key={s.document_id ?? idx}>
                    <span className="font-medium text-foreground">{s.filename || 'Document'}</span>
                    {s.content ? ` — ${s.content.slice(0, 120)}${s.content.length > 120 ? '…' : ''}` : null}
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
