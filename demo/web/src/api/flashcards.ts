import { apiClient } from './client'

export interface FlashcardDeck {
  id: string
  name: string
  document_id?: string | null
  project_id?: string | null
  created_at?: string
  card_count?: number
}

export interface FlashcardCard {
  id: string
  deck_id: string
  front: string
  back: string
  source_document_id?: string | null
}

interface ListDecksResponse {
  decks: FlashcardDeck[]
}

interface ListCardsResponse {
  cards: FlashcardCard[]
}

export async function listDecks(documentId?: string, projectId?: string): Promise<FlashcardDeck[]> {
  const params = new URLSearchParams()
  if (documentId) params.set('document_id', documentId)
  if (projectId) params.set('project_id', projectId)
  const query = params.toString() ? `?${params.toString()}` : ''
  const data = await apiClient.get<ListDecksResponse>(`/flashcards/decks${query}`)
  return data.decks ?? []
}

export async function listCards(deckId: string): Promise<FlashcardCard[]> {
  const data = await apiClient.get<ListCardsResponse>(`/flashcards/decks/${encodeURIComponent(deckId)}/cards`)
  return data.cards ?? []
}

export async function deleteDeck(deckId: string): Promise<void> {
  await apiClient.delete<void>(`/flashcards/decks/${encodeURIComponent(deckId)}`)
}

export async function generateFlashcardsFromDocument(options: {
  documentId: string
  count: number
  deckName?: string
  projectId?: string
}): Promise<void> {
  await apiClient.post('/flashcards/generate', {
    source_type: 'document',
    document_id: options.documentId,
    project_id: options.projectId,
    count: options.count,
    deck_name: options.deckName,
  })
}

export async function generateFlashcardsFromText(options: {
  text: string
  count: number
  deckName?: string
  projectId?: string
}): Promise<void> {
  await apiClient.post('/flashcards/generate', {
    source_type: 'text',
    text: options.text,
    count: options.count,
    deck_name: options.deckName,
    project_id: options.projectId,
  })
}

export async function createDeckFromPairs(options: {
  name: string
  pairs: { question: string; answer: string }[]
  documentId?: string
  projectId?: string
}): Promise<void> {
  await apiClient.post('/flashcards/decks', {
    name: options.name,
    pairs: options.pairs,
    document_id: options.documentId,
    project_id: options.projectId,
  })
}

