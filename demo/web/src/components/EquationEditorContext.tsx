import { createContext, useContext } from 'react'

export interface EquationEditorContextValue {
  openEquationModal: (opts: {
    initialLatex?: string
    onSave?: (latex: string) => void
    onInsert?: (latex: string, type: 'inline' | 'block') => void
  }) => void
}

export const EquationEditorContext = createContext<EquationEditorContextValue | null>(null)

export function useEquationEditorContext(): EquationEditorContextValue | null {
  return useContext(EquationEditorContext)
}
