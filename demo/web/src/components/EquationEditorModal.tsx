import React, { useEffect, useRef, useCallback } from 'react'
import 'mathlive'
import type { MathfieldElement } from 'mathlive'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'

export interface EquationEditorModalProps {
  open: boolean
  onClose: () => void
  initialLatex?: string
  onSave?: (latex: string) => void
  onInsert?: (latex: string, type: 'inline' | 'block') => void
}

export function EquationEditorModal({
  open,
  onClose,
  initialLatex = '',
  onSave,
  onInsert,
}: EquationEditorModalProps) {
  const mathFieldRef = useRef<MathfieldElement | null>(null)
  const isEditMode = Boolean(onSave)

  useEffect(() => {
    if (open && mathFieldRef.current) {
      mathFieldRef.current.setValue(initialLatex, { silenceNotifications: true })
      mathFieldRef.current.focus()
    }
  }, [open, initialLatex])

  const getLatex = useCallback(() => {
    const mf = mathFieldRef.current
    return (mf?.getValue?.() ?? mf?.value ?? '').trim()
  }, [])

  const handleInsert = useCallback(
    (type: 'inline' | 'block') => {
      const latex = getLatex()
      if (latex && onInsert) {
        onInsert(latex, type)
      }
      onClose()
    },
    [getLatex, onInsert, onClose]
  )

  const handleSave = useCallback(() => {
    const latex = getLatex()
    if (onSave) {
      onSave(latex)
    }
    onClose()
  }, [getLatex, onSave, onClose])

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent showClose className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{isEditMode ? 'Edit equation' : 'Insert equation'}</DialogTitle>
          <DialogDescription>
            {isEditMode
              ? 'Edit the equation below, then choose Save.'
              : 'Type or build your equation, then insert it inline or as a display block.'}
          </DialogDescription>
        </DialogHeader>
        <div className="equation-editor-modal-body">
          {React.createElement('math-field', {
            ref: mathFieldRef,
            className: 'equation-math-field',
            'virtual-keyboard-mode': 'auto',
            'smart-fence': true,
          }, initialLatex)}
        </div>
        <DialogFooter className="flex-row gap-2 sm:justify-end">
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>
          {isEditMode ? (
            <Button type="button" onClick={handleSave}>
              Save
            </Button>
          ) : (
            <>
              <Button type="button" variant="secondary" onClick={() => handleInsert('inline')}>
                Insert inline
              </Button>
              <Button type="button" onClick={() => handleInsert('block')}>
                Insert as block
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
