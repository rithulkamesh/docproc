import { useMemo } from 'react'
import type { Transition } from 'framer-motion'
import { theme } from '../design/theme'

export interface AnimatedPresenceRouteResult {
  /** Key for AnimatePresence (e.g. canvasMode) so content animates on change */
  presenceKey: string
  /** Transition for enter/exit (spring for panels) */
  transition: Transition
}

/**
 * Returns a key and transition config for use with AnimatePresence
 * when switching workspace panel content (e.g. by canvasMode).
 */
export function useAnimatedPresenceRoute(modeKey: string): AnimatedPresenceRouteResult {
  const transition = useMemo<Transition>(
    () => ({
      type: theme.motion.springPanel.type,
      stiffness: theme.motion.springPanel.stiffness,
      damping: theme.motion.springPanel.damping,
    }),
    []
  )
  return {
    presenceKey: modeKey,
    transition,
  }
}
