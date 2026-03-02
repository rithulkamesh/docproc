/**
 * Design system tokens from design.json — neo-brutalist meets minimalist.
 * Spacing: 4/8/16/24/32px scale. Radius: cards 8–12px, buttons 6–8px.
 * Visual tokens (colors, spacing values) are defined in index.css; prefer CSS vars and classes in UI.
 */

/** Spacing scale (px) — multiples of 4/8. Use for margins, padding, gaps. */
export const spacingScale = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
} as const

/** Backward-compatible: factor * 8px scale (0.5→4, 1→8, 2→16, 3→24, 4→32). */
export function spacing(factor: number): string {
  const px = Math.round(factor * 8)
  return `${Math.max(0, px)}px`
}

export const radius = {
  sm: '6px',
  md: '8px',
  lg: '10px',
  card: '10px',
  panel: '10px',
  button: '8px',
  input: '8px',
  badge: '6px',
} as const

export const motion = {
  durationMicro: 120,
  durationPanel: 240,
  easeMicro: 'ease' as const,
  easePanel: 'cubic-bezier(0.4, 0, 0.2, 1)' as const,
  easeFramer: 'easeInOut' as const,
  transitionPanel: { duration: 0.24, ease: [0.4, 0, 0.2, 1] as const },
  springPanel: { type: 'spring' as const, stiffness: 300, damping: 35 },
} as const

/** Component tokens from design.json */
export const componentTokens = {
  cardPadding: '22px',
  cardMargin: '28px',
  buttonHeight: '38px',
} as const
