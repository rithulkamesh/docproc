/**
 * Design tokens derived from design.json for use in JS/TS (motion, layout).
 * Colors and typography live in CSS variables; prefer var(--*) in components.
 */
export const motion = {
  durationMicro: 120,
  durationStandard: 240,
  durationPanel: 280,
  durationOverlay: 320,
  easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  easingFramer: [0.4, 0, 0.2, 1] as const,
  springPanel: { type: 'spring' as const, stiffness: 300, damping: 35 },
} as const

export const layout = {
  appMaxWidth: 'min(90vw, 120rem)',
  appHeight: 'min(92vh, 80rem)',
  appRadius: '1rem',
  outerPadding: 'clamp(1rem, 2vw, 2rem)',
  contentMaxWidth: 'min(80ch, 65vw)',
  topBarHeight: 'clamp(3.5rem, 4vw, 4.5rem)',
  navRailWidth: 'clamp(4rem, 5vw, 5rem)',
  sidePanelWidth: 'min(32rem, 40vw)',
  flashcardWidth: 'min(42rem, 60vw)',
  sectionGap: '3rem',
} as const

export const spacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
  '2xl': '3rem',
} as const
