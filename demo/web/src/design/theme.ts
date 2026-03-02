import { radius as systemRadius, motion, spacing, spacingScale, componentTokens } from './system'

/**
 * Theme object for layout/motion and legacy usage.
 * Visual design tokens (colors, typography, spacing scale) live in index.css and design.json.
 * Prefer CSS classes and var(--*) in components; use theme for motion (e.g. Framer) or layout constants in JS.
 */
/** Surface tokens — map to design.json backgrounds (primary, card, interactive, hover). */
export const surfaces = {
  base: 'var(--color-bg)',
  muted: 'var(--color-bg-interactive)',
  hover: 'var(--color-bg-hover)',
  elevated: 'var(--color-bg-alt)',
  borderSubtle: 'var(--border-subtle)',
} as const

export const theme = {
  colors: {
    background: 'var(--color-bg)',
    backgroundAlt: 'var(--color-bg-alt)',
    backgroundElevated: 'var(--color-bg-elevated)',
    backgroundInteractive: 'var(--color-bg-interactive)',
    borderStrong: 'var(--color-border-strong)',
    borderLight: 'var(--color-border-light)',
    text: 'var(--color-text)',
    textMuted: 'var(--color-text-muted)',
    textTertiary: 'var(--color-text-tertiary)',
    accent: 'var(--color-accent)',
    accentSoft: 'var(--color-accent-soft)',
    danger: 'var(--color-danger)',
    dangerSoft: 'var(--color-danger-soft)',
    success: 'var(--color-success)',
    successSoft: 'var(--color-success-soft)',
    badge: 'var(--color-badge)',
    btnPrimaryText: 'var(--color-btn-primary-text)',
  },
  surfaces,
  fonts: {
    heading: 'var(--font-family)',
    body: 'var(--font-family)',
    mono: 'var(--font-mono)',
  },
  fontSizes: {
    xs: 'var(--text-xs)',
    sm: 'var(--text-sm)',
    base: 'var(--text-base)',
    lg: 'var(--text-lg)',
    xl: 'var(--text-xl)',
    '2xl': 'var(--text-2xl)',
    md: 'var(--text-sm)',
    '3xl': 'var(--text-2xl)',
  },
  lineHeight: {
    body: 1.5,
    heading: 1.25,
  },
  borderWidth: {
    thin: '1px',
    strong: '1px',
    accent: '1px',
  },
  radius: {
    ...systemRadius,
  },
  shadow: {
    sm: 'var(--shadow-card)',
    md: 'var(--shadow-card)',
    lg: '0 4px 16px rgba(0, 0, 0, 0.08)',
  },
  spacing,
  spacingScale,
  componentTokens,
  transition: `${motion.durationMicro}ms ${motion.easeMicro}`,
  transitionPanel: `${motion.durationPanel}ms ${motion.easePanel}`,
  motion: {
    durationMicro: motion.durationMicro,
    durationPanel: motion.durationPanel,
    easeMicro: motion.easeMicro,
    easePanel: motion.easePanel,
    easeFramer: motion.easeFramer,
    transitionPanel: motion.transitionPanel,
    springPanel: motion.springPanel,
  },
  layout: {
    sidebarWidth: 'clamp(16rem, 18vw, 22rem)',
    mainMaxWidth: 'min(72ch, 60vw)',
    utilityWidth: 'min(28rem, 35vw)',
    paddingX: 'clamp(1rem, 2vw, 2rem)',
    sectionGap: '3rem',
  },
  contentGap: '32px',
} as const
