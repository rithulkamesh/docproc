/**
 * UI theme preference persisted in localStorage.
 */

const STORAGE_KEY = 'docproc-theme'

export const THEME_IDS = [
  'paper',
  'snow',
  'cream',
  'rose',
  'lavender',
  'ocean',
  'mint',
  'sand',
  'nordlight',
  'pearl',
  'dark',
  'midnight',
  'solarized',
  'nord',
  'dracula',
  'monokai',
  'onedark',
  'gruvbox',
  'tokyo',
  'slate',
] as const

export type ThemeId = (typeof THEME_IDS)[number]

export type ThemeVariant = 'light' | 'dark'

export const THEME_LABELS: Record<ThemeId, string> = {
  paper: 'Paper',
  snow: 'Snow',
  cream: 'Cream',
  rose: 'Rose',
  lavender: 'Lavender',
  ocean: 'Ocean',
  mint: 'Mint',
  sand: 'Sand',
  nordlight: 'Nord Light',
  pearl: 'Pearl',
  dark: 'Dark',
  midnight: 'Midnight',
  solarized: 'Solarized',
  nord: 'Nord',
  dracula: 'Dracula',
  monokai: 'Monokai',
  onedark: 'One Dark',
  gruvbox: 'Gruvbox',
  tokyo: 'Tokyo Night',
  slate: 'Slate',
}

export const THEME_VARIANT: Record<ThemeId, ThemeVariant> = {
  paper: 'light',
  snow: 'light',
  cream: 'light',
  rose: 'light',
  lavender: 'light',
  ocean: 'light',
  mint: 'light',
  sand: 'light',
  nordlight: 'light',
  pearl: 'light',
  dark: 'dark',
  midnight: 'dark',
  solarized: 'dark',
  nord: 'dark',
  dracula: 'dark',
  monokai: 'dark',
  onedark: 'dark',
  gruvbox: 'dark',
  tokyo: 'dark',
  slate: 'dark',
}

const THEME_SET = new Set<string>(THEME_IDS)
const DEFAULT_THEME: ThemeId = 'dark'

export function loadTheme(): ThemeId {
  if (typeof window === 'undefined') return DEFAULT_THEME
  const stored = window.localStorage.getItem(STORAGE_KEY)
  if (stored && THEME_SET.has(stored)) return stored as ThemeId
  return DEFAULT_THEME
}

export function saveTheme(themeId: ThemeId): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(STORAGE_KEY, themeId)
}

export function applyTheme(themeId: ThemeId): void {
  if (typeof document === 'undefined') return
  document.documentElement.dataset.theme = themeId
}

export function themeIdsByVariant(): { light: ThemeId[]; dark: ThemeId[] } {
  const light = THEME_IDS.filter((id) => THEME_VARIANT[id] === 'light')
  const dark = THEME_IDS.filter((id) => THEME_VARIANT[id] === 'dark')
  return { light, dark }
}

// Apply saved theme immediately so no flash of wrong theme
if (typeof document !== 'undefined') {
  applyTheme(loadTheme())
}
