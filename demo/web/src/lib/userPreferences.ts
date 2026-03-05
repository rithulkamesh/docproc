/**
 * User preferences persisted in localStorage (display name, avatar).
 */

const STORAGE_KEY = 'docproc-user-preferences'

export type AvatarStyle = 'avataaars' | 'initials' | 'lorelei'

export interface UserPreferences {
  displayName: string
  avatarSeed: string
  avatarStyle: AvatarStyle
}

const DEFAULT_PREFERENCES: UserPreferences = {
  displayName: '',
  avatarSeed: 'user',
  avatarStyle: 'avataaars',
}

function parseStored(raw: string | null): UserPreferences {
  if (!raw?.trim()) return { ...DEFAULT_PREFERENCES }
  try {
    const parsed = JSON.parse(raw) as Partial<UserPreferences>
    return {
      displayName: typeof parsed.displayName === 'string' ? parsed.displayName : DEFAULT_PREFERENCES.displayName,
      avatarSeed: typeof parsed.avatarSeed === 'string' && parsed.avatarSeed.trim()
        ? parsed.avatarSeed.trim()
        : DEFAULT_PREFERENCES.avatarSeed,
      avatarStyle: ['avataaars', 'initials', 'lorelei'].includes(parsed.avatarStyle ?? '')
        ? (parsed.avatarStyle as AvatarStyle)
        : DEFAULT_PREFERENCES.avatarStyle,
    }
  } catch {
    return { ...DEFAULT_PREFERENCES }
  }
}

export function loadUserPreferences(): UserPreferences {
  if (typeof window === 'undefined') return { ...DEFAULT_PREFERENCES }
  return parseStored(window.localStorage.getItem(STORAGE_KEY))
}

const PREFERENCES_CHANGED_EVENT = 'docproc-user-preferences-changed'

export function saveUserPreferences(partial: Partial<UserPreferences>): void {
  if (typeof window === 'undefined') return
  const current = loadUserPreferences()
  const next: UserPreferences = {
    displayName: partial.displayName !== undefined ? partial.displayName : current.displayName,
    avatarSeed: partial.avatarSeed !== undefined ? partial.avatarSeed : current.avatarSeed,
    avatarStyle: partial.avatarStyle !== undefined ? partial.avatarStyle : current.avatarStyle,
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  window.dispatchEvent(new CustomEvent(PREFERENCES_CHANGED_EVENT))
}

export function onUserPreferencesChange(callback: () => void): () => void {
  const handler = () => callback()
  const storageHandler = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) handler()
  }
  window.addEventListener(PREFERENCES_CHANGED_EVENT, handler)
  window.addEventListener('storage', storageHandler)
  return () => {
    window.removeEventListener(PREFERENCES_CHANGED_EVENT, handler)
    window.removeEventListener('storage', storageHandler)
  }
}

export function hasUserPreferences(): boolean {
  if (typeof window === 'undefined') return false
  return window.localStorage.getItem(STORAGE_KEY) != null
}

const DICEBEAR_BASE = 'https://api.dicebear.com/7.x'

/** Build avatar URL for the given seed and style. */
export function dicebearAvatarUrl(seed: string, style: AvatarStyle = 'avataaars'): string {
  const s = encodeURIComponent(seed || 'user')
  return `${DICEBEAR_BASE}/${style}/svg?seed=${s}`
}

/** Get initials from display name (e.g. "Jane Doe" -> "JD", "Alice" -> "A"). */
export function initialsFromDisplayName(displayName: string): string {
  const trimmed = displayName.trim()
  if (!trimmed) return 'U'
  const parts = trimmed.split(/\s+/).filter(Boolean)
  if (parts.length >= 2) {
    const first = parts[0][0]
    const last = parts[parts.length - 1][0]
    return `${first}${last}`.toUpperCase()
  }
  return trimmed.slice(0, 2).toUpperCase()
}
