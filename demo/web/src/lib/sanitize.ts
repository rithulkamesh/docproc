import DOMPurify from 'dompurify'

const ALLOWED_TAGS = [
  'p', 'br', 'strong', 'b', 'em', 'i', 'u', 's', 'code', 'pre',
  'ul', 'ol', 'li', 'blockquote', 'sub', 'sup', 'span', 'div',
]
const ALLOWED_ATTR: string[] = [] // no href/style/on* to prevent injection

/**
 * Sanitize HTML before sending to backend.
 * Prevents script injection, inline JS, style injection.
 */
export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
  })
}

/**
 * Sanitize for display (e.g. result feedback). Stricter.
 */
export function sanitizeHtmlForDisplay(html: string): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['p', 'br', 'strong', 'b', 'em', 'i', 'u', 'code', 'pre', 'ul', 'ol', 'li', 'span'],
    ALLOWED_ATTR: [],
  })
}
