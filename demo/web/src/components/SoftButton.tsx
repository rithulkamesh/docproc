import type { ButtonHTMLAttributes, ReactNode } from 'react'

interface SoftButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
  active?: boolean
}

/** Text-style button: soft background on hover/active, no filled accent. */
export function SoftButton({
  children,
  active,
  disabled,
  className: userClass,
  ...rest
}: SoftButtonProps) {
  const className = ['soft-btn', active ? 'soft-btn--active' : '', userClass].filter(Boolean).join(' ')
  return (
    <button type="button" {...rest} disabled={disabled} className={className}>
      {children}
    </button>
  )
}
