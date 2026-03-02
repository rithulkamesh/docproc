import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { Spinner } from './Spinner'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
  variant?: ButtonVariant
  fullWidth?: boolean
  loading?: boolean
}

export function Button({
  children,
  variant = 'primary',
  fullWidth,
  loading,
  disabled,
  className: userClass,
  ...rest
}: ButtonProps) {
  const isDisabled = disabled || loading
  const className = [
    'btn',
    `btn--${variant}`,
    fullWidth ? 'btn--full-width' : '',
    userClass,
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button
      {...rest}
      className={className}
      disabled={isDisabled}
    >
      {loading ? (
        <>
          <Spinner size="sm" />
          <span>Loading…</span>
        </>
      ) : (
        children
      )}
    </button>
  )
}
