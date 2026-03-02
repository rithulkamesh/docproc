interface SpinnerProps {
  size?: 'sm' | 'md'
  className?: string
}

export function Spinner({ size = 'md', className = '' }: SpinnerProps) {
  return (
    <span
      className={`spinner ${size === 'sm' ? 'spinner-sm' : ''} ${className}`.trim()}
      aria-hidden
    />
  )
}
