import type { ReactNode, HTMLAttributes } from 'react'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
}

/** Card per design.json: background card, 20–24px padding, 8–12px radius, subtle shadow (light). */
export function Card({ children, style, className = '', ...rest }: CardProps) {
  return (
    <div className={`card ${className}`.trim()} style={style} {...rest}>
      {children}
    </div>
  )
}
