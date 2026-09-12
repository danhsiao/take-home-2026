import { cn } from 'cn'

import type { Price } from '@/types/product'

/**
 * Currency comes from the data, never from a hardcoded symbol - the catalogue mixes
 * USD with GBP, and a hardcoded "$" is wrong on sight for the Nike page.
 */
function format(amount: number, currency: string): string {
  try {
    return new Intl.NumberFormat(undefined, { style: 'currency', currency }).format(amount)
  } catch {
    // An unrecognised currency code should not take the card down with it.
    return `${currency} ${amount.toFixed(2)}`
  }
}

export function PriceDisplay({
  price,
  size = 'sm',
  className,
}: {
  price: Price
  size?: 'sm' | 'lg'
  className?: string
}) {
  const onSale =
    price.compare_at_price !== null && price.compare_at_price > price.price

  return (
    <div className={cn('flex flex-wrap items-baseline gap-2', className)}>
      <span
        className={cn(
          'font-medium tabular-nums',
          size === 'lg' ? 'text-2xl' : 'text-sm',
          onSale && 'text-destructive',
        )}
      >
        {format(price.price, price.currency)}
      </span>
      {onSale && (
        <span
          className={cn(
            'text-muted-foreground line-through tabular-nums',
            size === 'lg' ? 'text-base' : 'text-xs',
          )}
        >
          {format(price.compare_at_price!, price.currency)}
        </span>
      )}
    </div>
  )
}
