import { cn } from 'cn'

import {
  getValueImage,
  getVisibleDimensions,
  isValueEnabled,
  type Selection,
} from '@/lib/variants'
import type { Variant } from '@/types/product'

/**
 * One labelled chip row per option dimension, entirely derived from the data.
 *
 * Impossible values are disabled rather than hidden: removing chips as the selection
 * changes makes rows jump around and hides the shape of what the merchant offers.
 */
export function VariantSelector({
  variants,
  selection,
  onSelect,
}: {
  variants: Variant[]
  selection: Selection
  onSelect: (dimension: string, value: string) => void
}) {
  const dimensions = getVisibleDimensions(variants)
  if (dimensions.length === 0) return null

  return (
    <div className="flex flex-col gap-5">
      {dimensions.map(({ key, values }) => (
        <div key={key} className="flex flex-col gap-2">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-medium">{key}</span>
            {selection[key] && (
              <span className="text-sm text-muted-foreground">{selection[key]}</span>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {values.map((value) => {
              const enabled = isValueEnabled(variants, selection, key, value)
              const selected = selection[key] === value
              const swatch = getValueImage(variants, key, value)
              return (
                <button
                  key={`${key}:${value}`}
                  type="button"
                  disabled={!enabled}
                  aria-pressed={selected}
                  onClick={() => onSelect(key, value)}
                  className={cn(
                    'min-w-11 rounded-md border px-3 py-2 text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring',
                    selected
                      ? 'border-foreground bg-foreground text-background'
                      : 'border-border hover:border-foreground/40',
                    // Struck through so an unavailable combination reads as
                    // deliberately unavailable, not merely styled differently.
                    !enabled &&
                      'cursor-not-allowed text-muted-foreground/50 line-through hover:border-border',
                    swatch && 'flex items-center gap-2 py-1.5 pl-1.5',
                  )}
                >
                  {swatch && (
                    <img
                      src={swatch}
                      alt=""
                      aria-hidden="true"
                      loading="lazy"
                      className="size-7 shrink-0 rounded-sm object-cover"
                    />
                  )}
                  {value}
                </button>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
