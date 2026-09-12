/**
 * Generic variant selection.
 *
 * `Variant.options` is an open `Record<string, string>` because the dimensions a PDP
 * exposes are category-dependent: Color/Size for apparel, Kit for a power tool,
 * Memory/Storage for a laptop. Nothing here names a dimension - every dimension, its
 * values, and their ordering are derived from the variants themselves.
 *
 * The other rule this module enforces is that only combinations the merchant actually
 * publishes are offered. A value is enabled when some real variant carries it
 * alongside the rest of the current selection, so the UI never implies the Cartesian
 * product of the dimensions.
 *
 * Pure functions, no React - which is what makes `variants.test.ts` possible.
 */

import type { Price, Variant } from '@/types/product'

export interface Dimension {
  key: string
  values: string[]
}

export type Selection = Record<string, string>

/**
 * Every option dimension and its values, in first-appearance order.
 *
 * Source order is preserved deliberately. Sizes are strings with no correct generic
 * comparator ("5.5" before "10", "Medium" before "Small"), and the merchant's own
 * order is almost always the intended one.
 */
export function getDimensions(variants: Variant[]): Dimension[] {
  const dimensions = new Map<string, Set<string>>()
  for (const variant of variants) {
    for (const [key, value] of Object.entries(variant.options)) {
      const values = dimensions.get(key) ?? new Set<string>()
      values.add(value)
      dimensions.set(key, values)
    }
  }
  return [...dimensions].map(([key, values]) => ({ key, values: [...values] }))
}

/**
 * Dimensions worth rendering: those where the user has a real choice.
 *
 * A dimension with one value across every variant ("Fit: Regular") is a label, not a
 * control, and a row of one permanently-selected chip is noise. Hiding it is safe only
 * because `resolveVariant` treats the selection as a partial constraint.
 */
export function getVisibleDimensions(variants: Variant[]): Dimension[] {
  return getDimensions(variants).filter((dimension) => dimension.values.length > 1)
}

/**
 * Can `value` still be chosen for `dimension`?
 *
 * The current value of `dimension` itself is excluded from the constraint, so a
 * selected chip never disables its own siblings' alternatives - every row stays
 * navigable no matter what is already chosen.
 */
export function isValueEnabled(
  variants: Variant[],
  selection: Selection,
  dimension: string,
  value: string,
): boolean {
  return variants.some(
    (variant) =>
      variant.options[dimension] === value &&
      Object.entries(selection).every(
        ([key, selected]) => key === dimension || variant.options[key] === selected,
      ),
  )
}

/**
 * Choose `value` for `dimension`, dropping any other selection it invalidates.
 *
 * Pruning rather than blocking the click is what keeps the user out of dead ends:
 * picking a Color that does not come in the currently selected Size clears the Size
 * instead of refusing the Color. Selecting the already-selected value clears it.
 */
export function applySelection(
  variants: Variant[],
  selection: Selection,
  dimension: string,
  value: string,
): Selection {
  if (selection[dimension] === value) {
    const { [dimension]: _cleared, ...rest } = selection
    return rest
  }

  const next: Selection = { [dimension]: value }
  for (const [key, selected] of Object.entries(selection)) {
    if (key === dimension) continue
    if (isValueEnabled(variants, next, key, selected)) next[key] = selected
  }
  return next
}

/**
 * The variant to show on open: the first one the page did not mark unavailable.
 *
 * Seeding a complete selection means price, availability and images are live
 * immediately - which matters because some pages carry a sale price only on the
 * variant, never at product level.
 */
export function initialSelection(variants: Variant[]): Selection {
  if (variants.length === 0) return {}
  const seed = variants.find((variant) => variant.available !== false) ?? variants[0]
  const visible = new Set(getVisibleDimensions(variants).map((d) => d.key))

  const selection: Selection = {}
  for (const [key, value] of Object.entries(seed.options)) {
    if (visible.has(key)) selection[key] = value
  }
  return selection
}

/**
 * The single variant matching the current selection, or null if it is still ambiguous.
 *
 * The selection is a *partial constraint*, not a lookup key. Exact equality would
 * break the moment a constant dimension is hidden from the UI: the selection would
 * carry Color and Size while the variant also carries `Fit`, and no variant would ever
 * match. Filtering and requiring a unique match also resolves early - on a product
 * where Color alone identifies the variant, one click is enough.
 */
export function resolveVariant(
  variants: Variant[],
  selection: Selection,
): Variant | null {
  const matches = variants.filter((variant) =>
    Object.entries(selection).every(([key, value]) => variant.options[key] === value),
  )
  return matches.length === 1 ? matches[0] : null
}

/**
 * The price to show, given the product and the currently resolved variant.
 *
 * Pages disagree about where a sale lives, and the two halves must be combined rather
 * than one shadowing the other. One page states the sale only at product level while
 * its variants carry the sale price with no `compare_at_price`; another states it only
 * on the variant while the product level looks like full price. Preferring the variant
 * wholesale silently drops the first case's discount.
 *
 * So: the variant wins on amount, but a product-level `compare_at_price` is carried
 * over when the variant does not have one and the two agree on the amount - i.e. when
 * they are describing the same sale.
 */
export function displayPrice(product: Price, variant: Price | null | undefined): Price {
  if (!variant) return product
  if (
    variant.compare_at_price === null &&
    product.compare_at_price !== null &&
    variant.price === product.price &&
    variant.currency === product.currency
  ) {
    return { ...variant, compare_at_price: product.compare_at_price }
  }
  return variant
}

/**
 * The image a given option value displays, if the page supplied one.
 *
 * Some merchants publish a swatch per value - a photograph of that colourway, that
 * finish, that material. Where one exists it says more than the value's name does,
 * and a shopper picks a colour by looking at it rather than by reading "Mint Green".
 *
 * Returns the image of the *first* variant carrying the value, which is the right
 * choice for the only case that matters: a swatch is a property of the value, so
 * every variant carrying that value shows the same one.
 */
export function getValueImage(
  variants: Variant[],
  dimension: string,
  value: string,
): string | null {
  const match = variants.find(
    (variant) => variant.options[dimension] === value && variant.image_urls.length > 0,
  )
  return match?.image_urls[0] ?? null
}
