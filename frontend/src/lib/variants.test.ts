import { describe, expect, it } from 'vitest'

import {
  applySelection,
  displayPrice,
  getVisibleDimensions,
  initialSelection,
  isValueEnabled,
  resolveVariant,
} from '@/lib/variants'
import type { Price, Variant } from '@/types/product'

function variant(options: Record<string, string>, extra: Partial<Variant> = {}): Variant {
  return {
    options,
    sku: null,
    gtin: null,
    price: null,
    available: null,
    image_urls: [],
    url: null,
    ...extra,
  }
}

// Black comes in 9 and 10, White only in 9. White/10 is a combination the merchant
// does not publish, and must never be offered.
const apparel = [
  variant({ Color: 'Black', Size: '9' }),
  variant({ Color: 'Black', Size: '10' }),
  variant({ Color: 'White', Size: '9' }),
]

describe('isValueEnabled', () => {
  it('enables a combination that exists', () => {
    expect(isValueEnabled(apparel, { Color: 'Black' }, 'Size', '10')).toBe(true)
  })

  it('disables a Cartesian combination the data does not contain', () => {
    expect(isValueEnabled(apparel, { Color: 'White' }, 'Size', '10')).toBe(false)
  })

  it('ignores the dimension being tested, so every value stays reachable', () => {
    // White is selected and 10 is impossible under it, but switching Color is allowed.
    expect(isValueEnabled(apparel, { Color: 'White', Size: '9' }, 'Color', 'Black')).toBe(
      true,
    )
  })
})

describe('applySelection', () => {
  it('prunes a size that the newly chosen color does not come in', () => {
    const selection = applySelection(apparel, { Color: 'Black', Size: '10' }, 'Color', 'White')
    expect(selection).toEqual({ Color: 'White' })
  })

  it('keeps a still-valid selection on other dimensions', () => {
    const selection = applySelection(apparel, { Color: 'Black', Size: '9' }, 'Color', 'White')
    expect(selection).toEqual({ Color: 'White', Size: '9' })
  })

  it('clears the dimension when its selected value is chosen again', () => {
    expect(applySelection(apparel, { Color: 'Black' }, 'Color', 'Black')).toEqual({})
  })
})

describe('resolveVariant', () => {
  it('returns null while the selection is still ambiguous', () => {
    expect(resolveVariant(apparel, { Color: 'Black' })).toBeNull()
  })

  it('resolves a complete selection', () => {
    expect(resolveVariant(apparel, { Color: 'Black', Size: '10' })?.options.Size).toBe('10')
  })

  it('resolves a partial selection that is already unique', () => {
    // Only one White variant exists, so Color alone identifies it.
    expect(resolveVariant(apparel, { Color: 'White' })?.options.Size).toBe('9')
  })

  it('resolves when a constant dimension is hidden from the selection', () => {
    const withConstantFit = [
      variant({ Color: 'Black', Size: '10', Fit: 'Regular' }),
      variant({ Color: 'White', Size: '10', Fit: 'Regular' }),
    ]
    // Fit is constant, so the UI hides it and it never enters the selection. Exact
    // equality against `options` would fail here; a partial constraint must not.
    expect(getVisibleDimensions(withConstantFit).map((d) => d.key)).toEqual(['Color'])
    expect(resolveVariant(withConstantFit, { Color: 'Black' })?.options.Fit).toBe('Regular')
  })
})

describe('single-dimension products', () => {
  const tool = [
    variant({ Kit: 'Battery & Charger' }, { available: true }),
    variant({ Kit: 'Tool Only' }, { available: true }),
  ]

  it('derives the dimension without any hardcoded Color/Size assumption', () => {
    expect(getVisibleDimensions(tool)).toEqual([
      { key: 'Kit', values: ['Battery & Charger', 'Tool Only'] },
    ])
  })

  it('resolves on a single click', () => {
    expect(resolveVariant(tool, { Kit: 'Tool Only' })).toBe(tool[1])
  })
})

describe('initialSelection', () => {
  it('treats available: null as unknown, not unavailable', () => {
    // Every variant here is `null`, as on a real page that never declares stock.
    // Skipping them would leave the PDP with nothing selected.
    expect(initialSelection(apparel)).toEqual({ Color: 'Black', Size: '9' })
  })

  it('skips variants the page explicitly marked unavailable', () => {
    const withSoldOut = [
      variant({ Size: 'S' }, { available: false }),
      variant({ Size: 'M' }, { available: true }),
    ]
    expect(initialSelection(withSoldOut)).toEqual({ Size: 'M' })
  })

  it('omits hidden constant dimensions', () => {
    const single = [variant({ Color: 'White Terrazzo' })]
    expect(initialSelection(single)).toEqual({})
  })

  it('handles a product with no variants', () => {
    expect(initialSelection([])).toEqual({})
  })
})

describe('displayPrice', () => {
  const price = (amount: number, compareAt: number | null = null): Price => ({
    price: amount,
    currency: 'GBP',
    compare_at_price: compareAt,
  })

  it('falls back to the product price when nothing is resolved', () => {
    expect(displayPrice(price(76.99, 109.99), null)).toEqual(price(76.99, 109.99))
  })

  it('keeps a product-level sale the variant does not repeat', () => {
    // The Nike page states the discount only at product level; its variants carry the
    // sale amount with no compare_at. Preferring the variant wholesale loses the sale.
    expect(displayPrice(price(76.99, 109.99), price(76.99))).toEqual(price(76.99, 109.99))
  })

  it('uses a variant-level sale the product does not have', () => {
    // The Ace page is the mirror image: sale lives only on the variant.
    expect(displayPrice(price(129), price(129, 159))).toEqual(price(129, 159))
  })

  it('does not carry a product sale onto a differently priced variant', () => {
    expect(displayPrice(price(76.99, 109.99), price(89.99))).toEqual(price(89.99))
  })
})
