/**
 * Mirrors the backend schema in `models.py`, one to one.
 *
 * `source` (the input filename) is deliberately absent: it is extraction metadata,
 * not product data, and nothing in the UI should surface it.
 */

export interface Price {
  price: number
  currency: string
  compare_at_price: number | null
}

export interface Variant {
  options: Record<string, string>
  sku: string | null
  gtin: string | null
  price: Price | null
  /** Tri-state. `null` means the page never said, which is not the same as false. */
  available: boolean | null
  image_urls: string[]
  url: string | null
}

export interface Product {
  id: string
  name: string
  price: Price
  description: string
  key_features: string[]
  image_urls: string[]
  video_url: string | null
  category: { name: string }
  brand: string
  colors: string[]
  variants: Variant[]
}

/**
 * What `GET /api/products` returns per item.
 *
 * Not derived from `Product` with `Pick`, because the two endpoints genuinely
 * disagree: the list endpoint flattens `category` to a string, the detail endpoint
 * keeps it as `{ name }`.
 */
export interface ProductSummary {
  id: string
  name: string
  brand: string
  price: Price
  category: string
  image_url: string | null
  colors: string[]
  variant_count: number
}
