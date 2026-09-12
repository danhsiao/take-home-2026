/**
 * The entire data layer: two GETs against the FastAPI catalogue.
 *
 * Paths are origin-relative because Vite proxies `/api` to the API process in
 * development (see `vite.config.ts`), so there is no base URL to configure.
 */

import type { Product, ProductSummary } from '@/types/product'

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? 'That product could not be found.'
        : `Request failed (${response.status}). Is the API running on :8000?`,
    )
  }
  return (await response.json()) as T
}

export async function fetchProducts(): Promise<ProductSummary[]> {
  const data = await getJson<{ items: ProductSummary[]; total: number }>(
    '/api/products',
  )
  return data.items
}

export function fetchProduct(id: string): Promise<Product> {
  return getJson<Product>(`/api/products/${id}`)
}
