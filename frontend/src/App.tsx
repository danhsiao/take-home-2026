import { useEffect, useState } from 'react'

import { AspectRatio } from '@/components/ui/aspect-ratio'
import { Skeleton } from '@/components/ui/skeleton'
import { ProductDialog } from '@/components/ProductDialog'
import { ProductGrid } from '@/components/ProductGrid'
import { fetchProducts } from '@/api/products'
import type { ProductSummary } from '@/types/product'

export default function App() {
  const [products, setProducts] = useState<ProductSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  useEffect(() => {
    fetchProducts()
      .then(setProducts)
      .catch((cause: Error) => setError(cause.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="mx-auto flex max-w-7xl items-baseline justify-between px-6 py-5">
          <span className="text-lg font-semibold tracking-tight">Channel3</span>
          {!loading && !error && (
            <span className="text-sm text-muted-foreground">
              {products.length} {products.length === 1 ? 'product' : 'products'}
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-10">
        {loading && <GridSkeleton />}

        {error && (
          <div className="py-24 text-center">
            <p className="text-sm text-destructive">{error}</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Start it with <code className="font-mono">npm run api</code>.
            </p>
          </div>
        )}

        {/* An empty catalogue is the expected state before the pipeline has run, so
            it gets the command rather than an indefinite spinner. */}
        {!loading && !error && products.length === 0 && (
          <div className="py-24 text-center">
            <p className="text-sm font-medium">No products yet</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Run <code className="font-mono">uv run python main.py</code> to extract the
              catalogue, then restart the API.
            </p>
          </div>
        )}

        {!loading && !error && products.length > 0 && (
          <ProductGrid products={products} onSelect={setSelectedId} />
        )}
      </main>

      <ProductDialog productId={selectedId} onClose={() => setSelectedId(null)} />
    </div>
  )
}

function GridSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-10 md:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: 8 }, (_, index) => (
        <div key={index} className="flex flex-col gap-3">
          <AspectRatio ratio={4 / 5}>
            <Skeleton className="size-full rounded-lg" />
          </AspectRatio>
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-16" />
        </div>
      ))}
    </div>
  )
}
