import { ProductCard } from '@/components/ProductCard'
import type { ProductSummary } from '@/types/product'

export function ProductGrid({
  products,
  onSelect,
}: {
  products: ProductSummary[]
  onSelect: (id: string) => void
}) {
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-10 md:grid-cols-3 xl:grid-cols-4">
      {products.map((product) => (
        <ProductCard
          key={product.id}
          product={product}
          onSelect={() => onSelect(product.id)}
        />
      ))}
    </div>
  )
}
