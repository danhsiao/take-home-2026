import { AspectRatio } from '@/components/ui/aspect-ratio'
import { PriceDisplay } from '@/components/PriceDisplay'
import { ProductImage } from '@/components/ProductImage'
import type { ProductSummary } from '@/types/product'

/**
 * One catalogue tile: image, brand, name, price. Nothing else.
 *
 * The summary endpoint also carries category, colours and a variant count. They stay
 * off the card on purpose - a storefront grid shows the product, and surfacing
 * extraction metadata is what makes a catalogue read like a debug table.
 */
export function ProductCard({
  product,
  onSelect,
}: {
  product: ProductSummary
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="group flex flex-col text-left outline-none"
    >
      <AspectRatio
        ratio={4 / 5}
        className="overflow-hidden rounded-lg bg-muted ring-offset-2 ring-offset-background group-focus-visible:ring-2 group-focus-visible:ring-ring"
      >
        <ProductImage
          src={product.image_url}
          alt={product.name}
          className="transition-transform duration-300 group-hover:scale-[1.03]"
        />
      </AspectRatio>

      <div className="mt-3 flex flex-col gap-1">
        <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          {product.brand}
        </span>
        <span className="line-clamp-2 text-sm leading-snug text-foreground">
          {product.name}
        </span>
        <PriceDisplay price={product.price} className="mt-0.5" />
      </div>
    </button>
  )
}
