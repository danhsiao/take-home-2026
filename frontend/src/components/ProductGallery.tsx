import { cn } from 'cn'

import { AspectRatio } from '@/components/ui/aspect-ratio'
import { ProductImage } from '@/components/ProductImage'

/**
 * Main image plus thumbnail strip.
 *
 * Controlled from the dialog so that selecting a variant can drive the main image.
 * The thumbnail strip is hidden for single-image products rather than rendering a
 * lone thumbnail under its own larger copy.
 */
export function ProductGallery({
  images,
  activeUrl,
  onSelect,
  alt,
}: {
  images: string[]
  activeUrl: string | null
  onSelect: (url: string) => void
  alt: string
}) {
  return (
    <div className="flex flex-col gap-3">
      <AspectRatio ratio={1} className="overflow-hidden rounded-lg bg-muted">
        <ProductImage src={activeUrl} alt={alt} className="object-contain" />
      </AspectRatio>

      {images.length > 1 && (
        // Some pages expose dozens of images. Capping the strip at roughly two rows
        // keeps the gallery from pushing the rest of the PDP off screen.
        <div className="flex max-h-36 flex-wrap gap-2 overflow-y-auto pr-1">
          {images.map((url) => (
            <button
              key={url}
              type="button"
              onClick={() => onSelect(url)}
              aria-label="Show image"
              aria-pressed={url === activeUrl}
              className={cn(
                'size-16 overflow-hidden rounded-md bg-muted ring-1 ring-border transition-all outline-none focus-visible:ring-2 focus-visible:ring-ring',
                url === activeUrl && 'ring-2 ring-foreground',
              )}
            >
              <ProductImage src={url} alt={alt} />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
