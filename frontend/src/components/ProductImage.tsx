import { ImageOff } from 'lucide-react'
import { useState } from 'react'

import { cn } from 'cn'

/**
 * An `<img>` that degrades to a neutral tile instead of a broken-image glyph.
 *
 * Extracted image URLs can 404 or be hotlink-blocked, and one dead URL should not put
 * a jagged hole in an otherwise finished grid. This is presentation only - a URL that
 * points at the wrong product is a pipeline bug, not something to patch here.
 */
export function ProductImage({
  src,
  alt,
  className,
}: {
  src: string | null
  alt: string
  className?: string
}) {
  // Which src failed, rather than a boolean: the gallery reuses this component
  // instance for every image it shows, so a new src must not inherit the previous
  // one's failure.
  const [failedSrc, setFailedSrc] = useState<string | null>(null)

  if (!src || failedSrc === src) {
    return (
      <div
        className={cn(
          'flex size-full items-center justify-center bg-muted text-muted-foreground',
          className,
        )}
      >
        <ImageOff className="size-5 opacity-40" />
      </div>
    )
  }

  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailedSrc(src)}
      className={cn('size-full object-cover', className)}
    />
  )
}
