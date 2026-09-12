import { useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { PriceDisplay } from '@/components/PriceDisplay'
import { ProductGallery } from '@/components/ProductGallery'
import { VariantSelector } from '@/components/VariantSelector'
import { fetchProduct } from '@/api/products'
import {
  applySelection,
  displayPrice,
  initialSelection,
  resolveVariant,
  type Selection,
} from '@/lib/variants'
import type { Product } from '@/types/product'

interface Loaded {
  id: string
  product: Product | null
  error: string | null
}

/** Descriptions arrive with CRLFs and markdown-ish bullets; render them as lines. */
function descriptionLines(description: string): string[] {
  return description
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*[-•]\s*/, '').trim())
    .filter(Boolean)
}

export function ProductDialog({
  productId,
  onClose,
}: {
  productId: string | null
  onClose: () => void
}) {
  // Keyed by the id it was fetched for, so a response for a product the user has
  // already navigated away from is ignored rather than flashing into the dialog.
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [selection, setSelection] = useState<Selection>({})
  const [videoFailed, setVideoFailed] = useState(false)
  const [activeImage, setActiveImage] = useState<string | null>(null)

  useEffect(() => {
    if (!productId) return

    let current = true
    fetchProduct(productId)
      .then((product) => {
        if (!current) return
        setLoaded({ id: productId, product, error: null })
        setSelection(initialSelection(product.variants))
        setActiveImage(product.image_urls[0] ?? null)
      })
      .catch((cause: Error) => {
        if (current) setLoaded({ id: productId, product: null, error: cause.message })
      })

    return () => {
      current = false
    }
  }, [productId])

  const fresh = loaded?.id === productId ? loaded : null
  const product = fresh?.product ?? null
  const error = fresh?.error ?? null
  const variant = product ? resolveVariant(product.variants, selection) : null

  /** Selecting a variant may change which image is shown, so both move together. */
  function selectOption(variants: Product['variants'], dimension: string, value: string) {
    const next = applySelection(variants, selection, dimension, value)
    setSelection(next)
    const image = resolveVariant(variants, next)?.image_urls[0]
    if (image) setActiveImage(image)
  }

  // Variant images are merged into the product gallery rather than replacing it, so
  // the full set of photography stays reachable while the selection drives which one
  // is shown.
  const images = useMemo(() => {
    if (!product) return []
    return [...new Set([...product.image_urls, ...product.variants.flatMap((v) => v.image_urls)])]
  }, [product])

  return (
    <Dialog open={productId !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="top-0 left-0 h-dvh max-h-none w-screen max-w-none translate-x-0 translate-y-0 gap-0 overflow-hidden rounded-none p-0 sm:top-1/2 sm:left-1/2 sm:h-auto sm:max-h-[88vh] sm:w-[95vw] sm:max-w-[1100px] sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-xl"
      >
        <div className="max-h-dvh overflow-y-auto p-6 sm:max-h-[88vh] sm:p-8">
          {error && <p className="py-16 text-center text-sm text-destructive">{error}</p>}

          {!product && !error && <ProductDialogSkeleton />}

          {product && (
            <div className="grid gap-8 md:grid-cols-2 md:gap-10">
              <div className="flex flex-col gap-4">
                <ProductGallery
                  images={images}
                  activeUrl={activeImage}
                  onSelect={setActiveImage}
                  alt={product.name}
                />
                {/* A merchant's video URL is often signed and access-bound - tied to
                    the session or origin that issued it - so it resolves for the
                    merchant's own page and returns an error body for anyone else. The
                    URL is still the right extraction; it simply cannot be played
                    here. Hiding the element on failure is better than leaving a grey
                    box reading "no supported format", which looks like our bug. */}
                {product.video_url && !videoFailed && (
                  <video
                    controls
                    src={product.video_url}
                    onError={() => setVideoFailed(true)}
                    className="w-full rounded-lg bg-muted"
                  />
                )}
              </div>

              <div className="flex flex-col gap-6 md:pr-6">
                <div className="flex flex-col gap-2">
                  <span className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
                    {product.brand}
                  </span>
                  <DialogTitle className="text-2xl leading-tight font-semibold">
                    {product.name}
                  </DialogTitle>
                  <PriceDisplay
                    price={displayPrice(product.price, variant?.price)}
                    size="lg"
                    className="mt-1"
                  />
                  {/* `available` is tri-state: null means the page never said, and
                      only an explicit false is worth telling the shopper about. */}
                  {variant?.available === false && (
                    <span className="text-sm text-muted-foreground">Out of stock</span>
                  )}
                </div>

                {product.variants.length > 1 && (
                  <VariantSelector
                    variants={product.variants}
                    selection={selection}
                    onSelect={(dimension, value) =>
                      selectOption(product.variants, dimension, value)
                    }
                  />
                )}

                {/* Some merchants model each configuration as its own page, so this
                    page carries every swatch but only the selected one's photography.
                    Linking out is the honest presentation: the rest of that
                    configuration's detail genuinely lives elsewhere, and inventing a
                    gallery for it would be worse than sending the shopper to it. */}
                {variant?.url && variant.image_urls.length <= 1 && (
                  <a
                    href={variant.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="w-fit text-sm underline underline-offset-4 hover:text-muted-foreground"
                  >
                    View this option on the retailer&rsquo;s site
                  </a>
                )}

                {product.description && (
                  <section className="flex flex-col gap-2">
                    <Separator />
                    <h3 className="pt-2 text-sm font-medium">Description</h3>
                    <DialogDescription className="flex flex-col gap-1 text-sm leading-relaxed">
                      {descriptionLines(product.description).map((line) => (
                        <span key={line}>{line}</span>
                      ))}
                    </DialogDescription>
                  </section>
                )}

                {product.key_features.length > 0 && (
                  <section className="flex flex-col gap-2">
                    <h3 className="text-sm font-medium">Key features</h3>
                    <ul className="flex list-disc flex-col gap-1 pl-5 text-sm text-muted-foreground">
                      {product.key_features.map((feature) => (
                        <li key={feature}>{feature}</li>
                      ))}
                    </ul>
                  </section>
                )}

                <Badge variant="outline" className="h-auto w-fit py-1 text-muted-foreground">
                  {product.category.name}
                </Badge>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

function ProductDialogSkeleton() {
  return (
    <div className="grid gap-8 md:grid-cols-2 md:gap-10">
      <Skeleton className="aspect-square w-full rounded-lg" />
      <div className="flex flex-col gap-4">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-7 w-3/4" />
        <Skeleton className="h-7 w-28" />
        <Skeleton className="mt-4 h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    </div>
  )
}
