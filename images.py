"""Image and media URL harvesting, with full-resolution selection.

The assignment asks for *all full resolution images*, and that is harder than it
sounds. A single product photo typically appears in one PDP a dozen times over: in a
`srcset` at several widths, again as an OpenGraph image, again inside embedded app
state, each time at a different size. One of the provided pages carries 315 `srcset`
attributes for roughly eight actual photographs.

So this module does two things:

  1. RECALL  - gather image URLs from every surface that can carry one.
  2. DEDUPE  - group URLs that point at the same underlying asset, and keep only the
               highest-resolution form of each.

Both are fully deterministic. No model is ever asked about an image URL, which is
deliberate: language models reliably mangle long CDN URLs full of transform tokens
and UUIDs, and a mangled URL is indistinguishable from a hallucinated one.

Two design rules worth stating explicitly, because both are easy to get wrong:

  * We harvest from the *pruned, product-scored* JSON subtrees rather than scanning the
    whole document for image-looking URLs. A raw scan would pull in every cross-sell
    and recommendation thumbnail on the page. Sourcing images from the same evidence
    the rest of the pipeline trusts keeps image extraction aligned with the
    evidence-pruning architecture.
  * We normalise URLs to compute asset *identity*, but we always emit a URL the page
    actually exposed. Manufacturing a URL by stripping parameters off a rendition can
    produce a link that 404s; picking the largest real rendition cannot.

Genericity: every rule here keys off resolution-encoding conventions shared across
image CDNs (Cloudinary-style `w_800` transforms, `?width=` query parameters,
`1200x800` path segments, `-large`/`-thumb` filename suffixes). None reference a
domain, and the module never branches on which site it is parsing.
"""

import re
from typing import Any
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse

from selectolax.parser import HTMLParser

from models import ImageAsset

# Budgets. A pathological PDP (an infinite-scroll category page misfiled as a PDP,
# say) must not be able to produce an unbounded image list.
MAX_CANDIDATES = 5_000  # raw URLs considered before dedupe
MAX_ASSETS = 200  # distinct assets emitted

IMAGE_EXT = r"(?:jpg|jpeg|png|webp|avif|gif)"
_URL_WITH_EXT = re.compile(r"^https?://[^\s]+\." + IMAGE_EXT + r"(?:\?.*)?$", re.I)

# Keys whose values are images by convention. Used when walking embedded JSON, where
# there is no markup to tell us what a string means. Generic commerce/CMS vocabulary.
_IMAGE_CONTAINER_KEYS = re.compile(
    r"^(image|images|img|imgs|photo|photos|picture|pictures|thumbnail|thumbnails|"
    r"thumb|thumbs|gallery|galleries|media|assets?|swatch|swatches|shot|shots)",
    re.I,
)
# Inside such a container, these keys hold the URL itself. Checked only when nested
# under an image container, since `src`/`url` alone are far too broad.
_URL_VALUE_KEYS = re.compile(r"^(src|url|href|uri|@id|path|source|link)$", re.I)

# Query parameters that request a *rendition* of an image rather than identify it.
# Stripped when computing identity; never stripped from the URL we emit. The
# abbreviated forms (`wid`, `hei`, `qlt`, `resmode`) are Scene7/Dynamic Media
# conventions, used by many large retailers' image servers.
_SIZING_PARAMS = {
    "w", "width", "h", "height", "size", "sz", "q", "quality", "fit", "dpr",
    "sw", "sh", "max", "maxwidth", "maxheight", "resize", "scale", "crop", "fm",
    "wid", "hei", "qlt", "resmode", "op_sharpen", "fmt",
}

# Attribute used to mark DOM regions that hold other products' content (navigation,
# carousels, "you may also like" strips). `extract.py` applies the mark; both modules
# read it.
#
# It is an attribute rather than a set of node ids on purpose: selectolax creates a
# fresh Python wrapper object on every tree access, so `id(node)` is not a stable
# identity - the ids are freed and silently reused by unrelated nodes, which
# mass-misclassifies content. Marking the tree itself is the only durable handle.
SKIP_ATTR = "data-evidence-skip"


def in_skipped_region(node) -> bool:
    """Whether a node sits inside a region marked as belonging to other products."""
    current = node
    hops = 0
    while current is not None and hops < 25:
        if current.attributes.get(SKIP_ATTR) is not None:
            return True
        current = current.parent
        hops += 1
    return False

# Generic size vocabulary used in filename suffixes, ranked largest-first. When two
# URLs differ only by one of these words and neither declares a pixel width, we keep
# the one claiming to be biggest. Ordinary English sizing words, not a naming scheme
# belonging to any merchant.
_SIZE_WORDS = [
    "original", "max", "full", "xxlarge", "xlarge", "large",
    "medium", "small", "thumb", "thumbnail", "mini", "tiny",
]
_SIZE_WORD_RANK = {word: index for index, word in enumerate(_SIZE_WORDS)}

# Resolution tokens stripped when computing an asset's identity. Each is a convention
# shared across many image CDNs.
# `\d{1,5}` on both sides because renditions like `200x0` (width-constrained, height
# auto) are common and must group with their full-size sibling.
_DIMENSION_SEGMENT = re.compile(r"/\d{1,5}x\d{1,5}/")  # /2890x1500/, /200x0/
_TRANSFORM_TOKEN = re.compile(r"\b[whcq]_\d+(?:\.\d+)?\b", re.I)  # Cloudinary w_800
# Named transformation presets and format-negotiation tokens, also Cloudinary-style.
# Two URLs differing only by preset (`t_default` vs `t_web_pdp_936_v2`) are renditions
# of one photograph and must share an identity.
_NAMED_TRANSFORM = re.compile(r"\b(?:t_[a-z0-9_]+|f_auto|fl_[a-z_]+)\b", re.I)
# Only strongly resolution-shaped filename suffixes. Deliberately does NOT match a
# bare `-1234.jpg`, which is far more often an asset id than a size.
_TRAILING_SIZE = re.compile(
    r"[-_](?:" + "|".join(_SIZE_WORDS) + r"|\d{2,5}w|\d{2,5}x\d{2,5})(?=\.\w+$)",
    re.I,
)

# Assets that are never product photography. Applied only to low-confidence sources:
# a URL nominated by Product JSON-LD or OpenGraph is the merchant's own statement of
# what the product looks like, and we do not second-guess it with a name heuristic.
_NON_PRODUCT = re.compile(
    r"(sprite|logo|icon|favicon|placeholder|pixel|beacon|spacer|blank|loading|"
    r"spinner|payment|visa|mastercard|paypal|facebook|twitter|instagram|"
    r"pinterest|youtube|avatar|star-?rating|review|testimonial|ugc|user[-_]photo)",
    re.I,
)


def collect(
    tree: HTMLParser,
    product_items: list[dict[str, Any]],
    meta: dict[str, str],
    json_subtrees: list[dict[str, Any]],
    base_url: str | None,
) -> list[ImageAsset]:
    """Gather every product image on the page, deduped to full resolution.

    Args:
        tree: the parsed document.
        product_items: schema.org items already filtered to Product/ProductGroup.
        meta: OpenGraph and meta tag values.
        json_subtrees: the pruned, product-scored embedded-JSON subtrees. Using these
            rather than a whole-document scan is what keeps recommendation imagery out.
        base_url: canonical page URL, for resolving relative references.

    DOM regions marked with `SKIP_ATTR` by `extract.py` are ignored, since those hold
    other products' photography.

    Returns:
        One ImageAsset per distinct underlying photograph, holding the
        highest-resolution URL the page actually exposed for it. Ordered by first
        appearance, so the page's own ordering (hero image first) is preserved.
    """
    # (url, declared_width, trusted) in discovery order. schema.org and OpenGraph come
    # first because those are the merchant's own nomination of the primary image,
    # which is what a catalogue grid should show.
    found: list[tuple[str, int | None, bool]] = []

    _from_structured(product_items, found)
    _from_meta(meta, found)
    canonical_path = _url_path(base_url)
    _from_json_subtrees(json_subtrees, found, canonical_path)
    _from_markup(tree, found, canonical_path)

    return _dedupe_to_full_resolution(found[:MAX_CANDIDATES], base_url)


# --------------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------------


def _from_structured(
    product_items: list[dict[str, Any]], out: list[tuple[str, int | None, bool]]
) -> None:
    """Pull image URLs out of schema.org Product items.

    `image` is specified as either a URL string, an array of URL strings, or an
    ImageObject with a `url`/`contentUrl`, so all three shapes are handled. An
    ImageObject may also declare `width`, which we keep as a resolution hint.

    `logo` is deliberately not read here: a brand logo is not product photography.
    """
    for item in product_items:
        for key in ("image", "images", "photo", "thumbnailUrl"):
            _walk_image_value(item.get(key), out)


def _walk_image_value(value: Any, out: list[tuple[str, int | None, bool]]) -> None:
    """Recursively read image URLs from a schema.org image-valued property."""
    if isinstance(value, str):
        if value.strip():
            out.append((value.strip(), None, True))
    elif isinstance(value, list):
        for entry in value:
            _walk_image_value(entry, out)
    elif isinstance(value, dict):
        parsed_width = _to_int(value.get("width"))
        for key in ("url", "contentUrl", "@id"):
            url = value.get(key)
            if isinstance(url, str) and url.strip():
                out.append((url.strip(), parsed_width, True))
                return


def _from_meta(meta: dict[str, str], out: list[tuple[str, int | None, bool]]) -> None:
    """Pull the OpenGraph and Twitter Card images.

    `og:image:width` is a standard companion tag, used as a resolution hint when
    present. These are trusted sources: OpenGraph images are chosen by the merchant to
    represent this specific page.
    """
    declared_width = _to_int(meta.get("og:image:width"))
    for key in ("og:image", "og:image:secure_url", "og:image:url", "twitter:image"):
        url = meta.get(key)
        if url:
            out.append((url, declared_width if key.startswith("og:image") else None, True))


# Keys by which a record states which page it belongs to.
_RECORD_URL_KEYS = re.compile(r"^(url|uri|href|link|path|pdp_?url|canonical_?url)$", re.I)


def _record_describes_another_page(node: dict[str, Any], canonical_path: str) -> bool:
    """Whether a state record identifies itself as belonging to a different page.

    The JSON counterpart of `_links_to_another_page`, and the same principle: a record
    that carries its own page URL is telling us what it describes. If that URL is not
    this page, the record's images are another product's images.

    This matters because a PDP's application state routinely contains sibling products
    - other colourways, related items, recently viewed - as fully-formed records. They
    are product-shaped by every structural test, so shape alone cannot separate them;
    self-declared identity can.

    Matching is by containment in either direction, not by prefix. Two things make
    prefix comparison wrong here, and both are common:

      * A record for *this* product often carries a more specific URL than the
        canonical one - the page plus a variant selector appended.
      * A record often carries a bare relative slug while the canonical URL carries a
        locale or section prefix, so neither is a prefix of the other even though they
        denote the same page.

    Containment handles both, while still separating genuinely distinct slugs.
    """
    if not canonical_path:
        return False

    canonical = canonical_path.strip("/").lower()
    if not canonical:
        return False

    for key, value in node.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not _RECORD_URL_KEYS.match(key):
            continue
        path = _url_path(value).strip("/").lower()
        if not path:
            continue
        if path in canonical or canonical in path:
            return False  # same page, a variant selector on it, or a bare slug of it
        return True

    return False


def _from_json_subtrees(
    subtrees: list[dict[str, Any]],
    out: list[tuple[str, int | None, bool]],
    canonical_path: str = "",
) -> None:
    """Harvest image URLs from the pruned, product-scored embedded JSON.

    Two matching rules, because CDNs are inconsistent about file extensions:
      - any string that looks like an image URL by extension, anywhere in the subtree;
      - any URL-valued string nested under an image-ish container key, even with no
        extension at all. Real PDPs serve images from paths like
        `/is/image/wim/224626_1176_41`, which no extension rule would ever catch.
    """
    for subtree in subtrees:
        _walk_json_for_images(subtree, out, under_image_key=False, canonical_path=canonical_path)


def _walk_json_for_images(
    node: Any,
    out: list[tuple[str, int | None, bool]],
    under_image_key: bool,
    depth: int = 0,
    canonical_path: str = "",
) -> None:
    """Recursively collect image URLs from parsed JSON, tracking key context."""
    if depth > 12 or len(out) >= MAX_CANDIDATES:
        return

    if isinstance(node, dict):
        # A record that names a different page describes a different product; its
        # images and everything nested beneath it belong to that product, not this one.
        if _record_describes_another_page(node, canonical_path):
            return

        for key, value in node.items():
            if not isinstance(key, str):
                continue
            # Once inside an image container, stay inside it: `media` -> `[{src: ...}]`
            # needs the flag to survive the intermediate list and object.
            nested = under_image_key or bool(_IMAGE_CONTAINER_KEYS.match(key))
            if nested and isinstance(value, str) and _URL_VALUE_KEYS.match(key):
                _append_json_url(value, out)
            else:
                _walk_json_for_images(value, out, nested, depth + 1, canonical_path)
        return

    if isinstance(node, list):
        for entry in node:
            _walk_json_for_images(entry, out, under_image_key, depth + 1, canonical_path)
        return

    if isinstance(node, str):
        # Extension match works anywhere; the key-context match is handled above.
        if _URL_WITH_EXT.match(node.strip()):
            _append_json_url(node, out)


def _append_json_url(value: str, out: list[tuple[str, int | None, bool]]) -> None:
    """Record a URL found in embedded JSON, undoing JSON's escaped slashes."""
    url = value.strip().replace("\\/", "/")
    if url.startswith("http") or url.startswith("//"):
        # Not trusted: pruned subtrees are product-scoped but still machine-selected,
        # so chrome filtering still applies to them.
        out.append((url, None, False))


def _links_to_another_page(node, canonical_path: str) -> bool:
    """Whether an image sits inside a link pointing at a different page.

    A universal structural rule, and a sharper one for images than link density: a
    photograph wrapped in a link to somewhere else depicts *that* thing, not this
    product. Gallery images are either unlinked or linked to a zoom view on the same
    page, so this costs nothing on real galleries.

    It also catches what density cannot. A single-item cross-sell - one image, one
    link, a few words - is not a link farm by any threshold, but it is unambiguously
    another product, and one of the provided pages embeds exactly that.
    """
    current = node.parent
    hops = 0
    while current is not None and hops < 6:
        if current.tag == "a":
            href = current.attributes.get("href")
            if href and not href.startswith("#"):
                path = _url_path(href)
                if path and path != canonical_path:
                    return True
            return False  # linked, but to this same page: a zoom or gallery anchor
        current = current.parent
        hops += 1
    return False


def _url_path(url: str | None) -> str:
    """Reduce a URL to its path, for comparing 'is this the same page'."""
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.path.rstrip("/")


def _from_markup(
    tree: HTMLParser, out: list[tuple[str, int | None, bool]], canonical_path: str = ""
) -> None:
    """Pull image URLs from `<img>`, `<source>`, and preload hints.

    Covers lazy-loading conventions (`data-src`, `data-srcset`, `data-original`)
    because a curl-captured page has not run its JavaScript, so the real image URL is
    frequently sitting in a data attribute rather than in `src`.

    Images inside flagged navigation or recommendation containers are skipped. Those
    regions hold *other products'* photography, and one of the provided pages carries
    cross-sell `srcset` entries for a different SKU entirely.
    """
    for node in tree.css("img, source, link[rel='preload'][as='image']"):
        if in_skipped_region(node) or _links_to_another_page(node, canonical_path):
            continue

        attrs = node.attributes
        declared = _to_int(attrs.get("width"))

        # Single-URL attributes. The `data-*` variants are lazy-loading conventions
        # shared by many lazy-load libraries, not one site's markup.
        for attr in ("src", "href", "data-src", "data-original", "data-lazy", "data-image"):
            url = attrs.get(attr)
            if url and url.strip():
                out.append((url.strip(), declared, False))

        # srcset carries several renditions of one image with explicit descriptors,
        # making it the most reliable place to learn true dimensions.
        for attr in ("srcset", "data-srcset", "imagesrcset"):
            if attrs.get(attr):
                out.extend((url, width, False) for url, width in _parse_srcset(attrs[attr]))


def _parse_srcset(value: str) -> list[tuple[str, int | None]]:
    """Parse a `srcset` attribute into (url, width) pairs.

    The format is a comma-separated list of `URL descriptor`, where the descriptor is
    a width (`800w`) or a pixel density (`2x`). Splitting naively on commas breaks on
    CDN URLs that contain commas in transform lists; splitting only before `http`
    breaks on relative candidates like `foo-320.jpg 320w, foo-640.jpg 640w`.

    So we tokenise on whitespace instead, which is unambiguous: in a valid srcset the
    URL and its descriptor are always whitespace-separated, and commas appearing
    inside a URL never have whitespace around them.
    """
    results: list[tuple[str, int | None]] = []
    pending: str | None = None

    for token in value.split():
        descriptor = re.fullmatch(r"(\d+(?:\.\d+)?)([wx]),?", token)
        if descriptor and pending is not None:
            # Only a `w` descriptor states a real pixel width; `2x` is a density ratio.
            width = _to_int(descriptor.group(1)) if descriptor.group(2) == "w" else None
            results.append((pending.rstrip(","), width))
            pending = None
            continue

        if pending is not None:
            # The previous candidate had no descriptor, which is legal (it means 1x).
            results.append((pending.rstrip(","), None))
        pending = token

    if pending is not None:
        results.append((pending.rstrip(","), None))

    return [(url, width) for url, width in results if url]


# --------------------------------------------------------------------------------
# Dedupe and full-resolution selection
# --------------------------------------------------------------------------------


def _dedupe_to_full_resolution(
    found: list[tuple[str, int | None, bool]], base_url: str | None
) -> list[ImageAsset]:
    """Group URLs by underlying asset and keep the highest-resolution real URL.

    Selection order, best signal first:
      1. Largest explicitly declared pixel width (from `srcset` or `og:image:width`).
      2. Largest effective width encoded in the URL itself.
      3. Highest-ranked generic size word in the filename (`-original` beats `-thumb`).

    The URL we emit is always one the page actually exposed. We never reconstruct a
    URL by stripping parameters: `?w=2600` is a real, fetchable rendition, whereas the
    bare path with the query removed may not resolve at all.
    """
    # asset_key -> (rank, url, width, first_seen_index)
    best: dict[str, tuple[tuple[int, int, int], str, int | None, int]] = {}

    for index, (raw_url, declared_width, trusted) in enumerate(found):
        url = _absolutise(raw_url, base_url)
        if not url:
            continue
        # Chrome filtering applies only to machine-selected sources. A URL nominated by
        # Product JSON-LD or OpenGraph is the merchant's own statement about this
        # product, and we do not overrule it with a filename heuristic.
        if not trusted and not _is_product_image(url):
            continue

        key = _asset_key(url)
        width = declared_width or _effective_width(url)
        # Negated so larger sorts first under plain tuple comparison; the final
        # element prefers the shorter URL as a stable, arbitrary tiebreak.
        rank = (-(width or 0), -_size_word_rank(url), len(url))

        current = best.get(key)
        if current is None:
            best[key] = (rank, url, width, index)
        elif rank < current[0]:
            # Preserve the earliest discovery index so ordering reflects the page's own
            # gallery order, not which rendition happened to win.
            best[key] = (rank, url, width, current[3])

    ordered = sorted(best.items(), key=lambda pair: pair[1][3])[:MAX_ASSETS]
    return [
        ImageAsset(url=url, asset_key=key, width=width)
        for key, (_rank, url, width, _index) in ordered
    ]


def _asset_key(url: str) -> str:
    """Reduce a URL to an identity shared by every rendition of the same image.

    Strips the conventions CDNs use to encode a requested size:
      - sizing query parameters      `?w=320&q=80&fit=max`
      - dimension path segments      `/2890x1500/`
      - transform tokens             `w_1536`, `h_600`
      - filename size suffixes       `-large`, `_thumb`, `-800w`, `-1200x800`

    Each is a cross-CDN convention; none identifies a particular merchant, which is
    what keeps this generic. Note this value is used *only* for grouping - it is never
    emitted as a URL.
    """
    parsed = urlparse(_strip_sizing_params(url))

    path = _DIMENSION_SEGMENT.sub("/", parsed.path)
    path = _NAMED_TRANSFORM.sub("", path)
    path = _TRANSFORM_TOKEN.sub("", path)
    path = _TRAILING_SIZE.sub("", path)
    # Transform removal can leave doubled separators behind.
    path = re.sub(r"[,/]{2,}", "/", path).strip(",")

    return f"{parsed.netloc}{path}".lower()


def _strip_sizing_params(url: str) -> str:
    """Drop rendition-requesting query parameters. Used for identity only."""
    parsed = urlparse(url)
    if not parsed.query:
        return url

    kept = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _SIZING_PARAMS
    ]
    return urlunparse(parsed._replace(query=urlencode(kept)))


def _effective_width(url: str) -> int | None:
    """Recover the width this URL will actually render at.

    Precedence matters, and is the opposite of "take the largest number present". A
    URL like `/2890x1500/photo.jpg?w=320` serves a 320px image: the path segment
    describes the *source* asset while the query requests a downscale. Reading the
    largest number would rank a thumbnail as the best rendition available.

    `w_1.0` style relative scales are excluded, since those are ratios, not pixels.
    """
    query_widths = [
        int(match.group(1))
        for match in re.finditer(r"[?&](?:w|wid|width|sw|maxwidth)=(\d+)", url, re.I)
    ]
    if query_widths:
        return max(query_widths)

    transform_widths = [
        int(match.group(1)) for match in re.finditer(r"\bw_(\d+)(?![\d.])", url, re.I)
    ]
    if transform_widths:
        return max(transform_widths)

    suffix = re.search(r"[-_](\d{3,5})w(?=\.\w+$)", url, re.I)
    if suffix:
        return int(suffix.group(1))

    dimension = _DIMENSION_SEGMENT.search(url)
    if dimension:
        return int(dimension.group(0).strip("/").split("x")[0])

    return None


def _size_word_rank(url: str) -> int:
    """Score a URL by any generic size word in its filename; higher is larger.

    Used only as a tiebreak when no pixel width is declared anywhere, which is common
    on CDNs that name their renditions `-max` and `-full`.
    """
    filename = urlparse(url).path.rsplit("/", 1)[-1].lower()
    for word, index in _SIZE_WORD_RANK.items():
        if re.search(rf"[-_]{word}\b", filename):
            return len(_SIZE_WORDS) - index
    return 0


def _is_product_image(url: str) -> bool:
    """Filter out chrome: logos, icons, payment badges, tracking pixels.

    Matches well-known asset naming conventions in the URL path only. Checking the
    path rather than the whole URL means a merchant whose *domain* contains "social"
    or "pixel" does not have its entire gallery discarded.
    """
    if url.startswith("data:"):
        return False  # inline base64 is always a placeholder or icon at this size
    parsed = urlparse(url)
    if not parsed.netloc:
        return False
    return not _NON_PRODUCT.search(parsed.path)


def _absolutise(url: str, base_url: str | None) -> str | None:
    """Resolve protocol-relative and root-relative URLs into absolute ones.

    `//cdn.example.com/a.jpg` and `/media/a.jpg` both appear in the wild, and
    downstream validation compares emitted URLs against evidence by exact string, so
    they must be normalised consistently here.
    """
    url = url.strip()
    if not url or url.startswith("data:"):
        return None
    if url.startswith("//"):
        scheme = urlparse(base_url).scheme if base_url else "https"
        return f"{scheme or 'https'}:{url}"
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if base_url:
        return urljoin(base_url, url)
    return None


def _to_int(value: Any) -> int | None:
    """Best-effort integer parse, tolerant of `"800px"` and `None`."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None
