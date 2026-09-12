"""
Finds product images, picks the biggest version, and drops the rest.
"""

import re
from typing import Any
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse

from selectolax.parser import HTMLParser

import identity as identity_mod
from identity import PageIdentity
from models import ImageAsset

# Budgets. A pathological PDP (an infinite-scroll category page misfiled as a PDP,
# say) must not be able to produce an unbounded image list.
MAX_CANDIDATES = 5_000  # raw URLs considered before dedupe
MAX_ASSETS = 200  # distinct assets emitted

# Below this width an image is a swatch, a nav thumbnail or a badge, not a product
# photo. 
MIN_PRODUCT_WIDTH = 200

# Path segments compared when deciding whether two URLs come from the same image
# store
FAMILY_SEGMENTS = 2

# How far out to look for the card an image sits in. 
CARD_HOPS = 12

TITLE_REGION_DEPTH = 3

_LEADING_ID = re.compile(r"^(\d{4,})(?:\D|$)")

IMAGE_EXT = r"(?:jpg|jpeg|png|webp|avif|gif)"
_URL_WITH_EXT = re.compile(r"^https?://[^\s]+\." + IMAGE_EXT + r"(?:\?.*)?$", re.I)

# Keys whose values are images by convention. 
_IMAGE_CONTAINER_KEYS = re.compile(
    r"^(image|images|img|imgs|photo|photos|picture|pictures|thumbnail|thumbnails|"
    r"thumb|thumbs|gallery|galleries|media|assets?|swatch|swatches|shot|shots)",
    re.I,
)
# Inside such a container, these keys hold the URL itself. 
_URL_VALUE_KEYS = re.compile(r"^(src|url|href|uri|@id|path|source|link)$", re.I)


_SIZING_PARAMS = {
    "w", "width", "h", "height", "size", "sz", "q", "quality", "fit", "dpr",
    "sw", "sh", "max", "maxwidth", "maxheight", "resize", "scale", "crop", "fm",
    "wid", "hei", "qlt", "resmode", "op_sharpen", "fmt",
}

_SIZING_PARAM_SHAPE = re.compile(
    r"(width|height|quality|scale|resize|crop|dpr|maxdim|^sz$|size$)", re.I
)


def _is_sizing_param(name: str) -> bool:
    """Is this query parameter asking for a size, rather than naming the image?

    We match by shape as well as by name. Miss one and the sizes never group, so "keep
    the biggest" quietly turns into "keep whichever we saw first".
    """
    lowered = name.lower()
    return lowered in _SIZING_PARAMS or bool(_SIZING_PARAM_SHAPE.search(lowered))

SKIP_ATTR = "data-evidence-skip"

TITLE_ATTR = "data-evidence-title-depth"


def in_skipped_region(node) -> bool:
    """Is this node inside a region marked as belonging to other products?"""
    current = node
    hops = 0
    while current is not None and hops < 25:
        if current.attributes.get(SKIP_ATTR) is not None:
            return True
        current = current.parent
        hops += 1
    return False

# Generic size vocabulary used in filename suffixes, ranked largest-first.
_SIZE_WORDS = [
    "original", "max", "full", "xxlarge", "xlarge", "large",
    "medium", "small", "thumb", "thumbnail", "mini", "tiny",
]
_SIZE_WORD_RANK = {word: index for index, word in enumerate(_SIZE_WORDS)}

# Resolution tokens stripped when computing an asset's identity.
_DIMENSION_SEGMENT = re.compile(r"/\d{1,5}x\d{1,5}/")  # /2890x1500/, /200x0/
_TRANSFORM_TOKEN = re.compile(r"\b[whcq]_\d+(?:\.\d+)?\b", re.I)  # Cloudinary w_800

_NAMED_TRANSFORM = re.compile(r"\b(?:t_[a-z0-9_]+|f_auto|fl_[a-z_]+)\b", re.I)
# Only strongly resolution-shaped filename suffixes.
_TRAILING_SIZE = re.compile(
    r"[-_](?:" + "|".join(_SIZE_WORDS) + r"|\d{2,5}w|\d{2,5}x\d{2,5})(?=\.\w+$)",
    re.I,
)

# Assets that are never product photography. 
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
    declared_media: list[str] | None = None,
    identity: PageIdentity | None = None,
) -> list[ImageAsset]:
    # Find every product image on the page, deduped to the largest version.

    # (url, declared_width, trusted) in discovery order. schema.org and OpenGraph come
    # first because those are the merchant's own nomination of the primary image,
    # which is what a catalogue grid should show.
    found: list[tuple[str, int | None, bool]] = []

    _from_structured(product_items, found)
    _from_meta(meta, found)
    # Those two sources, and only those, are the merchant's own statement about this
    # product. They anchor the relevance filter below.
    anchors = [
        absolute
        for url, _width, _trusted in found
        if (absolute := _absolutise(url, base_url))
    ]

    canonical_path = _url_path(base_url)
    # Gathered by their own walk, deliberately unpruned: a template only ever restates
    # an image already accepted, so admitting one can add nothing.
    templates: list[str] = []
    for subtree in json_subtrees:
        _collect_size_templates(subtree, templates)

    _from_json_subtrees(json_subtrees, found, canonical_path, identity)

    _from_markup(tree, found, canonical_path, identity)

    if templates:
        concrete = [_absolutise(url, base_url) or url for url, _w, _t in found]
        accepted = {_asset_key(url) for url in concrete}
        found.extend(
            (url, None, False)
            for url in _resolve_size_templates(templates, concrete)
            if _asset_key(url) in accepted
        )

    region_urls = _product_region_urls(tree, base_url)
    found = found[:MAX_CANDIDATES]

    assets = _dedupe_to_full_resolution(found, base_url)
    kept = _keep_relevant(assets, anchors, declared_media or [], region_urls)

    # fallback
    if region_urls and len(kept) <= 1:
        region_keys = {_asset_key(url) for url in region_urls}
        relaxed = _dedupe_to_full_resolution(found, base_url, region_keys)
        rescued = _keep_relevant(
            relaxed, anchors, declared_media or [], region_urls, region_keys
        )
        if len(rescued) > len(kept):
            return rescued

    return kept


# --------------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------------


def _from_structured(
    product_items: list[dict[str, Any]], out: list[tuple[str, int | None, bool]]
) -> None:
    # Read image URLs out of schema.org Product items.

    for item in product_items:
        for key in ("image", "images", "photo", "thumbnailUrl"):
            _walk_image_value(item.get(key), out)


def _walk_image_value(value: Any, out: list[tuple[str, int | None, bool]]) -> None:
    # Read image URLs out of a schema.org image field, whatever shape it takes.
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
    # Read the OpenGraph and Twitter images. 
    declared_width = _to_int(meta.get("og:image:width"))
    for key in ("og:image", "og:image:secure_url", "og:image:url", "twitter:image"):
        url = meta.get(key)
        if url:
            out.append((url, declared_width if key.startswith("og:image") else None, True))


# Keys by which a record states which page it belongs to.
_RECORD_URL_KEYS = re.compile(r"^(url|uri|href|link|path|pdp_?url|canonical_?url)$", re.I)


def _record_is_foreign(
    node: dict[str, Any], canonical_path: str, identity: PageIdentity | None
) -> bool:
    # Does this app state record belong to some other product?

    if identity is not None:
        verdict = identity_mod.ownership(node, identity)
        if verdict == identity_mod.Ownership.FOREIGN:
            return True
        if verdict == identity_mod.Ownership.OWNED:
            return False
    return _record_describes_another_page(node, canonical_path)


def _record_describes_another_page(node: dict[str, Any], canonical_path: str) -> bool:
    # Does this record's own URL point at a different page?

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
    identity: PageIdentity | None = None,
    templates: list[str] | None = None,
) -> None:
    # Read image URLs out of the embedded JSON.
    for subtree in subtrees:
        _walk_json_for_images(
            subtree,
            out,
            under_image_key=False,
            canonical_path=canonical_path,
            identity=identity,
            templates=templates,
        )


def _walk_json_for_images(
    node: Any,
    out: list[tuple[str, int | None, bool]],
    under_image_key: bool,
    depth: int = 0,
    canonical_path: str = "",
    identity: PageIdentity | None = None,
    templates: list[str] | None = None,
) -> None:
    # Walk parsed JSON for image URLs, keeping track of the key context.
    if depth > 12 or len(out) >= MAX_CANDIDATES:
        return

    if isinstance(node, dict):
        # A record that names a different page describes a different product; its
        # images and everything nested beneath it belong to that product, not this one.
        if _record_is_foreign(node, canonical_path, identity):
            return

        for key, value in node.items():
            if not isinstance(key, str):
                continue
            # Once inside an image container, stay inside it: `media` -> `[{src: ...}]`
            # needs the flag to survive the intermediate list and object.
            nested = under_image_key or bool(_IMAGE_CONTAINER_KEYS.match(key))
            if nested and isinstance(value, str) and _URL_VALUE_KEYS.match(key):
                _append_json_url(value, out, templates)
            else:
                _walk_json_for_images(
                    value, out, nested, depth + 1, canonical_path, identity, templates
                )
        return

    if isinstance(node, list):
        for entry in node:
            _walk_json_for_images(
                entry, out, under_image_key, depth + 1, canonical_path, identity, templates
            )
        return

    if isinstance(node, str):
        text = node.strip()
        if _URL_WITH_EXT.match(text):
            _append_json_url(node, out)
        elif templates is not None and text.startswith(("http", "//")):

            if _SIZE_TEMPLATE.match(text.replace("\\/", "/")):
                templates.append(text.replace("\\/", "/"))


def _append_json_url(
    value: str,
    out: list[tuple[str, int | None, bool]],
    templates: list[str] | None = None,
) -> None:
    #Record a URL found in JSON, undoing the escaped slashes.

    url = value.strip().replace("\\/", "/")
    if templates is not None and _SIZE_TEMPLATE.match(url):
        templates.append(url)
        return
    if url.startswith("http") or url.startswith("//"):

        out.append((url, None, False))


def _links_to_another_page(node, canonical_path: str) -> bool:
    #Is this image inside a link to a different page?

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

    return _enclosed_by_foreign_card(node, canonical_path)


def _enclosed_by_foreign_card(node, canonical_path: str) -> bool:
    # Is the smallest region around this image a card for another page?

    current = node
    for _ in range(CARD_HOPS):
        current = current.parent
        if current is None:
            return False

        # Reaching a region that tightly holds the product title means the walk has
        # left the tiles and arrived at the product's own part of the page.
        depth = current.attributes.get(TITLE_ATTR)
        if depth is not None and depth.isdigit() and int(depth) <= TITLE_REGION_DEPTH:
            return False

        destinations = set()
        for link in current.css("a"):
            href = link.attributes.get("href")
            if not href or href.startswith("#"):
                continue
            path = _url_path(href)
            if path:
                destinations.add(path)

        if canonical_path in destinations:
            return False

        siblings = {path for path in destinations if _is_sibling_page(path, canonical_path)}

        if siblings and len(siblings) >= len(current.css("img")):
            return True

    return False


def _is_sibling_page(path: str, canonical_path: str) -> bool:
    #Does this link point at another page of the same kind as this one?
    here = [segment for segment in path.split("/") if segment]
    canonical = [segment for segment in canonical_path.split("/") if segment]
    if not here or not canonical:
        return False
    return here[0].lower() == canonical[0].lower() and here != canonical


def _url_path(url: str | None) -> str:
    #Cut a URL down to its path, so we can ask if two links are the same page.
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.path.rstrip("/")


def _product_region_urls(tree: HTMLParser, base_url: str | None) -> list[str]:
    #Image URLs rendered inside the product's own part of the page.

    urls: list[str] = []
    for node in tree.css("img, source"):
        if in_skipped_region(node):
            continue
        current, hops, inside = node, 0, False
        while current is not None and hops < CARD_HOPS:
            if current.attributes.get(TITLE_ATTR) is not None:
                inside = True
                break
            current, hops = current.parent, hops + 1
        if not inside:
            continue
        for attr in ("src", "data-src", "data-original"):
            value = node.attributes.get(attr)
            if value and value.strip():
                if absolute := _absolutise(value.strip(), base_url):
                    urls.append(absolute)
    return urls


def _inside_foreign_card(node, identity: PageIdentity | None) -> bool:
    #Is this image inside an element advertising a different product?

    if identity is None or not identity:
        return False
    current, hops = node.parent, 0
    while current is not None and hops < CARD_HOPS:
        if current.attributes.get(TITLE_ATTR) is not None:
            return False
        if identity_mod.element_declares_foreign_id(current, identity):
            return True
        current, hops = current.parent, hops + 1
    return False


def _from_markup(
    tree: HTMLParser,
    out: list[tuple[str, int | None, bool]],
    canonical_path: str = "",
    identity: PageIdentity | None = None,
) -> None:
    # Read image URLs from `<img>`, `<source>` and preload hints.

    # `as="image"` is a fetch-priority hint, not what makes an href an image; the
    # href's own shape decides instead (checked per-node below).
    for node in tree.css("img, source, link[rel='preload'], link[rel='prefetch']"):
        if node.tag == "link" and not _URL_WITH_EXT.match(
            (node.attributes.get("href") or "").strip()
        ):
            continue
        if in_skipped_region(node) or _links_to_another_page(node, canonical_path):
            continue
        if _inside_foreign_card(node, identity):
            continue

        attrs = node.attributes

        declared = _to_int(attrs.get("width"))

        for attr in ("src", "href", "data-src", "data-original", "data-lazy", "data-image"):
            url = attrs.get(attr)
            if url and url.strip():
                out.append((url.strip(), declared, False))

        for attr in ("srcset", "data-srcset", "imagesrcset"):
            if attrs.get(attr):
                out.extend((url, width, False) for url, width in _parse_srcset(attrs[attr]))


def _parse_srcset(value: str) -> list[tuple[str, int | None]]:
    #Parse a `srcset` attribute into (url, width) pairs.
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
    found: list[tuple[str, int | None, bool]],
    base_url: str | None,
    region_keys: set[str] | None = None,
) -> list[ImageAsset]:
    # Group URLs by which photo they point at and keep the largest real one.

    # asset_key -> (rank, url, width, first_seen_index)
    best: dict[str, tuple[tuple[int, int, int], str, int | None, int]] = {}
    # asset_key -> every candidate URL seen for it, so a group whose renditions differ
    # only by a bare number can be resolved from the page's own evidence afterwards.
    siblings: dict[str, list[str]] = {}

    for index, (raw_url, declared_width, trusted) in enumerate(found):
        url = _absolutise(raw_url, base_url)
        if not url:
            continue

        if not trusted and not _is_product_image(url):
            continue

        key = _asset_key(url)
        width = declared_width or _effective_width(url)

        if (
            not trusted
            and width is not None
            and width < MIN_PRODUCT_WIDTH
            and not (region_keys and key in region_keys)
        ):
            continue
        # Negated so larger sorts first under plain tuple comparison
        rank = (-(width or 0), -_size_word_rank(url), len(url))

        current = best.get(key)
        if current is None:
            best[key] = (rank, url, width, index)
        elif rank < current[0]:
            # Preserve the earliest discovery index
            best[key] = (rank, url, width, current[3])

        siblings.setdefault(key, []).append(url)

    best = _prefer_larger_numbered_rendition(best, siblings)

    ordered = sorted(best.items(), key=lambda pair: pair[1][3])[:MAX_ASSETS]
    return [
        ImageAsset(url=url, asset_key=key, width=width)
        for key, (_rank, url, width, _index) in ordered
    ]


# A filename ending in a separator and a run of digits, before the extension:
# `..._100.jpg`, `...-1000.jpg`. 
_TRAILING_NUMBER = re.compile(r"^(?P<base>.*[-_])(?P<number>\d{2,5})(?P<ext>\.\w+)$")

# A URL published with the size left as a placeholder, like `..._<SIZE>.jpg`. 
_SIZE_TEMPLATE = re.compile(
    r"^(?P<base>.*[-_])(?:<[A-Za-z_]+>|\{[A-Za-z_]+\}|%[sd]|\$\{?[A-Za-z_]+\}?)(?P<ext>\.\w+)$"
)


def _collect_size_templates(node: Any, out: list[str], depth: int = 0) -> None:
    #Find URLs the page publishes with the size left as a placeholder.
    if depth > 12 or len(out) >= MAX_ASSETS:
        return
    if isinstance(node, dict):
        for value in node.values():
            _collect_size_templates(value, out, depth + 1)
    elif isinstance(node, list):
        for entry in node:
            _collect_size_templates(entry, out, depth + 1)
    elif isinstance(node, str):
        text = node.strip().replace("\\/", "/")
        if text.startswith(("http", "//")) and _SIZE_TEMPLATE.match(text):
            out.append(text)


def _resolve_size_templates(templates: list[str], concrete: list[str]) -> list[str]:
    #Fill in a page's own URL templates using sizes that page already uses.

    sizes = sorted(
        {
            match.group("number")
            for url in concrete
            if (match := _TRAILING_NUMBER.match(url)) and not match.group("number").startswith("0")
        },
        key=int,
        reverse=True,
    )
    if not sizes:
        return []

    resolved: list[str] = []
    for template in templates:
        match = _SIZE_TEMPLATE.match(template)
        if not match:
            continue
        for size in sizes:
            candidate = f"{match.group('base')}{size}{match.group('ext')}"
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def _prefer_larger_numbered_rendition(
    best: dict[str, tuple[tuple[int, int, int], str, int | None, int]],
    siblings: dict[str, list[str]],
) -> dict[str, tuple[tuple[int, int, int], str, int | None, int]]:
    #Pick the biggest when sizes differ only by a plain number in the filename.

    for key, urls in siblings.items():
        entry = best.get(key)
        if entry is None or entry[2] is not None:
            continue  # a real width is known; it already decided the winner

        numbered: dict[int, str] = {}
        bases: set[tuple[str, str]] = set()
        for url in set(urls):
            match = _TRAILING_NUMBER.match(url)
            if not match:
                bases.add((url, ""))
                continue
            bases.add((match.group("base"), match.group("ext")))
            numbered.setdefault(int(match.group("number")), url)

        # Every member must be the same filename bar the number, or these are not
        # renditions of one another and the page is telling us something else.
        if len(bases) != 1 or len(numbered) < 2:
            continue

        largest, smallest = max(numbered), min(numbered)
        if smallest <= 0 or largest < smallest * 2:
            continue

        rank, _url, _width, index = entry
        best[key] = (rank, numbered[largest], None, index)

    return best


def align_to_assets(urls: list[str], assets: list[ImageAsset]) -> list[str]:
    #Restate URLs using the biggest version the page published for the same photo.
    best = {asset.asset_key: asset.url for asset in assets}

    aligned: list[str] = []
    for url in urls:
        candidate = best.get(_asset_key(url), url)
        if candidate not in aligned:
            aligned.append(candidate)
    return aligned


def _keep_relevant(
    assets: list[ImageAsset],
    anchors: list[str],
    declared_media: list[str],
    region_urls: list[str] | None = None,
    region_keys: set[str] | None = None,
) -> list[ImageAsset]:
    #Drop images that are not photos of this product.

    if not assets:
        return assets

    region_urls = region_urls or []
    anchor_urls = set(anchors)

    families = {_family(url) for url in anchors}
    anchor_keys = {_asset_key(url) for url in anchors}

    region_keys = {_asset_key(url) for url in region_urls}

    if region_urls:
        remainder = [url for url in set(region_urls) if _asset_key(url) not in anchor_keys]
        remainder_families = {_family(url) for url in remainder}
        if remainder and not (remainder_families & families):
            counts: dict[tuple, int] = {}
            for url in remainder:
                counts[_family(url)] = counts.get(_family(url), 0) + 1
            families |= {family for family, count in counts.items() if count > 1}

    families |= {_family(asset.url) for asset in assets if asset.asset_key in anchor_keys}

    inventory = {_asset_key(url) for url in declared_media}
    if inventory and not any(_asset_key(url) in inventory for url in anchors):
        inventory = set()  # not this product's gallery; ignore it

    anchor_ids = [_leading_id(url) for url in anchors]
    # A single id, agreed on by every anchor, or nothing.
    expected_id = anchor_ids[0] if anchor_ids and len(set(anchor_ids)) == 1 else None

    kept: list[ImageAsset] = []
    for asset in assets:
        if asset.url in anchor_urls:
            kept.append(asset)
            continue
        if inventory and _asset_key(asset.url) not in inventory:
            continue
        rendered = _rendered_width(asset)
        if (
            rendered is not None
            and rendered < MIN_PRODUCT_WIDTH
            and not (region_keys and _asset_key(asset.url) in region_keys)
        ):
            continue
        if families and _family(asset.url) not in families:
            continue
        if expected_id is not None:
            asset_id = _leading_id(asset.url)
            if asset_id is not None and len(asset_id) == len(expected_id):
                if asset_id != expected_id:
                    continue
        kept.append(asset)

    return kept or assets


def _rendered_width(asset: ImageAsset) -> int | None:
    #The width this image will render at. Used only for filtering.
    if asset.width is not None:
        return asset.width
    bounds = [
        int(match.group(1))
        for match in re.finditer(r"[?&](?:max|hei|[a-z]*height)=(\d+)", asset.url, re.I)
    ]
    return max(bounds) if bounds else None


def _family(url: str) -> tuple[str, tuple[str, ...]]:
    #Which image store a URL comes from: the host plus its top folders.
    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    directory = segments[:-1]  # drop the filename
    return parsed.netloc.lower(), tuple(directory[:FAMILY_SEGMENTS])


def _leading_id(url: str) -> str | None:
    #The leading number in a URL's filename, if it has one.
    filename = urlparse(url).path.rsplit("/", 1)[-1]
    match = _LEADING_ID.match(filename)
    return match.group(1) if match else None


# A content addressed filename: a UUID or a long hex digest. 
_CONTENT_TOKEN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,}", re.I
)


def _asset_key(url: str) -> str:
    #Reduce a URL to an id shared by every size of the same photo.

    parsed = urlparse(_strip_sizing_params(url))

    # A content-addressed URL identifies itself, and does so across routes.
    tokens = _CONTENT_TOKEN.findall(parsed.path)
    if tokens:
        unique = dict.fromkeys(token.lower() for token in tokens)
        return f"{parsed.netloc.lower()}|{'+'.join(unique)}"

    path = _DIMENSION_SEGMENT.sub("/", parsed.path)
    path = _NAMED_TRANSFORM.sub("", path)
    path = _TRANSFORM_TOKEN.sub("", path)
    path = _TRAILING_SIZE.sub("", path)
    path = re.sub(r"[,/]{2,}", "/", path).strip(",")

    return f"{parsed.netloc}{path}".lower()


def _strip_sizing_params(url: str) -> str:
    #Drop the query parameters that ask for a size. For grouping only.
    parsed = urlparse(url)
    if not parsed.query:
        return url

    kept = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_sizing_param(key)
    ]
    return urlunparse(parsed._replace(query=urlencode(kept)))


def _effective_width(url: str) -> int | None:
    #Work out the width a URL will actually render at.
    query_widths = [
        int(match.group(1))
        for match in re.finditer(r"[?&](?:w|wid|sw|[a-z]*width)=(\d+)", url, re.I)
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
    #Score a URL by any size word in its filename. Higher means bigger.
    filename = urlparse(url).path.rsplit("/", 1)[-1].lower()
    for word, index in _SIZE_WORD_RANK.items():
        if re.search(rf"[-_]{word}\b", filename):
            return len(_SIZE_WORDS) - index
    return 0


def _is_product_image(url: str) -> bool:
    #Filter out logos, icons, payment badges and tracking pixels.
    if url.startswith("data:"):
        return False  # inline base64 is always a placeholder or icon at this size
    parsed = urlparse(url)
    if not parsed.netloc:
        return False

    if parsed.path.lower().endswith(".svg"):
        return False
    return not _NON_PRODUCT.search(parsed.path)


def _absolutise(url: str, base_url: str | None) -> str | None:
    #Turn protocol relative and root relative URLs into absolute ones.
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
    #Parse an int if we can, coping with things like `"800px"` and None.
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None
