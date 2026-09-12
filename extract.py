"""Pulls evidence out of raw HTML and returns a PageEvidence.

The rule this module follows: finding a structure generically is fine, but reaching
into it by a known path like `data["product"]["skus"][0]` is not. We find things by
shape, and nothing branches on a domain name.
"""

import json
import re
from typing import Any

from urllib.parse import urljoin

from selectolax.parser import HTMLParser

import controls
import identity as identity_mod
import images
from identity import PageIdentity
from models import PageEvidence, VariantRecord

# Budgets bounding how much evidence reaches the model - the biggest lever on token cost. 
MAX_SUBTREE_CHARS = 64_000  # largest single coherent subtree we will keep whole
MAX_TOTAL_SUBTREE_CHARS = 64_000  # ceiling across all subtrees for one page
MAX_JSON_SUBTREES = 16
MAX_STRING_CHARS = 2000  # long strings in app state are marketing HTML, not data
MAX_ARRAY_ITEMS = 1000 # a variant table longer than this is not a variant table
MAX_TEXT_BLOCKS = 200
MAX_TEXT_BLOCK_CHARS = 2_000  # generous: product descriptions are legitimately long

# A region is navigation/recommendations when this share of its text is link text.
# Menus sit near 0.9 and a product body near 0.05, so the threshold is not delicate.
LINK_DENSITY_THRESHOLD = 0.6
# Below this the ratio is computed over too few characters to mean anything. Kept low:
# compact recommendation strips are common and genuinely short.
MIN_LINK_FARM_CHARS = 20

# Generic HTML semantics.
BLOCK_TAGS = {
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "dd", "dt",
    "figcaption", "blockquote", "span", "div", "label", "button", "option", "a",
}

# Tags that never contain product information (from my experience at Expedia Group and Research)
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "head"}

# An element's distance from the product title, recorded on the tree itself because
# selectolax node ids are not stable across accesses.
TITLE_DEPTH_ATTR = images.TITLE_ATTR


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------


def run(html: str, url: str | None = None) -> PageEvidence:
    """Harvest every deterministic fact from a page of HTML.

    `url` is only a fallback for the canonical link. Empty fields are normal, not a bug.
    """
    tree = HTMLParser(html)

    meta = _parse_meta(tree)
    structured = _parse_jsonld(html)
    canonical = meta.get("canonical") or meta.get("og:url") or url

    # Marks the tree once, in place. 
    _mark_link_farms(tree, canonical)
    # Marks the chain from the product title up to the root. Text ranking uses it for
    # proximity; image relevance uses it to know where this product's own region ends.
    _mark_title_ancestors(tree, meta)

    # Parsed once, used two ways: `json_subtrees` is budgeted for the model, while
    # image harvesting gets every blob - image URLs never reach the model, so rationing
    # them saves no tokens and loses galleries.
    json_blobs = _find_json_blobs(html)

    # Who this document says it is. Built before any blob is read, because it is what
    # decides whether a blob is about this page at all.
    page_identity = _build_identity(meta, structured, canonical)

    # A payload declaring another route is stale or foreign state - a leftover SPA
    # payload, a redirect, an interstitial. Its products are *real* products, which is
    # exactly why nothing downstream can tell them from this one by shape.
    trusted_blobs = [
        blob
        for blob in json_blobs
        if not identity_mod.blob_describes_another_page(blob, page_identity)
    ]
    stale_blobs = len(json_blobs) - len(trusted_blobs)

    json_subtrees = _select_subtrees(trusted_blobs)

    # Resolved before images, because a page that publishes a variant graph thereby
    # publishes an inventory of its own product photography, and image harvesting is
    # better off knowing it.
    variant_graph = harvest_variant_graph(trusted_blobs, canonical)

    return PageEvidence(
        url=canonical,
        structured=structured,
        microdata=_parse_microdata(tree),
        meta=meta,
        breadcrumbs=_parse_breadcrumbs(structured, tree, canonical),
        json_subtrees=json_subtrees,
        text_blocks=_harvest_text_blocks(tree, meta),
        images=images.collect(
            tree=tree,
            product_items=find_typed(structured, "Product", "ProductGroup"),
            meta=meta,
            json_subtrees=trusted_blobs,
            base_url=canonical,
            declared_media=[url for row in variant_graph for url in row.image_urls],
            identity=page_identity,
        ),
        videos=_harvest_videos(tree, structured, meta),
        variant_graph=variant_graph,
        # The only evidence surface that survives client-side rendering: when a page
        # fetches its options over an API, the accessibility tree is all that remains.
        option_groups=controls.harvest(tree, canonical),
        identity_ids=sorted(page_identity.ids),
        stale_blobs=stale_blobs,
    )


def _build_identity(
    meta: dict[str, str],
    structured: list[dict[str, Any]],
    canonical: str | None,
) -> PageIdentity:
    #Collect what this page says its own identity is.
    extra: set[str] = set()
    for key in ("og:url", "twitter:url", "al:web:url"):
        if value := meta.get(key):
            extra |= identity_mod.id_tokens(value)

    for item in find_typed(structured, "Product", "ProductGroup"):
        for key, value in item.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower().replace("_", "").replace("-", "")
            if lowered in {"@id", "id", "sku", "productid", "mpn", "url"} or lowered.startswith(
                "gtin"
            ):
                extra |= identity_mod.id_tokens(value)
            elif lowered == "offers":
                # An Offer carries the purchasable identity (`sku`, `gtin13`) on many pages
                for offer in value if isinstance(value, list) else [value]:
                    if isinstance(offer, dict):
                        extra |= identity_mod.record_identity(offer)

    return PageIdentity(
        canonical_url=canonical,
        extra_ids=extra,
        title=meta.get("og:title") or meta.get("title"),
    )


# --------------------------------------------------------------------------------
# 1. schema.org JSON-LD
# --------------------------------------------------------------------------------


def _parse_jsonld(html: str) -> list[dict[str, Any]]:
    """Find and flatten every schema.org JSON-LD block on the page.

    We scan the raw text instead of walking the DOM, which copes better with `</` inside
    script bodies.
    """
    items: list[dict[str, Any]] = []

    # A raw text scan rather than a DOM walk: script bodies containing "</" sequences
    # or unusual escaping survive this more reliably than tree parsing.
    for match in re.finditer(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.S | re.I,
    ):
        try:
            parsed = json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            continue
        _flatten_jsonld(parsed, items)

    return items


def _flatten_jsonld(node: Any, out: list[dict[str, Any]]) -> None:
    """Flatten JSON-LD containers into one flat list of typed items."""
    if isinstance(node, list):
        for child in node:
            _flatten_jsonld(child, out)
        return

    if not isinstance(node, dict):
        return

    # `@graph` is the standard container for multiple items in a single block.
    if "@graph" in node:
        _flatten_jsonld(node["@graph"], out)

    if "@type" in node:
        out.append(node)

    # A ProductGroup's `hasVariant` entries are configurations *of this product*
    if "hasVariant" in node:
        _flatten_jsonld(node["hasVariant"], out)

    # schema.org defines `breadcrumb` as a property of `WebPage` holding a
    # `BreadcrumbList`, and pages commonly nest it there. 
    if "breadcrumb" in node:
        _flatten_jsonld(node["breadcrumb"], out)


def find_typed(items: list[dict[str, Any]], *types: str) -> list[dict[str, Any]]:
    """Return the JSON-LD items whose `@type` matches any of `types`.

    `@type` can be a string or a list, and a bare name or a full URL. All forms work.
    """
    wanted = {t.lower() for t in types}
    found = []
    for item in items:
        raw = item.get("@type")
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            # Accepts "https://schema.org/Product" as well as a bare "Product".
            if candidate.rsplit("/", 1)[-1].lower() in wanted:
                found.append(item)
                break
    return found


# --------------------------------------------------------------------------------
# 2. microdata and meta tags
# --------------------------------------------------------------------------------


def _parse_microdata(tree: HTMLParser) -> list[dict[str, Any]]:
    """Read microdata itemscopes as flat property maps."""
    scopes: list[dict[str, Any]] = []

    for node in tree.css("[itemscope]"):
        props: dict[str, Any] = {}
        itemtype = node.attributes.get("itemtype")
        if itemtype:
            props["@type"] = itemtype.rsplit("/", 1)[-1]

        for prop_node in node.css("[itemprop]"):
            name = prop_node.attributes.get("itemprop")
            if not name:
                continue
            value = _microdata_value(prop_node)
            if not value:
                continue
            # Repeated properties are legal in the spec; keep every occurrence.
            if name in props:
                existing = props[name]
                props[name] = (
                    existing + [value] if isinstance(existing, list) else [existing, value]
                )
            else:
                props[name] = value

        if len(props) > 1:  # more than just the @type we injected
            scopes.append(props)

    unscoped = _unscoped_itemprops(tree)
    if unscoped:
        scopes.append(unscoped)

    return scopes


def _unscoped_itemprops(tree: HTMLParser) -> dict[str, Any]:
    """Gather `itemprop` elements with no `itemscope` around them, as one item."""
    props: dict[str, Any] = {}

    for node in tree.css("[itemprop]"):
        name = node.attributes.get("itemprop")
        if not name or name in props or _within_itemscope(node):
            continue
        value = _microdata_value(node)
        if value:
            props[name] = value

    return props


def _within_itemscope(node) -> bool:
    """Is this element already inside a declared item?"""
    current = node.parent
    hops = 0
    while current is not None and hops < 25:
        if current.attributes.get("itemscope") is not None:
            return True
        current = current.parent
        hops += 1
    return False


def _microdata_value(node) -> str | None:
    """Read a microdata value from whichever attribute the tag implies."""
    attrs = node.attributes
    for attr in ("content", "datetime"):
        if attrs.get(attr):
            return attrs[attr].strip()
    if node.tag in ("a", "link", "area") and attrs.get("href"):
        return attrs["href"].strip()
    if node.tag in ("img", "audio", "video", "source", "embed") and attrs.get("src"):
        return attrs["src"].strip()
    text = (node.text() or "").strip()
    return text[:500] if text else None


def _parse_meta(tree: HTMLParser) -> dict[str, str]:
    """Collect OpenGraph, Twitter and plain meta tags, plus the canonical link."""
    meta: dict[str, str] = {}

    for node in tree.css("meta"):
        attrs = node.attributes
        key = attrs.get("property") or attrs.get("name") or attrs.get("itemprop")
        content = attrs.get("content")
        if key and content:
            # First occurrence wins: duplicated OG tags later in a document are
            # usually injected by third-party widgets.
            meta.setdefault(key.strip().lower(), content.strip())

    canonical = tree.css_first('link[rel="canonical"]')
    if canonical and canonical.attributes.get("href"):
        meta["canonical"] = canonical.attributes["href"].strip()

    title = tree.css_first("title")
    if title and title.text():
        meta.setdefault("title", title.text().strip())

    return meta


def _parse_breadcrumbs(
    structured: list[dict[str, Any]],
    tree: HTMLParser,
    canonical: str | None = None,
) -> list[str]:
    #Find the page's breadcrumb trail, the best category hint a PDP gives us.

    for item in find_typed(structured, "BreadcrumbList"):
        elements = item.get("itemListElement")
        if not isinstance(elements, list):
            continue
        names = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            name = element.get("name")
            if not name and isinstance(element.get("item"), dict):
                name = element["item"].get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        if names:
            return names

    # The microdata serialisation of the same specified vocabulary.
    for node in tree.css('[itemtype$="BreadcrumbList" i]'):
        names = [
            text
            for element in node.css('[itemprop="name"]')
            if (text := (element.attributes.get("content") or element.text() or "").strip())
        ]
        if names:
            return _dedupe_trail(names)

    if trail := _trail_by_path_prefix(tree, canonical):
        return trail

    for node in tree.css(
        'nav[aria-label*="readcrumb" i], [role="navigation"][aria-label*="readcrumb" i]'
    ):
        names = [text for link in node.css("a, li") if (text := (link.text() or "").strip())]
        if names:
            return _dedupe_trail(names)

    return []


def _dedupe_trail(names: list[str]) -> list[str]:
    """Drop the repeats that nested `li > a` markup creates, keeping order."""
    seen: set[str] = set()
    return [name for name in names if not (name in seen or seen.add(name))][:10]


# The smallest number of links that describes a *path* rather than a single reference.
MIN_TRAIL_LINKS = 2


def _trail_by_path_prefix(tree: HTMLParser, canonical: str | None) -> list[str]:
    #Find a breadcrumb by the one thing every breadcrumb has, in any language.
    if not canonical:
        return []
    segments = [segment for segment in _url_path(canonical).split("/") if segment]
    if len(segments) < MIN_TRAIL_LINKS:
        return []

    ancestors = ["/" + "/".join(segments[:depth]) for depth in range(1, len(segments))]

    best: list[str] = []
    for container in tree.css("nav, ol, ul"):
        if images.in_skipped_region(container):
            continue
        matched: list[tuple[int, str]] = []
        for link in container.css("a[href]"):
            path = _url_path(link.attributes.get("href")).rstrip("/")
            if path in ancestors and (text := (link.text() or "").strip()):
                matched.append((ancestors.index(path), text))
        # Strictly increasing depth: a trail, not a block that happens to hold
        # several ancestor links.
        depths = [depth for depth, _ in matched]
        if len(matched) >= MIN_TRAIL_LINKS and depths == sorted(set(depths)):
            if len(matched) > len(best):
                best = [text for _, text in matched]

    return _dedupe_trail(best) if best else []


# --------------------------------------------------------------------------------
# 3. embedded application JSON
# --------------------------------------------------------------------------------


def _select_subtrees(blobs: list[Any]) -> list[dict[str, Any]]:
    # Pick the product shaped subtrees to send to the model, within budget.
    collected: list[dict[str, Any]] = []
    for blob in blobs:
        _collect_subtrees(blob, collected)

    # Smallest-first, so the most specific records claim the budget before their
    # bulkier containers do.
    kept: list[dict[str, Any]] = []
    total = 0
    seen_fingerprints: set[str] = set()

    for subtree in sorted(collected, key=_serialised_size):
        size = _serialised_size(subtree)
        if len(kept) >= MAX_JSON_SUBTREES or total + size > MAX_TOTAL_SUBTREE_CHARS:
            break
        # Identical subtrees often appear in several blobs (SSR state duplicated into
        # a hydration payload), so keep only one copy of each.
        fingerprint = json.dumps(subtree, sort_keys=True, default=str)[:2000]
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        kept.append(subtree)
        total += size

    return kept


def _find_json_blobs(html: str) -> list[Any]:
    # Find JSON payloads embedded in the page.
    blobs: list[Any] = []

    for match in re.finditer(
        r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
        html,
        re.S | re.I,
    ):
        try:
            blobs.append(json.loads(match.group(1).strip()))
        except (json.JSONDecodeError, ValueError):
            continue

    # Brace-matched rather than regexed, because the payload contains arbitrary nested
    # braces inside string literals.
    for match in re.finditer(r"window\.([A-Za-z_$][\w$]*)\s*=\s*(?=[\{\[])", html):
        parsed = _brace_match(html, match.end())
        if parsed is not None:
            blobs.append(parsed)

    return blobs


def _brace_match(text: str, start: int) -> Any | None:
    # Read one JSON value starting at `start` by matching brackets.
    if text[start] not in "{[":
        return None

    depth = 0
    in_string = False
    escaped = False
    quote = ""

    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue

        if char in "\"'":
            in_string = True
            quote = char
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# Generic commerce field-name vocabulary. 
_PRICE_KEYS = re.compile(r"^[a-z]*_?(price|prices|amount|value|cost)$", re.I)
_NAME_KEYS = re.compile(r"^[a-z]*_?(name|title|label|description)s?$", re.I)
_ID_KEYS = re.compile(
    r"^[a-z]*_?(sku|skus|gtin\d*|upc|ean|mpn|part_?number|item_?id|product_?id|"
    r"variant_?id)$",
    re.I,
)
_OPTION_KEYS = re.compile(
    r"^(option|options|variant|variants|attribute|attributes|selection|swatch|"
    r"choices|choice|answers|values|skus)$",
    re.I,
)

# Keys whose contents are usually not product data. 
_NOISE_KEYS = re.compile(
    r"(analytic|telemetr|tracking|gtm|ga4|experiment|abtest|session|csrf|nonce|"
    r"feature_?flag|metrics|review|question|recommend|seo|html|markup|script|"
    r"style|icon|warning|policy|faq|__typename|breadcrumbJson)",
    re.I,
)


def _collect_subtrees(node: Any, out: list[dict[str, Any]], depth: int = 0) -> None:
    """Walk the blob and collect product shaped subtrees, whole ones first."""
    if depth > 12:
        return

    if isinstance(node, list):
        for child in node:
            _collect_subtrees(child, out, depth + 1)
        return

    if not isinstance(node, dict):
        return

    if _is_product_shaped(node):
        denoised = _denoise(node)
        if denoised and _serialised_size(denoised) <= MAX_SUBTREE_CHARS:
            out.append(denoised)
            return

        # Too large even after denoising. Try to decompose into finer records.
        before = len(out)
        for child in node.values():
            _collect_subtrees(child, out, depth + 1)

        # Nothing finer qualified, so this record is atomic. Keep it anyway, denoised
        # harder - emitting nothing here would mean emitting nothing for the page.
        if len(out) == before and denoised:
            out.append(_denoise(node, max_string=200))
        return

    for child in node.values():
        _collect_subtrees(child, out, depth + 1)


def _carries_options(node: Any, depth: int = 0) -> bool:
    #This guards the cleanup step, so a configurator built out of questions and answers does not get deleted as noise.

    if depth > 4:
        return False

    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and _OPTION_KEYS.match(key) and _is_record_array(value):
                return True
            if _carries_options(value, depth + 1):
                return True
        return False

    if isinstance(node, list):
        return any(_carries_options(child, depth + 1) for child in node[:20])

    return False


def _denoise(node: Any, depth: int = 0, max_string: int = MAX_STRING_CHARS) -> Any:
    # Strip the bulk out of a subtree without losing product structure.
    if depth > 12:
        return None

    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            if (
                isinstance(key, str)
                and _NOISE_KEYS.search(key)
                and not _carries_options(value)
            ):
                continue
            child = _denoise(value, depth + 1, max_string)
            if child is not None and child != {} and child != []:
                result[key] = child
        return result

    if isinstance(node, list):
        trimmed = [_denoise(x, depth + 1, max_string) for x in node[:MAX_ARRAY_ITEMS]]
        return [x for x in trimmed if x is not None and x != {} and x != []]

    if isinstance(node, str):
        return node[:max_string] + "..." if len(node) > max_string else node

    return node


def _serialised_size(node: Any) -> int:
    #How big a subtree is once serialised. A rough stand in for token cost.
    try:
        return len(json.dumps(node, default=str))
    except (TypeError, ValueError):
        return MAX_SUBTREE_CHARS + 1


def _is_product_shaped(node: dict[str, Any]) -> bool:
    #Decide whether an object looks like a product or a variant record.
    has_name = has_price = has_id = has_option_table = False

    for key, value in node.items():
        if not isinstance(key, str):
            continue
        if _NAME_KEYS.match(key) and isinstance(value, str) and 2 < len(value) < 300:
            has_name = True
        elif _PRICE_KEYS.match(key) and _is_price_like(value):
            has_price = True
        elif _ID_KEYS.match(key) and isinstance(value, (str, int)) and str(value).strip():
            has_id = True
        elif _OPTION_KEYS.match(key) and _is_record_array(value):
            has_option_table = True

    # A product record: something named, carrying a price or an identifier.
    # A configurator record: something named that enumerates its options.
    return has_name and (has_price or has_id or has_option_table)


def _is_price_like(value: Any) -> bool:
    #Could this value be a price?
    if isinstance(value, bool):  # bool subclasses int; exclude explicitly
        return False
    if isinstance(value, (int, float)):
        return 0 < value < 10_000_000
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.]", "", value)
        try:
            return 0 < float(cleaned) < 10_000_000
        except ValueError:
            return False
    return False


def _is_record_array(value: Any) -> bool:
    #Is this a list of similar records, meaning an option table?
    if not isinstance(value, list) or len(value) < 2:
        return False
    dicts = [v for v in value if isinstance(v, dict)]
    if len(dicts) < 2:
        return False
    first, second = set(dicts[0].keys()), set(dicts[1].keys())
    if not first or not second:
        return False
    return len(first & second) / max(len(first), len(second)) > 0.6


# --------------------------------------------------------------------------------
# 4. visible text
# --------------------------------------------------------------------------------


def _harvest_text_blocks(tree: HTMLParser, meta: dict[str, str]) -> list[str]:
    #Collect visible text, with the main product content ranked first.
    body = tree.css_first("body")
    if body is None:
        return []

    chain_length = _mark_title_ancestors(tree, meta)

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()

    for node in body.traverse(include_text=False):
        if node.tag in SKIP_TAGS or node.tag not in BLOCK_TAGS:
            continue
        if images.in_skipped_region(node):
            continue

        text = _normalise(node.text(deep=True, separator=" ") or "")
        if not text or text in seen:
            continue
        # Truncate rather than drop
        if len(text) > MAX_TEXT_BLOCK_CHARS:
            text = text[:MAX_TEXT_BLOCK_CHARS] + "..."
            if text in seen:
                continue
        seen.add(text)
        scored.append((_proximity_score(node, chain_length), text))

    # Highest proximity first, so the budget retains main-product content.
    scored.sort(key=lambda pair: -pair[0])
    return [text for _score, text in scored[:MAX_TEXT_BLOCKS]]


def _mark_link_farms(tree: HTMLParser, canonical: str | None) -> None:
    #Mark containers that are nav or recommendation strips rather than content.

    canonical_path = _url_path(canonical)

    for node in tree.css("ul, ol, nav, section, aside, div"):
        total_chars = len(_normalise(node.text(deep=True, separator=" ") or ""))
        # Too small to judge: a two-word container is not evidence of anything.
        if total_chars < MIN_LINK_FARM_CHARS:
            continue

        link_chars = 0
        paths = set()
        for link in node.css("a"):
            link_chars += len(_normalise(link.text(deep=True, separator=" ") or ""))
            href = link.attributes.get("href")
            if not href or href.startswith("#"):
                continue
            path = _url_path(href)
            # Ignore in-page anchors and links back to this same product.
            if path and path != canonical_path:
                paths.add(path)

        if len(paths) >= 3 and link_chars / total_chars > LINK_DENSITY_THRESHOLD:
            node.attrs[images.SKIP_ATTR] = "1"


def _url_path(url: str | None) -> str:
    #Cut a URL down to its path, so we can ask if two links are the same page.
    if not url:
        return ""
    stripped = re.sub(r"^https?://[^/]+", "", url)
    return stripped.split("?")[0].split("#")[0].rstrip("/")


def _mark_title_ancestors(tree: HTMLParser, meta: dict[str, str]) -> int:
    #Mark the ancestors of the element showing the page title, nearest first.

    title = meta.get("og:title") or meta.get("title")
    if not title:
        return 0

    needle = _normalise(title).lower()[:60]
    if len(needle) < 4:
        return 0

    best = None
    best_length = None
    for node in tree.css("h1, h2, span, div, p, a"):
        text = _normalise(node.text(deep=True, separator=" ") or "").lower()
        # The smallest matching element is the tightest wrapper around the title.
        if needle in text and (best_length is None or len(text) < best_length):
            best, best_length = node, len(text)

    if best is None:
        return 0

    depth = 0
    parent = best.parent
    while parent is not None and depth < 25:
        parent.attrs[TITLE_DEPTH_ATTR] = str(depth)
        parent = parent.parent
        depth += 1
    return depth


def _proximity_score(node, chain_length: int) -> int:
    #Score a node by how close its ancestors are to the title element.
    if chain_length == 0:
        return 0

    current = node
    hops = 0
    while current is not None and hops < 25:
        marked = current.attributes.get(TITLE_DEPTH_ATTR)
        if marked is not None:
            return chain_length - int(marked)
        current = current.parent
        hops += 1
    return 0


def _normalise(text: str) -> str:
    #Collapse whitespace so text blocks compare and dedupe cleanly.
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------------
# 5. the variant graph
# --------------------------------------------------------------------------------

# Everything below matches on shape and on generic commerce vocabulary. No key path,
# no merchant, no CDN.

MAX_GRAPH_VARIANTS = 300
MIN_GRAPH_DIMENSIONS = 2

# The identity of a record, so rows can be found again by the ids that reference them.
_RECORD_ID_KEYS = re.compile(r"^(id|sku|code|key)$", re.I)
# Where a SKU row says this configuration is bought. 
_ROW_URL_KEYS = re.compile(r"^[a-z]*_?(url|uri|href|link|permalink|pdpurl|canonicalurl)$", re.I)
# Fields on a SKU row that point at its imagery, its price, and its stock state.
_MEDIA_REF_KEYS = re.compile(
    r"^(media|image|images|img|photo|photos|picture|pictures|asset|assets|"
    r"gallery|shot|shots)(_?ids?)?$",
    re.I,
)
_PRICE_REF_KEYS = re.compile(r"^(price|prices|pricing|cost)(_?ids?)?$", re.I)
_STOCK_KEYS = re.compile(
    r"^(availability|available|in_?stock|stock|inventory|inventory_?status|status)$", re.I
)
_AMOUNT_KEYS = re.compile(r"^(amount|value|price|current|now|sale|final)$", re.I)
_URL_VALUE_KEYS = re.compile(r"^(src|url|href|uri|source|link|path)$", re.I)

_IN_STOCK = re.compile(r"^(in ?stock|instock|in|available|purchasable|true|yes|y)$", re.I)
_OUT_OF_STOCK = re.compile(
    r"^(out ?of ?stock|outofstock|out|sold ?out|soldout|unavailable|discontinued|"
    r"backorder(ed)?|false|no|n)$",
    re.I,
)


def harvest_variant_graph(blobs: list[Any], base_url: str | None = None) -> list[VariantRecord]:
    #Join a page's option, SKU, media and price tables into one row per variant.

    groups = [group for blob in blobs for group in _option_tables(blob)]
    if len(groups) < MIN_GRAPH_DIMENSIONS:
        return []

    # identifier -> {dimension: choice}
    sku_to_options: dict[str, dict[str, str]] = {}
    for dimension, choices in groups:
        for choice, identifiers in choices:
            for identifier in identifiers:
                sku_to_options.setdefault(identifier, {})[dimension] = choice

    # Only now, once there is something to join against, is it worth indexing the page.
    records_by_id: dict[str, dict[str, Any]] = {}
    for blob in blobs:
        _index_records(blob, records_by_id)

    variants: list[VariantRecord] = []
    for sku, options in sku_to_options.items():
        if len(options) < MIN_GRAPH_DIMENSIONS:
            continue

        row = records_by_id.get(sku) or {}
        variants.append(
            VariantRecord(
                sku=sku,
                options=options,
                image_urls=_resolve_media(row, records_by_id, base_url),
                amount=_resolve_amount(row, records_by_id),
                available=_resolve_availability(row),
                url=_resolve_url(row, base_url),
            )
        )

    return variants[:MAX_GRAPH_VARIANTS]


def _resolve_url(row: dict[str, Any], base_url: str | None) -> str | None:
    #The buy URL a SKU row names, resolved against the page.
    for key, value in row.items():
        if isinstance(key, str) and _ROW_URL_KEYS.match(key) and isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return urljoin(base_url, candidate) if base_url else candidate
    return None


def _option_tables(
    node: Any, depth: int = 0
) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    #Find option tables, as (dimension, [(choice, ids)]).
    if depth > 8:
        return []

    if isinstance(node, list):
        return [table for child in node for table in _option_tables(child, depth + 1)]

    if not isinstance(node, dict):
        return []

    found: list[tuple[str, list[tuple[str, list[str]]]]] = []
    dimension = _label_of(node)

    for key, value in node.items():
        if not (isinstance(key, str) and _OPTION_KEYS.match(key) and isinstance(value, list)):
            continue

        choices: list[tuple[str, list[str]]] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            choice = _label_of(entry)
            identifiers = _identifier_list(entry)
            if choice and identifiers:
                choices.append((choice, identifiers))

        if dimension and choices:
            found.append((dimension, choices))

    for value in node.values():
        found.extend(_option_tables(value, depth + 1))

    return found


def _label_of(record: dict[str, Any]) -> str | None:
    #The readable name of a record, if it has one.
    for key in ("title", "name", "label", "displayName", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _identifier_list(entry: dict[str, Any]) -> list[str]:
    #The item ids one option choice applies to.
    for key, value in entry.items():
        if not (isinstance(key, str) and _ID_LIST_KEYS.match(key) and isinstance(value, list)):
            continue
        identifiers = [
            str(item) for item in value if isinstance(item, (str, int)) and str(item).strip()
        ]
        if identifiers:
            return identifiers
    return []


_ID_LIST_KEYS = re.compile(
    r"^(skus|sku_?ids|ids|item_?ids|variant_?ids|product_?ids|member_?ids)$", re.I
)


def _index_records(node: Any, out: dict[str, dict[str, Any]], depth: int = 0) -> None:
    #Index every record that has an id, so references can be followed.
    if depth > 12:
        return

    if isinstance(node, list):
        for child in node:
            _index_records(child, out, depth + 1)
        return

    if not isinstance(node, dict):
        return

    for key, value in node.items():
        if isinstance(key, str) and _RECORD_ID_KEYS.match(key):
            if isinstance(value, (str, int)) and str(value).strip():
                out.setdefault(str(value), node)
                break

    for value in node.values():
        _index_records(value, out, depth + 1)


def _resolve_media(
    row: dict[str, Any], records: dict[str, dict[str, Any]], base_url: str | None
) -> list[str]:
    #Follow a SKU row's media references out to absolute image URLs.
    urls: list[str] = []

    for key, value in row.items():
        if not (isinstance(key, str) and _MEDIA_REF_KEYS.match(key)):
            continue
        for reference in value if isinstance(value, list) else [value]:
            url = _dereference_url(reference, records)
            absolute = images._absolutise(url, base_url) if url else None
            if absolute and absolute not in urls:
                urls.append(absolute)

    return urls


def _dereference_url(reference: Any, records: dict[str, dict[str, Any]]) -> str | None:
    #Turn one media reference into a URL. It may be a record, a URL, or an id.
    if isinstance(reference, dict):
        return _url_field(reference)

    if not isinstance(reference, (str, int)):
        return None

    value = str(reference).strip()
    if value.startswith(("http", "//", "/")):
        return value

    target = records.get(value)
    return _url_field(target) if target else None


def _url_field(record: dict[str, Any]) -> str | None:
    #The URL a media record exposes.#
    for key, value in record.items():
        if not (isinstance(key, str) and _URL_VALUE_KEYS.match(key)):
            continue
        if isinstance(value, str) and value.strip().startswith(("http", "//", "/")):
            return value.strip()
    return None


def _resolve_amount(row: dict[str, Any], records: dict[str, dict[str, Any]]) -> float | None:
    #Follow a SKU row's price reference to a number.
    for key, value in row.items():
        if not (isinstance(key, str) and _PRICE_REF_KEYS.match(key)):
            continue

        target: Any = value
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
                target = records.get(value)
            elif _is_price_like(value):
                return float(value)

        if isinstance(target, dict):
            for inner_key, inner_value in target.items():
                if (
                    isinstance(inner_key, str)
                    and _AMOUNT_KEYS.match(inner_key)
                    and _is_price_like(inner_value)
                ):
                    return float(inner_value)

    return None


def _resolve_availability(row: dict[str, Any]) -> bool | None:
    #Read a SKU row's stock state, leaving it unknown if the page is silent.
    for key, value in row.items():
        if not (isinstance(key, str) and _STOCK_KEYS.match(key)):
            continue

        candidates = (
            list(value.values()) if isinstance(value, dict) else [value]
        )
        for candidate in candidates:
            if isinstance(candidate, bool):
                return candidate
            if isinstance(candidate, str):
                text = candidate.strip()
                if _IN_STOCK.match(text):
                    return True
                if _OUT_OF_STOCK.match(text):
                    return False

    return None


# --------------------------------------------------------------------------------
# 6. video
# --------------------------------------------------------------------------------


def _harvest_videos(
    tree: HTMLParser, structured: list[dict[str, Any]], meta: dict[str, str]
) -> list[str]:
    #Collect video URLs from schema.org, OpenGraph and HTML media tags.
    videos: list[str] = []

    for item in structured:
        for key in ("video", "contentUrl", "embedUrl"):
            value = item.get(key)
            if isinstance(value, str) and value.startswith("http"):
                videos.append(value)
            elif isinstance(value, dict):
                for nested in ("contentUrl", "embedUrl", "url"):
                    if isinstance(value.get(nested), str):
                        videos.append(value[nested])
                        break

    for key in ("og:video", "og:video:url", "og:video:secure_url"):
        if meta.get(key):
            videos.append(meta[key])

    for node in tree.css("video[src], video source[src]"):
        src = node.attributes.get("src")
        if src:
            videos.append(src)

    # Preserve discovery order while removing duplicates.
    seen: set[str] = set()
    return [v for v in videos if not (v in seen or seen.add(v))]
