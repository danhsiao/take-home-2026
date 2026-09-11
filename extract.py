"""Deterministic evidence harvesting: raw HTML -> PageEvidence.

This is the first and largest stage of the pipeline, and it runs before any model is
involved. Its job is *recall*: find every place the page might have stated a product
fact, and hand them onward. It deliberately does not decide what anything means,
except where a published standard already decided for us.

Four evidence surfaces are harvested, in descending order of trustworthiness:

1. schema.org JSON-LD          - semantics defined by a spec
2. microdata + OpenGraph/meta  - semantics defined by a spec
3. embedded application JSON   - shape is arbitrary, so we prune rather than interpret
4. visible text                - last resort, ranked by proximity to the page title

THE COMPLIANCE LINE (the rule this whole module is built around):
discovering a JSON blob generically - a `<script type="application/ld+json">`, a
`<script type="application/json">`, a `window.__X__ = {...}` assignment - is
*framework* generic and fine, because those are conventions of Next.js/Redux/Apollo
and of the schema.org and OpenGraph specs, not of any merchant. Navigating a
discovered blob by a known path (`data["product"]["skus"][0]["price"]`) would be
site-specific and is forbidden. So: we locate structures by SHAPE, never by PATH, and
we never branch on a domain name.

Image collection lives in `images.py`; this module supplies it with the parsed tree,
the product-typed JSON-LD items, and the pruned subtrees.
"""

import json
import re
from typing import Any

from selectolax.parser import HTMLParser

import images
from models import PageEvidence

# --------------------------------------------------------------------------------
# Budgets.
#
# These bound how much evidence we forward to the model, which is the single biggest
# lever on token cost - and the entire justification for this architecture over
# "send the page to a big model". They are not arbitrary: they were set by measuring
# the qualifying-subtree size distribution across the provided PDPs. See the README
# cost table.
#
# The important property is that they prune NOISE rather than truncate SIGNAL. An
# oversized product record is decomposed or denoised, never silently dropped - an
# earlier cut of this module discarded a 70KB subtree and produced *zero* evidence for
# the one page that has no JSON-LD at all.
# --------------------------------------------------------------------------------

# Sized from measurement, not guesswork. The largest atomic product record across the
# provided PDPs denoises to ~59K chars, so the per-subtree cap sits just above that:
# below it, that page's only evidence would be discarded or shattered into fragments.
MAX_SUBTREE_CHARS = 64_000  # largest single coherent subtree we will keep whole
MAX_TOTAL_SUBTREE_CHARS = 64_000  # ceiling across all subtrees for one page
MAX_JSON_SUBTREES = 16
MAX_STRING_CHARS = 2000  # long strings in app state are marketing HTML, not data
MAX_ARRAY_ITEMS = 1000 # a variant table longer than this is not a variant table
MAX_TEXT_BLOCKS = 200
MAX_TEXT_BLOCK_CHARS = 2_000  # generous: product descriptions are legitimately long

# A region is navigation/recommendations when this share of its text is link text.
# Menus and carousels sit near 0.9; a product body sits near 0.05, so the threshold is
# not delicate.
LINK_DENSITY_THRESHOLD = 0.6
# Below this, the density ratio is computed over too few characters to be meaningful.
# Kept low because compact recommendation strips ("Alpha | Beta | Gamma") are common
# and genuinely short - a high threshold lets exactly the wrong content through.
MIN_LINK_FARM_CHARS = 20

# Tags whose text is a meaningful standalone unit. Generic HTML semantics.
BLOCK_TAGS = {
    "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "dd", "dt",
    "figcaption", "blockquote", "span", "div", "label", "button", "option", "a",
}

# Tags that never contain product information.
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "head"}

# Attribute recording an element's distance from the product title. Set on the tree
# for the same reason as `images.SKIP_ATTR`: selectolax node ids are not stable across
# accesses, so the tree is the only durable place to record a structural annotation.
TITLE_DEPTH_ATTR = "data-evidence-title-depth"


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------


def run(html: str, url: str | None = None) -> PageEvidence:
    """Harvest all deterministic evidence from a raw HTML string.

    Args:
        html: the raw page source, exactly as fetched.
        url: the page URL if known. Used only as a fallback for the canonical URL and
            to recognise links pointing away from this product; never branched on.

    Returns:
        A PageEvidence containing everything we found. Fields may legitimately be
        empty - a page with no structured data is a normal, expected input.
    """
    tree = HTMLParser(html)

    meta = _parse_meta(tree)
    structured = _parse_jsonld(html)
    canonical = meta.get("canonical") or meta.get("og:url") or url

    # Marks the tree once, in place. Both text and image harvesting need to know which
    # DOM regions are navigation or recommendation strips rather than product content.
    _mark_link_farms(tree, canonical)

    # Parsed once, then used two different ways. `json_subtrees` is a *budgeted*
    # selection for the model, because context costs money. Image harvesting gets the
    # full set of blobs instead: image URLs are never sent to the model, so there is no
    # token reason to ration them, and rationing them actively loses galleries - an
    # oversized product record gets decomposed for budget reasons, and its own media
    # can drop out while smaller sibling records survive.
    json_blobs = _find_json_blobs(html)
    json_subtrees = _select_subtrees(json_blobs)

    return PageEvidence(
        url=canonical,
        structured=structured,
        microdata=_parse_microdata(tree),
        meta=meta,
        breadcrumbs=_parse_breadcrumbs(structured, tree),
        json_subtrees=json_subtrees,
        text_blocks=_harvest_text_blocks(tree, meta),
        images=images.collect(
            tree=tree,
            product_items=find_typed(structured, "Product", "ProductGroup"),
            meta=meta,
            json_subtrees=json_blobs,
            base_url=canonical,
        ),
        videos=_harvest_videos(tree, structured, meta),
    )


# --------------------------------------------------------------------------------
# 1. schema.org JSON-LD
# --------------------------------------------------------------------------------


def _parse_jsonld(html: str) -> list[dict[str, Any]]:
    """Extract and flatten every schema.org JSON-LD item on the page.

    JSON-LD legitimately appears in several shapes - a bare object, an array of
    objects, or an object wrapping an `@graph` list - so all three are normalised into
    one flat list of typed items. We also recurse into `hasVariant`, so a
    ProductGroup's nested Products surface as items in their own right.

    Malformed blocks are skipped silently: a broken analytics blob elsewhere on the
    page is not a reason to fail the whole extraction.
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
    """Recursively flatten JSON-LD containers into a flat list of typed items."""
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

    # Surface nested Products (a ProductGroup's `hasVariant` entries) as first-class
    # items too, so downstream code finds them without knowing the nesting depth.
    for key in ("hasVariant", "isSimilarTo", "isRelatedTo"):
        if key in node:
            _flatten_jsonld(node[key], out)


def find_typed(items: list[dict[str, Any]], *types: str) -> list[dict[str, Any]]:
    """Return JSON-LD items whose `@type` matches any of `types`.

    `@type` may be a string or a list of strings, and either a bare name or a full
    schema.org URL, so every form is normalised before comparison.
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
    """Extract microdata itemscopes as flat property maps.

    Microdata is the older schema.org serialisation and is still common. Each
    `itemscope` element's `itemprop` descendants are collected into a dict, keeping
    `itemtype` so consumers can tell a Product from a BreadcrumbList.

    Nested itemscopes are flattened rather than modelled precisely: a flat map is
    enough to recover prices and identifiers, which is realistically all microdata
    carries.
    """
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

    return scopes


def _microdata_value(node) -> str | None:
    """Read a microdata property value from the attribute its element type implies.

    The spec puts the value in `content` for meta, `href` for links, `src` for media,
    `datetime` for time, and the text content otherwise.
    """
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
    """Collect OpenGraph, Twitter Card, and plain meta tags, plus the canonical link.

    OpenGraph is a published standard, so `og:title` and `og:image` carry real
    semantics we are entitled to rely on - unlike, say, a CSS class name.
    """
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


def _parse_breadcrumbs(structured: list[dict[str, Any]], tree: HTMLParser) -> list[str]:
    """Recover the page's breadcrumb trail.

    Breadcrumbs are the single strongest taxonomy signal a PDP offers - the merchant's
    own categorisation of the product - and they feed the taxonomy retrieval query
    directly.

    schema.org `BreadcrumbList` is preferred because its meaning is specified. Only if
    that is absent do we fall back to the ARIA `breadcrumb` landmark, which is an
    accessibility standard rather than a site-specific selector.
    """
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

    # Fallback: the ARIA breadcrumb landmark, defined by the ARIA spec and used across
    # the web, so this is a standard rather than a merchant selector.
    for selector in ('nav[aria-label*="readcrumb"]', '[class*="readcrumb"]'):
        node = tree.css_first(selector)
        if not node:
            continue
        names = [
            (link.text() or "").strip()
            for link in node.css("a, li")
            if (link.text() or "").strip()
        ]
        # Nested li>a markup produces repeats; dedupe while preserving order.
        seen: set[str] = set()
        ordered = [n for n in names if not (n in seen or seen.add(n))]
        if ordered:
            return ordered[:10]

    return []


# --------------------------------------------------------------------------------
# 3. embedded application JSON
# --------------------------------------------------------------------------------


def _select_subtrees(blobs: list[Any]) -> list[dict[str, Any]]:
    """Select the product-shaped subtrees to forward to the model, within budget.

    This is the hardest part of the module and the one that matters most: several real
    PDPs put their price, images, and full variant matrix *only* in a framework state
    blob, with no JSON-LD at all. Those blobs run to 100-450KB, far too large to
    forward to a model - so we prune.

    The rule is: KEEP THE LARGEST COHERENT SUBTREE THAT FITS THE BUDGET.

      1. Find JSON blobs by framework convention. Those conventions belong to
         Next.js/Redux/Apollo, not to any merchant.
      2. Score object subtrees on product-likeness using generic commerce field names
         (`price`, `name`, `sku`) - vocabulary shared across the whole industry.
      3. Denoise a qualifying subtree, then keep it WHOLE if it fits. Coherence
         matters enormously: real state blobs are normalised relational graphs where a
         SKU references its price and images by foreign key into sibling arrays.
         Splitting that apart destroys the only thing that makes the variants
         recoverable, so we hand the model the whole structure and let it resolve the
         references.
      4. Only if a subtree cannot fit do we decompose it into its qualifying children.
         If decomposition yields nothing, keep the denoised parent anyway - something
         beats nothing, and this case is exactly the page that has no JSON-LD.

    We never navigate to a known key path, and we never branch on a domain.

    Known limitation: variant recovery from foreign-key-joined state is partial. That
    is a deliberate, documented trade-off, not an oversight.
    """
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
    """Locate parsed JSON payloads embedded in the page.

    Two generic mechanisms:
      - `<script type="application/json">` bodies, how Next.js, Nuxt, and Remix ship
        server-rendered state.
      - `window.SOMETHING = {...}` assignments, the older SSR convention.

    Both are framework conventions. Neither encodes which merchant we are on.
    """
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
    """Parse one JSON value beginning at `start` by matching brackets.

    Tracks string state and backslash escapes so braces inside string literals do not
    corrupt the depth count. Returns None if the value is unterminated or is not valid
    JSON (a JS object literal with unquoted keys, for instance).
    """
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


# Generic commerce field-name vocabulary. These are ordinary words used by essentially
# every ecommerce framework and by the schema.org vocabulary itself; they identify no
# particular merchant. This is the closest the module comes to the compliance line and
# it stays on the right side of it: we match field *names* that are industry-standard,
# never a path, a domain, or a CSS class.
# An optional leading qualifier absorbs the camelCase and snake_case compounds that
# every framework produces - `currentPrice`, `colorDescription`, `product_name` - so a
# record is recognised by the *kind* of field it carries rather than by an exact
# spelling. Without this, records whose price field is named `prices` or whose only
# label is `colorDescription` are invisible, which loses a real page's entire gallery.
_PRICE_KEYS = re.compile(r"^[a-z]*_?(price|prices|amount|value|cost)$", re.I)
_NAME_KEYS = re.compile(r"^[a-z]*_?(name|title|label|description)s?$", re.I)
_ID_KEYS = re.compile(
    r"^[a-z]*_?(sku|skus|gtin\d*|upc|ean|mpn|part_?number|item_?id|product_?id|"
    r"variant_?id)$",
    re.I,
)
_OPTION_KEYS = re.compile(
    r"^(option|options|variant|variants|attribute|attributes|selection|swatch|choices|skus)$",
    re.I,
)

# Keys whose contents are never product data. Dropping these is how an oversized
# subtree is shrunk without losing product information - reviews, Q&A, analytics
# payloads, and experiment buckets routinely account for most of a state blob's bulk.
_NOISE_KEYS = re.compile(
    r"(analytic|telemetr|tracking|gtm|ga4|experiment|abtest|session|csrf|nonce|"
    r"feature_?flag|metrics|review|question|recommend|seo|html|markup|script|"
    r"style|icon|warning|policy|faq|__typename|breadcrumbJson)",
    re.I,
)


def _collect_subtrees(node: Any, out: list[dict[str, Any]], depth: int = 0) -> None:
    """Recursively collect product-shaped subtrees, preferring coherent wholes.

    See `_select_subtrees` for the rationale behind the keep/decompose rule.
    """
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


def _denoise(node: Any, depth: int = 0, max_string: int = MAX_STRING_CHARS) -> Any:
    """Strip non-product bulk from a subtree without losing product structure.

    Removes keys matching the generic noise vocabulary, truncates long strings
    (invariably marketing HTML rather than data), and caps very long arrays. The
    relational shape - which records exist and how they reference each other - is
    preserved, because that is what makes variants recoverable.
    """
    if depth > 12:
        return None

    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            if isinstance(key, str) and _NOISE_KEYS.search(key):
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
    """Size of a subtree once serialised, as a proxy for its token cost."""
    try:
        return len(json.dumps(node, default=str))
    except (TypeError, ValueError):
        return MAX_SUBTREE_CHARS + 1


def _is_product_shaped(node: dict[str, Any]) -> bool:
    """Decide whether one object looks like a product or variant record.

    Purely structural: a name-like string alongside a price-like number or a commerce
    identifier, or a named object enumerating a homogeneous set of option records. No
    merchant, domain, or category knowledge is involved.
    """
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
    """Whether a value could plausibly be a price.

    Accepts numbers and numeric strings in a sane commerce range. The upper bound
    rejects timestamps and numeric IDs that happen to sit under a `value` key.
    """
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
    """Whether a value is a homogeneous array of records - i.e. an option table."""
    if not isinstance(value, list) or len(value) < 2:
        return False
    dicts = [v for v in value if isinstance(v, dict)]
    if len(dicts) < 2:
        return False
    # Homogeneous means the first two records share most of their keys. That is what
    # distinguishes a data table from an incidental list of mixed objects.
    first, second = set(dicts[0].keys()), set(dicts[1].keys())
    if not first or not second:
        return False
    return len(first & second) / max(len(first), len(second)) > 0.6


# --------------------------------------------------------------------------------
# 4. visible text
# --------------------------------------------------------------------------------


def _harvest_text_blocks(tree: HTMLParser, meta: dict[str, str]) -> list[str]:
    """Collect visible text, ranked so main product content comes first.

    Needed because some PDPs state their price only in rendered markup, with no
    machine-readable copy anywhere on the page. But visible text is also where
    cross-sell and recommendation modules live, and those carry *other products'*
    names and prices - a naive text scrape invites the model to describe the wrong
    product entirely.

    Two generic defences, neither using a selector tied to a merchant:

      1. Drop link-farm containers (see `_mark_link_farms`).
      2. Rank by proximity to the title. Blocks sharing a closer DOM ancestor with the
         element containing `og:title` are likelier to belong to the main product
         region, so they survive the budget cut while distant blocks do not.
    """
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
        # Truncate rather than drop: a long block is usually the product description,
        # and discarding it outright would lose the best description on the page.
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
    """Mark containers that are navigation/recommendation regions, not content.

    Structural rule: LINK DENSITY. A region where most of the visible text is itself
    link text, pointing at three or more distinct other pages, is a list of other
    things - a menu, a carousel, a "you may also like" strip. A product body is mostly
    prose with a few links (size guide, review anchors), so its density is low.

    Density rather than a raw link count is essential. Counting links alone marks any
    outer container that merely *encloses* the site header, which on a real PDP is
    nearly every wrapper on the page - that cascades until the entire document is
    suppressed. Density is self-limiting, because a container large enough to hold the
    product description is diluted by that description's text.

    This is a property of the DOM that holds on any site, which is what lets it defend
    against cross-sell contamination without naming a merchant or a CSS class. One of
    the provided pages carries a cross-sell block with a different product's name and
    price, and another exposes `srcset` entries for an entirely different SKU.

    Regions are marked with an attribute ON THE TREE rather than collected into a set
    of node ids. selectolax hands out a fresh Python wrapper on every tree access, so
    `id(node)` is not a stable identity: the ids are freed as soon as the wrappers go
    out of scope and are then reused by unrelated nodes, which silently misclassifies
    most of the page. The tree itself is the only durable place to record this.
    """
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
    """Reduce a URL to its path, for comparing 'is this the same page'."""
    if not url:
        return ""
    stripped = re.sub(r"^https?://[^/]+", "", url)
    return stripped.split("?")[0].split("#")[0].rstrip("/")


def _mark_title_ancestors(tree: HTMLParser, meta: dict[str, str]) -> int:
    """Mark the ancestors of the element rendering the page title, nearest first.

    This is the anchor for proximity ranking. The title element is located by matching
    the text of `og:title` (a standard) rather than by looking for an `<h1>`, because
    a real PDP in the provided data uses its only `<h1>` for a breadcrumb -
    heading-based heuristics are not dependable across sites.

    Like `_mark_link_farms`, the chain is recorded as an attribute on the tree rather
    than as node ids, because selectolax node identity is not stable across accesses.

    Returns:
        The chain length, so scores can be normalised. Zero if no title was found.
    """
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
    """Score a node by how closely it shares a DOM ancestor with the title element.

    A higher score means the node sits in the same region of the page as the product
    title - a generic structural proxy for "part of the main product block". Walking
    up from the node, the first marked ancestor found is the nearest common ancestor
    with the title, and a smaller recorded depth means a tighter relationship.
    """
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
    """Collapse whitespace so text blocks compare and deduplicate cleanly."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------------
# 5. video
# --------------------------------------------------------------------------------


def _harvest_videos(
    tree: HTMLParser, structured: list[dict[str, Any]], meta: dict[str, str]
) -> list[str]:
    """Collect video URLs from schema.org, OpenGraph, and HTML media elements.

    All three sources are standards-defined, so this needs no interpretation and never
    reaches a model.
    """
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
