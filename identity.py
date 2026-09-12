"""Works out which product a page is actually about.

A PDP usually carries other real products too, like a "you may also like" rail or
leftover app state from the page you came from. Those records look exactly like
products because they are products. The one thing that separates them is that they
say who they are, and it is not who this page says it is.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

# --------------------------------------------------------------------------------
# Identifier tokens
# --------------------------------------------------------------------------------

# What makes a string usable as an identity token. Deliberately conservative, because
# a false *match* silently merges two products while a false *miss* only costs recall.

_ID_TOKEN = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9_.-]{5,}$")

# A pure version-ish or dimension-ish number, which appears in asset paths ("v1",
# "2890x1500") and identifies a rendition rather than a product.
_NOT_AN_ID = re.compile(r"^(?:v\d+|\d+x\d+)$", re.I)

# Keys whose value names a resource, meaning the page or record it describes. Matched
# on the key's shape, never by a path into a known blob. 
_RESOURCE_KEYS = re.compile(
    r"^[a-z]*_?(url|uri|canonical|canonicalurl|href|link|permalink|slug|path|route|page|aspath)$"
)

# Keys whose value identifies a *product*. schema.org vocabulary plus the
# industry-generic spellings of the same concepts.
_IDENTIFIER_KEYS = re.compile(
    r"^[a-z]*_?(id|ids|sku|skus|itemid|productid|offerid|variantid|gtin|gtin8|gtin12|"
    r"gtin13|gtin14|upc|ean|isbn|asin|mpn|partnumber|modelnumber|productcode)$"
)

# Keys that carry a human name. Used only to recognise that a record is
# product-shaped enough for its identity claim to be meaningful.
_NAME_KEYS = re.compile(r"^[a-z]*_?(name|title|productname|producttitle|displayname)$")


# A URL naming a file rather than a page, like an image or a video. These never count
# as identity.
_ASSET_URL = re.compile(
    r"\.(?:jpg|jpeg|png|webp|avif|gif|svg|bmp|ico|mp4|webm|mov|m3u8|pdf|css|js)"
    r"(?:[?#]|$)",
    re.I,
)


def id_tokens(value: Any) -> set[str]:
    #Pull every usable id token out of a value.

    if value is None or isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        return set()

    text = text.strip()
    if not text:
        return set()
    if _ASSET_URL.search(text):
        return set()  # names a file, not a product or a page

    # Keep only the path of anything URL-shaped; a bare identifier is left alone.
    if "//" in text or text.startswith("/") or "?" in text:
        text = urlparse(text).path or text

    tokens = set()
    for token in re.split(r"[^A-Za-z0-9_.-]+", text):
        token = token.strip("._-")
        if not token or _NOT_AN_ID.match(token):
            continue
        if _ID_TOKEN.match(token):
            tokens.add(token.lower())
    return tokens


class PageIdentity:
    #The identity a page claims for itself.

    def __init__(
        self,
        canonical_url: str | None = None,
        extra_ids: set[str] | None = None,
        title: str | None = None,
    ) -> None:
        self.canonical_url = canonical_url
        self.title = title
        self.path = urlparse(canonical_url).path if canonical_url else None
        self.ids: set[str] = set(extra_ids or set())
        if canonical_url:
            self.ids |= id_tokens(canonical_url)

    def __bool__(self) -> bool:
        return bool(self.ids)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PageIdentity(path={self.path!r}, ids={sorted(self.ids)!r})"

    def claims(self, tokens: set[str]) -> bool:
        """True if these tokens include an id this page claims."""
        return bool(tokens & self.ids)


# --------------------------------------------------------------------------------
# Ownership
# --------------------------------------------------------------------------------


class Ownership:
    #How a record relates to the main product. Three states, never a score.

    OWNED = "owned"  # declares an identifier this page claims
    FOREIGN = "foreign"  # declares identifiers, none of which this page claims
    UNKNOWN = "unknown"  # declares nothing that identifies anything


def record_identity(record: dict[str, Any]) -> set[str]:
    #The id tokens a record declares about itself.

    tokens: set[str] = set()
    for key, value in record.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower().replace("_", "").replace("-", "")
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if _RESOURCE_KEYS.match(lowered) or _IDENTIFIER_KEYS.match(lowered):
                tokens |= id_tokens(value)
        elif isinstance(value, list) and _IDENTIFIER_KEYS.match(lowered):
            # `skus: ["A1", "A2"]` and `gtins: [...]`; scalars only, one level.
            for item in value:
                if isinstance(item, (str, int, float)) and not isinstance(item, bool):
                    tokens |= id_tokens(item)
    return tokens


def is_product_shaped(record: dict[str, Any]) -> bool:
    #Whether a record has a name, so its id claim means something.

    return any(
        _NAME_KEYS.match(key.lower().replace("_", "").replace("-", ""))
        and isinstance(value, str)
        and value.strip()
        for key, value in record.items()
        if isinstance(key, str)
    )


def ownership(record: dict[str, Any], identity: PageIdentity) -> str:
    #Sort one record into OWNED, FOREIGN or UNKNOWN.

    if not identity:
        return Ownership.UNKNOWN
    if not isinstance(record, dict):
        return Ownership.UNKNOWN

    tokens = record_identity(record)
    if not tokens:
        return Ownership.UNKNOWN
    if identity.claims(tokens):
        return Ownership.OWNED
    if not is_product_shaped(record):
        return Ownership.UNKNOWN
    return Ownership.FOREIGN


def element_declares_foreign_id(node, identity: PageIdentity) -> bool:
    #Whether a DOM element declares an id this page does not claim.

    if not identity:
        return False

    tokens: set[str] = set()
    for key, value in node.attributes.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            continue
        if _names_a_product(key):
            tokens |= id_tokens(value)

    return bool(tokens) and not identity.claims(tokens)


# Attribute names that name a product. An exact set, matched against the end of the
# attribute name, rather than the loose prefix pattern we use for JSON keys.
_PRODUCT_ATTR_NAMES = frozenset(
    {
        "itemid", "productid", "offerid", "variantid", "skuid", "itemnumber",
        "productnumber", "productcode", "sku", "upc", "ean", "isbn", "asin", "mpn",
        "partnumber", "gtin", "gtin8", "gtin12", "gtin13", "gtin14",
        "href", "canonicalurl", "pdpurl", "producturl", "itemurl",
    }
)


def _names_a_product(attribute: str) -> bool:
    #Whether an attribute name refers to a product rather than to markup.

    segments = [
        segment
        for segment in re.split(r"[^a-z0-9]+", attribute.lower())
        if segment and segment != "data"
    ]
    if not segments:
        return False

    candidates = {"".join(segments), segments[-1]}
    if len(segments) >= 2:
        candidates.add(segments[-2] + segments[-1])
    return bool(candidates & _PRODUCT_ATTR_NAMES)


# --------------------------------------------------------------------------------
# Blob-level trust
# --------------------------------------------------------------------------------

# How deep we look for the route a blob claims. 
_ROUTE_SEARCH_DEPTH = 4

# Keys naming the route a rendered payload belongs to.
_ROUTE_KEYS = re.compile(r"^(page|pathname|aspath|route|canonical|canonicalurl|metacanon|url)$")


def _declared_routes(blob: Any, depth: int = 0) -> list[str]:
    #The routes a payload says it belongs to.

    found: list[str] = []
    if depth > _ROUTE_SEARCH_DEPTH:
        return found
    if isinstance(blob, dict):
        for key, value in blob.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower().replace("_", "").replace("-", "")
            if _ROUTE_KEYS.match(lowered) and isinstance(value, str) and value.strip():
                found.append(value.strip())
            else:
                found.extend(_declared_routes(value, depth + 1))

    return found


def _route_shape(route: str) -> str | None:
    #The first path segment of a route, with framework placeholders removed.

    path = urlparse(route).path if ("//" in route or route.startswith("http")) else route
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None
    first = segments[0]
    if re.search(r"[\[\]{}]|^:", first):
        return None
    return first.lower()


def blob_describes_another_page(blob: Any, identity: PageIdentity) -> bool:
    # True when a payload says it belongs to some other page.
    if not identity or not identity.path:
        return False
    expected = _route_shape(identity.path)
    if expected is None:
        return False

    shapes = {shape for route in _declared_routes(blob) if (shape := _route_shape(route))}
    if not shapes:
        return False
    return expected not in shapes
