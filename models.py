"""Pydantic models for the PDP extraction pipeline.

This module holds every model in the system, per the take-home instruction that
additional models live here for legibility. They fall into three groups:

1. The graded output contract - `Category`, `Price`, `Product`. Unchanged from the
   supplied schema except that `Product.variants` is now typed `list[Variant]`.
2. Internal pipeline representations - `ImageAsset`, `PageEvidence`,
   `DeterministicFacts`, `ProductCandidate`. These exist so that deterministic
   extraction, LLM interpretation, and final assembly stay separable.
3. Reliability types - `ValidationIssue`, `ExtractionResult`.

The pipeline direction is:

    raw HTML -> PageEvidence -> DeterministicFacts + ProductCandidate -> Product

Each arrow narrows what is trusted. `PageEvidence` is everything we found,
`DeterministicFacts` is only what a published standard told us, `ProductCandidate`
is what a model inferred, and `Product` is what survived validation.
"""

from typing import Any, Literal
from pathlib import Path
from pydantic import BaseModel, field_validator

# Load categories once at module level
CATEGORIES_FILE = Path(__file__).parent / "categories.txt"
VALID_CATEGORIES = set()
if CATEGORIES_FILE.exists():
    with open(CATEGORIES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                VALID_CATEGORIES.add(line)


class Category(BaseModel):
    # A category from Google's Product Taxonomy
    # https://www.google.com/basepages/producttype/taxonomy.en-US.txt
    name: str

    @field_validator("name")
    @classmethod
    def validate_name_exists(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"Category '{v}' is not a valid category in categories.txt")
        return v


class Price(BaseModel):
    price: float
    currency: str
    # If a product is on sale, this is the original price
    compare_at_price: float | None = None


class Variant(BaseModel):
    """One discrete, purchasable-or-declared configuration of a product.

    `options` is an open dict rather than named fields (size/color/...) because the
    dimensions a PDP exposes are category-dependent and unbounded: apparel varies by
    size and color, laptops by memory and storage, furniture by fabric and
    configuration, tools by voltage and kit contents. Naming the dimensions in the
    schema would force category-specific logic, which the assignment forbids. The
    LLM names the dimension, we only guarantee it is a flat string->string map.

    Every commercial field is nullable *by design*. Some PDPs declare the set of
    configurations (e.g. a color swatch strip) without exposing a per-configuration
    price, SKU, or availability anywhere in the HTML. Requiring those fields would
    force us to drop such variants entirely and silently tank variant recall, so we
    keep the configuration and leave the commercial data null.

    Known limitation: a flat `dict[str, str]` cannot express ordering (size "5.5"
    sorts before "10" lexically, which is wrong) and cannot carry a color's swatch
    hex. Both are accepted trade-offs for category-agnosticism.
    """

    options: dict[str, str]
    sku: str | None = None
    gtin: str | None = None
    price: Price | None = None
    available: bool | None = None
    image_urls: list[str] = []
    url: str | None = None


# This is the final product schema that you need to output.
# You may add additional models as needed.
class Product(BaseModel):
    name: str
    price: Price
    description: str
    key_features: list[str]
    image_urls: list[str]
    video_url: str | None = None
    category: Category
    brand: str
    colors: list[str]
    variants: list[Variant]


class ImageAsset(BaseModel):
    """One logical image, deduped across every resolution the page offers.

    A single product photo typically appears many times in one PDP: in a `srcset`
    at six widths, again as an OpenGraph image, again inside embedded app state.
    The assignment asks for *full resolution* images, so we group every URL that
    points at the same underlying asset and keep only the largest.

    `asset_key` is that grouping identity - the URL with generic resolution tokens
    stripped. See `images.py` for how it is derived.
    """

    url: str  # highest-resolution variant found for this asset
    asset_key: str  # identity used for dedup
    width: int | None = None  # largest declared width, when the page declares one


class PageEvidence(BaseModel):
    """Everything deterministic extraction found in the HTML, before interpretation.

    This is intentionally a *harvest*, not an analysis. Fields are grouped by the
    standard or mechanism they came from, never by what we think they mean. There is
    no `titles` or `brands` list here: on a page with no structured data, a generic
    "find the brand" heuristic produces noise, and handing noise to a model labelled
    as evidence is worse than handing it raw text.

    Everything the model is later allowed to claim must be traceable back into this
    object - that is what `validate.py` checks.
    """

    url: str | None = None
    structured: list[dict[str, Any]] = []  # schema.org JSON-LD items, @graph flattened
    microdata: list[dict[str, Any]] = []  # itemscope/itemprop trees
    meta: dict[str, str] = {}  # og:*, twitter:*, <meta name=...>, canonical
    breadcrumbs: list[str] = []  # strongest available taxonomy signal
    json_subtrees: list[dict[str, Any]] = []  # pruned subtrees of embedded app state
    text_blocks: list[str] = []  # visible text, ranked by proximity to the title
    images: list[ImageAsset] = []
    videos: list[str] = []


class DeterministicFacts(BaseModel):
    """Facts asserted by a published standard (schema.org, OpenGraph, microdata).

    These are high-confidence because their semantics come from a spec, not from a
    heuristic: when JSON-LD says `offers.price`, that is the price, full stop.

    The pipeline gives these strict precedence over anything the LLM returns. That
    single rule is the main hallucination guardrail - a model cannot overwrite a
    price that the page stated in machine-readable form, it can only fill gaps.
    """

    name: str | None = None
    brand: str | None = None
    description: str | None = None
    price: Price | None = None
    gtin: str | None = None
    sku: str | None = None
    image_urls: list[str] = []
    video_url: str | None = None
    variants: list[Variant] = []


class ProductCandidate(BaseModel):
    """The LLM's output: an interpretation, not yet a Product.

    Every field is optional. This is deliberate - a permissive intermediate lets a
    partial page produce a partial answer that we can reject cleanly, instead of
    pressuring the model to invent a value just to satisfy a required field. "I
    could not find this" is a valid, useful answer; a hallucinated price is not.
    """

    name: str | None = None
    brand: str | None = None
    description: str | None = None
    key_features: list[str] = []
    colors: list[str] = []
    price: Price | None = None

    # Must be copied verbatim from the shortlist we supply in the prompt. We do not
    # let the model free-form a category, because `Category` requires exact
    # membership in a 5,596-entry taxonomy and near-misses fail validation.
    category: str | None = None

    # The model's escape hatch. If none of the retrieved candidates fit, it says so
    # here rather than picking the least-bad option. This is how we tell a bad
    # *retrieval* apart from a bad *choice* - without it, the two are
    # indistinguishable and neither can be fixed.
    category_confident: bool = True

    # Two to four generic nouns for what this product *is* ("floor lamp", "pants").
    # Returned by the main call at no extra cost, and used only on the repair path to
    # re-run taxonomy retrieval.
    #
    # This exists because lexical retrieval cannot cross a synonym gap: a page saying
    # "Trousers" will never lexically match the taxonomy entry "Pants", so the correct
    # category is absent from the shortlist entirely and no amount of widening finds
    # it. Rather than hand-maintaining a synonym table - which would drift toward
    # being tuned to whichever pages we happened to test - we let the model supply the
    # vocabulary, since knowing that trousers are pants is exactly what a language
    # model is for.
    category_keywords: list[str] = []

    variants: list[Variant] = []


class ValidationIssue(BaseModel):
    """One structured validation finding.

    Structured rather than a bare exception or `assert` so that issues can be
    counted, aggregated into metrics, and routed by severity - a warning annotates
    the result, an error drops a value or fails the extraction.
    """

    field: str
    code: str  # machine-readable; doubles as the metric key
    message: str
    severity: Literal["error", "warning"] = "error"


class ExtractionResult(BaseModel):
    """The pipeline's return type. Never raises to the caller.

    `product` is None when a required field could not be resolved from the page.
    Returning an explicit failure is the point: several real PDPs do not expose a
    machine-readable price, and the correct behaviour there is to say so loudly
    rather than let a model guess a number that looks plausible.
    """

    product: Product | None = None
    source: str | None = None  # which input this came from, for batch reporting
    errors: list[str] = []
    warnings: list[str] = []
    usage: dict[str, float] = {}  # tokens and extrapolated cost, from ai.py
