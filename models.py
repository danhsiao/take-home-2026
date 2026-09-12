"""All the Pydantic models. The output contract, the internal pipeline types, and
the reliability types.

The flow is HTML to PageEvidence to DeterministicFacts plus ProductCandidate to
Product. Each step trusts less than the one before it.
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
    # One configuration of a product, like a size or a color.

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
    # One image, deduped across every size the page offers.

    url: str  # highest-resolution variant found for this asset
    asset_key: str  # identity used for dedup
    width: int | None = None  # largest declared width, when the page declares one


class VariantRecord(BaseModel):
    """One joined row of a page's variant graph, before we know the currency."""

    sku: str
    options: dict[str, str]
    image_urls: list[str] = []
    amount: float | None = None
    available: bool | None = None
    url: str | None = None


class OptionValue(BaseModel):
    # One choice inside a rendered option group. See `controls.py`.

    label: str
    selected: bool = False
    image_url: str | None = None
    url: str | None = None
    available: bool | None = None


class OptionGroup(BaseModel):
    #A set of choices the rendered page offers, like a colour picker.

    name: str | None = None
    values: list[OptionValue] = []
    # Distance from the product title's ancestor chain, when the group sits inside it.
    title_depth: int | None = None


class PageEvidence(BaseModel):
    # Everything the deterministic harvest found, before anything interprets it.

    url: str | None = None
    structured: list[dict[str, Any]] = []  # schema.org JSON-LD items, @graph flattened
    microdata: list[dict[str, Any]] = []  # itemscope/itemprop trees
    meta: dict[str, str] = {}  # og:*, twitter:*, <meta name=...>, canonical
    breadcrumbs: list[str] = []  # strongest available taxonomy signal
    json_subtrees: list[dict[str, Any]] = []  # pruned subtrees of embedded app state
    text_blocks: list[str] = []  # visible text, ranked by proximity to the title
    images: list[ImageAsset] = []
    videos: list[str] = []

    # The page's own variant graph, joined from embedded state. 
    variant_graph: list[VariantRecord] = []

    # Choice sets the rendered document declares, from the accessibility tree.
    option_groups: list[OptionGroup] = []

    # The identifiers this document asserts about itself, from `rel=canonical`,
    # `og:url` and schema.org identity fields.
    identity_ids: list[str] = []

    # Embedded payloads that declared they belong to a different page than this one
    stale_blobs: int = 0


class DeterministicFacts(BaseModel):
    # Facts a published standard stated: schema.org, OpenGraph, or microdata.


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
    #What the model returned. An interpretation, not a Product yet.

    name: str | None = None
    brand: str | None = None
    description: str | None = None
    key_features: list[str] = []
    colors: list[str] = []
    price: Price | None = None

    # Must be copied verbatim from the shortlist we supply in the prompt. 
    category: str | None = None

    # The model's escape hatch. 
    category_confident: bool = True

    # Two to four generic nouns for what this product *is* ("floor lamp", "pants").
    # Returned by the main call at no extra cost, and used only on the repair path to
    # re-run taxonomy retrieval.

    category_keywords: list[str] = []

    variants: list[Variant] = []


class ValidationIssue(BaseModel):
    """One validation finding."""

    field: str
    code: str  # machine-readable; doubles as the metric key
    message: str
    severity: Literal["error", "warning"] = "error"


class ExtractionResult(BaseModel):
    """What the pipeline returns. It never raises."""

    product: Product | None = None
    source: str | None = None  # which input this came from, for batch reporting
    errors: list[str] = []
    warnings: list[str] = []
    usage: dict[str, float] = {}  # tokens and extrapolated cost, from ai.py
