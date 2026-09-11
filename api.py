"""HTTP API over the extracted catalogue.

Serves the products produced by `main.py` from `out/products.json`. There is no
database, per the assignment - the snapshot is loaded once at startup and held in
memory, which is adequate for a catalogue of this size and keeps the deployment story
to a single process.

Exactly two endpoints, one per page the frontend needs: a catalogue grid and a product
detail page. Filtering, sorting, faceting, and a live-extraction endpoint are all
deliberately absent. The assignment asks for two pages and then says not to build
features it did not ask for, and none of those are needed to render either page.

Run with:
    uv run uvicorn api:app --reload
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

PRODUCTS_FILE = Path(__file__).parent / "out" / "products.json"

app = FastAPI(
    title="Channel3 Take-Home API",
    description="Structured product data extracted from raw PDP HTML.",
)

# The frontend runs on a different port in development. Permissive because this is a
# local demo with no authentication and no private data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at import. Re-running ingestion requires an API restart, which is the
# right trade-off for a read-only snapshot that changes only when the batch job runs.
_CATALOGUE: list[dict[str, Any]] = []
_BY_ID: dict[str, dict[str, Any]] = {}


def _load() -> None:
    """Read the extraction snapshot into memory.

    A missing file is not fatal: the API starts and reports an empty catalogue rather
    than crashing, so the frontend can be developed before ingestion has been run.
    """
    global _CATALOGUE, _BY_ID

    if not PRODUCTS_FILE.exists():
        logger.warning(
            "No snapshot at %s - run `uv run python main.py` first.", PRODUCTS_FILE
        )
        _CATALOGUE, _BY_ID = [], {}
        return

    _CATALOGUE = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
    _BY_ID = {product["id"]: product for product in _CATALOGUE}
    logger.info("Loaded %d products from %s", len(_CATALOGUE), PRODUCTS_FILE)


_load()


def _summarise(product: dict[str, Any]) -> dict[str, Any]:
    """Reduce a product to what a catalogue grid needs.

    The grid omits the description, key features, and full variant list. On a real
    catalogue that difference dominates response size, and the detail endpoint is one
    request away.
    """
    return {
        "id": product["id"],
        "name": product["name"],
        "brand": product["brand"],
        "price": product["price"],
        "category": product["category"]["name"],
        # Image ordering is preserved from the page, with schema.org and OpenGraph
        # nominations first, so the first image is the merchant's own hero shot.
        "image_url": product["image_urls"][0] if product["image_urls"] else None,
        "colors": product["colors"],
        "variant_count": len(product["variants"]),
    }


@app.get("/api/products")
def list_products() -> dict[str, Any]:
    """The catalogue grid: every product, in summary form."""
    return {
        "items": [_summarise(product) for product in _CATALOGUE],
        "total": len(_CATALOGUE),
    }


@app.get("/api/products/{product_id}")
def get_product(product_id: str) -> dict[str, Any]:
    """One product in full, including every variant and image."""
    product = _BY_ID.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"No product with id {product_id}")
    return product
