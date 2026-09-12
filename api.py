"""A small HTTP API over the extracted catalogue."""

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

    A missing file is not an error. The API still starts and the shop shows an empty
    state rather than failing.
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
    """Cut a product down to what a catalogue grid needs."""
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
    """One product in full, with every variant and image."""
    product = _BY_ID.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"No product with id {product_id}")
    return product
