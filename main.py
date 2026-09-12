"""Batch CLI. Reads the HTML in `data/` and writes `out/products.json`."""

import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pipeline
from models import ExtractionResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "out"
OUTPUT_FILE = OUTPUT_DIR / "products.json"

# The source URLs live in data/README.md rather than in the HTML filenames. We read
# the canonical URL out of each page instead, so nothing here depends on a filename.


async def process(path: Path) -> ExtractionResult:
    """Run one page through the pipeline. Never raises."""
    html = path.read_text(encoding="utf-8", errors="replace")
    result = await pipeline.run_html(html)
    result.source = result.source or path.name
    return result


async def main(paths: list[Path]) -> int:
    """Process pages at the same time and report. Returns a shell exit code."""
    if not paths:
        logger.error("No HTML files found in %s", DATA_DIR)
        return 1

    # Pages are independent, so they run concurrently. This is also the shape the
    # production design assumes: the unit of work is one page, with no shared state.
    results = await asyncio.gather(*(process(path) for path in paths))

    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = [
        {
            "id": pipeline.product_id(result.source, result.product.name),
            "source": result.source,
            **result.product.model_dump(),
        }
        for result in results
        if result.product
    ]
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _report(results)
    print(f"\nWrote {len(payload)} product(s) to {OUTPUT_FILE}")

    # A page that could not be extracted is a real failure and should be visible in
    # the exit code, not buried in the log.
    return 0 if all(result.product for result in results) else 1


def _report(results: list[ExtractionResult]) -> None:
    """Print what happened per page, plus the totals for each issue type."""
    print("\n" + "=" * 78)
    print(f"{'page':<20}{'status':<10}{'variants':>9}{'images':>8}  category")
    print("-" * 78)

    for result in results:
        source = (result.source or "?").rsplit("/", 1)[-1][:19]
        if result.product:
            product = result.product
            print(
                f"{source:<20}{'ok':<10}{len(product.variants):>9}"
                f"{len(product.image_urls):>8}  {product.category.name[:34]}"
            )
        else:
            print(f"{source:<20}{'FAILED':<10}{'-':>9}{'-':>8}  {result.errors[:1]}")

    issues = Counter(
        message.split(":", 1)[0]
        for result in results
        for message in result.warnings + result.errors
    )
    if issues:
        print("\nValidation issues by code:")
        for code, count in issues.most_common():
            print(f"  {count:>4}  {code}")

    print(
        "\nPer-product token usage and cost extrapolation are logged by ai.py "
        "(set logging to INFO to see them)."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Quieten the HTTP client, which would otherwise log a line per request.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    arguments = [Path(argument) for argument in sys.argv[1:]]
    targets = arguments or sorted(DATA_DIR.glob("*.html"))
    sys.exit(asyncio.run(main(targets)))
