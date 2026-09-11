"""The AI stage: pruned evidence -> ProductCandidate.

Everything deterministic has already happened by the time this module runs. The model
is not the scraper; it is the interpreter. It is asked to do the things that genuinely
need judgement and cannot be derived from a specification:

  * decide which of several conflicting titles is the canonical product name
  * read variant dimensions out of an arbitrary state blob, including resolving
    foreign-key references between SKU, price, and media records
  * turn prose into discrete key features
  * pick a category from a shortlist we retrieved for it

It is NOT asked for image URLs, video URLs, GTINs, or a price that a structured offer
already stated. Those come from `extract.py` and are merged in by `pipeline.py`, where
deterministic facts take precedence. A model that never gets to state an image URL
cannot mangle one.

There is one call on the normal path. A second, cheap call fires only when category
selection fails validation - and it receives just the failing field, the validation
issues, and a freshly retrieved shortlist, never the page again.

PROMPT COMPLIANCE: the prompts below contain no site names, no domains, and no
examples drawn from `data/`. The only worked examples are invented ones illustrating
the shape of a variant dimension, which is a statement about the output format rather
than a hint about any page we were given.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel

import ai
import taxonomy
from models import DeterministicFacts, PageEvidence, ProductCandidate, ValidationIssue

logger = logging.getLogger(__name__)


class CategoryChoice(BaseModel):
    """Response schema for the repair call, which answers one question only.

    Kept as narrow as possible: a repair that could also rewrite the name or price
    would let a cheap model overwrite fields that were already validated.
    """

    category: str | None = None
    # The repair must be able to decline. Forcing a choice from a shortlist that does
    # not contain the right answer produces a category that *passes* validation while
    # being wrong - which is worse than no answer, because nothing downstream can
    # detect it. A declined repair fails the product explicitly instead.
    fits: bool = True

# The main call does the heavy interpretation and benefits from a stronger model.
PRIMARY_MODEL = "google/gemini-3-flash-preview"
# The repair call answers a single multiple-choice question and does not need one.
REPAIR_MODEL = "google/gemini-2.5-flash-lite"

# Ceiling on the evidence packet handed to the model. Beyond this, additional context
# reliably costs more than it adds, and cost per product is a first-class concern at
# the scale this is meant to run at.
MAX_PACKET_CHARS = 90_000


SYSTEM_PROMPT = """\
You extract structured product data from evidence harvested off a product detail page.

You will receive:
- STRUCTURED DATA: schema.org/OpenGraph values, whose meaning is defined by a spec.
- EMBEDDED DATA: pruned fragments of the page's application state. Arbitrary shape.
- TEXT: visible text from the page's main product region.
- ALREADY EXTRACTED: facts read deterministically from published standards.
- CATEGORY CANDIDATES: a shortlist retrieved from the Google Product Taxonomy.

Rules:

1. Use ONLY the supplied evidence. Never invent, infer, or fill a gap with a plausible
   value. If the evidence does not state something, return null or an empty list.
   Absence is a correct and useful answer.

2. ALREADY EXTRACTED values came from machine-readable standards and are more reliable
   than anything you will read in free text. Do not contradict them.

3. category MUST be copied verbatim, character for character, from CATEGORY
   CANDIDATES. Do not compose, adjust, or reformat a category path. If none of the
   candidates genuinely fits the product, set category to the closest one but set
   category_confident to false.

4. category_keywords: 2-4 short generic nouns naming what this product IS, ignoring
   brand and model ("floor lamp", "drill", "dress shirt"). Use the plainest,
   most common word for the thing, not the merchant's stylistic wording.

5. variants: one entry per discrete configuration the page actually offers.
   - options is an open map of dimension name to value. Name dimensions from the page's
     own vocabulary, e.g. {"Color": "...", "Size": "..."} or {"Voltage": "...",
     "Package": "..."}, whichever the evidence supports.
   - CRITICAL: do not generate combinations. If the page lists colors and sizes as two
     independent lists without saying which pairings exist, you must NOT emit every
     colour crossed with every size. Emit only configurations the evidence shows as
     real records.
   - Embedded data often references prices and images indirectly, by id. Resolve those
     references where the evidence lets you do so unambiguously; leave the field null
     where it does not.
   - price, sku, gtin and availability may be null. Many pages declare configurations
     without exposing per-configuration commercial data, and a configuration with no
     price is still a real configuration worth recording.
   - Only use image URLs copied exactly from the evidence. Never edit or reconstruct one.

6. description: the product's own description, cleaned of navigation and boilerplate.
   key_features: short factual bullets, each stating one attribute.
   colors: colour names offered for the product, if any.

7. Ignore anything belonging to a different product - recommendations, "customers also
   bought", recently viewed. Extract only the page's primary product.
"""


def _compact(value: Any, limit: int) -> str:
    """Serialise evidence compactly, truncating at a character budget.

    Separators are stripped of whitespace because pretty-printed JSON spends a
    meaningful share of its tokens on indentation, and the model does not need it.
    """
    try:
        text = json.dumps(value, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def build_packet(
    evidence: PageEvidence, facts: DeterministicFacts, candidates: list[str]
) -> str:
    """Assemble the evidence packet sent to the model.

    This function is the cost lever for the whole system. The raw pages average about
    138,000 tokens; the packet this produces averages roughly a tenth of that, and the
    difference is what makes per-product economics work at scale.

    Evidence is ordered by trustworthiness so that the most reliable material is
    nearest the instructions, and each section is separately budgeted so that one
    enormous state blob cannot crowd out the visible text that another page depends on.
    """
    sections: list[str] = []

    if facts:
        # Given first: these are the values the model must not contradict.
        sections.append(
            "ALREADY EXTRACTED (from published standards - authoritative):\n"
            + _compact(facts.model_dump(exclude_none=True, exclude_defaults=True), 4_000)
        )

    if evidence.structured:
        sections.append("STRUCTURED DATA:\n" + _compact(evidence.structured, 40_000))

    if evidence.meta:
        # Only the standards-defined keys; the rest is analytics and viewport config.
        interesting = {
            key: value
            for key, value in evidence.meta.items()
            if key.startswith(("og:", "twitter:", "product:")) or key in ("title", "description")
        }
        if interesting:
            sections.append("PAGE METADATA:\n" + _compact(interesting, 2_000))

    if evidence.breadcrumbs:
        sections.append("BREADCRUMBS:\n" + " > ".join(evidence.breadcrumbs))

    if evidence.json_subtrees:
        sections.append("EMBEDDED DATA:\n" + _compact(evidence.json_subtrees, 40_000))

    if evidence.text_blocks:
        sections.append("TEXT:\n" + "\n".join(evidence.text_blocks)[:20_000])

    sections.append(
        "CATEGORY CANDIDATES (copy one verbatim):\n" + "\n".join(candidates)
    )

    return "\n\n".join(sections)[:MAX_PACKET_CHARS]


async def run(
    evidence: PageEvidence, facts: DeterministicFacts, candidates: list[str]
) -> ProductCandidate | None:
    """Interpret the evidence into a ProductCandidate. One LLM call.

    Args:
        evidence: the deterministic harvest.
        facts: high-confidence values already read from published standards.
        candidates: the retrieved taxonomy shortlist.

    Returns:
        The model's interpretation, or None if the call failed outright. A None here
        is a transport failure, not a data problem - a page the model could not read
        still returns a populated candidate with null fields.
    """
    packet = build_packet(evidence, facts, candidates)

    try:
        return await ai.responses(
            PRIMARY_MODEL,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": packet},
            ],
            text_format=ProductCandidate,
        )
    except Exception as error:  # noqa: BLE001 - transport failures must not crash a batch
        logger.error("Normalization call failed: %s", error)
        return None


REPAIR_PROMPT = """\
A product was assigned a category that failed validation.

Choose the single best category for the product described.

Rules:
- category MUST be copied verbatim, character for character, from the candidate list.
  These are the only permitted values. Do not compose or adjust a path.
- If none of the candidates genuinely describes this product, set fits to false and
  leave category null. Do not settle for a category that is merely in the right
  general area. A confidently wrong category is worse than no answer, because nothing
  downstream can tell that it is wrong.
"""


async def repair_category(
    candidate: ProductCandidate, issues: list[ValidationIssue]
) -> str | None:
    """Re-select a category after the first attempt failed validation. One cheap call.

    Repair is deliberately limited to `category`, because it is the only field where
    asking again is the right response. Category has a closed vocabulary and a hard
    validator, so a failure is usually a formatting or shortlist problem that a second
    look genuinely fixes.

    Nothing else is repaired. If the model returned an image URL that is not in the
    evidence, the correct action is to drop that URL, not to ask for another guess -
    re-asking would just resample the same distribution that produced the error.

    The shortlist is rebuilt from the model's own `category_keywords` rather than
    merely widened. When the first retrieval missed because the page's vocabulary
    differs from the taxonomy's, widening the same query returns more of the same wrong
    branch; re-querying with the model's plain-language nouns actually reaches the
    right one.
    """
    query = " ".join(candidate.category_keywords) if candidate.category_keywords else ""
    if not query:
        # No keywords to re-query with, so fall back to widening the original signal.
        query = " ".join(filter(None, [candidate.name, candidate.description]))[:200]

    retried = taxonomy.retrieve(query, taxonomy.WIDE_CANDIDATE_COUNT)
    if not retried:
        return None

    summary = {
        "name": candidate.name,
        "description": (candidate.description or "")[:300],
        "keywords": candidate.category_keywords,
        "rejected": candidate.category,
        "reasons": [issue.message for issue in issues],
    }

    try:
        result = await ai.responses(
            REPAIR_MODEL,
            [
                {"role": "system", "content": REPAIR_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT:\n{_compact(summary, 2_000)}\n\n"
                        f"CANDIDATES (copy one verbatim):\n" + "\n".join(retried)
                    ),
                },
            ],
            text_format=CategoryChoice,
        )
        if result is None or not result.fits:
            return None
        # Verify against the taxonomy here rather than trusting the reply: the repair
        # runs on a cheaper model, and "copy this verbatim" is exactly the instruction
        # a small model is most likely to approximate.
        if result.category and taxonomy.is_valid(result.category):
            return result.category
        return None
    except Exception as error:  # noqa: BLE001
        logger.error("Category repair call failed: %s", error)
        return None
