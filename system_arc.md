# System Architecture

Raw HTML goes through one deterministic wrangle that looks everywhere the page might have stated a fact: schema.org, microdata, OpenGraph, embedded app JSON, the rendered accessibility tree, and visible text. Anything a published standard already defined, like price, brand, GTIN, images and variants, is read straight out of that evidence, and the model is never asked about it. A lexical search over the Google Product Taxonomy narrows about 5,600 categories down to a shortlist of 30. One LLM call then
does interpretation only: the name, description, features, what the variant dimensions mean, and picking one category from that shortlist. A pure validator checks every claimback against the evidence, and anything it cannot find is dropped instead of trusted.
Assembly merges the two halves, and the deterministic facts always win. The goal of this architecture is to reduce the LLM cost and maintain reliability. When scaling, the LLM costs can pile up very easily, so reducing the LLM cost is crucial. 

```
                            raw HTML
                               |
              +----------------+----------------+
              |     DETERMINISTIC HARVEST       |
              |  schema.org / microdata / OG    |
              |  embedded JSON / ARIA / text    |
              +----------------+----------------+
                               |
                         PageEvidence
                               |
        +----------------------+----------------------+
        |                                             |
  DeterministicFacts                          taxonomy search
  price, brand, GTIN,                         5,600 categories
  images, variants                                   |
        |                                     30 candidates
        |                                             |
        |                                     +-------+-------+
        |                                     |  ONE LLM CALL |
        |                                     |  name, desc,  |
        |                                     |  features,    |
        |                                     |  category     |
        |                                     +-------+-------+
        |                                             |
        |                                     ProductCandidate
        |                                             |
        |                                     +-------+-------+
        |                                     |   VALIDATE    |
        |                                     | drop anything |
        |                                     | not in the    |
        |                                     | evidence      |
        |                                     +-------+-------+
        |                                             |
        +----------------------+----------------------+
                               |
                          ASSEMBLE
                    (facts beat model output)
                               |
                            Product
```

Downstream, the batch CLI writes `out/products.json`, FastAPI serves it, and a React
storefront renders the grid and the product page. The goal of the frontend is to look like if you were shopping. If I had more time, I would implement a search feature, as mentioned in @README.md 

```
data/*.html -> main.py -> out/products.json -> api.py -> frontend/
```
