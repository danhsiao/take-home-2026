## Channel3 Take Home Assignment

Structured product data extracted from raw PDP HTML, served over a small FastAPI layer
and browsed through a React storefront.

```
data/*.html  →  extraction pipeline  →  out/products.json  →  FastAPI  →  React
```

### Running it

Two steps. The first produces the catalogue, the second serves and browses it.

```bash
# 1. Extract. Writes out/products.json. Needs OPEN_ROUTER_API_KEY in .env. If you want to add more html files, just add it to the data folder. The run command will extract all products in the data folder. 
uv run python main.py

# 2. Run the API and the frontend together on http://localhost:5173
cd frontend && npm install && npm start
```

### Tests

```bash
uv run pytest          # backend: extraction, validation, golden and generality suites
cd frontend && npm test # frontend: variant selection
```

### System Architecture Write Up

To scale from 5 to 50 million+ products, I would separate crawling, extraction, and normalization into stateless workers backed by durable storage. It would store raw HTML, screenshots, and images in S3, persist normalized products and variants in a relational database, and maintain a separate search index over fields in the Product schema. Because raw HTML does not always contain the complete rendered PDP, I would selectively use Selenium or Playwright to capture rendered DOM and screenshots. I have experience with this from Expedia Group, where I built an internal tool using Selenium web drivers and LLM APIs to automate testing and documentation workflows. I would also introduce indexing to the most common searched results, and caching so that searches can stay under a second when scaling. To control inference cost, I would fingerprint each product’s normalized evidence and reuse cached results when the underlying PDP has not meaningfully changed rather than rerunning the LLM on every crawl. I would also cache popular searches and incrementally update the search index so common queries can stay under a second at scale. The current deterministic extraction, validation, and taxonomy retrieval scale well horizontally, while products.json, unconditional LLM calls, and assuming raw HTML always contains the full PDP do not. 

For agentic shopping apps, the main additional APIs I would focus on are the search, product/variant retrieval, and checkout endpoints. For checkout, using something like Stripe and Visa's TAP API would be beneficial for allowing AI agents to buy. The search API should support both NL or multimodal queries and structured filters, like “outdoor grills from Weber, less than $1000, with 4 burners,” “running sneakers under 300g, size 12,” or “this couch [image] but in green”. The product API would expose normalized attributes, variants, current price and availability, merchant URLs, and freshness so agents can reason over products without understanding each retailer’s website. Beyond the API, I would provide SDKs, bulk feeds, webhooks for price or inventory changes, and semantic-search or embedding endpoints. Developer tooling should also make extraction debuggable by showing the source evidence behind each normalized field. This keeps the merchant-specific complexity inside the platform while giving shopping agents a stable interface across millions of products.