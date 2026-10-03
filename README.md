# Finsight

Finsight is a single-turn financial research agent over a local Markdown corpus.
It combines vector search and Unicode BM25, retries searches when evidence is
insufficient, and answers with citations to the exact context shown to the model.
The same retrieval functions are exposed through MCP.

This is a **local proof of concept for experimentation and evaluation**, scoped to
macOS/Linux with a local POSIX filesystem and a small corpus held in memory. It
uses Ollama by default; OpenAI, Anthropic and OpenAI-compatible chat endpoints are
optional. Local inference can run offline after dependencies and models are downloaded.

## Release boundary

The core POC roadmap is implemented: dated provenance and filtering, optional
reranking and semantic support checks, runtime controls and traces, resumable
category evaluation, paired report comparison, incremental vector reuse, and
leased generation cleanup. Automated tests exercise these paths with model
doubles, local Chroma and MCP stdio.

**Live-model quality, latency, cost and hardware testing remain explicitly deferred.**
No measured answer-quality improvement or production-readiness claim is made.
The sample data comprises **11 sourced historical summaries and 40 authored cases**,
with 20 development and 20 held-out cases. Both partitions are public and share the
same corpus; they are **not an independent benchmark**. See [dataset provenance
and limitations](docs/dataset.md).

## Quickstart

Requires `uv` and a running Ollama service. Run commands from the repository root;
a wheel installation alone does not include the sample corpus.

```bash
git clone https://github.com/wif1c0ldspot/finsight.git
cd finsight
uv sync --frozen

ollama pull qwen3:8b
ollama pull nomic-embed-text
uv run finsight doctor
uv run finsight ingest
uv run finsight ask "Where and when was Airwallex founded?"
uv run finsight mcp-demo "What valuation did Stripe announce in February 2024?"
uv run finsight evaluate --split development --output reports/baseline.json
```

`doctor` checks configuration, paths and local service reachability. It does not
certify model availability or answer accuracy. For code verification without live
model calls:

```bash
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy src
uv run --frozen --no-sync pytest -q
```

## Questions, filters and optional quality checks

```bash
uv run finsight ask "What valuation did Stripe announce in February 2024?" \
  --doc stripe --published-after 2024-01-01 --published-before 2024-12-31

uv run finsight ask "Where and when was Airwallex founded?" \
  --timeout 120 --max-model-calls 12 --trace reports/airwallex-trace.json

uv run finsight ask "What valuation did Stripe announce in February 2024?" \
  --rerank --verify-claims
```

`--doc` and `--source-url` accept repeated values. Filters combine fields with AND
and values within each list with OR. Source URLs match exactly; publication-date
bounds are inclusive and exclude undated documents. Filtering applies to both
vector and sparse retrieval before fusion.

Reranking and semantic verification are **disabled by default**. The optional LLM
reranker orders a bounded candidate set; invalid rankings retain the RRF order.
Semantic verification checks each answer segment against its cited evidence and
requires complete verdict coverage. If enabled checks still fail after bounded
retries, the final answer abstains. These are fallible model judgments, not proven
accuracy improvements or a general prompt-injection defense.

Without semantic verification, grounding checks citation existence, not factual
entailment. Invalid citations trigger bounded retries or a visible warning. When
no usable evidence fits, the agent abstains without an answer-generation call.

## Configuration and runtime limits

Copy [`.env.example`](.env.example) to `.env` for local configuration. Settings use
`FINSIGHT_<FIELD>` names; `.env` and `reports/` are Git-ignored. Defaults are defined
in [`config.py`](src/finsight/config.py).

Optional controls include per-run timeout, logical model-call count, input/output
budgets, per-call output caps and configured cost budgets. The numbers in the CLI
example are illustrative controls, not measured hardware recommendations.

- Deadlines and cancellation are **cooperative boundary checks**. They cannot
  preempt an in-flight synchronous provider call; request timeouts still matter.
  Internal SDK retries are not separately counted as logical model calls.
- `FINSIGHT_CHUNK_MAX_TOKENS` and `FINSIGHT_CONTEXT_MAX_TOKENS` add content bounds
  alongside character limits. Without an injected exact counter, token accounting
  uses UTF-8 content bytes, conservatively bounding byte-based tokenization of that
  content. It excludes provider framing/tool overhead and is not a universal
  tokenizer or a guaranteed model-window measurement.
- Run accounting tracks provider-reported usage separately from fallback budget
  counts. Costs require operator-supplied input/output rates; no prices are built
  in. Agent cost estimates exclude embeddings, evaluation-judge calls and hidden
  provider retries; they are not a billing cap.
- Traces contain statuses, node/model timings, counts and configured cost estimates,
  not prompts, source text, responses, credentials or raw endpoint URLs. Evaluation
  reports additionally contain questions and answers and may contain private data.

The Python `RunContext` supports a caller-supplied cancellation event and token
counter; each invocation requires a fresh context. See [architecture](docs/architecture.md).

### Hosted or compatible providers

Install the appropriate optional dependencies:

```bash
uv sync --frozen --extra all
```

Set `FINSIGHT_LLM_PROVIDER`, `FINSIGHT_LLM_MODEL` and credentials for your chosen
provider. Embeddings support Ollama, OpenAI and OpenAI-compatible endpoints.
For a local compatible chat gateway while retaining Ollama embeddings:

```bash
export FINSIGHT_LLM_PROVIDER=openai_compatible
export FINSIGHT_LLM_BASE_URL=http://localhost:8000/v1
export FINSIGHT_LLM_MODEL=your-served-model
# The client needs a nonempty key. This placeholder is for an unauthenticated
# local gateway only; use your own credential for an authenticated endpoint.
export FINSIGHT_LLM_API_KEY=local-placeholder
```

Compatible providers require an explicit base URL. A default same-provider judge
inherits the answering endpoint. Set a separate `FINSIGHT_JUDGE_MODEL` and, when
needed, its provider/endpoint/key to reduce self-grading bias. Independent judges
can still be wrong.

## Index updates, upgrades and cleanup

`ingest` always publishes a new immutable generation atomically. When the previous
index is compatible, it reuses vectors for unchanged chunk text and embeds changed
text. Changed provenance is still written into the new registry. Removed documents
are absent from the new generation. Failed builds preserve the published index.

**Manifest v4 requires a one-time rebuild for v1–3 indexes:**

```bash
uv run finsight ingest
```

Provider, model, endpoint hash and optional `FINSIGHT_EMBED_REVISION` bind vectors
to their embedding identity. The revision is supplied by the operator, not verified
against model weights. When an unchanged model alias serves different weights,
update its revision and ingest, or explicitly bypass reuse:

```bash
uv run finsight ingest --full
```

Readers hold POSIX leases on their generation. Running MCP search reloads a newly
published generation on its next request. Cleanup protects the published generation,
active readers/builders, recently created generations and the configured retained
set. It only deletes owned artifacts; legacy/unmarked artifacts are left alone.

```bash
# Preview first; defaults retain two newest owned generations and a 24-hour age floor.
uv run finsight index-cleanup --keep 2 --min-age-hours 24
uv run finsight index-cleanup --keep 2 --min-age-hours 24 --apply
```

These locks and atomic publication assumptions target a **local POSIX filesystem**,
not Windows, network filesystems or distributed storage. Cleanup is explicit, not
automatically scheduled; retention and age limits can leave substantial data on disk.

## Evaluation, resume and comparison

Cases support `category` and `split` (`development` or `held_out`). Categories cover
single facts, numbers, multiple documents, time, misleading premises, unavailable
facts and adversarial requests. Omitted metadata defaults to `general` and
`development` for custom legacy cases. Repeat `--category` to select several groups.

```bash
uv run finsight evaluate --split development --category numerical \
  --output reports/numerical-baseline.json

# Resume using the SAME split/category flags; completed cases are skipped.
uv run finsight evaluate --resume reports/numerical-baseline.json \
  --split development --category numerical

# Explicitly rerun recorded operational or malformed-judge errors.
uv run finsight evaluate --resume reports/numerical-baseline.json --retry-failures \
  --split development --category numerical

# After changing model/configuration, start a NEW report, not a resume.
uv run finsight evaluate --split development --category numerical \
  --output reports/numerical-candidate.json
uv run finsight compare reports/numerical-baseline.json reports/numerical-candidate.json
```

Reports are atomically checkpointed before model setup and after every fully processed
case. An interrupted case is rerun; earlier completed cases survive. Resume writes
back to its source unless a different `--output` is provided. It refuses changed
golden/model/config/index fingerprints, runtime versions or selections. Recorded
failures are retained unless `--retry-failures` is set; a low quality score alone
is not a retryable execution error. Without `--output` or `--resume`, no checkpoint
is saved.

Malformed cases fail before model setup. Agent failures are isolated per case and
judge failures per metric. Operational failures yield exit code 1 after saving;
an interrupted CLI run yields 130. Malformed judge verdicts remain unscored.
Zero exit status is not a quality pass: inspect completion, scored counts and errors.

Metrics include recall@k, MRR, document-level nDCG, citation validity,
model-judged faithfulness/correctness, and semantic abstention on negative cases.
Duplicate chunks earn nDCG relevance once. Missing labels and unavailable verdicts
remain unscored; empty answers fail explicit abstention and reference correctness.

Comparison requires unchanged golden and index-content hashes and the same selected
case IDs. Each metric delta uses **only IDs scored in both reports**, with paired
counts and IDs shown separately from each report's independent averages and coverage.
Missing pairs produce `null` deltas. Category results, errors and model/configuration
changes remain visible. These descriptive differences imply neither statistical
significance nor a release-quality threshold.

## MCP tools

The shared tool layer exposes `search_documents` (including metadata filters),
`list_documents`, `get_document` (text), and `get_document_record` (text plus recorded
provenance). Listing/getting reads current local corpus files; search reads the
published index, so ingest after editing the corpus.

```bash
uv run python -m finsight.mcp.server          # stdio
uv run python -m finsight.mcp.server --http   # local streamable HTTP demo
```

Graph retrieval uses these functions in-process by default, not a remote tool call.
The stdio client forwards supported configuration/credentials and preserves tool
errors. HTTP has no application authentication, tenant isolation or service quotas.
Remote deployment requires a separate security and operational design.

## Remaining scope

The registry and BM25 index are memory-resident; chunking, Unicode tokenization and
PII redaction remain heuristics. Semantic checks are fallible and cannot replace
human label review. Embedding revisions need operator discipline, especially with
mutable aliases. Human adjudication and independent data are still needed before
benchmark claims. Live model/hardware evaluation remains deferred by request.

Authentication, isolation, TLS, quotas and backup/restore are gates for a remote
service. UI, streaming, conversation memory and calculator tools remain optional
extensions. See [architecture](docs/architecture.md), [dataset](docs/dataset.md), and
[completed roadmap and remaining gates](docs/learning-plan.md).
