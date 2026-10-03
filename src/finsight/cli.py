"""Command-line interface for Finsight.

Commands wrap their work in a single error boundary. Previously an unreachable
model service produced a 60-frame traceback ending in ``ConnectionError``; now it
produces one legible line naming the likely cause and the check to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finsight.config import Settings, get_settings
from finsight.eval.harness import (
    EvalSplit,
    EvaluationCheckpointError,
    run_eval,
    summarize,
)
from finsight.graph.builder import build_agent
from finsight.guardrails.validation import (
    GroundingAssessment,
    assess_grounding,
    validate_query,
)
from finsight.observability.tracing import MetricsCollector, timed, write_trace
from finsight.rag.index import IndexIntegrityError, IndexMissingError
from finsight.rag.ingest import BuildStats, build_index
from finsight.rag.lifecycle import cleanup_generations
from finsight.rag.models import RetrievalFilter
from finsight.runtime import RunLimitError
from finsight.structured import StructuredOutputError

app = typer.Typer(help="Finsight -- financial research agent over a local corpus.")
console = Console()

#: Errors we expect from a misconfigured environment, mapped to a fix.
_ERROR_HINTS: list[tuple[type[BaseException], str]] = [
    (
        ConnectionError,
        "Could not reach the model service. Check that it is running and that the "
        "configured base URL is correct (see `finsight doctor`).",
    ),
    (
        IndexMissingError,
        "No index found. Build one first: finsight ingest",
    ),
    (
        IndexIntegrityError,
        "The index is inconsistent. Rebuild it: finsight ingest",
    ),
    (
        FileNotFoundError,
        "A required file is missing. Check the configured paths with `finsight doctor`.",
    ),
    (
        StructuredOutputError,
        "A model call did not return valid structured output. Re-run, or use a model "
        "with tool-calling support.",
    ),
]


def _fail(exc: Exception) -> None:
    """Print a legible error instead of a traceback, then exit non-zero."""
    for exc_type, hint in _ERROR_HINTS:
        if isinstance(exc, exc_type):
            console.print(
                Panel(
                    f"{type(exc).__name__}\n\n[bold]What to do:[/bold] {hint}",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1) from exc
    console.print(
        Panel(
            str(exc) if isinstance(exc, RunLimitError)
            else f"{type(exc).__name__}: check inputs, configuration and service availability.",
            title="Error", border_style="red",
        )
    )
    raise typer.Exit(code=1) from exc


def _endpoint_for_display(value: str | None) -> str:
    """Keep endpoint diagnostics useful without exposing URL auth/query secrets."""
    if value is None:
        return "not configured"
    try:
        parts = urlsplit(value)
        if not parts.scheme or not parts.netloc:
            return "invalid URL"
        return urlunsplit(parts._replace(
            netloc=parts.netloc.rsplit("@", 1)[-1],
            query="<redacted>" if parts.query else "", fragment="",
        ))
    except ValueError:
        return "invalid URL"


def _describe(settings: Settings) -> Table:
    table = Table(title="finsight configuration", show_header=False)
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("llm", f"{settings.llm_provider} / {settings.llm_model}")
    table.add_row("llm base url", _endpoint_for_display(settings.resolved_llm_base_url()))
    table.add_row("embeddings", f"{settings.embed_provider} / {settings.embed_model}")
    table.add_row("embed base url", _endpoint_for_display(settings.resolved_embed_base_url()))
    table.add_row(
        "judge",
        (
            f"{settings.judge_model} (separate)"
            if not settings.judge_is_same_model
            else "same as llm (self-graded)"
        ),
    )
    table.add_row("collection", settings.collection_name)
    table.add_row("corpus", str(settings.corpus_dir))
    table.add_row("index", str(settings.index_dir))
    table.add_row("mcp tools", "enabled" if settings.use_mcp_tools else "disabled")
    table.add_row("grounding", "enforced" if settings.enforce_grounding else "advisory")
    return table


@app.command()
def doctor() -> None:
    """Show the resolved configuration and check that paths and services exist."""
    try:
        settings = get_settings()
    except Exception as exc:  # pydantic validation errors
        _fail(exc)
        return

    console.print(_describe(settings))

    checks = Table(title="checks", show_header=True)
    checks.add_column("Check")
    checks.add_column("Status")

    corpus_files = (
        sorted(p.name for p in settings.corpus_dir.glob("*.md"))
        if settings.corpus_dir.exists()
        else []
    )
    checks.add_row("corpus directory", "ok" if corpus_files else "missing or empty")
    checks.add_row("corpus documents", f"{len(corpus_files)} file(s)")

    try:
        from finsight.rag.index import read_manifest

        manifest = read_manifest(settings.index_dir)
        checks.add_row(
            "index manifest",
            f"ok ({manifest.chunk_count} chunks, {manifest.created_at})",
        )
        checks.add_row("index embedding model", manifest.embed_model)
    except (IndexMissingError, IndexIntegrityError) as exc:
        checks.add_row("index manifest", f"not usable: {exc}")

    # Chat and embeddings can run on different services. Probe each configured
    # Ollama endpoint independently; hosted checks would require authentication.
    for label, provider, base_url in (
        ("chat endpoint", settings.llm_provider, settings.resolved_llm_base_url()),
        ("embedding endpoint", settings.embed_provider, settings.resolved_embed_base_url()),
    ):
        if provider != "ollama":
            checks.add_row(label, f"not checked ({provider}; authenticated probe required)")
            continue
        try:
            import httpx

            response = httpx.get(f"{str(base_url).rstrip('/')}/api/tags", timeout=3.0)
            if response.is_success:
                status = f"reachable (HTTP {response.status_code})"
            else:
                status = f"failed (HTTP {response.status_code})"
            checks.add_row(label, status)
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            checks.add_row(label, f"unreachable ({type(exc).__name__})")

    console.print(checks)


@app.command()
def ingest(full: Annotated[bool, typer.Option(help="Re-embed all chunks without reuse.")] = False
           ) -> None:
    """Chunk, embed, and index the corpus under data/corpus."""
    try:
        settings = get_settings()
        metrics = MetricsCollector()
        stats: list[BuildStats] = []
        with timed(metrics, "ingest"):
            n = build_index(settings, incremental=not full, on_stats=stats.append)
    except Exception as exc:
        _fail(exc)
        return
    console.print(f"[green]Indexed {n} chunks[/green] from {settings.corpus_dir}")
    if stats:
        console.print(f"Embedded {stats[-1].embedded_chunks}; reused {stats[-1].reused_chunks}.")
    console.print(metrics.summary())


@app.command()
def index_cleanup(
    apply: Annotated[bool, typer.Option("--apply", help="Delete eligible inactive generations.")]
    = False,
    keep: Annotated[int, typer.Option(min=0, help="Retain this many newest generations.")] = 2,
    min_age_hours: Annotated[float, typer.Option(min=0, help="Minimum age before cleanup.")] = 24,
) -> None:
    """Preview old-generation cleanup; deletion requires --apply and an inactive lease."""
    try:
        records = cleanup_generations(
            get_settings(), keep=keep, min_age_seconds=min_age_hours * 3600, dry_run=not apply,
        )
    except Exception as exc:
        _fail(exc)
        return
    console.print("Cleanup applied." if apply else "Dry run: no generations deleted.")
    console.print_json(json.dumps([
        {"generation": r.generation, "action": r.action} for r in records
    ]))


def _render_grounding(grounding: GroundingAssessment) -> None:
    if grounding.is_grounded:
        console.print(f"[dim]grounding: {grounding.reason}[/dim]")
        return
    console.print(f"[yellow]grounding: {grounding.reason}[/yellow]")


@app.command()
def ask(
    question: str,
    doc: Annotated[list[str] | None, typer.Option("--doc", help="Restrict document IDs.")] = None,
    source_url: Annotated[list[str] | None, typer.Option(help="Restrict source URLs.")] = None,
    published_after: Annotated[str | None, typer.Option(help="Inclusive YYYY-MM-DD.")] = None,
    published_before: Annotated[str | None, typer.Option(help="Inclusive YYYY-MM-DD.")] = None,
    trace: Annotated[Path | None, typer.Option(help="Save sanitized telemetry as JSON.")] = None,
    timeout: Annotated[float | None, typer.Option(help="Cooperative run deadline in seconds.")]
    = None,
    max_model_calls: Annotated[int | None, typer.Option(help="Logical model-call budget.")] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    verify_claims: Annotated[bool | None, typer.Option("--verify-claims/--no-verify-claims")]
    = None,
) -> None:
    """Answer a single question through the agent graph."""
    try:
        settings = get_settings()
        overrides = {key: value for key, value in {
            "run_timeout_s": timeout, "run_max_model_calls": max_model_calls,
            "rerank_enabled": rerank, "semantic_verification": verify_claims,
        }.items() if value is not None}
        if overrides:
            settings = Settings.model_validate({**settings.model_dump(), **overrides})
        question = validate_query(question)
        filters = None
        if any(value is not None for value in (doc, source_url, published_after, published_before)):
            filters = RetrievalFilter(
                doc_ids=doc, source_urls=source_url,
                published_after=published_after, published_before=published_before,
            )
        agent = (
            build_agent(settings, filters=filters) if filters is not None else build_agent(settings)
        )
        metrics = MetricsCollector()
        with timed(metrics, "agent"):
            final: dict[str, Any] = agent.invoke(
                {"question": question, "current_query": question, "attempts": 0}
            )
        if trace is not None:
            write_trace(trace, final.get("runtime", {}))
    except KeyboardInterrupt as exc:
        runtime = getattr(exc, "runtime_summary", {"status": "cancelled"})
        if trace is not None:
            try:
                write_trace(trace, runtime)
            except OSError:
                console.print("Could not save the run trace.", markup=False)
        console.print("Run cancelled.", markup=False)
        raise typer.Exit(code=130) from None
    except Exception as exc:
        runtime = (
            exc.summary if isinstance(exc, RunLimitError) else getattr(exc, "runtime_summary", None)
        )
        if trace is not None and isinstance(runtime, dict):
            try:
                write_trace(trace, runtime)
            except OSError:
                console.print("Could not save the run trace.", markup=False)
        _fail(exc)
        return

    grounded = bool(final.get("grounded", False))
    console.print(
        Panel(
            final.get("answer", "(no answer)"),
            title="Answer",
            border_style="green" if grounded else "yellow",
        )
    )

    retrieved = final.get("retrieved", [])
    table = Table(title="Retrieved sources")
    table.add_column("#", justify="right")
    table.add_column("Chunk")
    table.add_column("Doc")
    table.add_column("Published")
    table.add_column("Source")
    table.add_column("RRF", justify="right")
    evidence = final.get("context_citations", dict(enumerate(retrieved, start=1)))
    for i, chunk in evidence.items():
        components = chunk.component_scores or {}
        detail = " ".join(f"{k}={v}" for k, v in sorted(components.items()))
        table.add_row(
            str(i),
            chunk.chunk.chunk_id,
            chunk.chunk.doc_id,
            chunk.chunk.published_at or "unknown",
            chunk.chunk.source_url or "local file",
            f"{chunk.score:.4f} {detail}".strip(),
        )
    console.print(table)

    _render_grounding(assess_grounding(final.get("answer", ""), evidence))

    note = final.get("verification_note")
    if note:
        console.print(f"[dim]verification: {note}[/dim]")
    if final.get("attempts"):
        console.print(f"[dim]reformulation attempts: {final['attempts']}[/dim]")
    console.print(metrics.summary())
    if final.get("rerank_note"):
        console.print(final["rerank_note"], markup=False)
    if final.get("semantic_supported") is not None:
        console.print(f"Semantic support passed: {final['semantic_supported']}", markup=False)
    if final.get("runtime"):
        runtime = final["runtime"]
        console.print(f"Model calls: {runtime['model_calls']}; run status: {runtime['status']}")
    if trace is not None:
        console.print(f"Run trace saved: {trace}", markup=False)


@app.command()
def evaluate(
    output: Annotated[Path | None, typer.Option(help="Save an atomic JSON checkpoint.")] = None,
    resume: Annotated[Path | None, typer.Option(help="Resume a matching checkpoint.")] = None,
    retry_failures: Annotated[bool, typer.Option(help="Retry errors when resuming.")] = False,
    category: Annotated[list[str] | None, typer.Option(help="Category (repeatable).")] = None,
    split: Annotated[EvalSplit | None, typer.Option(help="Select development or held_out.")] = None,
) -> None:
    """Evaluate selected golden cases, checkpointing every completed case."""
    destination = output or resume
    try:
        settings = get_settings()
        metrics = MetricsCollector()
        with timed(metrics, "eval"):
            results = run_eval(
                settings, output=output, resume=resume, retry_failures=retry_failures,
                categories=category, split=split.value if split is not None else None,
            )
        summary = summarize(results, judge_is_same_model=settings.judge_is_same_model)
        if destination is not None:
            console.print(f"Evaluation report saved: {destination}", markup=False)
    except EvaluationCheckpointError as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        message = (f"Evaluation interrupted; completed cases saved in {destination}."
                   if destination is not None else "Evaluation interrupted.")
        console.print(message, markup=False)
        raise typer.Exit(code=130) from None
    except Exception as exc:
        # Provider/validation exception messages may contain credentials or input data.
        console.print(f"Evaluation failed ({type(exc).__name__}).", markup=False)
        raise typer.Exit(code=1) from None
    console.print_json(json.dumps(summary, indent=2))
    console.print(metrics.summary())
    if summary.get("n_operational_failures", 0):
        raise typer.Exit(code=1)


@app.command()
def compare(baseline: Path, candidate: Path) -> None:
    """Compare matching evaluation reports, including category scoring coverage."""
    from finsight.eval.compare import ReportComparisonError, compare_reports

    try:
        comparison = compare_reports(baseline, candidate)
    except ReportComparisonError as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from None
    console.print_json(json.dumps(comparison, indent=2))


@app.command()
def mcp_demo(query: str = typer.Argument("What is Airwallex?")) -> None:
    """List the MCP tools and call search_documents through the MCP client."""
    try:
        from finsight.mcp.client import call_tool, list_tools

        tools = list_tools()
        result = call_tool("search_documents", query=query, top_k=3)
    except Exception as exc:
        _fail(exc)
        return

    console.print("[bold]Tools exposed by the finsight MCP server:[/bold]")
    for tool in tools:
        console.print(f"  - {tool['name']}: {tool['description']}")

    console.print(f"\n[bold]search_documents({query!r})[/bold]")
    console.print_json(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
