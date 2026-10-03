"""Command-line interface for Finsight.

Commands wrap their work in a single error boundary. Previously an unreachable
model service produced a 60-frame traceback ending in ``ConnectionError``; now it
produces one legible line naming the likely cause and the check to run.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from finsight.config import Settings, get_settings
from finsight.eval.harness import run_eval, summarize
from finsight.graph.builder import build_agent
from finsight.guardrails.validation import (
    GroundingAssessment,
    assess_grounding,
    validate_query,
)
from finsight.observability.tracing import MetricsCollector, timed
from finsight.rag.index import IndexIntegrityError, IndexMissingError
from finsight.rag.ingest import build_index
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
                    f"{exc}\n\n[bold]What to do:[/bold] {hint}",
                    title="Error",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1) from exc
    console.print(
        Panel(f"{type(exc).__name__}: {exc}", title="Unexpected error", border_style="red")
    )
    raise typer.Exit(code=1) from exc


def _describe(settings: Settings) -> Table:
    table = Table(title="finsight configuration", show_header=False)
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("llm", f"{settings.llm_provider} / {settings.llm_model}")
    table.add_row("llm base url", str(settings.resolved_llm_base_url()))
    table.add_row("embeddings", f"{settings.embed_provider} / {settings.embed_model}")
    table.add_row("embed base url", str(settings.resolved_embed_base_url()))
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
            checks.add_row(label, f"unreachable: {exc}")

    console.print(checks)


@app.command()
def ingest() -> None:
    """Chunk, embed, and index the corpus under data/corpus."""
    try:
        settings = get_settings()
        metrics = MetricsCollector()
        with timed(metrics, "ingest"):
            n = build_index(settings)
    except Exception as exc:
        _fail(exc)
        return
    console.print(f"[green]Indexed {n} chunks[/green] from {settings.corpus_dir}")
    console.print(metrics.summary())


def _render_grounding(grounding: GroundingAssessment) -> None:
    if grounding.is_grounded:
        console.print(f"[dim]grounding: {grounding.reason}[/dim]")
        return
    console.print(f"[yellow]grounding: {grounding.reason}[/yellow]")


@app.command()
def ask(question: str) -> None:
    """Answer a single question through the agent graph."""
    try:
        settings = get_settings()
        question = validate_query(question)
        agent = build_agent(settings)
        metrics = MetricsCollector()
        with timed(metrics, "agent"):
            final: dict[str, Any] = agent.invoke(
                {"question": question, "current_query": question, "attempts": 0}
            )
    except Exception as exc:
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
    table.add_column("RRF", justify="right")
    evidence = final.get("context_citations", dict(enumerate(retrieved, start=1)))
    for i, chunk in evidence.items():
        components = chunk.component_scores or {}
        detail = " ".join(f"{k}={v}" for k, v in sorted(components.items()))
        table.add_row(
            str(i),
            chunk.chunk.chunk_id,
            chunk.chunk.doc_id,
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


@app.command()
def evaluate() -> None:
    """Run the golden-set evaluation and print aggregate metrics."""
    try:
        settings = get_settings()
        metrics = MetricsCollector()
        with timed(metrics, "eval"):
            results = run_eval(settings)
        summary = summarize(results, judge_is_same_model=settings.judge_is_same_model)
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(summary, indent=2))
    console.print(metrics.summary())


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
