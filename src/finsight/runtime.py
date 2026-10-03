"""Per-invocation cooperative controls and content-free execution telemetry.

The fallback token counter counts UTF-8 bytes of content, an upper bound for
byte-based tokenization of that content, NOT provider framing/tool overhead.
Reported provider usage is tracked separately and reconciles budget accounting.
Deadlines/cancellation are checked at boundaries; an arbitrary synchronous call
cannot be interrupted. Real provider requests retain their configured timeouts.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from finsight.config import Settings


def _generated_content(message: Any) -> str:
    """Count tool arguments when providers omit usage and plain text is empty.

    LangChain often retains both normalized and legacy copies of the same calls.
    Prefer normalized calls (including invalid ones) and use legacy fields only
    as a fallback. Keep payloads local to counting, never in telemetry records.
    """
    content = getattr(message, "content", None)
    if content is None:
        return message.model_dump_json() if isinstance(message, BaseModel) else json.dumps(message)
    calls = [*getattr(message, "tool_calls", []), *getattr(message, "invalid_tool_calls", [])]
    if not calls:
        additional = getattr(message, "additional_kwargs", {})
        if isinstance(additional, Mapping):
            calls = additional.get("tool_calls") or []
            if not calls and additional.get("function_call"):
                calls = [additional["function_call"]]
    if isinstance(content, list) and calls:
        # Anthropic-style content blocks can duplicate normalized calls by id.
        call_ids = {call.get("id") for call in calls if isinstance(call, Mapping)} - {None}
        content = [block for block in content if not (
            isinstance(block, Mapping) and block.get("type") in ("tool_use", "tool_call")
            and block.get("id") in call_ids
        )]
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return text + (json.dumps(calls, ensure_ascii=False) if calls else "")


@dataclass(frozen=True)
class RunLimits:
    timeout_s: float | None = None
    max_model_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_output_tokens_per_call: int | None = None
    max_cost_usd: float | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_model_calls", "max_input_tokens", "max_output_tokens",
                     "max_output_tokens_per_call"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("timeout_s", "max_cost_usd", "input_cost_per_million",
                     "output_cost_per_million"):
            value = getattr(self, name)
            minimum_inclusive = name.endswith("per_million")
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)
                    or value < 0 or (value == 0 and not minimum_inclusive)):
                raise ValueError(f"{name} must be finite and within its nonnegative domain")
        if self.max_cost_usd is not None and (
            self.input_cost_per_million is None or self.output_cost_per_million is None
        ):
            raise ValueError("Cost budget requires explicit input and output rates")

    def tightened_by(self, requested: RunLimits) -> RunLimits:
        """Combine optional per-run limits without weakening configured controls.

        Rates are accounting inputs, not ceilings: keep their meaning stable and
        reject conflicting supplied rates instead of silently repricing a run.
        """
        values: dict[str, Any] = {}
        for name in ("timeout_s", "max_model_calls", "max_input_tokens", "max_output_tokens",
                     "max_output_tokens_per_call", "max_cost_usd"):
            configured, supplied = getattr(self, name), getattr(requested, name)
            candidates = [value for value in (configured, supplied) if value is not None]
            values[name] = min(candidates) if candidates else None
        for name in ("input_cost_per_million", "output_cost_per_million"):
            configured, supplied = getattr(self, name), getattr(requested, name)
            if configured is not None and supplied is not None and configured != supplied:
                raise ValueError(f"RunContext {name} must match the configured cost rate")
            values[name] = configured if configured is not None else supplied
        return RunLimits(**values)

    @classmethod
    def from_settings(cls, settings: Settings) -> RunLimits:
        return cls(
            timeout_s=settings.run_timeout_s,
            max_model_calls=settings.run_max_model_calls,
            max_input_tokens=settings.run_max_input_tokens,
            max_output_tokens=settings.run_max_output_tokens,
            max_output_tokens_per_call=settings.llm_max_output_tokens,
            max_cost_usd=settings.run_max_cost_usd,
            input_cost_per_million=settings.input_cost_per_million,
            output_cost_per_million=settings.output_cost_per_million,
        )


class RunLimitError(RuntimeError):
    """A control boundary stopped execution; summary contains no request text."""

    def __init__(self, reason: str, summary: dict[str, Any]) -> None:
        self.reason = reason
        self.summary = summary
        super().__init__(f"Run stopped: {reason}")


@dataclass
class RunContext:
    limits: RunLimits = field(default_factory=RunLimits)
    token_counter: Callable[[str], int] | None = None
    cancellation: threading.Event = field(default_factory=threading.Event)
    clock: Callable[[], float] = time.monotonic
    model_calls: int = 0
    input_budget_tokens: int = 0
    output_budget_tokens: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    reported_usage_calls: int = 0
    status: str = "running"
    nodes: list[dict[str, Any]] = field(default_factory=list)
    model_events: list[dict[str, Any]] = field(default_factory=list)
    _started: float = field(init=False)
    _finished: float | None = None
    _claimed: bool = False

    def __post_init__(self) -> None:
        self._started = self.clock()

    def cancel(self) -> None:
        self.cancellation.set()

    def stop(self, reason: str) -> None:
        self.status = reason
        raise RunLimitError(reason, self.summary())

    def check(self) -> None:
        if self.cancellation.is_set():
            self.stop("cancelled")
        timeout = self.limits.timeout_s
        if timeout is not None and self.clock() - self._started >= timeout:
            self.stop("deadline_exceeded")

    def count(self, text: str) -> int:
        value = self.token_counter(text) if self.token_counter else len(text.encode("utf-8"))
        if type(value) is not int or value < 0:
            raise ValueError("Token counter must return a nonnegative integer")
        return value

    def cost(self, inputs: int, outputs: int) -> float | None:
        if (self.limits.input_cost_per_million is None
                or self.limits.output_cost_per_million is None):
            return None
        return (inputs * self.limits.input_cost_per_million
                + outputs * self.limits.output_cost_per_million) / 1_000_000

    def begin_call(self, content: str) -> tuple[int, int | None]:
        self.check()
        limits = self.limits
        if limits.max_model_calls is not None and self.model_calls >= limits.max_model_calls:
            self.stop("model_call_budget_exceeded")
        estimated_input = self.count(content)
        projected_input = self.input_budget_tokens + estimated_input
        if limits.max_input_tokens is not None and projected_input > limits.max_input_tokens:
            self.stop("input_token_budget_exceeded")
        output_cap = limits.max_output_tokens_per_call
        if limits.max_output_tokens is not None:
            remaining = limits.max_output_tokens - self.output_budget_tokens
            output_cap = remaining if output_cap is None else min(output_cap, remaining)
        if limits.max_cost_usd is not None:
            projected_cost = self.cost(projected_input, self.output_budget_tokens)
            if projected_cost is None:
                raise ValueError("Cost budget requires explicit input and output rates")
            if projected_cost >= limits.max_cost_usd:
                self.stop("cost_budget_exceeded")
            if limits.output_cost_per_million:
                affordable = int((limits.max_cost_usd - projected_cost)
                                 * 1_000_000 / limits.output_cost_per_million)
                output_cap = affordable if output_cap is None else min(output_cap, affordable)
        if output_cap is not None and output_cap <= 0:
            self.stop("output_token_budget_exceeded")
        self.model_calls += 1
        self.input_budget_tokens = projected_input
        return estimated_input, output_cap

    def finish_call(
        self, raw: Any, estimated_input: int, output_cap: int | None = None
    ) -> None:
        message = raw.get("raw") if isinstance(raw, dict) and "raw" in raw else raw
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, Mapping) and all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens")
        ):
            inputs, outputs = usage["input_tokens"], usage["output_tokens"]
            self.reported_input_tokens += inputs
            self.reported_output_tokens += outputs
            self.reported_usage_calls += 1
            self.input_budget_tokens += max(0, inputs - estimated_input)
        else:
            outputs = self.count(_generated_content(message))
        self.output_budget_tokens += outputs
        limits = self.limits
        if (limits.max_input_tokens is not None
                and self.input_budget_tokens > limits.max_input_tokens):
            self.stop("input_token_budget_exceeded")
        if (limits.max_output_tokens is not None
                and self.output_budget_tokens > limits.max_output_tokens):
            self.stop("output_token_budget_exceeded")
        if output_cap is not None and outputs > output_cap:
            self.stop("output_token_budget_exceeded")
        cost = self.cost(self.input_budget_tokens, self.output_budget_tokens)
        if limits.max_cost_usd is not None and cost is not None and cost > limits.max_cost_usd:
            self.stop("cost_budget_exceeded")
        self.check()

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "elapsed_ms": max(0.0, ((self._finished if self._finished is not None
                                      else self.clock()) - self._started) * 1000),
            "model_calls": self.model_calls,
            "input_budget_tokens": self.input_budget_tokens,
            "output_budget_tokens": self.output_budget_tokens,
            "reported_input_tokens": self.reported_input_tokens,
            "reported_output_tokens": self.reported_output_tokens,
            "reported_usage_calls": self.reported_usage_calls,
            "estimated_cost_usd": self.cost(self.reported_input_tokens, self.reported_output_tokens)
            if self.reported_usage_calls == self.model_calls else None,
            "budget_cost_usd": self.cost(self.input_budget_tokens, self.output_budget_tokens),
            "token_count_basis": "injected_content_counter" if self.token_counter else
            "utf8_content_bytes_excludes_provider_overhead",
            "nodes": list(self.nodes), "model_events": list(self.model_events),
        }


_ACTIVE: ContextVar[RunContext | None] = ContextVar("finsight_run", default=None)


def current_run() -> RunContext | None:
    return _ACTIVE.get()


def timed_node(name: str, node: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap any graph node (including extensions) with the same control boundary."""
    def wrapped(state: Any) -> Any:
        context = current_run()
        if context is None:
            return node(state)
        start = context.clock()
        status = "ok"
        try:
            context.check()
            result = node(state)
            context.check()
            return result
        except RunLimitError as exc:
            status = exc.reason
            raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            context.cancel()
            context.status = status = "cancelled"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            context.nodes.append({"name": name, "status": status,
                                  "elapsed_ms": max(0.0, (context.clock() - start) * 1000)})
    return wrapped


class BudgetedModel:
    """Duck-typed chat adapter: counts native, plain, and repair invocations alike."""
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def invoke(self, prompt: str) -> Any:
        return self._invoke(prompt)

    def with_structured_output(self, schema: type[BaseModel]) -> Any:
        owner = self

        class Structured:
            def invoke(self, prompt: str) -> Any:
                return owner._invoke(prompt, schema)

        return Structured()

    def _invoke(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        context = current_run()
        model = self.model
        # Test doubles may not support include_raw; production LangChain clients do.
        native = isinstance(model, BaseChatModel)
        target: Any = model
        if schema is not None:
            target = (model.with_structured_output(schema, include_raw=True) if native
                      else model.with_structured_output(schema))
        if context is None:
            result = target.invoke(prompt)
        else:
            content = prompt + (json.dumps(schema.model_json_schema()) if schema else "")
            estimated, output_cap = context.begin_call(content)
            started = context.clock()
            status = "ok"
            try:
                if output_cap is not None and native:
                    field_name = ("num_predict" if "num_predict" in type(model).model_fields
                                  else "max_tokens")
                    if field_name not in type(model).model_fields:
                        raise ValueError("Configured model does not expose an output token limit")
                    existing_cap = getattr(model, field_name, None)
                    if type(existing_cap) is int and existing_cap > 0:
                        output_cap = min(output_cap, existing_cap)
                    model = model.model_copy(update={field_name: output_cap})
                    target = (model.with_structured_output(schema, include_raw=True)
                              if schema else model)
                result = target.invoke(prompt)
                context.finish_call(result, estimated, output_cap)
            except RunLimitError as exc:
                status = exc.reason
                raise
            except (KeyboardInterrupt, asyncio.CancelledError):
                context.cancel()
                context.status = status = "cancelled"
                raise
            except Exception:
                status = "error"
                raise
            finally:
                context.model_events.append({"status": status,
                    "elapsed_ms": max(0.0, (context.clock() - started) * 1000)})
        if schema is not None and native:
            if result.get("parsing_error") is not None:
                raise result["parsing_error"]
            return result["parsed"]
        return result


class AgentRunner:
    """Compiled graph facade with fresh accounting for every invoke/ainvoke.

    Pass ``runtime=RunContext(...)`` for cancellation or a custom token counter.
    Supplied contexts inherit configured limits and can only tighten them; explicit
    cost rates must agree with configured rates. Contexts are single-use.
    Streaming is intentionally not exposed: it must not
    bypass the invocation boundary or accidentally hide terminal budget errors.
    """
    def __init__(self, graph: Any, limits: RunLimits) -> None:
        self.graph = graph
        self.limits = limits

    def _context(self, runtime: RunContext | None) -> RunContext:
        context = runtime if runtime is not None else RunContext(self.limits)
        if context._claimed:
            raise ValueError("RunContext is single-use; create one per invocation")
        if runtime is not None:
            context.limits = self.limits.tightened_by(context.limits)
        context._claimed = True
        context._started = context.clock()
        return context

    def invoke(self, state: Any, config: Any = None, *, runtime: RunContext | None = None,
               **kwargs: Any) -> Any:
        context = self._context(runtime)
        token = _ACTIVE.set(context)
        try:
            context.check()
            result = self.graph.invoke(state, config=config, **kwargs)
            context.check()
            context.status = "completed"
            return {**result, "runtime": context.summary()}
        except RunLimitError as exc:
            exc.summary = context.summary()
            raise
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            context.cancel()
            context.status = "cancelled"
            exc.runtime_summary = context.summary()  # type: ignore[union-attr]
            raise
        except Exception as exc:
            context.status = "error"
            exc.runtime_summary = context.summary()  # type: ignore[attr-defined]
            raise
        finally:
            context._finished = context.clock()
            _ACTIVE.reset(token)

    async def ainvoke(self, state: Any, config: Any = None, *, runtime: RunContext | None = None,
                      **kwargs: Any) -> Any:
        context = self._context(runtime)
        token = _ACTIVE.set(context)
        try:
            context.check()
            result = await self.graph.ainvoke(state, config=config, **kwargs)
            context.check()
            context.status = "completed"
            return {**result, "runtime": context.summary()}
        except RunLimitError as exc:
            exc.summary = context.summary()
            raise
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            context.cancel()
            context.status = "cancelled"
            exc.runtime_summary = context.summary()  # type: ignore[union-attr]
            raise
        except Exception as exc:
            context.status = "error"
            exc.runtime_summary = context.summary()  # type: ignore[attr-defined]
            raise
        finally:
            context._finished = context.clock()
            _ACTIVE.reset(token)

    def with_config(self, config: Any = None, **kwargs: Any) -> AgentRunner:
        return AgentRunner(self.graph.with_config(config, **kwargs), self.limits)

    def get_graph(self, **kwargs: Any) -> Any:
        return self.graph.get_graph(**kwargs)
