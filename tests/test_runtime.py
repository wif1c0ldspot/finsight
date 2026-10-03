"""Deterministic runtime controls; no provider or service calls."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from finsight.config import Settings
from finsight.graph.builder import build_agent
from finsight.llm import build_chat_model
from finsight.rag.models import Chunk, RetrievedChunk
from finsight.runtime import (
    AgentRunner,
    BudgetedModel,
    RunContext,
    RunLimitError,
    RunLimits,
    current_run,
    timed_node,
)
from finsight.structured import invoke_structured


class Graph:
    def __init__(self, function):
        self.function = function

    def invoke(self, state, **kwargs):
        return self.function(state)

    async def ainvoke(self, state, **kwargs):
        return self.function(state)


class Model:
    def __init__(self, response="ok", usage=None):
        self.calls = 0
        self.response = response
        self.usage = usage

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.response, usage_metadata=self.usage)

    def with_structured_output(self, schema):
        raise NotImplementedError


def runner_for(model, limits=None):
    wrapped = BudgetedModel(cast(Any, model))
    return AgentRunner(Graph(timed_node("answer", lambda state: {
        "answer": wrapped.invoke(state["question"]).content,
    })), limits or RunLimits())


def test_pre_cancelled_context_stops_before_model_and_resets_scope():
    model = Model()
    context = RunContext()
    context.cancel()
    with pytest.raises(RunLimitError, match="cancelled") as error:
        runner_for(model).invoke({"question": "secret"}, runtime=context)
    assert model.calls == 0
    assert error.value.summary["status"] == "cancelled"
    assert current_run() is None
    with pytest.raises(ValueError, match="single-use"):
        runner_for(model).invoke({}, runtime=context)


def test_delayed_node_checks_deadline_before_next_model():
    now = [0.0]
    context = RunContext(RunLimits(timeout_s=1), clock=lambda: now[0])
    model = Model()

    def delayed(state):
        now[0] += 2
        return state

    graph = Graph(lambda state: runner_for(model).graph.invoke(
        timed_node("retrieve", delayed)(state)))
    with pytest.raises(RunLimitError, match="deadline_exceeded") as error:
        AgentRunner(graph, context.limits).invoke({"question": "secret"}, runtime=context)
    assert model.calls == 0
    assert error.value.summary["nodes"] == [
        {"name": "retrieve", "status": "deadline_exceeded", "elapsed_ms": 2000.0}
    ]


def test_deadline_after_blocking_call_records_usage_and_stops_cooperatively():
    now = [0.0]
    context = RunContext(RunLimits(timeout_s=1), clock=lambda: now[0])

    class Delayed(Model):
        def invoke(self, prompt):
            now[0] += 2
            return super().invoke(prompt)

    model = Delayed(usage={"input_tokens": 2, "output_tokens": 1})
    with pytest.raises(RunLimitError, match="deadline_exceeded") as error:
        runner_for(model).invoke({"question": "x"}, runtime=context)
    assert model.calls == 1
    assert error.value.summary["reported_input_tokens"] == 2
    assert error.value.summary["model_events"][0]["status"] == "deadline_exceeded"


def test_model_cap_applies_to_native_fallback_and_repair_without_swallowing():
    class Schema(BaseModel):
        valid: bool

    class Invalid(Model):
        def with_structured_output(self, schema):
            return self

    model = Invalid(response="not JSON")
    wrapped = cast(Any, BudgetedModel(cast(Any, model)))
    graph = Graph(lambda state: {"value": invoke_structured(wrapped, "x", Schema)})
    with pytest.raises(RunLimitError, match="model_call_budget_exceeded") as error:
        AgentRunner(graph, RunLimits(max_model_calls=2)).invoke({})
    assert model.calls == 2  # native + plain; repair is blocked
    assert error.value.summary["model_calls"] == 2


def test_input_budget_prevents_call_and_names_counting_basis():
    model = Model()
    with pytest.raises(RunLimitError, match="input_token_budget_exceeded") as error:
        runner_for(model, RunLimits(max_input_tokens=2)).invoke({"question": "界"})
    assert model.calls == 0
    assert error.value.summary["token_count_basis"] == (
        "utf8_content_bytes_excludes_provider_overhead"
    )


def test_output_overrun_is_detected_after_uncontrolled_fake_returns():
    model = Model(response="too long")
    with pytest.raises(RunLimitError, match="output_token_budget_exceeded") as error:
        runner_for(model, RunLimits(max_output_tokens=1)).invoke({"question": "x"})
    assert model.calls == 1
    assert error.value.summary["output_budget_tokens"] == len("too long")


def test_provider_usage_costs_and_sanitized_records():
    model = Model(response="secret-output", usage={"input_tokens": 10, "output_tokens": 3})
    limits = RunLimits(input_cost_per_million=2, output_cost_per_million=4)
    context = RunContext(limits, token_counter=lambda _: 1)
    result = runner_for(model).invoke({"question": "secret-key"}, runtime=context)
    runtime = result["runtime"]
    assert runtime["input_budget_tokens"] == 10
    assert runtime["reported_input_tokens"] == 10
    assert runtime["reported_output_tokens"] == 3
    assert runtime["estimated_cost_usd"] == pytest.approx(32 / 1_000_000)
    assert runtime["token_count_basis"] == "injected_content_counter"
    assert "secret" not in json.dumps(runtime)
    assert runtime["nodes"][0]["status"] == "ok"


def test_missing_usage_never_claims_provider_cost():
    result = runner_for(Model(), RunLimits(
        input_cost_per_million=2, output_cost_per_million=4,
    )).invoke({"question": "x"})
    assert result["runtime"]["estimated_cost_usd"] is None
    assert result["runtime"]["budget_cost_usd"] is not None


def test_failure_attaches_sanitized_telemetry_and_resets_context():
    class Failed(Model):
        def invoke(self, prompt):
            raise RuntimeError("secret provider response")

    with pytest.raises(RuntimeError) as error:
        runner_for(Failed()).invoke({"question": "secret question"})
    summary = error.value.runtime_summary
    assert summary["status"] == "error"
    assert summary["nodes"][0]["status"] == "error"
    assert "secret" not in json.dumps(summary)
    assert current_run() is None


@pytest.mark.parametrize("provider,module,field", [
    ("ollama", "langchain_ollama", "num_predict"),
    ("openai", "langchain_openai", "max_tokens"),
    ("anthropic", "langchain_anthropic", "max_tokens"),
])
def test_factory_and_runtime_set_supported_provider_output_fields(
    monkeypatch, provider, module, field,
):
    pytest.importorskip(module)
    model = build_chat_model(provider, "test-model", None, "test-key", temperature=0,
                             timeout_s=2, max_retries=0, max_output_tokens=12)
    assert getattr(model, field) == 12
    caps = []

    def invoke(self, prompt):
        caps.append(getattr(self, field))
        return AIMessage(content="ok", usage_metadata={
            "input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
        })

    monkeypatch.setattr(type(model), "invoke", invoke)
    result = runner_for(model, RunLimits(max_output_tokens=5)).invoke({"question": "x"})
    assert caps == [5]
    assert result["runtime"]["reported_output_tokens"] == 2
    assert getattr(model, field) == 12  # copies do not mutate shared providers


def test_cost_budget_reserves_affordable_output_and_checks_reported_overrun(monkeypatch):
    pytest.importorskip("langchain_openai")
    model = build_chat_model("openai", "test", None, "test-key", temperature=0,
                             timeout_s=2, max_retries=0)
    caps = []

    def invoke(self, prompt):
        caps.append(self.max_tokens)
        return AIMessage(content="ok", usage_metadata={
            "input_tokens": 4, "output_tokens": 2, "total_tokens": 6,
        })

    monkeypatch.setattr(type(model), "invoke", invoke)
    limits = RunLimits(max_cost_usd=5, input_cost_per_million=1_000_000,
                       output_cost_per_million=1_000_000)
    context = RunContext(limits, token_counter=lambda _: 1)
    with pytest.raises(RunLimitError, match="cost_budget_exceeded"):
        runner_for(model).invoke({"question": "x"}, runtime=context)
    assert caps == [4]
    assert context.summary()["estimated_cost_usd"] == 6


def test_each_graph_invocation_has_independent_context_and_node_timings():
    class Retriever:
        def retrieve(self, query, top_k=None):
            return [RetrievedChunk(chunk=Chunk(chunk_id="a:0", doc_id="a", title="t",
                                              text="body", position=0), score=1, rank=1)]

    class AnswerModel(Model):
        def invoke(self, prompt):
            response = '{"sufficient": true}' if "verifying" in prompt else "Answer [1]"
            return SimpleNamespace(content=response)

    graph = build_agent(Settings(run_max_model_calls=2), retriever=cast(Any, Retriever()),
                        llm=cast(Any, AnswerModel()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: graph.invoke({"question": "who?"}), range(2)))
    for result in results:
        assert result["runtime"]["model_calls"] == 2
        assert [step["name"] for step in result["runtime"]["nodes"]] == [
            "retrieve", "verify", "answer", "grade", "finalize",
        ]
        assert all(step["elapsed_ms"] >= 0 for step in result["runtime"]["nodes"])
    assert graph.get_graph() is not None
    assert graph.with_config({"recursion_limit": 20}).invoke({"question": "who?"})["grounded"]
    assert current_run() is None


def test_async_entrypoint_uses_same_budget_boundary():
    result = asyncio.run(runner_for(Model()).ainvoke({"question": "x"}))
    assert result["runtime"]["model_calls"] == 1
    assert current_run() is None


@pytest.mark.parametrize("kwargs", [
    {"run_timeout_s": float("inf")}, {"run_max_model_calls": 0},
    {"run_max_input_tokens": -1}, {"run_max_output_tokens": True},
    {"llm_max_output_tokens": 0}, {"run_max_cost_usd": 1},
    {"input_cost_per_million": -1},
])
def test_invalid_control_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"timeout_s": float("nan")}, {"max_model_calls": 0}, {"max_cost_usd": 1},
])
def test_direct_limits_are_validated(kwargs):
    with pytest.raises(ValueError):
        RunLimits(**kwargs)


def test_native_structured_usage_is_kept_with_raw_response(monkeypatch):
    pytest.importorskip("langchain_openai")

    class Schema(BaseModel):
        valid: bool

    model = build_chat_model("openai", "test", None, "test-key", temperature=0,
                             timeout_s=2, max_retries=0, max_output_tokens=7)
    caps = []

    def structured(self, schema, *, include_raw):
        assert include_raw
        current_model = self

        class Runnable:
            def invoke(self, prompt):
                caps.append(current_model.max_tokens)
                return {"raw": AIMessage(content="", usage_metadata={
                    "input_tokens": 11, "output_tokens": 3, "total_tokens": 14,
                }), "parsed": schema(valid=True), "parsing_error": None}

        return Runnable()

    monkeypatch.setattr(type(model), "with_structured_output", structured)
    wrapped = cast(Any, BudgetedModel(model))
    graph = Graph(lambda state: {"verdict": invoke_structured(wrapped, "x", Schema)})
    result = AgentRunner(graph, RunLimits(max_output_tokens=100)).invoke({})
    assert caps == [7]  # cumulative budget cannot relax the existing per-call cap
    assert result["verdict"].valid
    assert result["runtime"]["model_calls"] == 1
    assert result["runtime"]["reported_input_tokens"] == 11
    assert result["runtime"]["reported_output_tokens"] == 3


def test_async_task_cancellation_marks_context_and_resets_scope():
    class WaitingGraph:
        async def ainvoke(self, state, **kwargs):
            await asyncio.sleep(60)

    async def scenario():
        context = RunContext()
        task = asyncio.create_task(AgentRunner(WaitingGraph(), RunLimits()).ainvoke(
            {}, runtime=context,
        ))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert context.summary()["status"] == "cancelled"
        assert context.cancellation.is_set()
        assert current_run() is None

    asyncio.run(scenario())


def test_native_tool_arguments_without_usage_cannot_bypass_output_budget(monkeypatch):
    pytest.importorskip("langchain_openai")

    class Schema(BaseModel):
        rationale: str

    model = build_chat_model("openai", "test", None, "test-key", temperature=0,
                             timeout_s=2, max_retries=0)

    def structured(self, schema, *, include_raw):
        assert include_raw

        class Runnable:
            def invoke(self, prompt):
                return {
                    "raw": AIMessage(content="", tool_calls=[{
                        "name": "Schema", "args": {"rationale": "x" * 1000}, "id": "call-1",
                    }]),
                    "parsed": schema(rationale="x" * 1000), "parsing_error": None,
                }

        return Runnable()

    monkeypatch.setattr(type(model), "with_structured_output", structured)
    wrapped = cast(Any, BudgetedModel(model))
    graph = Graph(lambda state: {"verdict": invoke_structured(wrapped, "x", Schema)})
    with pytest.raises(RunLimitError, match="output_token_budget_exceeded") as error:
        AgentRunner(graph, RunLimits(max_output_tokens=5)).invoke({})
    assert error.value.summary["output_budget_tokens"] >= 1000
    assert error.value.summary["reported_usage_calls"] == 0
    assert error.value.summary["model_calls"] == 1


@pytest.mark.parametrize("legacy", [
    {"function_call": {"name": "Schema", "arguments": '{"rationale": "long content"}'}},
    {"tool_calls": [{"id": "call-1", "type": "function", "function": {
        "name": "Schema", "arguments": '{"rationale": "long content"}',
    }}]},
])
def test_legacy_tool_arguments_are_included_in_fallback_cost(legacy):
    # SimpleNamespace avoids LangChain normalizing these legacy fields itself.
    message = SimpleNamespace(content="", additional_kwargs=legacy)
    context = RunContext(RunLimits(max_cost_usd=0.000001,
                                   input_cost_per_million=0, output_cost_per_million=1))
    with pytest.raises(RunLimitError, match="cost_budget_exceeded"):
        context.finish_call(message, 0)
    assert context.output_budget_tokens > 10


def test_normalized_tool_calls_are_not_counted_twice_with_legacy_copy():
    call = {"id": "call-1", "name": "Schema", "args": {"rationale": "content"}}
    modern = SimpleNamespace(content="", tool_calls=[call])
    duplicated = SimpleNamespace(content="", tool_calls=[call], additional_kwargs={
        "tool_calls": [{"id": "call-1", "function": {
            "name": "Schema", "arguments": json.dumps(call["args"]),
        }}],
    })
    one, two = RunContext(), RunContext()
    one.finish_call(modern, 0)
    two.finish_call(duplicated, 0)
    assert one.output_budget_tokens == two.output_budget_tokens > 0


def test_provider_usage_remains_authoritative_for_tool_responses():
    context = RunContext(RunLimits(max_output_tokens=5))
    context.finish_call(AIMessage(content="", tool_calls=[{
        "id": "call-1", "name": "Schema", "args": {"rationale": "x" * 1000},
    }], usage_metadata={"input_tokens": 1, "output_tokens": 3, "total_tokens": 4}), 1)
    assert context.output_budget_tokens == 3


@pytest.mark.parametrize("supplied_limits", [RunLimits(), RunLimits(max_model_calls=10)])
def test_custom_context_cannot_disable_or_relax_configured_model_cap(supplied_limits):
    model = Model()
    wrapped = BudgetedModel(cast(Any, model))
    graph = Graph(lambda state: {"answers": [wrapped.invoke("x"), wrapped.invoke("x")]})
    configured = RunLimits.from_settings(Settings(run_max_model_calls=1))
    context = RunContext(supplied_limits, token_counter=lambda _: 1)
    with pytest.raises(RunLimitError, match="model_call_budget_exceeded"):
        AgentRunner(graph, configured).invoke({}, runtime=context)
    assert model.calls == 1
    assert context.limits.max_model_calls == 1
    assert context.token_counter("text") == 1


def test_custom_context_inherits_deadline_and_keeps_its_clock():
    now = [0.0]
    context = RunContext(clock=lambda: now[0])

    class Delayed(Model):
        def invoke(self, prompt):
            now[0] += 2
            return super().invoke(prompt)

    model = Delayed()
    with pytest.raises(RunLimitError, match="deadline_exceeded"):
        runner_for(model, RunLimits(timeout_s=1)).invoke({"question": "x"}, runtime=context)
    assert model.calls == 1
    assert context.summary()["elapsed_ms"] == 2000
    assert context.limits.timeout_s == 1


@pytest.mark.parametrize("limits,expected_calls", [
    (RunLimits(max_input_tokens=2), 0),
    (RunLimits(max_output_tokens=2), 1),
    (RunLimits(max_output_tokens_per_call=2), 1),
])
def test_cancellation_context_inherits_configured_token_controls(limits, expected_calls):
    model = Model(response="long output")
    context = RunContext()
    with pytest.raises(RunLimitError, match="token_budget_exceeded"):
        runner_for(model, limits).invoke({"question": "long question"}, runtime=context)
    assert model.calls == expected_calls
    assert context.cancellation.is_set() is False


def test_custom_counter_inherits_cost_budget_and_configured_rates():
    context = RunContext(token_counter=lambda _: 4)
    model = Model()
    limits = RunLimits(max_cost_usd=3, input_cost_per_million=1_000_000,
                       output_cost_per_million=1_000_000)
    with pytest.raises(RunLimitError, match="cost_budget_exceeded"):
        runner_for(model, limits).invoke({"question": "x"}, runtime=context)
    assert model.calls == 0
    assert context.limits == limits


@pytest.mark.parametrize("name", ["input_cost_per_million", "output_cost_per_million"])
def test_context_cannot_reprice_configured_cost_accounting(name):
    model = Model()
    context = RunContext(RunLimits(**{name: 0}))
    runner = runner_for(model, RunLimits(max_cost_usd=3, input_cost_per_million=2,
                                         output_cost_per_million=4))
    with pytest.raises(ValueError, match="must match the configured cost rate"):
        runner.invoke({"question": "x"}, runtime=context)
    assert model.calls == 0
    assert not context._claimed
    assert current_run() is None


def test_stricter_context_limits_are_honored_without_changing_runner_defaults():
    model = Model()
    wrapped = BudgetedModel(cast(Any, model))
    graph = Graph(lambda state: {"answers": [wrapped.invoke("x"), wrapped.invoke("x")]})
    runner = AgentRunner(graph, RunLimits(max_model_calls=2))
    context = RunContext(RunLimits(max_model_calls=1))
    with pytest.raises(RunLimitError, match="model_call_budget_exceeded"):
        runner.invoke({}, runtime=context)
    assert model.calls == 1
    assert runner.invoke({})["runtime"]["model_calls"] == 2
    assert runner.limits.max_model_calls == 2


def test_async_context_inherits_configured_model_budget():
    model = Model()
    wrapped = BudgetedModel(cast(Any, model))
    graph = Graph(lambda state: {"answers": [wrapped.invoke("x"), wrapped.invoke("x")]})
    with pytest.raises(RunLimitError, match="model_call_budget_exceeded"):
        asyncio.run(AgentRunner(graph, RunLimits(max_model_calls=1)).ainvoke(
            {}, runtime=RunContext(),
        ))
    assert model.calls == 1
