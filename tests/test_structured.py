"""Tests for the structured-output helper.

The code this replaces read model output with ``text.upper().startswith("YES")``
and ``re.search(r"\\d+", text)``. Both silently produced a wrong answer on
realistic model replies, so the important cases here are the *messy* ones.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from finsight.structured import (
    StructuredOutputError,
    _extract_json_object,
    invoke_structured,
)


class Verdict(BaseModel):
    sufficient: bool
    reason: str = ""


class FakeLLM:
    def __init__(
        self, *responses: str, native: Any | None = None, native_raises: Any = None
    ) -> None:
        self._responses = list(responses)
        self._index = 0
        self._native = native
        self._native_raises = native_raises

    def invoke(self, prompt: str) -> SimpleNamespace:
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
        else:
            response = self._responses[-1] if self._responses else ""
        return SimpleNamespace(content=response)

    def with_structured_output(self, schema: Any) -> Any:
        if self._native_raises is not None:
            raise self._native_raises
        payload = self._native

        class _Runner:
            def invoke(self, prompt: str) -> Any:
                if payload is None:
                    raise NotImplementedError
                return payload

        return _Runner()


# --- JSON extraction -------------------------------------------------------


def test_extract_plain_object():
    assert _extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_object_wrapped_in_prose():
    text = 'Sure! Here it is:\n{"a": 1}\nLet me know if you need more.'
    assert _extract_json_object(text) == '{"a": 1}'


def test_extract_object_from_fenced_block():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json_object(text) == '{"a": 1}'


def test_extract_nested_object_is_balanced():
    text = 'prefix {"a": {"b": 2}} suffix'
    assert _extract_json_object(text) == '{"a": {"b": 2}}'


def test_extract_ignores_braces_inside_strings():
    text = '{"a": "not a } brace"}'
    assert _extract_json_object(text) == '{"a": "not a } brace"}'


def test_extract_returns_none_when_absent():
    assert _extract_json_object("no json here") is None


# --- invoke_structured -----------------------------------------------------


def test_native_structured_output_is_preferred():
    llm = FakeLLM(native=Verdict(sufficient=True, reason="native"))
    result = invoke_structured(llm, "prompt", Verdict)
    assert result.sufficient is True
    assert result.reason == "native"


def test_falls_back_to_text_when_native_is_unsupported():
    llm = FakeLLM('{"sufficient": true, "reason": "parsed"}', native=None)
    result = invoke_structured(llm, "prompt", Verdict)
    assert result.sufficient is True
    assert result.reason == "parsed"


def test_native_dict_payload_is_validated():
    llm = FakeLLM(native={"sufficient": False, "reason": "d"})
    result = invoke_structured(llm, "prompt", Verdict)
    assert result.sufficient is False


def test_messy_reply_still_parses():
    """This is the reply that broke startswith("YES"): prose before the answer."""
    llm = FakeLLM('The context looks sufficient — YES.\n{"sufficient": true, "reason": "ok"}')
    result = invoke_structured(llm, "prompt", Verdict)
    assert result.sufficient is True


def test_repair_attempt_recovers_from_bad_first_reply():
    llm = FakeLLM("not json at all", '{"sufficient": true, "reason": "fixed"}')
    result = invoke_structured(llm, "prompt", Verdict)
    assert result.sufficient is True


def test_raises_rather_than_defaulting_when_output_is_unusable():
    """Silent fallbacks are exactly the failure mode being fixed."""
    llm = FakeLLM("still not json", "still not json")
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, "prompt", Verdict)


def test_raises_when_json_is_valid_but_schema_violated():
    llm = FakeLLM('{"sufficient": "maybe"}', '{"sufficient": "still maybe"}')
    with pytest.raises(StructuredOutputError):
        invoke_structured(llm, "prompt", Verdict)


def test_content_blocks_are_flattened():
    class BlockLLM(FakeLLM):
        def invoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(
                content=[{"text": '{"sufficient": tr'}, {"text": 'ue, "reason": "blocks"}'}]
            )

    result = invoke_structured(BlockLLM(), "prompt", Verdict)
    assert result.sufficient is True
    assert result.reason == "blocks"
