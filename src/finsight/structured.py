"""Provider-agnostic structured output.

Prompts that ask a model for a single word ("reply YES or NO") are parsed with
string matching that silently misreads anything unexpected: a reply of *"The
context is sufficient — YES"* fails a ``startswith("YES")`` check and is treated
as *insufficient*. A reply of ``""`` fails the same way, with no error.

This module makes those extractions typed. It prefers the model's native
structured-output path and falls back to extracting JSON from free text, so it
works on tool-calling APIs and plain chat completions alike.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from finsight.runtime import RunLimitError

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """The model could not be coaxed into a valid instance of the schema."""


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced JSON object found in ``text``.

    Scans for balance rather than using a greedy regex so that trailing prose
    after the object does not corrupt the match.
    """
    fenced = _FENCED.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(candidate)):
                char = candidate[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return candidate[start : idx + 1]
            start = candidate.find("{", start + 1)
    return None


def _raw_text(llm: BaseChatModel, prompt: str) -> str:
    raw = llm.invoke(prompt)
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        # Some providers return content blocks rather than a plain string.
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip() if content is not None else ""


def invoke_structured(
    llm: BaseChatModel,
    prompt: str,
    schema: type[SchemaT],
    *,
    repair_attempts: int = 1,
) -> SchemaT:
    """Return a validated instance of ``schema`` from ``llm``.

    Order of attempts:
      1. the model's native structured-output path;
      2. a plain completion with the JSON object extracted from the text;
      3. up to ``repair_attempts`` retries that show the model its own error.

    Raises :class:`StructuredOutputError` rather than returning a default — a
    silent fallback is what the previous string-matching code did wrong.
    """
    # 1. Native structured output.
    try:
        structured = llm.with_structured_output(schema)
        result = structured.invoke(prompt)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
    except RunLimitError:
        raise
    except NotImplementedError:
        pass
    except Exception:  # provider-specific tool errors; fall through to text parsing
        pass

    # 2. Free-text completion, JSON extracted and validated.
    last_error: Exception | None = None
    active_prompt = prompt
    for attempt in range(repair_attempts + 1):
        text = _raw_text(llm, active_prompt)
        payload = _extract_json_object(text)
        if payload is not None:
            try:
                return schema.model_validate(json.loads(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
        else:
            last_error = StructuredOutputError(
                f"No JSON object found in model output: {text[:200]!r}"
            )

        if attempt < repair_attempts:
            active_prompt = (
                f"{prompt}\n\nYour previous reply was invalid: {last_error}\n"
                f"Reply with ONLY a JSON object matching this schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}"
            )

    raise StructuredOutputError(
        f"Could not obtain a valid {schema.__name__} from the model: {last_error}"
    )
