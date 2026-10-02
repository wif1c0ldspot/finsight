"""MCP client: connect to the finsight MCP server and invoke its tools.

Uses the official MCP SDK's stdio client — the standard pattern for attaching
an agent to a local MCP server.
"""

import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolError(RuntimeError):
    """A server tool reported failure rather than a successful payload."""

    def __init__(self, tool: str, content: list[Any]) -> None:
        self.tool = tool
        self.content = content
        super().__init__(f"MCP tool {tool!r} failed: {_extract_text(content)}")


def _server_params() -> StdioServerParameters:
    # The SDK supplies its own process allowlist (PATH, HOME, etc.). Add only
    # application settings and the provider credentials our factories consume.
    provider_keys = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    env = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("FINSIGHT_") or name in provider_keys
    }
    return StdioServerParameters(
        command=sys.executable, args=["-m", "finsight.mcp.server"], env=env
    )


async def _call_tool_async(tool: str, **kwargs: Any) -> list[Any]:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments=kwargs)
            content = list(result.content)
    # Raise after closing SDK task groups so callers receive the typed error
    # directly, rather than an ExceptionGroup wrapping it.
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)
    if is_error:
        raise MCPToolError(tool, content)
    return content


def _extract_text(content: list[Any]) -> Any:
    texts = [str(getattr(block, "text", "")) for block in content if getattr(block, "text", None)]
    if not texts:
        return []
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return texts


def call_tool(tool: str, **kwargs: Any) -> Any:
    """Call a tool on the finsight MCP server; JSON results are auto-parsed."""
    content = asyncio.run(_call_tool_async(tool, **kwargs))
    return _extract_text(content)


def list_tools() -> list[dict[str, str]]:
    """Return the tools exposed by the finsight MCP server."""
    return asyncio.run(_list_tools_async())


async def _list_tools_async() -> list[dict[str, str]]:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [{"name": t.name, "description": t.description or ""} for t in tools.tools]
