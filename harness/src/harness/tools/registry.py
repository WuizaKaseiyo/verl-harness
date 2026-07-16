"""Tool registry — collects Tool instances, dispatches by name, exposes schemas."""

from __future__ import annotations

from harness.tools.base import Tool, ToolContext, ToolError
from harness.tools.bash import BashTool
from harness.tools.edit import EditTool
from harness.tools.read import ReadTool
from harness.tools.todo import TodoTool
from harness.tools.web_fetch import WebFetchTool
from harness.tools.write import WriteTool


class ToolRegistry:
    """A name → Tool map that can emit Anthropic tool schemas and dispatch."""

    def __init__(self, tools: list[Tool]) -> None:
        seen: dict[str, Tool] = {}
        for t in tools:
            if t.name in seen:
                raise ValueError(f"duplicate tool name: {t.name}")
            seen[t.name] = t
        self._tools = seen

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def execute(
        self,
        name: str,
        input: dict,
        ctx: ToolContext,
    ) -> tuple[str, bool]:
        """Run tool `name`. Returns (content, is_error).

        - Unknown tool          → is_error=True, message.
        - ToolError raised      → is_error=True, message.
        - Any other exception   → is_error=True, wrapped for the model to see.
        """
        tool = self._tools.get(name)
        if tool is None:
            return (f"unknown tool: {name}", True)
        try:
            return (tool.execute(input, ctx), False)
        except ToolError as e:
            return (f"tool error: {e}", True)
        except Exception as e:  # pragma: no cover — defensive backstop
            return (f"internal tool crash: {type(e).__name__}: {e}", True)


def default_registry() -> ToolRegistry:
    """The 6-tool built-in registry (M3):

    bash / read / edit / write / todo (M1) + web_fetch (M3-T5).
    """
    return ToolRegistry(
        [BashTool(), ReadTool(), EditTool(), WriteTool(), TodoTool(), WebFetchTool()]
    )
