"""The parts of inspect_ai the sidecar needs that inspect_ai doesn't yet make public.

Every import of a private inspect_ai module in the sidecar is here and nowhere else, each beside the public function asked of inspect_ai in its place (design/sidecar.md, "Asked of inspect_ai"). When inspect_ai makes one public, it is swapped in here.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, cast

from inspect_ai.model import ChatMessage, GenerateConfig
from inspect_ai.tool import (
    Tool,
    ToolCall,
    ToolCallView,
    ToolCallViewer,
    ToolChoice,
    ToolInfo,
)
from pydantic import JsonValue

from ._wire import UnreadableError

# A request offers the model every tool it declares. The agent bridge withholds
# the tools a provider runs unless its eval grants them, so here every one is
# granted: the sidecar reads a request, and leaves what it offers to the client.
_EVERY_PROVIDER: dict[str, Any] = {}


def tool_call_view(call: ToolCall, tool: str | None = None) -> ToolCallView:
    """The view Inspect makes of a call: the tool's own, or the default view.

    Asked of inspect_ai: a public way to make the default view of a tool call, and to find a registered tool's viewer.

    Args:
        call: The call.
        tool: The registry name of the tool called, where it is known, e.g. `inspect_ai/bash`. Its viewer makes the view, if it has one.
    """
    from inspect_ai.tool._tool_call import default_tool_call_viewer

    viewer = _viewer(tool) if tool is not None else None
    if viewer is not None:
        try:
            return viewer(call)
        except Exception:
            # a viewer written for another tool's arguments
            pass
    return default_tool_call_viewer(call)


@functools.cache
def _viewer(tool: str) -> ToolCallViewer | None:
    # A tool's viewer is attached when its factory is called, so the tool is
    # made once, with no arguments. One that needs some has the default view.
    from inspect_ai._util.registry import (  # pyright: ignore[reportMissingTypeStubs]
        registry_info,
        registry_lookup,
    )
    from inspect_ai.tool._tool import TOOL_VIEWER

    factory = registry_lookup("tool", tool)
    if not callable(factory):
        return None
    try:
        made = factory()
    except Exception:
        return None
    viewer = registry_info(made).metadata.get(TOOL_VIEWER)
    return cast(ToolCallViewer, viewer) if callable(viewer) else None


# where inspect_ai's providers note what a message can't hold: each module
# with the name of its context variable and of the class it holds
_NOTED = (
    (
        "inspect_ai.model._openai_responses",
        "_openai_assistant_internal",
        "_AssistantInternal",
    ),
    (
        "inspect_ai.model._providers.anthropic",
        "_anthropic_assistant_internal",
        "_AssistantInternal",
    ),
)


@dataclass
class Noted:
    """What inspect_ai's converters noted within a `noting()` block, once it is left."""

    value: JsonValue | None = None


@contextmanager
def noting(start: JsonValue | None = None) -> Generator[Noted]:
    """Give inspect_ai's converters a place of their own to note what a message can't hold.

    The Anthropic and OpenAI Responses converters note each provider item an Inspect message can't hold, such as a tool call as the provider wrote it, so that a message written back puts the item back as it was. They note it in state that inspect_ai gives each sample. The sidecar runs no sample, so without this every conversion in the process would share one place, and a call id that two runs both use would put one run's call into the other's request.

    Args:
        start: What an earlier block noted, for the converters to start from. They start from nothing if it is None.

    Asked of inspect_ai: a public way to run its converters with state of their own.
    """
    from inspect_ai.model._assistant_internal import (
        dump_sample_assistant_internal,
        init_sample_assistant_internal,
    )

    bound: list[tuple[ContextVar[Any], Token[Any]]] = []
    for module_name, variable_name, class_name in _NOTED:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # that provider's SDK isn't installed, so nothing is noted for it
            continue
        variable = cast("ContextVar[Any]", getattr(module, variable_name))
        bound.append((variable, variable.set(getattr(module, class_name)())))
    noted = Noted()
    try:
        if start is not None:
            init_sample_assistant_internal(start)
        yield noted
        noted.value = dump_sample_assistant_internal()
    finally:
        for variable, token in reversed(bound):
            variable.reset(token)


def is_assistant_item(item: dict[str, Any]) -> bool:
    """Whether inspect_ai's Responses reader takes an input item for part of an assistant's turn, as it gathers a turn's items until another item ends it.

    Asked of inspect_ai: Responses readers that drop no assistant text unread.
    """
    from inspect_ai.model._openai_responses import is_assistant_message_param

    return is_assistant_message_param(cast(Any, item))


def without_reasoning(text: str) -> str:
    """A Chat Completions assistant's text with the reasoning in tags taken out, as inspect_ai's reader takes it out before anything else.

    Asked of inspect_ai: Chat Completions readers that keep an assistant's `internal` value.
    """
    from inspect_ai.model._reasoning import parse_content_with_reasoning

    try:
        rest, _ = parse_content_with_reasoning(text)
    except Exception:
        # the reader raises on it too, which makes the request unreadable
        return text
    return rest


def internal_key(value: object) -> str:
    """An `internal` value written as inspect_ai writes one to compare it."""
    from inspect_ai._util.json import to_json_str_safe

    return to_json_str_safe(value)


def internal_text(text: str) -> tuple[str, str] | None:
    """The `<content-internal>` value a text carries, as inspect_ai's Responses reader compares two of them, and the text it reads; None if the text carries none or one it can't read.

    Asked of inspect_ai: Responses readers that drop no assistant text unread.
    """
    from inspect_ai._util.json import to_json_str_safe
    from inspect_ai.model._internal import (
        CONTENT_INTERNAL_TAG,
        parse_content_with_internal,
    )

    try:
        kept, value = parse_content_with_internal(text, CONTENT_INTERNAL_TAG)
    except Exception:
        # the reader raises on it too, which makes the request unreadable
        return None
    if not value:
        return None
    return to_json_str_safe(value), kept


async def anthropic_message(message: ChatMessage) -> dict[str, Any]:
    """A message written in Anthropic's Messages format, as inspect_ai's provider writes it.

    Asked of inspect_ai: a public function that writes a message in Anthropic's format, as there is for OpenAI's.
    """
    from inspect_ai.model._providers.anthropic import (  # pyright: ignore[reportMissingTypeStubs]
        message_param,
    )

    return cast("dict[str, Any]", await message_param(message))


def anthropic_offered(
    request: dict[str, Any],
) -> tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]:
    """The tools, tool choice and configuration an Anthropic Messages request sends the model.

    Asked of inspect_ai: public functions that read these from a request in each provider's format.
    """
    from inspect_ai.agent._bridge.anthropic_api_impl import (
        generate_config_from_anthropic,
        tool_choice_from_anthropic_tool_choice,
        tools_from_anthropic_tools,
    )

    def read(declaration: Any) -> list[ToolInfo | Tool]:
        return tools_from_anthropic_tools(
            [declaration],
            None,
            cast(Any, _EVERY_PROVIDER),
            cast(Any, _EVERY_PROVIDER),
            allow_remote_mcp=True,
        )

    def read_servers(servers: Any) -> list[ToolInfo | Tool]:
        return tools_from_anthropic_tools(
            None,
            servers,
            cast(Any, _EVERY_PROVIDER),
            cast(Any, _EVERY_PROVIDER),
            allow_remote_mcp=True,
        )

    servers = request.get("mcp_servers")
    return (
        [
            *_readable("anthropic", request.get("tools"), read),
            *_readable("anthropic", [servers] if servers else [], read_servers),
        ],
        tool_choice_from_anthropic_tool_choice(request.get("tool_choice")),
        generate_config_from_anthropic(request),
    )


def chat_offered(
    request: dict[str, Any],
) -> tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]:
    """The tools, tool choice and configuration an OpenAI Chat Completions request sends the model.

    Asked of inspect_ai: as for `anthropic_offered`.
    """
    from inspect_ai.agent._bridge.completions import (
        generate_config_from_openai_completions,
        tool_choice_from_openai_tool_choice,
        tools_from_openai_tools,
    )

    return (
        _readable(
            "openai-chat",
            request.get("tools"),
            lambda declaration: list(tools_from_openai_tools([declaration])),
        ),
        tool_choice_from_openai_tool_choice(request.get("tool_choice")),
        generate_config_from_openai_completions(request),
    )


def responses_offered(
    request: dict[str, Any],
) -> tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]:
    """The tools, tool choice and configuration an OpenAI Responses request sends the model.

    Asked of inspect_ai: as for `anthropic_offered`.
    """
    from inspect_ai.agent._bridge.responses_impl import (
        generate_config_from_openai_responses,
        tool_choice_from_responses_tool_choice,
        tools_from_responses_tool,
    )

    def read(declaration: Any) -> list[ToolInfo | Tool]:
        # a kind inspect_ai doesn't know is left out here, with its own
        # warning, and _readable() refuses the request for it
        return tools_from_responses_tool(
            declaration,
            cast(Any, _EVERY_PROVIDER),
            cast(Any, _EVERY_PROVIDER),
            allow_remote_mcp=True,
        )

    return (
        _readable("openai-responses", _responses_tools(request), read),
        tool_choice_from_responses_tool_choice(request.get("tool_choice")),
        generate_config_from_openai_responses(request),
    )


def _responses_tools(request: dict[str, Any]) -> list[Any]:
    # A client can declare tools in `additional_tools` items of its input,
    # rather than or as well as in `tools`. They are offered the model just
    # the same, so they are merged as inspect_ai's bridge merges them, in
    # inspect_responses_api_request_impl() in its
    # src/inspect_ai/agent/_bridge/responses_impl.py.
    declared: object = request.get("tools")
    tools = list(cast("list[Any]", declared)) if isinstance(declared, list) else []
    known = {_key(tool) for tool in tools}
    entries: object = request.get("input")
    for entry in cast("list[object]", entries if isinstance(entries, list) else []):
        if not (
            isinstance(entry, dict)
            and cast("dict[str, Any]", entry).get("type") == "additional_tools"
        ):
            continue
        added: object = cast("dict[str, Any]", entry).get("tools")
        for tool in cast("list[Any]", added if isinstance(added, list) else []):
            if _key(tool) not in known:
                known.add(_key(tool))
                tools.append(tool)
    return tools


def _key(declaration: object) -> tuple[object, object]:
    if not isinstance(declaration, dict):
        return (None, None)
    found = cast("dict[str, Any]", declaration)
    return (found.get("type"), found.get("name"))


# what a function that declares no parameters takes
_NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


def _filled(api: str, declaration: object) -> object:
    # A provider lets a tool declaration leave out some fields that
    # inspect_ai's bridge readers index, so they are filled in here with what
    # leaving them out means: no description, no parameters, any of an MCP
    # server's tools, no headers, and input of any text.
    if not isinstance(declaration, dict):
        return declaration
    found = cast("dict[str, Any]", declaration)
    kind = found.get("type")
    if api == "openai-chat":
        function: object = found.get("function")
        if kind == "function" and isinstance(function, dict):
            return found | {
                "function": {"description": "", "parameters": _NO_PARAMETERS}
                | cast("dict[str, Any]", function)
            }
    elif api == "anthropic":
        if "input_schema" in found:
            return {"description": ""} | found
    elif kind == "function":
        return {"description": "", "parameters": _NO_PARAMETERS} | found
    elif kind == "custom":
        return {"description": "", "format": {"type": "text"}} | found
    elif kind == "mcp":
        return {"allowed_tools": None, "headers": None} | found
    elif kind == "namespace":
        # a namespace holds tools of its own, which are read the same way
        inner: object = found.get("tools")
        if isinstance(inner, list):
            return found | {
                "tools": [_filled(api, tool) for tool in cast("list[object]", inner)]
            }
    return found


def _complete(api: str, declaration: object) -> object:
    # A Responses declaration of a provider's tool can give the model content
    # of its own that inspect_ai's readers leave out: an MCP server's
    # description, and the files a code interpreter's container holds, or
    # the container itself where it is one the provider keeps. Such a
    # declaration can't be read whole.
    if api != "openai-responses" or not isinstance(declaration, dict):
        return declaration
    found = cast("dict[str, Any]", declaration)
    container: object = found.get("container")
    if (found.get("type") == "mcp" and found.get("server_description")) or (
        found.get("type") == "code_interpreter"
        and (
            (
                isinstance(container, dict)
                and bool(cast("dict[str, Any]", container).get("file_ids"))
            )
            # a container the provider keeps, named by its id, with whatever
            # files and state it holds
            or (isinstance(container, str) and container != "auto")
        )
    ):
        raise UnreadableError(
            f"The request offers a tool with content that would not be read: {api} {_kind(found)}."
        )
    return found


def _readable(
    api: str,
    declarations: object,
    read: Callable[[Any], Sequence[ToolInfo | Tool]],
) -> list[ToolInfo]:
    # A declaration that can't be read makes the request unreadable, rather
    # than a step being told the model is offered fewer tools than it is.
    # inspect_ai's readers raise on some kinds they don't know, and leave
    # others out with a warning of their own.
    from inspect_ai.tool._tool_util import tool_to_tool_info

    tools: list[ToolInfo] = []
    for declaration in cast("list[Any]", declarations or []):
        try:
            read_ = [
                tool if isinstance(tool, ToolInfo) else tool_to_tool_info(tool)
                for tool in read(_filled(api, _complete(api, declaration)))
            ]
        except UnreadableError:
            raise
        except Exception as ex:
            raise UnreadableError(
                f"The request offers a tool that can't be read: {api} {_kind(declaration)}."
            ) from ex
        if len(read_) < _declared(declaration):
            raise UnreadableError(
                f"The request offers a tool that can't be read: {api} {_kind(declaration)}."
            )
        tools.extend(read_)
    return tools


def _declared(declaration: object) -> int:
    # the tools in a declaration: a namespace's reader leaves out any of its
    # tools it doesn't know, with a warning of its own
    if isinstance(declaration, dict):
        found = cast("dict[str, Any]", declaration)
        if found.get("type") == "namespace" and isinstance(found.get("tools"), list):
            return len(cast("list[object]", found["tools"]))
    return 1


def _kind(declaration: object) -> str:
    if isinstance(declaration, list):
        return "mcp_servers"
    if isinstance(declaration, dict):
        return str(cast("dict[str, Any]", declaration).get("type", "custom"))
    return type(declaration).__name__
