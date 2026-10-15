from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from pydantic_core import to_jsonable_python
from starlette.responses import JSONResponse, Response

# JSON-RPC 2.0's own codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
# the sidecar's, in the range JSON-RPC leaves to a server. Its endpoints share
# one list, so that a code means one thing whichever endpoint gave it, and a
# code never changes, since the proxy hooks act on it
UNREADABLE = -32003
SENTINEL_FAILED = -32004
TIME_LIMIT = -32005
UNREGISTERED = -32006


class Call(BaseModel):
    """One JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"]
    method: str
    params: dict[str, Any] = Field(default_factory=dict[str, Any])
    id: int | str | None = None


class RpcError(Exception):
    """A call that is answered with a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def read_call(body: bytes) -> Call:
    """Read a request body as one JSON-RPC 2.0 request.

    Raises:
        RpcError: If the body is not JSON, or is not one request.
    """
    try:
        parsed: object = json.loads(body)
    except ValueError:
        raise RpcError(PARSE_ERROR, "The body is not JSON.") from None
    try:
        return Call.model_validate(parsed)
    except ValidationError:
        raise RpcError(
            INVALID_REQUEST, "The body is not one JSON-RPC 2.0 request."
        ) from None


def invalid_params(ex: ValidationError) -> RpcError:
    """The error for parameters that don't fit a method: where they are wrong and how, never the values sent."""
    where = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in ex.errors(include_input=False)
    )
    return RpcError(INVALID_PARAMS, f"The parameters don't fit: {where}")


def failed(id: int | str | None, ex: RpcError) -> JSONResponse:
    """The response to a call that is answered with an error."""
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": ex.code, "message": ex.message},
        }
    )


def answered(id: int | str | None, result: BaseModel) -> Response:
    """The response to a call that is answered with a result."""
    # a call without an id is a notification, which is not answered
    if id is None:
        return Response(status_code=204)
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": id,
            "result": to_jsonable_python(result, exclude_none=True),
        }
    )
