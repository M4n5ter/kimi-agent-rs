import contextlib
import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from mcp.server.fastmcp import FastMCP

from boxlite_mcp_common import fixture_context

mcp = FastMCP("boxlite-http-fixture", stateless_http=True, json_response=True)
mcp.settings.streamable_http_path = "/"


@mcp.tool()
def box_http_context() -> str:
    """Return the HTTP fixture execution context."""
    return fixture_context()


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield


async def health(_request) -> PlainTextResponse:
    return PlainTextResponse("ok")


app = Starlette(
    routes=[
        Route("/health", health),
        Mount("/mcp", app=mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)


def main() -> None:
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["MCP_HTTP_PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
