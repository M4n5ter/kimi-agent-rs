import contextlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse
from starlette.routing import Mount, Route

from mcp.server.fastmcp import FastMCP

from boxlite_mcp_common import fixture_context

ACCESS_TOKEN = "boxlite-oauth-access-token"
AUTHORIZATION_CODE = "boxlite-oauth-code"
CLIENT_ID = "boxlite-oauth-client"
REQUIRED_SCOPE = "box.read"
BASE_URL = f"http://127.0.0.1:{os.environ['MCP_OAUTH_PORT']}"
STATE_FILE = Path(os.environ["OAUTH_STATE_PATH"])

STATE = {
    "last_authorization_header": None,
    "authorized_mcp_requests": 0,
    "register_requests": 0,
    "authorize_requests": 0,
    "token_requests": 0,
    "resource_metadata_requests": 0,
    "authorization_metadata_requests": 0,
}

mcp = FastMCP("boxlite-oauth-http-fixture", stateless_http=True, json_response=True)
mcp.settings.streamable_http_path = "/"


@mcp.tool()
def box_oauth_http_context() -> str:
    """Return the protected OAuth MCP execution context."""
    payload = json.loads(fixture_context())
    payload["last_authorization_header"] = STATE["last_authorization_header"]
    payload["authorized_mcp_requests"] = STATE["authorized_mcp_requests"]
    return json.dumps(payload, separators=(",", ":"))


class ProtectedMcpApp:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        authorization = headers.get("authorization")
        expected = f"Bearer {ACCESS_TOKEN}"
        if authorization != expected:
            response = PlainTextResponse("", status_code=401)
            response.headers["WWW-Authenticate"] = (
                'Bearer error="invalid_token", '
                f'resource_metadata="{BASE_URL}/.well-known/oauth-protected-resource/mcp", '
                f'scope="{REQUIRED_SCOPE}"'
            )
            await response(scope, receive, send)
            return

        STATE["last_authorization_header"] = authorization
        STATE["authorized_mcp_requests"] += 1
        write_state()
        await self.app(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(_app: Starlette):
    write_state()
    async with mcp.session_manager.run():
        yield


async def health(_request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def protected_resource_metadata(_request: Request) -> JSONResponse:
    STATE["resource_metadata_requests"] += 1
    write_state()
    print("resource metadata requested", flush=True)
    return JSONResponse(
        {
            "authorization_servers": [
                f"{BASE_URL}/.well-known/oauth-authorization-server/oauth"
            ],
            "scopes_supported": [REQUIRED_SCOPE],
        }
    )


async def authorization_server_metadata(_request: Request) -> JSONResponse:
    STATE["authorization_metadata_requests"] += 1
    write_state()
    print("authorization server metadata requested", flush=True)
    return JSONResponse(
        {
            "issuer": f"{BASE_URL}/oauth",
            "authorization_endpoint": f"{BASE_URL}/oauth/authorize",
            "token_endpoint": f"{BASE_URL}/oauth/token",
            "registration_endpoint": f"{BASE_URL}/oauth/register",
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": [REQUIRED_SCOPE],
        }
    )


async def register_client(request: Request) -> JSONResponse:
    payload = await request.json()
    STATE["register_requests"] += 1
    write_state()
    print(f"register client request: {payload}", flush=True)
    return JSONResponse(
        {
            "client_id": CLIENT_ID,
            "client_secret": "",
            "client_name": payload.get("client_name"),
            "redirect_uris": payload.get("redirect_uris", []),
        }
    )


async def authorize(request: Request) -> RedirectResponse | PlainTextResponse:
    STATE["authorize_requests"] += 1
    write_state()
    print(f"authorize request: {dict(request.query_params)}", flush=True)
    redirect_uri = request.query_params.get("redirect_uri", "").strip()
    state = request.query_params.get("state", "").strip()
    if not redirect_uri or not state:
        return PlainTextResponse("missing redirect_uri or state", status_code=400)

    query = urlencode({"code": AUTHORIZATION_CODE, "state": state})
    return RedirectResponse(f"{redirect_uri}?{query}", status_code=302)


async def exchange_token(request: Request) -> JSONResponse | PlainTextResponse:
    form = await request.form()
    STATE["token_requests"] += 1
    write_state()
    print(f"token request: {dict(form)}", flush=True)
    if form.get("code") != AUTHORIZATION_CODE:
        return PlainTextResponse("invalid authorization code", status_code=400)

    return JSONResponse(
        {
            "access_token": ACCESS_TOKEN,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": REQUIRED_SCOPE,
        }
    )


def write_state() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(STATE, indent=2), encoding="utf-8")


app = Starlette(
    routes=[
        Route("/health", health),
        Route(
            "/.well-known/oauth-protected-resource/mcp",
            protected_resource_metadata,
        ),
        Route(
            "/.well-known/oauth-authorization-server/oauth",
            authorization_server_metadata,
        ),
        Route("/oauth/register", register_client, methods=["POST"]),
        Route("/oauth/authorize", authorize),
        Route("/oauth/token", exchange_token, methods=["POST"]),
        Mount("/mcp", app=ProtectedMcpApp(mcp.streamable_http_app())),
    ],
    lifespan=lifespan,
)


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("KIMI_OAUTH_BIND_HOST", "0.0.0.0"),
        port=int(os.environ["MCP_OAUTH_PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
