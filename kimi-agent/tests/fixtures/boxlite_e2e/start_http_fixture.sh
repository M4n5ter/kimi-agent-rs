#!/bin/sh
set -eu

cd /root/fixtures

: "${MCP_HTTP_PORT:?MCP_HTTP_PORT is required}"
: "${BOX_MCP_ENV:?BOX_MCP_ENV is required}"
: "${FIXTURE_TRANSPORT:?FIXTURE_TRANSPORT is required}"

/root/fixtures/venv/bin/python3 /root/fixtures/boxlite_mcp_http.py 2>&1 | tee /tmp/box-http.log
