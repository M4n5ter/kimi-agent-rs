#!/bin/sh
set -eu

cd /root/fixtures

: "${MCP_OAUTH_PORT:?MCP_OAUTH_PORT is required}"
: "${BOX_MCP_ENV:?BOX_MCP_ENV is required}"
: "${FIXTURE_TRANSPORT:?FIXTURE_TRANSPORT is required}"
: "${OAUTH_STATE_PATH:?OAUTH_STATE_PATH is required}"

exec /root/fixtures/venv/bin/python3 /root/fixtures/boxlite_oauth_mcp.py
