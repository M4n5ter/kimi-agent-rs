#!/bin/sh
set -eu

url="$1"
remote_authority="${KIMI_TEST_BROWSER_REMOTE_AUTHORITY:-}"
local_authority="${KIMI_TEST_BROWSER_LOCAL_AUTHORITY:-}"

if [ -n "$remote_authority" ] && [ -n "$local_authority" ]; then
  url="$(printf '%s' "$url" | sed "s|$remote_authority|$local_authority|")"
fi

headers_file="$(mktemp)"
trap 'rm -f "$headers_file"' EXIT

curl -fsS -D "$headers_file" -o /dev/null "$url"

location="$(
  grep -i -m 1 '^Location:' "$headers_file" \
    | sed -e 's/^[Ll][Oo][Cc][Aa][Tt][Ii][Oo][Nn]:[[:space:]]*//' -e 's/\r$//'
)"
if [ -z "$location" ]; then
  echo "missing redirect location from OAuth authorize response" >&2
  exit 1
fi

python3 - "$location" <<'PY'
import socket
import sys
from urllib.parse import urlsplit

target = urlsplit(sys.argv[1])
host = target.hostname
port = target.port or 80
path = target.path or "/"
if target.query:
    path = f"{path}?{target.query}"

request = (
    f"GET {path} HTTP/1.1\r\n"
    f"Host: {target.netloc}\r\n"
    "Connection: close\r\n"
    "\r\n"
)

with socket.create_connection((host, port), timeout=10) as sock:
    sock.sendall(request.encode("utf-8"))
    try:
        sock.recv(4096)
    except OSError:
        pass
PY
