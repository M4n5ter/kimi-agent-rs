import json
import os
import socket


def fixture_context() -> str:
    payload = {
        "transport": os.environ.get("FIXTURE_TRANSPORT", "unknown"),
        "cwd": os.getcwd(),
        "env_value": os.environ.get("BOX_MCP_ENV"),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    return json.dumps(payload, separators=(",", ":"))
