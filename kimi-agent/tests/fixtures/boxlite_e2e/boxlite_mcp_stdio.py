from mcp.server.fastmcp import FastMCP

from boxlite_mcp_common import fixture_context

mcp = FastMCP("boxlite-stdio-fixture")


@mcp.tool()
def box_stdio_context() -> str:
    """Return the stdio fixture execution context."""
    return fixture_context()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
