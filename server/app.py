"""Minimal MCP server entry point for OpenEnv validation."""

from fastmcp import FastMCP

mcp = FastMCP("inboxpilot-openenv")


@mcp.tool()
def openenv_health() -> str:
    """Return health status for validator checks."""
    return "ok"


def main() -> None:
    """Run MCP server without touching the FastAPI app."""
    mcp.run()


if __name__ == "__main__":
    main()
