"""
Minimal MCP server entry point to satisfy OpenEnv multi-mode validation.
This does NOT replace the main FastAPI app used by Hugging Face deployment.
"""

from fastmcp import FastMCP

mcp = FastMCP("inboxpilot-openenv")


@mcp.tool()
def openenv_health() -> str:
    """Return health status for OpenEnv multi-mode validator."""
    return "ok"


def main() -> None:
    """Run the minimal MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
