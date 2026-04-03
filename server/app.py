"""
Minimal MCP server entry point to satisfy OpenEnv multi-mode validation.
"""
from fastmcp import FastMCP

# Create a minimal MCP server definition to satisfy standard OpenEnv requirements
mcp = FastMCP("inboxpilot-openenv")

@mcp.tool()
def openenv_health() -> str:
    """Return health status for the OpenEnv multi-mode validator."""
    return "ok"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
