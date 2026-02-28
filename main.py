from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search-mcp")


@mcp.tool()
def hello(name: str) -> str:
    """A simple hello tool."""
    return f"Hello, {name}!"


def run():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
