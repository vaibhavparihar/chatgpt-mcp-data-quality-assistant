# server.py
"""
MCP server exposing the Data Quality & EDA tools (CSV only).

Tools:
- list_datasets()       -> list available CSV files + basic info
- profile_dataset(name) -> generate EDA summary & Markdown report
- detect_data_issues(name) -> data quality checks + score
- open_report(name)     -> read a report snippet for the model to reason about

We use the Python MCP SDK's FastMCP with the Streamable HTTP transport, which
creates an HTTP server exposing a /mcp endpoint compatible with the OpenAI Apps SDK
and ChatGPT connectors.
"""

from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

import core  # our local module

# Create the MCP server
mcp = FastMCP("data-quality-eda")


@mcp.tool(name="list_datasets")
def list_datasets_tool() -> Dict[str, Any]:
    """
    List available datasets and basic properties (rows, columns, file format).
    """
    infos = core.list_datasets()
    return {
        "datasets": [vars(info) for info in infos]
    }


@mcp.tool(name="profile_dataset")
def profile_dataset_tool(name: str) -> Dict[str, Any]:
    """
    Generate an EDA profile for the dataset and save a Markdown report.

    Args:
        name: Dataset name, e.g. "titanic" or "titanic.csv".
    """
    return core.profile_dataset(name)


@mcp.tool(name="detect_data_issues")
def detect_data_issues_tool(name: str) -> Dict[str, Any]:
    """
    Analyze a dataset for common data-quality issues and compute a quality score.

    Args:
        name: Dataset name, e.g. "titanic".
    """
    return core.detect_data_issues(name)


@mcp.tool(name="open_report")
def open_report_tool(name: str) -> Dict[str, Any]:
    """
    Open a previously generated Markdown profiling report and return a text snippet.
    """
    return core.open_report(name)


if __name__ == "__main__":
    # FastMCP with the "streamable-http" transport provides an HTTP server with a /mcp endpoint,
    # which is exactly what the Apps SDK and ChatGPT connectors expect.
    print("Starting Data Quality MCP server on http://127.0.0.1:8000/mcp")
    mcp.run(transport="streamable-http")
