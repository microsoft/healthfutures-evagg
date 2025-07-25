import logging
from collections.abc import Sequence
from typing import Any

from fastmcp import Client

from .interfaces import ISearchPapers

logger = logging.getLogger(__name__)


class MCPPaperSearcher(ISearchPapers):
    """Search for papers using MCP server tools."""

    def __init__(self, mcp_client: Client, tool_name: str, parameter_mapping: dict[str, str]) -> None:
        """Initialize MCP paper searcher.

        Args:
            mcp_client: FastMCP client for calling MCP tools.
            tool_name: Name of the MCP tool to call for paper search.
            parameter_mapping: Mapping from ISearchPapers parameters to MCP tool parameters.
        """
        self._client = mcp_client
        self._tool_name = tool_name
        self._param_mapping = parameter_mapping

    async def search_papers(self, gene_symbol: str, query_params: dict[str, Any]) -> Sequence[str]:
        """Search for paper IDs related to gene_symbol using MCP tools.

        Args:
            gene_symbol: Gene symbol to search for.
            query_params: Additional query parameters (retmax, min_date, max_date).

        Returns:
            Sequence of paper IDs (PMIDs) matching the search criteria.

        Raises:
            RuntimeError: If MCP tool call fails.
        """
        if not gene_symbol:
            raise ValueError("Gene symbol is required for paper search.")

        # Map parameters according to configuration
        mcp_params = {self._param_mapping["gene_symbol"]: gene_symbol}

        # Add optional parameters if provided
        retmax = query_params.get("retmax")
        if retmax:
            mcp_params["retmax"] = retmax
        if "min_date" in query_params:
            mcp_params["min_date"] = query_params["min_date"]
        if "max_date" in query_params:
            mcp_params["max_date"] = query_params["max_date"]

        logger.info(f"Calling MCP tool {self._tool_name} for gene {gene_symbol} with params: {mcp_params}")

        # Call MCP tool with proper connection management
        async with self._client:
            result = await self._client.call_tool(self._tool_name, {"request": mcp_params})

        if result.is_error:
            raise RuntimeError(f"MCP tool {self._tool_name} failed: {result}")

        data = result.data

        error = data.get("error")
        if error:
            raise RuntimeError(f"MCP tool {self._tool_name} returned error: {error}")

        full_count = data.get("full_count")
        if full_count:
            logger.info(f"Found {full_count} paper IDs for {gene_symbol}.")

            if retmax and retmax < full_count:
                logger.warning(
                    f"Reached the maximum number of papers ({retmax}) for {gene_symbol}. This may cause "
                    "an incomplete result, with not all papers matching the gene and date range being processed."
                )

        # Extract paper IDs from MCP response, converting numeric PMIDs to strings
        pmids = data.get("pmids", [])
        return [str(pmid) for pmid in pmids]
