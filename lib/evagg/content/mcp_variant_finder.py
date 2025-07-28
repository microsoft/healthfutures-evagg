"""MCP-based variant finder using configurable MCP server."""

import logging
from collections.abc import Sequence
from typing import Any

from fastmcp import Client

from .interfaces import IFindVariants

logger = logging.getLogger(__name__)


class MCPVariantFinder(IFindVariants):
    """Find variants using MCP server tools."""

    def __init__(self, mcp_client: Client, tool_name: str, parameter_mapping: dict[str, str]) -> None:
        """Initialize MCP variant finder.

        Args:
            mcp_client: FastMCP client for calling MCP tools.
            tool_name: Name of the MCP tool to call for variant extraction.
            parameter_mapping: Mapping from IFindVariants parameters to MCP tool parameters.
        """
        self._client = mcp_client
        self._tool_name = tool_name
        self._param_mapping = parameter_mapping

    async def find_variant_descriptions(
        self,
        full_text_xml: str,
        full_text: str,
        focus_texts: Sequence[str] | None,
        gene_symbol: str,
        metadata: dict[str, Any],
    ) -> Sequence[str]:
        """Identify the genetic variants relevant to the gene_symbol described in the full text of the paper.

        Returned variants will be _as described_ in the source text. Downstream manipulations to make them
        HGVS-compliant may be required.
        """
        # MCP tools like tmVar3 need XML format, use it if available
        text_to_process = full_text_xml if full_text_xml else full_text
        if not text_to_process and focus_texts:
            # Combine focus texts if no full text available
            text_to_process = "\n\n".join(focus_texts)

        if not text_to_process:
            logger.warning("No text provided for variant extraction")
            return []

        try:
            # Map parameters according to configuration
            mcp_params = {
                self._param_mapping["text"]: text_to_process,
                self._param_mapping["gene_symbol"]: gene_symbol,
            }

            logger.info(f"Calling MCP tool {self._tool_name} for gene {gene_symbol}")

            # Call MCP tool with proper connection management
            async with self._client:
                result = await self._client.call_tool(self._tool_name, {"request": mcp_params})

            if result.is_error:
                logger.error(f"MCP tool {self._tool_name} failed: {result}")
                return []

            data = result.data

            # Check for errors
            if isinstance(data, dict) and "error" in data:
                logger.error(f"MCP tool {self._tool_name} returned error: {data['error']}")
                return []

            # Extract variant texts from response
            variant_descriptions = []
            # Extract variant texts from response
            variants = data if isinstance(data, list) else data.get("variants", [])

            for variant in variants:
                variant_text = variant.get("text", "")
                if variant_text:
                    variant_descriptions.append(variant_text)

            logger.info(f"Found {len(variant_descriptions)} variants for gene {gene_symbol}")
            return variant_descriptions

        except Exception as e:
            logger.error(f"Failed to extract variants using MCP tool {self._tool_name}: {e}")
            return []
