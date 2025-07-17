#!/usr/bin/env python3
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "fastmcp",
# ]
# ///
"""Example client for PubTator MCP server."""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from fastmcp import Client

from pubtator_mcp.models import ErrorResult, GeneVariantSearchRequest, GeneVariantSearchResult, VariantExtractionRequest


def _expand_env_vars(text):
    """Expand environment variables in text using ${VAR} or ${VAR:-default} syntax."""

    def replacer(match):
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(
            var_name, default if default is not None else match.group(0)
        )

    # Pattern matches ${VAR} or ${VAR:-default}
    pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"
    return re.sub(pattern, replacer, text)


async def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: uv run --script example_client.py <GENE_SYMBOL>")
        print("Example: uv run --script example_client.py ACTC1")
        sys.exit(1)

    gene_symbol = sys.argv[1]

    # Load config
    config_path = Path(__file__).parent / "mcp_config.json"
    with open(config_path) as f:
        config_text = f.read()

    # Expand environment variables in the raw JSON
    config_text = _expand_env_vars(config_text)

    # Parse the expanded JSON
    config = json.loads(config_text)

    # Create client
    client = Client(config)

    async with client:
        # Create typed request
        request = GeneVariantSearchRequest(
            gene_symbol=gene_symbol,
            retmax=50
        )

        # Search for papers
        result = await client.call_tool(
            "search_gene_variant_papers", {"request": request.model_dump()}
        )

        # Check for errors
        if result.is_error:
            print("Tool execution failed")
            sys.exit(1)

        # Parse result as typed model
        data = result.data

        # Check if it's an error result
        if "error" in data:
            error_result = ErrorResult.model_validate(data)
            print(f"Error: {error_result.error}")
            sys.exit(1)

        # Parse as success result
        search_result = GeneVariantSearchResult.model_validate(data)

        # Display results using typed fields
        print(f"\nGene: {search_result.gene_symbol}")
        print(f"Papers with gene-disease associations and variants: {search_result.full_count}")

        # Show first 10 PMIDs
        if search_result.pmids:
            print(f"\nFirst 10 PMIDs: {search_result.pmids[:10]}")

        # Demo variant extraction
        print("\n" + "="*50)
        print("VARIANT EXTRACTION DEMO")
        print("="*50)

        # Create simple BioC XML for demo
        sample_bioc_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>demo</source>
  <date>2024-01-01</date>
  <key>demo.key</key>
  <document>
    <id>demo_doc</id>
    <passage>
      <infon key="section_type">ABSTRACT</infon>
      <infon key="type">abstract</infon>
      <offset>0</offset>
      <text>The {gene_symbol} gene p.Arg273His mutation is associated with cardiomyopathy.
             Another variant c.123G>A was also identified.</text>
    </passage>
  </document>
</collection>"""

        variant_request = VariantExtractionRequest(
            text=sample_bioc_xml,
            gene_symbol=gene_symbol
        )

        variant_result = await client.call_tool(
            "extract_variants", {"request": variant_request.model_dump()}
        )

        if variant_result.is_error:
            print("Variant extraction failed")
        else:
            variant_data = variant_result.data
            if "error" in variant_data:
                print(f"Variant extraction error: {variant_data['error']}")
            else:
                print("\nSample BioC XML processed")
                print(f"\nExtracted variants for {gene_symbol}:")
                if isinstance(variant_data, list):
                    variants = variant_data
                else:
                    variants = variant_data.get("variants", [])

                for i, variant in enumerate(variants, 1):
                    print(f"  {i}. Text: '{variant.get('text', 'N/A')}'")
                    print(f"     Position: {variant.get('start_pos', 'N/A')}-{variant.get('end_pos', 'N/A')}")
                    if variant.get('ncbi_gene_id'):
                        print(f"     NCBI Gene ID: {variant.get('ncbi_gene_id')}")
                    if variant.get('hgvs'):
                        print(f"     HGVS: {variant.get('hgvs')}")
                    if variant.get('rs_id'):
                        print(f"     dbSNP: {variant.get('rs_id')}")
                    print()


if __name__ == "__main__":
    asyncio.run(main())
