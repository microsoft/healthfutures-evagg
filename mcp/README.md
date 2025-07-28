# MCP Servers

The MCP servers in this directory are all purely optional, i.e. they're not required for the default EvAgg pipeline.
However, they allow individual pipeline components to be replaced with MCP tool based implementations, without tight coupling to the core pipeline.
Note that currently the EvAgg pipeline doesn't use agents, which benefit most from MCP, but this might change in the future.

## pubtator-search

An MCP server for searching PubTator3 data to find scientific papers describing gene-disease associations that also mention genetic variants.

