# PubTator MCP Server

MCP server for PubTator3 integration - search gene-variant-disease associations and extract variants from text.

The PubTator related code and datasets used by this MCP server are in the public domain in the US.

## Setup

### PubTator Search Setup

Before running the search functionality, you need to download and create the index files:

```bash
# Download and create indices
uv run pubtator-update-indices --index-dir ./data
```

This will download the latest PubTator3 data files and create the necessary indices. The process may take some time as it downloads and processes several large files.

#### Updating Indices

To check for updates and download newer versions of the PubTator3 data:

```bash
uv run pubtator-update-indices --index-dir ./data
```

The update script will only download files if newer versions are available. You can force a complete re-download and rebuild with:

```bash
uv run pubtator-mcp-update-indices --index-dir ./data --force
```

#### Automating Updates

For production environments, you may want to set up a cron job to periodically check for updates:

```bash
# Example crontab entry to check for updates weekly on Sunday at 2 AM
0 2 * * 0 cd /path/to/project && uv run pubtator-mcp-update-indices --index-dir /path/to/data
```

### GNorm2 Setup (for gene normalization)

GNorm2 is required for gene normalization before variant extraction. Follow these steps:

1. **Create cache directory and download GNorm2:**

   ```bash
   mkdir -p ../../.cache
   cd ../../.cache
   wget https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNorm2/GNorm2.tar.gz
   tar -xzf GNorm2.tar.gz
   rm GNorm2.tar.gz
   cd GNorm2
   ```

2. **Create requirements-py310.txt file:**

   ```bash
   cat > requirements-py310.txt << 'EOF'
   tensorflow==2.13.1
   transformers==4.37.2
   stanza==1.4.0
   spacy==3.2.4
   bioc==2.0.post4
   protobuf==3.20.3
   EOF
   ```

3. **Set up Python 3.10 virtual environment with uv:**

   ```bash
   uv venv --python=python3.10 .venv
   source .venv/bin/activate
   uv pip install -r requirements-py310.txt
   ```

4. **Apply required code fixes for Apple Silicon:**

   ```bash
   patch -p0 < ../../mcp/pubtator-mcp/gnorm2_fixes.patch
   ```

### tmVar3 Setup (for variant extraction)

To enable variant extraction functionality, you need to manually download and set up tmVar3:

1. **Create cache directory and download tmVar3:**

   You'll need about 6GB of free space.

   ```bash
   mkdir -p ../../.cache
   cd ../../.cache
   wget https://ftp.ncbi.nlm.nih.gov/pub/lu/tmVar3/tmVar3.tar.gz
   tar -xzf tmVar3.tar.gz
   rm tmVar3.tar.gz
   ```

2. **Download newer SQLite JDBC for Apple Silicon compatibility:**

   ```bash
   cd tmVar3
   wget https://repo1.maven.org/maven2/org/xerial/sqlite-jdbc/3.49.1.0/sqlite-jdbc-3.49.1.0.jar -P lib/
   ```

3. **Replace CRF++ with working version:**

   ```bash
   wget "https://drive.google.com/uc?id=1lEwSRLAXpPnlPMPv8fx48y13Xy5eHNU9&export=download" -O CRF++-0.58.tar.gz
   tar -xzf CRF++-0.58.tar.gz
   cd CRF++-0.58
   ./configure && make
   cd ../CRF
   rm crf_test
   ln -s ../CRF++-0.58/crf_test .
   ```

## Usage

### Running the MCP Server

Once the indices are created, you can run the server:

```bash
uv run pubtator-mcp-server --index-dir ./data
```

The server will fail to start if the required index files are not present. In this case, run the update script first.

### As MCP Server

Configuration in `mcp_config.json`.

### Example Client

```bash
uv run --script example_client.py ACTC1
```
