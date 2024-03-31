#!/bin/bash

# This script downloads and runs the Health Futures aoaiproxy client. The client is a Health
# Futures-internal Go application that runs a local AOAI API proxy server, listens for AOAI
# requests and load-balances them across a pool of HF-owned AOAI endpoints for improved throughout
# and token count management. See https://github.com/health-futures/aoaiproxy for more information.

# The script will check in the .bin/aoaiproxy directory for the aoaiproxy binary. If it doesn't exist,
# it will download the binary from the Health Futures aoaiproxy GitHub releases page. The script will
# then run the binary with the default HF pool configuration. To make use of the proxy in evagg experiments,
# set the AZURE_OPENAI_ENDPOINT environment variable in your .env to "http://localhost:2624".

# Prerequisites:
# - [download] Logged in to GitHub CLI with a Microsoft EMU identity that has been added to the health-futures group.
# - [running] .env file is configured correctly with AZURE_OPENAI_ENDPOINT set to "http://localhost:2624".
# - [running] Running as an Azure user that has AAD access to the HF keyvaults.

# Update this to new releases as they come out at
# https://github.com/health-futures/aoaiproxy/releases
VERSION=v0.4

# Check if the directory exists; if not, create it.
DIR="$(dirname $0)/.bin/aoaiproxy/$VERSION"
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
fi

FILE_PATH="$DIR/aoaiproxy"

# Source the environment variables from .env to check if AZURE_OPENAI_ENDPOINT is set correctly.
if [ -f ".env" ]; then
  source .env
fi
if [ "$AZURE_OPENAI_ENDPOINT" != "http://localhost:2624" ]; then
  echo "Warning: AZURE_OPENAI_ENDPOINT is not set to http://localhost:2624."
  echo "It is $AZURE_OPENAI_ENDPOINT. This may cause issues with the AOAI proxy."
fi

# Check if the file exists; if not, download it.
if [ ! -f "$FILE_PATH" ]; then
  echo "Downloading aoaiproxy $VERSION..."

  # Convert periods and dashes to underscores.
  VERSION_=$(echo $VERSION | sed 's/\./_/g' | sed 's/-/_/g')
  ARCHIVE="aoaiproxy_${VERSION_}_linux_amd64.tar.gz"
  DOWNLOAD_URL="https://github.com/health-futures/aoaiproxy/releases/download/${VERSION}/${ARCHIVE}"

  # Must login with a Microsoft EMU identity (https://github.com/enterprises/microsoft)
  err=$(2>&1 gh release download $VERSION --repo https://github.com/health-futures/aoaiproxy --pattern $ARCHIVE --output "$FILE_PATH.tar.gz")
  if [ $? -ne 0 ]; then
    if [[ $err == *"GraphQL: Could not resolve to a Repository with the name 'health-futures/aoaiproxy'"* || \
          $err == *"To get started with GitHub CLI, please run:  gh auth login"* ]]; then
      echo "Unauthorized - must login to GitHub with a Microsoft EMU identity that has been added to the health-futures group."
      echo "Run the following and choose your corp-connected account:"
      echo "gh auth login --hostname github.com --git-protocol https --web"
    else
      echo "Failed to download: $err"
    fi
    exit 1
  fi

  # Unzip/untar the file.
  tar -xzf "$FILE_PATH.tar.gz" -C "$DIR" --warning=no-unknown-keyword # 'LIBARCHIVE.xattr.com.apple.provenance'
  chmod +x "$FILE_PATH"
  rm "$FILE_PATH.tar.gz"
fi

# Run the proxy with the default HF pool configuration.
"$FILE_PATH" -config-url="https://hflabgeneral.blob.core.windows.net/aoaiproxy/health-futures-default/services.yaml"
