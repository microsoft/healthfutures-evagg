#!/bin/bash

# set -e
# set -o pipefail

# Update this to new releases as they come out at
# https://github.com/health-futures/aoaiproxy/releases
VERSION=v0.3

# Check if the directory exists; if not, create it.
DIR="$(dirname $0)/.bin/aoaiproxy/$VERSION"
if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
fi

# Check if the file exists; if not, download it.
FILE_PATH="$DIR/aoaiproxy"
if [ ! -f "$FILE_PATH" ]; then
  # Must login with a Microsoft EMU identity (https://github.com/enterprises/microsoft)
  # gh auth login --hostname github.com --git-protocol https --web

  echo "Downloading aoaiproxy $VERSION..."
  # Convert periods and dashes to underscores.
  VERSION_=$(echo $VERSION | sed 's/\./_/g' | sed 's/-/_/g')
  ARCHIVE="aoaiproxy_${VERSION_}_linux_amd64.tar.gz"
  DOWNLOAD_URL="https://github.com/health-futures/aoaiproxy/releases/download/${VERSION}/${ARCHIVE}"
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
