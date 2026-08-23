#!/usr/bin/env bash
# Downloads and extracts the CIFAR-10 binary dataset into data/,
# so cifar_cnn can find it at data/cifar-10-batches-bin/ regardless
# of whether you run the binary from the repo root or from build/.
#
# Usage (from anywhere in the repo):
#   ./scripts/download_cifar.sh
set -euo pipefail

# Resolve repo root relative to this script's own location, not the
# caller's cwd, so this works no matter where you invoke it from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data
cd data

if [ -d cifar-10-batches-bin ]; then
    echo "data/cifar-10-batches-bin/ already exists, skipping download."
    exit 0
fi

echo "Downloading CIFAR-10 (binary version, ~170MB)..."
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Extracting..."
tar -xzf cifar-10-binary.tar.gz

echo "Cleaning up archive..."
rm cifar-10-binary.tar.gz

echo "Done. CIFAR-10 binaries are in data/cifar-10-batches-bin/"
