#!/usr/bin/env bash
# Compile every fixture kernel to PTX and refresh tests/fixtures/.
# Needs a nightly toolchain with the nvptx64-nvidia-cuda target
# (see rust-toolchain.toml).
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --examples

out=../tests/fixtures
for ptx in target/nvptx64-nvidia-cuda/release/examples/*.ptx; do
    cp "$ptx" "$out/$(basename "$ptx")"
    echo "updated $out/$(basename "$ptx")"
done
