#!/usr/bin/env bash
# Compile every Rust fixture kernel to PTX and refresh cases/.
# Needs a nightly toolchain with the nvptx64-nvidia-cuda target
# (see rust-toolchain.toml). The hand-written cases in cases/ are left alone.
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --examples

for ptx in target/nvptx64-nvidia-cuda/release/examples/*.ptx; do
    cp "$ptx" "../cases/$(basename "$ptx")"
    echo "updated cases/$(basename "$ptx")"
done
