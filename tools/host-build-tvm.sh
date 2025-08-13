#! /usr/bin/env bash
# Reference: docker/setup.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

sudo apt update && sudo apt install -y \
	ninja-build zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev llvm-13 libopenblas-dev

cd "$SCRIPT_DIR"/../compilers/tvm-main

mkdir -p build && cd build
cp "$SCRIPT_DIR"/../docker/resources/tvm-main.config.cmake ./config.cmake
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release .. -G Ninja && ninja
