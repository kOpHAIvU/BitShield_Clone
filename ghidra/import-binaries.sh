#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd "$SCRIPT_DIR"

[ -n "$(ls -A ../built)" ] || exit 0

./exec-analyze-headless.sh -import ../built/*.so
