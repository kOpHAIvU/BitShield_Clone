#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

execute() {
	echo "> $*"
	"$@"
}

for file in "$@"; do
	echo "Removing $file..."
	execute rm -f "$file"
	echo 'Removing associated built-aux files...'
	fname=$(basename "$file")
	execute find "$PROJ_DIR"/built-aux -name "$fname"'*' -exec rm -f {} \; -print
	echo 'Removing file from Ghidra DB...'
	execute "$PROJ_DIR"/ghidra/remove-from-project.py "$fname"
done
