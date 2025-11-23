#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DB_DIR=${DB_DIR:-"$SCRIPT_DIR"/db}
PROJECT_NAME=debfa

# Find analyzeHeadless in common locations
analyze_exec=""

# 1. Try local installation (from install-ghidra.sh)
if [ -f "$SCRIPT_DIR/ghidra-app/support/analyzeHeadless" ]; then
	analyze_exec="$SCRIPT_DIR/ghidra-app/support/analyzeHeadless"
# 2. Try macOS Homebrew Cask installation (expand wildcard)
elif [ -d "/usr/local/Caskroom/ghidra" ]; then
	# Find the latest version
	GHIDRA_DIR=$(ls -td /usr/local/Caskroom/ghidra/*/ghidra_* 2>/dev/null | head -1)
	if [ -n "$GHIDRA_DIR" ] && [ -f "$GHIDRA_DIR/support/analyzeHeadless" ]; then
		analyze_exec="$GHIDRA_DIR/support/analyzeHeadless"
	fi
# 3. Try system-wide installation
elif [ -f "/usr/local/bin/analyzeHeadless" ]; then
	analyze_exec="/usr/local/bin/analyzeHeadless"
elif [ -f "/opt/ghidra/support/analyzeHeadless" ]; then
	analyze_exec="/opt/ghidra/support/analyzeHeadless"
# 4. Try to find in PATH
elif command -v analyzeHeadless &> /dev/null; then
	analyze_exec=$(command -v analyzeHeadless)
fi

# Fallback: use local installation if nothing found
if [ -z "$analyze_exec" ] || [ ! -f "$analyze_exec" ]; then
	analyze_exec="$SCRIPT_DIR/ghidra-app/support/analyzeHeadless"
fi

# Verify the executable exists
if [ ! -f "$analyze_exec" ]; then
	echo "Error: analyzeHeadless not found. Please install Ghidra first:" >&2
	echo "  ./ghidra/install-ghidra.sh" >&2
	exit 1
fi

exec() {
	echo "> $*"
	"$@"
}

mkdir -p "$DB_DIR"

exec $analyze_exec \
	"$DB_DIR" $PROJECT_NAME \
	-scriptPath "$SCRIPT_DIR" \
	"$@"
