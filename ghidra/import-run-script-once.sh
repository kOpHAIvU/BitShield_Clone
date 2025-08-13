#! /usr/bin/env bash

# Caller of this script is supposed to cd into a (possibly temporary) directory.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FILE=$1
SCRIPT=$2
shift 2

DB_DIR=./temp-db \
	"$SCRIPT_DIR"/exec-analyze-headless.sh \
	-import "$FILE" -postScript "$SCRIPT" "$@"
