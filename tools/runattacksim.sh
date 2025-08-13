#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
attacksim="${SCRIPT_DIR}/../attacksim.py"

run_attacksim() {
	local args=("$@" --skip-existing)
	echo "> Running with ${args[*]}"
	"$attacksim" "${args[@]}"
}

_run_with_drams() {
	run_attacksim "$@" --vuln-pct 0.16e-4 --zero-one-pct 51.15 --nexps 50
	run_attacksim "$@" --vuln-pct 0.39e-4 --zero-one-pct 48.89 --nexps 50
	run_attacksim "$@" --vuln-pct 3.04e-4 --zero-one-pct 50.59 --nexps 50
	run_attacksim "$@" --vuln-pct 26.40e-4 --zero-one-pct 50.75 --nexps 50
	run_attacksim "$@" --vuln-pct 64.54e-4 --zero-one-pct 51.16 --nexps 50
	# run_attacksim "$@" --vuln-pct 7.50e-4 --zero-one-pct 50.34 --nexps 50
	# run_attacksim "$@" --vuln-pct 13.60e-4 --zero-one-pct 50.74 --nexps 50
}

_run_with_protections_and_drams() {
	_run_with_drams "$@" --cig ncnp --dig nd
	_run_with_drams "$@" --cig cc2 --dig gn1
}

if [ "$#" = 0 ]; then
	for model in resnet50 googlenet densenet121; do
		for dataset in CIFAR10 MNISTC FashionC; do
			for attacker in w a s; do
				_run_with_protections_and_drams \
					--model-name "$model" --dataset "$dataset" --attacker-type "$attacker"
			done
		done
	done
elif [ "$#" = 1 ] && [ "$1" = "--check-sig-bypass" ]; then
	_run_with_drams "$1"
elif [ "$#" = 4 ]; then
	for attacker in w a s; do
		_run_with_protections_and_drams \
			"$@" --attacker-type "$attacker"
	done
else
	echo "Usage: $0 [ -m model-name -d dataset | --check-sig-bypass ]"
	exit 1
fi
