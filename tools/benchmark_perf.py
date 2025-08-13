#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import argparse

from eval import evalutils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pause', action='store_true')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    if args.pause:
        print(f'PID: {os.getpid()}')

    for f in args.files:
        print(f'Benchmarking {f}')
        if args.pause:
            input('Press enter to continue...')
        perf = evalutils.benchmark_perf(os.path.realpath(f))
        print(perf)
        print(f'----------------------------------')
