#! /usr/bin/env python3

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import analysis

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_file')
    parser.add_argument('-o', '--outfile')
    args = parser.parse_args()

    df = analysis.extract_dfs(args.sweep_file, quiet=True)[0]
    bitlines = (
        df[['g_offset', 'bitidx']]
        .astype(str)
        .apply(','.join, axis=1)
        .tolist()
    )

    if not args.outfile:
        [print(bitline) for bitline in bitlines]
    else:
        with open(args.outfile, 'w+') as f:
            [f.write(f'{bitline}\n') for bitline in bitlines]
