#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import utils
import cfg
from eval import evalutils

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('built_so_names', type=str, nargs='+')
    parser.add_argument('-c', '--coeff', type=float, default=None)
    parser.add_argument('-C', '--no-cache', action='store_true')
    parser.add_argument('-w', '--with-fp', action='store_true')
    args = parser.parse_args()

    for so_name in args.built_so_names:

        bi = utils.BinaryInfo.from_fname(so_name)
        print(f'Getting range for {cfg.built_dir}/{bi.fname}')
        kwargs = {}
        if args.no_cache:
            kwargs['use_cache'] = False
        if args.coeff is not None:
            kwargs['extend_coeff'] = args.coeff
        if args.with_fp:
            kwargs['with_fp'] = True
        ret = evalutils.get_sus_score_range(bi, **kwargs)

        if args.with_fp:
            sus_range, fp_range = ret
            print(f'Range: {sus_range}, FP: {fp_range:.2%}')
        else:
            sus_range = ret
            print(f'Range: {sus_range}')
