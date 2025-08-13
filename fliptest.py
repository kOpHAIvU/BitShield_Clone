#! /usr/bin/env python3

import os
import resource
import argparse
import random
import json

# TODO: If we import utils.py the program may crash with -6
# See https://discuss.tvm.apache.org/t/free-invalid-pointer-aborted/11357
import utils

import cfg
import dataman
from eval import evalutils
from typing import Any, NamedTuple
from fliputils import *
import utils

class ClassifierFlipResult(NamedTuple):
    # For legacy code
    base10_offset: int
    bitidx: int
    correct_pct: float
    top_labels: Any

def main(args):
    random.seed(args.seed)

    if args.pause:
        print(f'PID: {os.getpid()}')
        input('Press enter to continue...')

    bi = utils.BinaryInfo(
        args.compiler, args.compiler_version, args.model_name, args.dataset,
        args.cig, args.dig,
        not args.no_avx, args.opt_level
    )
    lmi = load_mod(bi, fpath=args.fpath)

    if args.gan:
        gan_evaluator = evalutils.GANEvaluator(bi.dataset, device=args.device)
        gan_evaluator.set_ref(lmi.mod)
    else:
        if args.no_fast:
            val_loader = dataman.get_benign_loader(args.dataset, bi.input_img_size, 'test', cfg.batch_size)
        else:
            val_loader = dataman.get_sampling_loader_v2(args.dataset, bi.input_img_size, 'test', cfg.batch_size, n_per_class=args.ncls_samples)

    flips = []
    if args.nbits: # Random flips
        flips = random_flip_bits(lmi, args.nbits)
    elif args.byteidx is not None and args.bitidx is not None: # Specific flips
        flips = [new_flip(args.byteidx, args.bitidx)]
        flip_bits(lmi, flips, quiet=args.quiet)
    elif args.flip_pairs: # One or more specific flips
        flip_bits(lmi, [new_flip(*x.split(',')) for x in args.flip_pairs], quiet=args.quiet)
    elif args.pause:
        utils.warn('No flips specified')
    else:
        assert False, 'Nothing to do and not pausing'

    if args.pause:
        input('Flipping done! Press enter to continue...')

    if args.gan:
        outputs = gan_evaluator.get_gan_outputs(
            lmi.mod, debug=True, debug_fname=f'{flips[0].byteidx:x}_{flips[0].bitidx}'
        )
        top_labels, lchange, lpips_avg, fid = gan_evaluator.eval(outputs)
        print(f'Top labels: {top_labels.flatten().tolist()}')
        print(f'Label change: {lchange:.2%}, LPIPS: {lpips_avg:.2f}, FID: {fid:.2f}')
        return

    if bi.has_dig:
        acc, top_labels, sus_score = evalutils.check_accuracyv2(lmi.mod, val_loader, sus_score=True)
        print(f'Suspicious score: {sus_score:.2f}')
    else:
        acc, top_labels = evalutils.check_accuracyv2(lmi.mod, val_loader, sus_score=False)
    print(f'Acc. after flipping: {acc:.2%}')
    print('Top labels:', [x[0] for x in top_labels])

    if args.json:
        print(json.dumps({
            'acc': acc,
            'sus_score': sus_score if bi.has_dig else None,
        }))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, default='tvm')
    parser.add_argument('-v', '--compiler-version', type=str, default='main')
    parser.add_argument('-m', '--model-name', type=str, default='googlenet')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10')
    parser.add_argument('-i', '--cig', default='ncnp')
    parser.add_argument('-I', '--dig', default='nd')
    parser.add_argument('-s', '--seed', help='Random seed', type=int, default=42)
    parser.add_argument('-n', '--nbits', help='Number of bits to flip', type=int)
    parser.add_argument('-p', '--pause', help='Pause before flipping', action='store_true')
    parser.add_argument('-b', '--byteidx', default=None)
    parser.add_argument('-B', '--bitidx', default=None)
    parser.add_argument('-P', '--flip-pairs', nargs='*', help='byteidx,bitidx pairs to flip')
    parser.add_argument('-X', '--no-avx', action='store_true', default=False)
    parser.add_argument('-O', '--opt-level', type=int, default=3)
    parser.add_argument('-g', '--gan', action='store_true', default=False)
    parser.add_argument('-f', '--fpath', help='Override the path to the built .so file')
    parser.add_argument('-F', '--no-fast', action='store_true', help="Don't use fewer images for faster testing")
    parser.add_argument('-N', '--ncls-samples', type=int, default=10, help='Number of test samples to use for each class')
    parser.add_argument('-J', '--json', action='store_true', help='Append a line of machine-readable JSON output')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-D', '--device', type=str, default='cpu')
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    main(args)
