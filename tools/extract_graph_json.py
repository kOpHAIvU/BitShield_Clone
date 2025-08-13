#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import json

import cfg
import utils

output_root = f'{cfg.debug_dir}/graph-jsons'

def get_graph_json(bi: utils.BinaryInfo):
    '''Deprecated.'''
    import modman
    irmod, params = modman.get_irmod_by_bi(bi)
    json_str, _ir = modman.get_json_and_ir(irmod, params)
    json_str = '\n'.join([x.rstrip() for x in json_str.splitlines()])
    return json.loads(json_str)

if __name__ == '__main__':
    files = sys.argv[1:]
    for f in files:
        print(f'Processing {f}')
        obj = utils.extract_graph_json(f)
        outfile = f'{output_root}/{os.path.basename(f)}.json'
        utils.save_json(obj, outfile)
        print(f'Wrote {outfile}')
