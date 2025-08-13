#! /usr/bin/env python3

import sys
import os
import shutil
import tempfile

import modman
import dataman
import record
import utils
import cfg
import inst
import prune
import cig
from eval import evalutils

def maybe_record_coverages(bi: utils.BinaryInfo, output_dir):
    model_name, dataset = bi.model_name, bi.dataset
    mode = bi.dig.split('_')[0]
    if mode == 'nd':
        print(f'No recording needed')
        return

    epb_mode = inst.cov_mode_configs[mode].epb_mode
    if not epb_mode:
        print(f'No recording needed')
        return

    output_file = f'{output_dir}/{dataset}-{model_name}-{mode}.pth'
    if os.path.exists(output_file):
        print(f'Skipping recording {output_file}')
        return

    print('Recording extra params to {output_file}')
    # Use the last xx% of data in each class, leaving the rest for threshold determining
    data_loader = dataman.get_sampling_benign_loader(
        dataset, bi.input_img_size, 'train', cfg.batch_size, 0.90, start_frac=0.10
    )
    # Get the irmod in epb (extra params building) mode so no extra params are required or loaded
    mod, params = modman.get_irmod(
        model_name, dataset, epb_mode, cfg.batch_size, bi.input_img_size,
        nchannels=bi.nchans, include_extra_params=(epb_mode != mode)
    )
    mod, extra_params_vars, output_defs = inst.instrument_module(
        mod, epb_mode, overall_cov=0, verbose=0
    )
    params = {**params, **modman.create_zeroed_extra_params_dict(extra_params_vars)}
    # Build the epb rtmod for extra params recording
    rtmod, lib = modman.build_module(mod, params)
    # Record extra params based on the data loader
    record.get_record_fn(epb_mode)(rtmod, output_defs, data_loader, outfile=output_file)

def finalise_built_mod(bi, fpath, output_defs, check_acc):
    if output_defs:
        utils.save_json(output_defs, bi.output_defs_file)

    # Check accuracy
    if check_acc and not bi.dataset.startswith('fake') and not bi.model_name.startswith('dcgan'):
        assert evalutils.check_so_acc(fpath) > 0.6

def get_dig_instrumented_mod(bi: utils.BinaryInfo):
    model_name, dataset = bi.model_name, bi.dataset
    mode = bi.dig.split('_r')[0]  # r for random
    frac = float(bi.dig.split('_r')[1]) if '_r' in bi.dig else 1.
    prune_frac = 1.0 - frac  # frac to insert -> frac to prune

    if prune_frac > 0:
        if model_name.startswith('Q'):
            qmod, qparams = modman.get_irmod(
                model_name, dataset, mode, cfg.batch_size, bi.input_img_size, nchannels=bi.nchans
            )
            omod, oparams = modman.get_irmod(
                model_name[1:], dataset, mode, cfg.batch_size, bi.input_img_size, nchannels=bi.nchans
            )

            skipped_eps, skipped_neurons = prune.get_ignored_components(
                oparams, prune_frac, as_eps=True, irmod=omod, is_random=True
            )
            qparams = prune.ignored_neurons_applied_to_extra_params(
                qparams, None, skipped_neurons, mode, eps_mode=True
            )

            params = qparams
            mod, _, output_defs = inst.instrument_module(
                qmod, mode, overall_cov=1, skipped_weights=skipped_eps,
                skipped_neurons=skipped_neurons, skip_as_eps=True, verbose=0
            )
        else:
            mod, params = modman.get_irmod(
                model_name, dataset, mode, cfg.batch_size, bi.input_img_size, nchannels=bi.nchans
            )

            skipped_weights, skipped_neurons = prune.get_ignored_components(
                params, prune_frac, is_random=True
            )
            params = prune.ignored_neurons_applied_to_extra_params(params, mod, skipped_neurons, mode)

            mod, _, output_defs = inst.instrument_module(
                mod, mode, overall_cov=1, skipped_weights=skipped_weights, skipped_neurons=skipped_neurons, verbose=0
            )
    else:  # No pruning
        mod, params = modman.get_irmod(
            bi.model_name, bi.dataset, mode, cfg.batch_size, bi.input_img_size, nchannels=bi.nchans
        )
        output_defs = bi.default_output_defs
        if mode != 'nd':
            mod, _, output_defs = inst.instrument_module(mod, mode, overall_cov=1, verbose=0)

    return mod, params, output_defs

def maybe_build_tvm_mod_dig_only(bi: utils.BinaryInfo, check_acc):
    output_file = f'{cfg.built_dir}/{bi.fname}'
    target = modman.targets['avx2' if bi.avx else 'llvm']

    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return
    print(f'Building {output_file}')

    mod, params, output_defs = get_dig_instrumented_mod(bi)

    _rtmod, _lib = modman.build_module(
        mod, params, export_path=output_file, cig_make_space=False,
        target=target, opt_level=bi.opt_level, is_qnn=bi.model_name.startswith('Q')
    )

    finalise_built_mod(bi, output_file, output_defs, check_acc)

def maybe_build_classic_cig_tvm(bi, check_acc, patcher_kwargs={}, plan_fn_kwargs={}, planner_kwargs={}):
    output_file = f'{cfg.built_dir}/{bi.fname}'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return
    print(f'Patching to generate {output_file}')

    output_defs = None
    # TODO: Maybe change nc/ncnp -> ncpp/nc?
    if bi.cig == 'nc':
        # Build the .so lib with extra params embedded and only do CIG prepatch
        mod, params, output_defs = get_dig_instrumented_mod(bi)
        target = modman.targets['avx2' if bi.avx else 'llvm']
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpout = f'{tmpdir}/{bi.fname}'
            _rtmod, _lib = modman.build_module(
                mod, params, export_path=tmpout, cig_make_space=True,
                target=target, opt_level=bi.opt_level, is_qnn=bi.model_name.startswith('Q')
            )
            cig.prepatch(tmpout, output_file, bi.cig_spots_file)
    else:
        # Otherwise, we will make use of the prepatched nc mod
        patcher = cig.CIGPatcher(bi._replace(cig='nc'), **patcher_kwargs)
        planner = cig.get_planner_cls(bi.cig)(**planner_kwargs)
        patcher.plan(planner, **plan_fn_kwargs)
        patcher.patch(output_file)

    finalise_built_mod(bi, output_file, output_defs, check_acc)

def maybe_build_ccN_cig_tvm(cig_name, bi: utils.BinaryInfo, check_acc):
    output_file = f'{cfg.built_dir}/{bi.fname}'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return
    print(f'Patching to generate {output_file}')

    irmod, params, output_defs = get_dig_instrumented_mod(bi)
    builder_cls = {
        'cc1': cig.CoopCIGV1ModuleBuilder,
        'cc2': cig.CoopCIGV2ModuleBuilder,
    }[cig_name]
    builder_cls().build(
        bi, irmod, params, output_file
    )

    finalise_built_mod(bi, output_file, output_defs, check_acc)

def maybe_build_tvm_mod_cig(bi, check_acc, **kwargs):
    if bi.cig.startswith('cc'):
        return maybe_build_ccN_cig_tvm(bi.cig, bi, check_acc, **kwargs)
    return maybe_build_classic_cig_tvm(bi, check_acc, **kwargs)

def maybe_build_glow_mod(bi, output_dir, weights_out_dir, check_acc):
    output_file = f'{output_dir}/{bi.fname}'
    weights_file = f'{weights_out_dir}/{bi.fname}.weights.bin'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return

    print(f'Building {output_file} with {weights_file}')
    utils.ensure_dir_of(output_file)
    utils.ensure_dir_of(f'{weights_out_dir}/.')
    with tempfile.TemporaryDirectory() as tmpdir:
        modman.build_glow_model(
            bi.model_name, bi.dataset, cfg.batch_size, bi.input_img_size, tmpdir, nchannels=bi.nchans
        )
        shutil.move(f'{tmpdir}/model.so', output_file)
        shutil.move(f'{tmpdir}/model.weights.bin', weights_file)

    # Check accuracy
    if check_acc and not bi.dataset.startswith('fake') and not bi.model_name.startswith('dcgan'):
        assert evalutils.check_so_acc(output_file) > 0.55  # TODO: Accuracy of glow

def maybe_build_nnf_mod(bi, output_dir, data_out_dir, check_acc):
    output_file = f'{output_dir}/{bi.fname}'
    data_file = f'{data_out_dir}/{bi.fname}.data.tar'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return

    print(f'Building {output_file} with {data_file}')
    utils.ensure_dir_of(output_file)
    utils.ensure_dir_of(f'{data_out_dir}/.')
    with tempfile.TemporaryDirectory() as tmpdir:
        modman.build_nnf_model(
            bi.model_name, bi.dataset, cfg.batch_size, bi.img_size, tmpdir, nchannels=bi.nchans
        )
        shutil.move(f'{tmpdir}/libnnfusion_cpu_rt.so', output_file)
        shutil.move(f'{tmpdir}/data.tar', data_file)

    # TODO: Check accuracy
    # if check_acc and not bi.dataset.startswith('fake') and not bi.model_name.startswith('dcgan'):
        # assert evalutils.check_so_acc(output_file) > 0.6

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, default='tvm')
    parser.add_argument('-v', '--compiler_ver', type=str, default='main')
    parser.add_argument('-m', '--model', type=str, default='resnet50')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10')
    parser.add_argument('-X', '--no-avx', action='store_false', dest='avx')
    parser.add_argument('-A', '--no-check-acc', action='store_false', dest='check_acc')
    parser.add_argument('-O', '--opt-level', type=int, default=3)
    parser.add_argument('-i', '--cig', default='nc')
    parser.add_argument('-I', '--dig', default='nd')
    args = parser.parse_args()

    bis_to_build = cfg.all_build_bis
    if len(sys.argv) > 1 and not (len(sys.argv) == 2 and not args.check_acc):
        bi = utils.BinaryInfo(
            args.compiler, args.compiler_ver, args.model, args.dataset,
            args.cig, args.dig, args.avx, args.opt_level
        )
        bis_to_build = [bi]

    for bi in bis_to_build:
        print(f'{bi.compiler=} {bi.compiler_ver=} {bi.model_name=} {bi.dataset=} {bi.cig=} {bi.dig=} {bi.avx=} {bi.opt_level=} {bi.input_img_size=}')
        try:
            if bi.has_dig:
                maybe_record_coverages(bi, cfg.coverages_dir)

            if bi.compiler == 'tvm':
                if bi.cig == 'ncnp':
                    maybe_build_tvm_mod_dig_only(bi, args.check_acc)
                else:
                    maybe_build_tvm_mod_cig(bi, args.check_acc)
            elif bi.compiler == 'glow':
                maybe_build_glow_mod(bi, cfg.built_dir, cfg.built_aux_dir, args.check_acc)
            elif bi.compiler == 'nnfusion':
                maybe_build_nnf_mod(bi, cfg.built_dir, cfg.built_aux_dir, args.check_acc)
            else:
                raise ValueError(f'Unknown compiler {bi.compiler}')
        except FileNotFoundError as e:
            if cfg.models_dir in e.filename:
                utils.warn(f'Skipping building due to lacking file(s): {e.filename}')
            else:
                raise
        print('-----------------')
