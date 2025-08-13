#! /usr/bin/env python3

import os
import subprocess
from dataclasses import dataclass
import argparse
from enum import Enum
from typing import Any, Set, NamedTuple, Iterable, Dict, List
import pickle
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from functional import seq
import torch

import analysis
import cfg
import utils
from eval import evalutils
import modman
import dataman
from support import torchdig

PAGE_SIZE = 4096

class BitFlip(NamedTuple):
    phy_byte_offset: int
    bitidx: int
    # If 1, the flip direction is from 0 to 1
    direction: int

    @property
    def phy_pagenum(self):
        return self.phy_byte_offset // PAGE_SIZE

    @property
    def in_page_offset(self):
        return self.phy_byte_offset % PAGE_SIZE

@dataclass
class PhyPage:
    '''A physical page.'''
    pagenum: int

class MemoryModel:
    def __init__(self, size_mib) -> None:
        self.size_mib = size_mib
        self.templates: Set[BitFlip] = set()

    def init_vuln_bits(self, rng, vuln_frac, zero_one_frac):
        self.rng = rng or np.random.default_rng()
        vuln_bit_locs = set()
        for _ in range(int(self.size_mib * 1024**2 * 8 * vuln_frac)):
            while True:
                phy_byte_offset = self.rng.integers(0, self.size_mib * 1024**2)
                bitidx = self.rng.integers(0, 8)
                if (phy_byte_offset, bitidx) not in vuln_bit_locs:
                    vuln_bit_locs.add((phy_byte_offset, bitidx))
                    break
        self.templates.clear()
        for phy_byte_offset, bitidx in vuln_bit_locs:
            self.templates.add(BitFlip(
                phy_byte_offset, bitidx,
                self.rng.choice((0, 1), p=(1-zero_one_frac, zero_one_frac))
            ))
        print(f'Initialised {len(self.templates)} templates')

    def has_template(self, phy_byte_offset, bitidx, direction):
        return BitFlip(phy_byte_offset, bitidx, direction) in self.templates

class AttackConsequence(str, Enum):
    NOT_APPLICABLE = 'not_applicable'
    SUCCESS = 'success'
    NO_EFFECT = 'no_effect'
    DIG_DETECTED = 'dig_detected'
    CIG_DETECTED_OR_CRASHED = 'cig_detected_or_crashed'

class VictimBinaryInfo:
    def __init__(self, bi: utils.BinaryInfo, ncls_samples: int) -> None:
        self.bi = bi
        self.so_path = f'{cfg.built_dir}/{bi.fname}'
        self.analysis = bi.get_analysis()
        self.offset2inst = analysis.get_offset2inst(self.analysis)

        sweepfile = f'{cfg.sweep_dir}/{bi.fname.replace(".so", "-sweep.pkl")}'
        dfs = analysis.extract_dfs(sweepfile)
        assert len(dfs) == 1
        self.sweep_df = dfs[0]
        # We use the ft_offset for attack simulations as it's  automatically
        # compatible with fliptest.py.
        self.sweep_df['ft_b10offset'] = self.sweep_df['ft_offset'].map(lambda x: int(x, 16))
        # Calculate the in-page offsets to match against the templates
        self.sweep_df['in_page_offset'] = self.sweep_df['ft_b10offset'] % PAGE_SIZE

        # The range of sus_score considered normal
        self.sus_score_range = (-2**31, 2**31 - 1)
        if bi.has_dig:
            self.sus_score_range = evalutils.get_sus_score_range(bi)
        # Original accuracy
        self.orig_acc = evalutils.check_so_acc(self.so_path, n_per_class=ncls_samples)

    def apply_memory_templates(self, memmodel: MemoryModel) -> pd.DataFrame:
        '''Applies the memory templates in the given memory model to obtain the
        actually flippable bits.'''
        # We turn the templates into a dataframe then do an inner join with
        # self.sweep_df.
        templates = set(
            (x.in_page_offset, x.bitidx, x.direction)
            for x in memmodel.templates
        )
        templates_df = pd.DataFrame(
            # Use the same column names as self.sweep_df
            templates, columns=['in_page_offset', 'bitidx', 'flip_direction']
        )
        flippable_df = pd.merge(
            self.sweep_df, templates_df, how='inner',
            on=['in_page_offset', 'bitidx', 'flip_direction']
        )
        # For some flips, we don't have flip direction info (e.g., because
        # Ghidra analysis doesn't see bytes that won't be executed), so we
        # don't consider the direction of these flips in order to represent a
        # stronger attacker.
        flippable_df = pd.concat([
            flippable_df, templates_df.drop(columns=['flip_direction']).merge(
                self.sweep_df[self.sweep_df['flip_direction'] == -1],
                on=['in_page_offset', 'bitidx'], how='inner'
            )
        ])
        return flippable_df

    def eval_attack_consequence(self, acc, sus_score):
        if sus_score is not None:
            if sus_score < self.sus_score_range[0] or \
                sus_score > self.sus_score_range[1]:
                return AttackConsequence.DIG_DETECTED
        if abs(acc - self.orig_acc) / self.orig_acc < 0.03:  # 3% accuracy change
            return AttackConsequence.NO_EFFECT
        return AttackConsequence.SUCCESS

@dataclass
class AttackResult:
    n_flips: int
    attack_plan: Iterable[BitFlip]
    correct_pct: float
    sus_score: float
    consequence: AttackConsequence

    def __setattr__(self, __name: str, __value: Any) -> None:
        assert __name in self.__dataclass_fields__, f'Cannot set attribute {__name}'
        super().__setattr__(__name, __value)

@dataclass
class AttackResultV2(AttackResult):
    evasion_frac: float

    @staticmethod
    def empty(plan):
        return AttackResultV2(len(plan), plan, -1., -1., None, 0.)

class Attacker:
    def __init__(
        self, attacker_type: str, ncls_samples: int, bi: utils.BinaryInfo
    ) -> None:
        self.attacker_type = attacker_type
        self.ncls_samples = ncls_samples
        self.vbi = VictimBinaryInfo(bi, ncls_samples)

    def run_validator(self, flips: Iterable[BitFlip]):
        bi = self.vbi.bi
        fliptest_path = f'{os.path.dirname(os.path.realpath(__file__))}/fliptest.py'
        # Translate offsets to ones accepted by fliptest.py, then run it
        try:
            return json.loads(subprocess.check_output([
                'python3', fliptest_path, '--json', '--quiet',
                '--compiler', bi.compiler, '--compiler-version', bi.compiler_ver,
                '--model-name', bi.model_name, '--dataset', bi.dataset,
                '--cig', bi.cig, '--dig', bi.dig,
                *(['--no-avx'] if not bi.avx else []), f'--opt-level={bi.opt_level}',
                '--ncls-samples', str(self.ncls_samples),
                '--flip-pairs', *[f'{x.phy_byte_offset:x},{x.bitidx}' for x in flips]
            ], text=True, timeout=10).splitlines()[-1])
        except subprocess.CalledProcessError as _:
            # Process crashed
            return None
        except subprocess.TimeoutExpired as _:
            # Process timed out
            return None

    def get_attack_plans(self, memmodel: MemoryModel) -> Iterable[Iterable[BitFlip]]:
        '''Gets all the attack plans for the given memory model.'''
        assert self.attacker_type in {'a', 's'}
        flippable_df = self.vbi.apply_memory_templates(memmodel)
        print(f'Narrowed down {len(self.vbi.sweep_df)} flips to {len(flippable_df)} with templates')
        flippable_df = utils.thread_first(
            flippable_df,
            (analysis.filter_df, 'sus_score', self.vbi.sus_score_range),  # Non DIG-detectable
            (analysis.filter_df, 'correct_pct', (1/self.vbi.bi.nclasses*100, None)),
            (analysis.filter_df, 'acc_drop_pct', (3, None)),
        ).sort_values('acc_drop_pct', ascending=(self.attacker_type == 's'))
        all_flips = []
        # A simple way to limit one bit per instruction in multi-flip attacks
        used_inst_offsets = set()
        flips_pool = []
        for _, row in flippable_df.iterrows():
            flip = BitFlip(row['ft_b10offset'], row['bitidx'], row['flip_direction'])
            all_flips.append(flip)
            inst = self.vbi.offset2inst.get(row['base10_offset'])
            inst_offset = inst['base10_offset'] if inst else -1
            if inst_offset in used_inst_offsets:
                continue
            used_inst_offsets.add(inst_offset)
            flips_pool.append(flip)
        return [
            [x] for x in all_flips
        ] + [
            flips_pool[:i+1] for i in range(len(flips_pool))
        ]

    def _simulate_wbbfa(self, memmodel: MemoryModel):
        # Unlike code-based attackers, WBBFA's logic is delegated to its own module
        from wbbfa import WBBFA  # Circular import
        assert self.attacker_type in {'w', 'r'}
        bi = self.vbi.bi
        tmod = modman.get_torch_mod(bi.model_name, bi.dataset)
        if bi.has_dig:
            tmod = torchdig.DIGProtectedModule(tmod)

        train_loader = dataman.get_benign_loader(
            bi.dataset, bi.input_img_size, 'train', bi.batch_size
        )
        test_loader = dataman.get_sampling_loader_v2(
            bi.dataset, bi.input_img_size, 'test', bi.batch_size,
            n_per_class=self.ncls_samples
        )

        attacker = WBBFA(memmodel=memmodel)
        tmod, nflipped, evasion_frac = attacker.attack(
            tmod, train_loader, test_loader, random_flips=(self.attacker_type == 'r'),
            sus_score_range=torchdig.cached_sus_score_ranges[(bi.model_name, bi.dataset)]
        )

        results = [AttackResult(0, [], self.vbi.orig_acc * 100, -1., AttackConsequence.NOT_APPLICABLE)]
        if nflipped > 0 or evasion_frac > 0:
            result = AttackResultV2.empty([BitFlip(-1, -1, -1)] * nflipped)
            results.append(result)
            result.consequence = AttackConsequence.DIG_DETECTED
            result.evasion_frac = evasion_frac
            if nflipped > 0:
                result.consequence = AttackConsequence.SUCCESS
                result.correct_pct = attacker.get_acc(tmod, test_loader) * 100
                if bi.has_dig:
                    result.sus_score = tmod.calc_sus_score(next(iter(test_loader))[0]).item()

        print(f'Final attack result: {results[-1]}')
        return results

    def simulate(self, memmodel: MemoryModel, max_cbbfa_plans=50):
        '''Runs the attack simulation on the given memory model by testing out
        all the attack plans using the validator. Returns attack metrics.'''
        if self.attacker_type in {'w', 'r'}:
            return self._simulate_wbbfa(memmodel)
        # Initialise the results with the original accuracy
        results = [AttackResult(0, [], self.vbi.orig_acc * 100, -1., AttackConsequence.NOT_APPLICABLE)]
        plans = self.get_attack_plans(memmodel)
        print(f'Got {len(plans)} attack plans (limited to {max_cbbfa_plans})')
        plans = plans[:max_cbbfa_plans]
        for plan_flips in tqdm(plans):
            result = AttackResultV2.empty(plan_flips)
            results.append(result)
            val_ret = self.run_validator(plan_flips)
            if val_ret is None:
                result.consequence = AttackConsequence.CIG_DETECTED_OR_CRASHED
                continue
            result.correct_pct = val_ret['acc'] * 100
            result.sus_score = val_ret['sus_score']
            result.consequence = self.vbi.eval_attack_consequence(val_ret['acc'], val_ret['sus_score'])
            if result.consequence == AttackConsequence.SUCCESS:
                # If the attack is successful, stop trying
                break

        print(f'Final attack result: {results[-1]}')
        return results

@dataclass
class AttackSimResult:
    args: argparse.Namespace
    retcoll_map: Dict[int, List[AttackResult]]  # seed -> results

    def __setattr__(self, __name: str, __value: Any) -> None:
        assert __name in self.__dataclass_fields__, f'Cannot set attribute {__name}'
        super().__setattr__(__name, __value)

class AttackSimResultUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return globals()[name] if name in globals() else super().find_class(module, name)

def load_attack_sim_result(fname) -> AttackSimResult:
    with open(fname, 'rb') as f:
        return AttackSimResultUnpickler(f).load()

def _do_check_sig_bypass(memmodel, threshold=4) -> bool:
    ntemplates_in_page_1 = np.zeros((4096, 8), dtype=int)
    ntemplates_in_page_0 = np.zeros((4096, 8), dtype=int)
    for t in memmodel.templates:
        if t.direction == 1:
            ntemplates_in_page_1[t.in_page_offset][t.bitidx] += 1
        else:
            ntemplates_in_page_0[t.in_page_offset][t.bitidx] += 1
    # If there's template on either direction for a bit, the attacker may be able to change it
    ntemplates_in_page = np.minimum(ntemplates_in_page_0, ntemplates_in_page_1)
    return (
        ntemplates_in_page
        .reshape(-1, 32)  # Each checksum has 4 bytes
        [::18,:]          # Distance between checksums
        .all(axis=1)      # Is every bit in the checksum flippable?
        .sum()            # How many checksums can be flipped?
    ) <= threshold        # Threshold should be no more than the total number of checksums

def check_sig_bypass(memmodel: MemoryModel, args, vuln_frac):
    '''Run a check to see if there are enough templates to allow direct SIG
    bypass.'''
    for i in tqdm(range(args.nexps)):
        seed = args.seed_start + i
        rng = np.random.default_rng(seed)
        memmodel.init_vuln_bits(rng, vuln_frac, args.zero_one_pct / 100)
        assert _do_check_sig_bypass, f'Check failed at seed {seed}!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, default='tvm')
    parser.add_argument('-v', '--compiler-version', type=str, default='main')
    parser.add_argument('-m', '--model-name', type=str, default='googlenet')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10')
    parser.add_argument('-i', '--cig', default='ncnp')
    parser.add_argument('-I', '--dig', default='nd')
    parser.add_argument('-t', '--attacker-type', choices=['a', 's', 'w', 'r'], default=None)
    parser.add_argument('-p', '--vuln-pct', type=float, default=26.4e-6 * 100, help='Percentage (0-100) of vulnerable bits in the DRAM module')
    parser.add_argument('-P', '--zero-one-pct', type=float, default=50.75, help='Percentage (0-100) of 0->1 flips')
    parser.add_argument('-s', '--seed-start', help='Initial random seed', type=int, default=42)
    parser.add_argument('-n', '--nexps', help='Number of experiments', type=int, default=1)
    parser.add_argument('-S', '--size-mib', help='Size of the DRAM module in MiB', type=int, default=256)
    parser.add_argument('-N', '--ncls-samples', type=int, default=None, help='Number of test samples to use for each class')
    parser.add_argument('--check-sig-bypass', action='store_true', help='Only check the possibilities for SIG bypass')
    parser.add_argument('-o', '--out-dir', type=str, default=cfg.attacksim_dir)
    parser.add_argument('-O', '--discard-output', action='store_true')
    parser.add_argument('-k', '--skip-existing', action='store_true')
    args = parser.parse_args()

    assert args.attacker_type or args.check_sig_bypass

    if args.discard_output:
        utils.warn('Output will be discarded')

    bi = utils.BinaryInfo(
        args.compiler, args.compiler_version, args.model_name, args.dataset,
        args.cig, args.dig, True, 3
    )

    if args.ncls_samples is None:
        args.ncls_samples = {
            10: 13,
        }.get(bi.fast_n_per_class, bi.fast_n_per_class)

    vuln_frac = args.vuln_pct / 100
    outfile = f'{args.out_dir}/{bi.fname}-{vuln_frac*1e6:.2f}-{args.attacker_type}.pkl'
    utils.ensure_dir_of(outfile)

    if args.skip_existing and os.path.exists(outfile):
        utils.warn(f'Skipping existing {outfile}')
        exit(0)

    memmodel = MemoryModel(args.size_mib)

    if args.check_sig_bypass:
        check_sig_bypass(memmodel, args, vuln_frac)
        print('Bypass is not poissible')
        exit(0)

    attacker = Attacker(args.attacker_type, args.ncls_samples, bi)

    retcolls = {}
    for i in range(args.nexps):
        seed = args.seed_start + i
        print(f'Running experiment {i+1}/{args.nexps} with seed {seed}')
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        memmodel.init_vuln_bits(rng, vuln_frac, args.zero_one_pct / 100)
        results = attacker.simulate(memmodel)
        retcolls[seed] = results

    if not args.discard_output:
        ret = AttackSimResult(args, retcolls)
        utils.save(ret, outfile, merge=False)
        print(f'Saved to {outfile}')
