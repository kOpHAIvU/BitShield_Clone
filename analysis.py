from functools import reduce
import os
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
from scipy.stats import mode
import pandas as pd
import math
from tqdm import tqdm

import flipsweep
import utils
import cfg

class BitFlip(NamedTuple):
    base10_offset: int
    bitidx: int
    # If 1, the flip direction is from 0 to 1
    direction: int

    def add_offset(self, offset):
        return self._replace(base10_offset=self.base10_offset + offset)

class Page:
    '''A page represents a chunk of data on which we want to perform bit flips.
    The page can be placed into the memory system, at which point the
    placement_offset should be set to a valid value.'''

    def __init__(self, pagenum) -> None:
        self.pagenum = pagenum
        # A map from BitFlip to a list of placement offsets for this page that
        # allow it to be flipped.
        self.bits_ploffs: Dict[BitFlip, List[int]] = {}
        self.placement_offset: int = None

    def unplace(self):
        self.placement_offset = None

    @property
    def placed(self):
        return self.placement_offset is not None

class RHModel:

    PAGE_SIZE = 4096

    def __init__(
        self, victim_bi: utils.BinaryInfo, target_bits: List[BitFlip],
        dram_profile_path=cfg.dram_profile_path
    ):
        # Bits that the attacker wants to flip
        self.target_bits = target_bits
        dram_flips = utils.load_json(dram_profile_path)['sweeps'][0]['flips']['details']
        # Bits that are flippable in DRAM
        self.flippable_bits = set()
        for flip in tqdm(dram_flips, desc='Loading DRAM profile'):
            bitidxs = [i for i in range(8) if flip['bitmask'] & (1 << i)]
            for bitidx in bitidxs:
                self.flippable_bits.add(BitFlip(
                    base10_offset=int(flip['addr'], 16), bitidx=bitidx,
                    direction=int(bool(flip['data'] & (1 << bitidx))),
                ))

        # Align to page size
        self.sweep_region_start = min(x.base10_offset for x in self.flippable_bits) // self.PAGE_SIZE * self.PAGE_SIZE
        # Exclusive
        self.sweep_region_end = max(x.base10_offset for x in self.flippable_bits) // self.PAGE_SIZE * self.PAGE_SIZE + self.PAGE_SIZE
        assert is_aligned_to_size(self.sweep_region_start, self.PAGE_SIZE)
        assert is_aligned_to_size(self.sweep_region_end, self.PAGE_SIZE)

        ana_path = f"{cfg.analysis_dir}/{victim_bi.fname}-analysis.json"
        mmap = utils.load_json(ana_path)['memory_map']
        self.init_start = mmap['.init']['base10_start']
        self.npages = math.ceil(sum(
            mmap[x]['size'] for x in ['.init', '.plt', '.plt.got', '.text']
        ) / self.PAGE_SIZE)
        self.pages = [self._new_page(i) for i in tqdm(range(self.npages), desc='Init pages')]

    def get_bit_page_info(self, bit):
        '''Returns (page_number, in_page_offset)'''
        return divmod(bit.base10_offset - self.init_start, self.PAGE_SIZE)

    def get_page_containing(self, bit):
        return self.pages[self.get_bit_page_info(bit)[0]]

    def _new_page(self, pagenum) -> Page:
        page = Page(pagenum)
        bits_in_page = [x for x in self.target_bits if self.get_bit_page_info(x)[0] == pagenum]
        min_ploff = self.sweep_region_start - self.init_start
        max_ploff = self.sweep_region_end - self.init_start - self.PAGE_SIZE
        for ploff in tqdm(
            range(min_ploff, max_ploff + 1, self.PAGE_SIZE),
            desc=f'Init page {pagenum+1}/{self.npages}', leave=False
        ):
            placed_bits = {x.add_offset(ploff) for x in bits_in_page}
            flippable_placed_bits = placed_bits & self.flippable_bits
            for placed_bit in flippable_placed_bits:
                page.bits_ploffs.setdefault(placed_bit.add_offset(-ploff), []).append(ploff)
        return page

    def unplace_all_pages(self):
        [p.unplace() for p in self.pages]

    def get_avail_placements(self, bit) -> Tuple[int, List[int]]:
        '''Returns (page_number, [placement_offsets])'''
        page = self.get_page_containing(bit)
        assert not page.placed
        # Make sure the ploffs are not already occupied
        ploffs = [
            ploff for ploff in page.bits_ploffs.get(bit, [])
            if not any(other_p.placement_offset == ploff for other_p in self.pages)
        ]
        return page.pagenum, ploffs

    def place_page(self, pagenum, placement_offset):
        page = self.pages[pagenum]
        assert not page.placed
        page.placement_offset = placement_offset

def is_aligned_to_size(base10_offset, size):
    return base10_offset % size == 0

def get_filter_key_range(filter_mode):
    filter_key, filter_range = {
        'none': (None, None),
        'acc': ('correct_pct', (5, 15)),
        'label': ('top_label_change_pct', (85, 999)),
        'lpips': ('lpips_avg', NotImplemented),
        'fid': ('fid', NotImplemented),
    }[filter_mode]
    return filter_key, filter_range

def superbits_fname2bis(fname):
    bi = utils.BinaryInfo.from_fname(os.path.basename(fname).replace('-superbits.csv', '.x'))
    base_dataset, other_dataset = bi.dataset.split('@')
    base_bi, other_bi = bi._replace(dataset=base_dataset), bi._replace(dataset=other_dataset)
    return base_bi, other_bi

def get_offset2inst(loaded_analysis):
    offset2inst = {}
    for _fname, fn in loaded_analysis['fns'].items():
        for inst in fn['insts']:
            for i in range(inst['nbytes']):
                offset2inst[inst['base10_offset'] + i] = inst
    return offset2inst

def _create_df_from_sweep(path, quiet=False, use_cache=True):
    '''Creates a dataframe from a sweep file containing the sweep data of one
    or more binaries. Cells may contain lists of values.'''

    sweepret = flipsweep.load_sweep_result(path)
    if not quiet:
        print('Args:', sweepret.args)
    any_bi = utils.BinaryInfo(
        sweepret.args.compiler, sweepret.args.compiler_version,
        sweepret.args.model_name, sweepret.args.datasets[0],
        sweepret.args.cig, sweepret.args.dig,
        not sweepret.args.no_avx, sweepret.args.opt_level
    )

    if use_cache:
        cache_path = f'{cfg.cache_dir}/analysis-dfs/' \
            f'{utils.sha1sum_file(path)}-{os.path.basename(path)}.pkl'
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path), any_bi

    ana = any_bi.get_analysis()
    init_start = ana['memory_map']['.init']['base10_start']
    fns = ana['fns']
    offset2inst = get_offset2inst(ana)

    total_nbits = len(list(flipsweep.get_all_bytes(fns))) * 8
    total_cf_nbits = len(list(flipsweep.get_all_bytes({k: v for k, v in fns.items() if v['is_compute_fn']}))) * 8
    assessed_nbits = len(sweepret.retcoll_map) * 8
    if not quiet:
        print(f'Assessed {assessed_nbits}/{total_nbits} bits ({assessed_nbits / total_nbits * 100:.2f}%)'
              f' (total compute fn bits: {total_cf_nbits})')

    results = pd.DataFrame(sweepret.flat_result_colls).to_dict(orient='records')
    for result in results:
        result['g_offset'] = hex(result['base10_offset'])
        inst_info = offset2inst.get(result['base10_offset'])
        if inst_info is None:
            result['asm'] = None
            result['flip_direction'] = -1
            continue
        result['asm'] = inst_info['asm']
        imask = int(inst_info['imask'].split('_')[result['base10_offset'] - inst_info['base10_offset']], 16)
        result['opcode_flipped'] = bool(imask & (1 << result['bitidx']))
        local_byte_offset = result['base10_offset'] - inst_info['base10_offset']
        byte = int(inst_info['bytes'].split('_')[local_byte_offset], 16)
        result['flip_direction'] = 1 - ((byte >> result['bitidx']) & 1)  # 1 for 0->1

    df = pd.DataFrame(results)
    df['flip_direction'] = df['flip_direction'].astype(int)
    df['ft_offset'] = df['base10_offset'].map(lambda x: hex(x - init_start))
    df['fn'] = df['base10_offset'].map(lambda x: [
        name for name, f in fns.items() if x >= f['base10_offset'] and x < f['base10_offset'] + f['size']
    ][0])
    df['fn_size'] = df['fn'].map(lambda x: fns[x]['size'] * 8)  # Size in bits
    cols = list(df.columns)
    a, b = cols.index('g_offset'), cols.index('base10_offset')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    # df.drop(columns=['base10_offset'], inplace=True)

    df['label_modes'] = df['top_labels_list'].map(
        lambda x: [mode(labels)[0][0].flatten().tolist()[0] for labels in x]
    )
    df.drop(columns=['top_labels_list'], inplace=True)

    # If older sweep files don't contain these metric columns, fill them with default values
    for metric_name in ['top_label_change_pcts', 'lpips_avgs', 'fids']:
        if metric_name not in df.columns:
            df[metric_name] = df['g_offset'].map(
                lambda _: [-1.] * len(sweepret.args.datasets)
            )

    if use_cache:
        utils.ensure_dir_of(cache_path)
        df.to_pickle(cache_path)

    return df, any_bi

def extract_dfs(sweep_or_bits_path, quiet=False):
    '''Turns a sweep/bits file containing flip results for one or more binaries
    into separate dataframes for each binary.'''

    if '.json' in sweep_or_bits_path:
        bitsinfo = utils.load_json(sweep_or_bits_path)
        df = pd.DataFrame.from_records(bitsinfo['results'])
        df['asm'] = df['inst'].map(lambda x: x['asm'] if x['valid'] else None)
        df['opcode_flipped'] = df['inst'].map(lambda x: x['opcode_flipped'] if x['valid'] else None)
        df.drop(columns=['inst'], inplace=True)
    else:
        df, any_bi = _create_df_from_sweep(sweep_or_bits_path, quiet=quiet)

    def extract_column(df, col):
        return pd.DataFrame(df.pop(col).to_list())

    correct_pcts = extract_column(df, 'correct_pcts')
    label_modes = extract_column(df, 'label_modes')
    top_label_change_pcts = extract_column(df, 'top_label_change_pcts')
    lpips_avgs = extract_column(df, 'lpips_avgs')
    fids = extract_column(df, 'fids')
    sus_scores = extract_column(df, 'sus_scores')

    def assign_bit_strengths_and_acc_drops(df):
        if any_bi.is_gan:
            raise NotImplementedError
        # Give correct_pct valid lower/upper bounds
        orig_acc = df[df['top_label_change_pct'] == 0]['correct_pct'].max()
        df['acc_drop_pct'] = (orig_acc - df['correct_pct']) / orig_acc * 100
        rand_guess_acc = 100 / any_bi.nclasses
        df['acc_reset'] = df['correct_pct']
        df.loc[df['acc_reset'] < rand_guess_acc, 'acc_reset'] = rand_guess_acc
        df.loc[df['acc_reset'] > orig_acc, 'acc_reset'] = orig_acc
        # Assign vuln. bit strength
        df['bit_strength'] = 100 - 100 * (
            # Scale to 0-1
            (df['acc_reset'] - rand_guess_acc) / (orig_acc - rand_guess_acc)
        )
        # Drop aux columns
        df.drop(columns=['acc_reset'], inplace=True)

    # Generate n dataframes, one for each dataset
    orig_df = df
    dfs = []
    for i in range(len(correct_pcts.columns)):
        df = orig_df.copy()
        df['correct_pct'] = correct_pcts[i]
        df['label_mode'] = label_modes[i]
        df['top_label_change_pct'] = top_label_change_pcts[i]
        df['lpips_avg'] = lpips_avgs[i]
        df['fid'] = fids[i]
        df['sus_score'] = sus_scores[i]
        assign_bit_strengths_and_acc_drops(df)
        df.sort_values('correct_pct', ignore_index=True, inplace=True)
        dfs.append(df)

    return dfs

def filter_df(
    df,
    filter_key,
    filter_: Union[Callable, List[float], Tuple[float], Any],
    not_=False
):
    orig_df = df
    if callable(filter_):
        df = df.loc[df[filter_key].map(filter_)]
    elif isinstance(filter_, list) or isinstance(filter_, tuple):
        filter_range = filter_
        if filter_range[0] is not None:
            df = df.loc[df[filter_key] >= filter_range[0]]
        if filter_range[1] is not None:
            df = df.loc[df[filter_key] < filter_range[1]]
    else:
        df = df.loc[df[filter_key] == filter_]
    if not_:
        df = orig_df.loc[~orig_df.index.isin(df.index)]
    return df

def group_count(df, group_by):
    '''Groups a dataframe by a column, and returns the count of each group.'''
    return df.groupby(group_by).size().reset_index(name='count')

def merge_dfs(*dfs, filter_key=None, filter_range=None, drop_metrics_cols=True):
    '''Merges multiple dataframes into one by calling pd.merge(), with optional
    filtering.'''

    if filter_key:
        assert filter_range

    def transform(df):
        if filter_key:
            df = filter_df(df, filter_key, filter_range)
        if drop_metrics_cols:
            df.drop(
                columns=[
                    'correct_pct', 'top_label_change_pct', 'label_mode',
                    'bit_strength', 'acc_drop_pct', 'sus_score',
                    'lpips_avg', 'fid'
                ],
                inplace=True
            )
        return df

    if len(dfs) == 1:
        return transform(dfs[0])
    if not drop_metrics_cols:
        utils.warn('Merging multiple dataframes without dropping metrics columns')
    return reduce(
        lambda x, y: x.merge(y, on=None),
        map(transform, dfs)
    )
