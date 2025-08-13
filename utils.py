from typing import Dict
import os
import subprocess
import tempfile
import sys
from typing import NamedTuple
import pickle
import json
from contextlib import contextmanager

class BinaryInfo(NamedTuple):

    compiler: str
    compiler_ver: str
    model_name: str
    dataset: str
    cig: str
    dig: str
    avx: bool
    opt_level: int

    @property
    def core_model_name(self):
        return self.model_name[1:] if self.model_name.startswith('Q') else self.model_name

    @property
    def fname(self):
        return (
            f'{self.compiler}-{self.compiler_ver}-{self.model_name}-{self.dataset}'
            f'-{self.cig}-{self.dig}'
            f'{"-noavx" if not self.avx else ""}'
            f'{f"-{self.opt_level}" if self.opt_level != 3 else ""}'
            '.so'
        )

    @property
    def fpath(self):
        import cfg
        return f'{cfg.built_dir}/{self.fname}'

    @property
    def nchans(self):
        return {
            'lenet1': 1,
            'lenet5': 1,
            'dcgan_g': 100,
        }.get(self.model_name, 3)

    @property
    def nclasses(self):
        return {
            'CIFAR10': 10,
            'MNIST': 10,
            'MNISTC': 10,
            'FashionC': 10,
            'ImageNet': 1000,
        }.get(self.dataset, 10)

    @property
    def input_img_size(self):
        return {
            ('lenet1', self.dataset): 28,
            ('dcgan_g', self.dataset): 1,
            (self.model_name, 'ImageNet'): 96,
            ('densenet121', 'ImageNet'): 72,
        }.get((self.model_name, self.dataset), 32)

    @property
    def output_img_size(self):
        return {
            'dcgan_g': 64,
        }[self.model_name]

    @property
    def fast_n_per_class(self):
        return 10 if self.nclasses <= 100 else 1

    @property
    def input_shape(self):
        return (self.batch_size, self.nchans, self.input_img_size, self.input_img_size)

    @property
    def is_gan(self):
        return self.model_name in {'dcgan_g'}

    @property
    def has_dig(self):
        return self.dig != 'nd'

    @property
    def has_cig(self):
        # Prepatched but w/o CIGs, and non-prepatched models
        return self.cig not in {'nc', 'ncnp'}

    @property
    def default_output_defs(self):
        import cfg
        if self.is_gan:
            ret = [{'shape': (cfg.batch_size, 1, 64, 64), 'dtype': 'float32'}]
        ret = [{'shape': (cfg.batch_size, self.nclasses), 'dtype': 'float32'}]
        if self.has_dig:
            # Note that currently suspicious score is just one value
            ret.append({'shape': (1,), 'dtype': 'float32'})
        return ret

    @property
    def output_defs_file(self):
        import cfg
        return f'{cfg.output_defs_dir}/{self.fname}.json'

    @property
    def output_defs(self):
        if os.path.exists(self.output_defs_file):
            return load_json(self.output_defs_file)
        if self.has_cig:
            return self._replace(cig='nc').output_defs
        return self.default_output_defs

    @property
    def batch_size(self):
        return self.output_defs[0]['shape'][0]

    @property
    def analysis_file(self):
        import cfg
        return f'{cfg.analysis_dir}/{self.fname}-analysis.json'

    def get_analysis(self):
        # See also: ghidra/export-analysis.py
        return load_json(self.analysis_file)

    @property
    def cig_spots_file(self):
        import cfg
        return f'{cfg.cig_spots_dir}/{self.fname}-cig-spots.json'

    def get_cig_spots(self):
        if os.path.exists(self.cig_spots_file):
            return load_json(self.cig_spots_file)
        if self.has_cig:
            return self._replace(cig='nc').get_cig_spots()
        return []

    def just_datasets_differ(self, other):
        return self.compiler == other.compiler and \
            self.compiler_ver == other.compiler_ver and \
            self.model_name == other.model_name and \
            self.cig == other.cig and \
            self.dig == other.dig and \
            self.avx == other.avx and \
            self.opt_level == other.opt_level

    @staticmethod
    def from_fname(fname):
        fname = os.path.basename(fname)
        fname = fname.rsplit('.', maxsplit=1)[0]  # Remove extension
        fsplit = fname.split('-')
        compiler, compiler_ver, model_name, dataset, cig, dig, *fsplit = fsplit
        avx = 'noavx' not in fsplit
        try:
            opt_level = int(fsplit[-1])
        except (ValueError, IndexError):
            opt_level = 3
        return BinaryInfo(compiler, compiler_ver, model_name, dataset, cig, dig, avx, opt_level)

@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def sha1sum_file(fname):
    import hashlib
    buf_size = 65536
    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def reimport(*modules):
    import importlib
    for module in modules:
        if isinstance(module, str):
            module = importlib.import_module(module)
        importlib.reload(module)

@contextmanager
def np_temp_seed(seed):
    import numpy as np
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def one_shot_ghidra_analyse(fpath) -> Dict:
    '''Runs Ghidra in headless mode to analyse the given binary. Returns the
    analysis result as a dict.'''
    # TODO: We assume the user is already running in docker?
    import cfg
    fpath = os.path.abspath(fpath)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.basename(fpath)
        subprocess.run([
            f'{cfg.ghidra_dir}/import-run-script-once.sh', fpath, 'export-analysis.py'
        ], check=True, cwd=tmpdir, stdout=subprocess.DEVNULL)
        analysis_path = f'{tmpdir}/ghidra/analysis/{fname}-analysis.json'
        return load_json(analysis_path)

def extract_graph_json(fname):
    import mmap
    import struct
    marker = b'GraphExecutorFactory'
    with open(fname, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        start = mm.find(marker)
        mm.seek(start + len(marker))
        # Read an 8-byte length
        length = struct.unpack('<Q', mm.read(8))[0]
        # Read the length-byte string
        lines = mm.read(length).decode('utf-8')
        mm.close()
    # Clean up trailing spaces
    lines = [x.rstrip() for x in lines.splitlines()]
    json_str = '\n'.join(lines)
    return json.loads(json_str)

def get_tvm_shape(ndarray):
    import tvm
    return [x.value if isinstance(x, tvm.tir.IntImm) else x for x in ndarray.shape]

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def save(obj, filepath, merge=True):
    assert not merge or isinstance(obj, dict)
    ensure_dir_of(filepath)
    if merge and os.path.exists(filepath):
        orig_obj = load(filepath)
        obj = {**orig_obj, **obj}
    with open(filepath, 'wb+') as f:
        pickle.dump(obj, f)

def load(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(obj, filepath, sorted=False):
    ensure_dir_of(filepath)
    with open(filepath, 'w+') as f:
        json.dump(obj, f, indent=2, sort_keys=sorted)

def save_str(s, filepath):
    ensure_dir_of(filepath)
    with open(filepath, 'w+') as f:
        f.write(str(s))

def warn(msg):
    print(f'⚠️  Warning: {msg}', file=sys.stderr)

def thread_first(data, *calls, thread_last=False):
    """An approximation of the "->" and "->>" macros in Clojure."""

    def tryexec(f, func_name):
        try:
            return f()
        except Exception as e:
            print(f'thread_{"last" if thread_last else "first"} failed to execute {func_name}: {e}')
            raise

    for c in calls:
        c = list(c) if isinstance(c, tuple) else [c]
        f = c[0]
        if len(c) == 2 and isinstance(c[1], dict):
            data = tryexec(lambda: f(data, **c[1]), f.__name__)
            continue
        if thread_last:
            args = c[1:] + [data]
        else:
            args = [data] + c[1:]
        data = tryexec(lambda: f(*args), f.__name__)
    return data

thread_last = lambda data, *calls: thread_first(data, *calls, thread_last=True)

tf = thread_first
tl = thread_last
