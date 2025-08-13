import os
import json
from typing import List
from inspect import isfunction
try:  # https://pypi.org/project/ghidra-stubs/
    import ghidra.ghidra_builtins as builtins
    from ghidra.ghidra_builtins import *
except:
    import builtins
from ghidra.program.model.listing import Instruction
from ghidra.util.task import TaskMonitor

DUMMY_MONITOR = TaskMonitor.DUMMY

# Ghidrathon v3.0.0 calling convention changes
monitor = lambda: builtins.monitor() if isfunction(builtins.monitor) else builtins.monitor
currentProgram = lambda: builtins.currentProgram() if isfunction(builtins.currentProgram) else builtins.currentProgram
currentAddress = lambda: builtins.currentAddress() if isfunction(builtins.currentAddress) else builtins.currentAddress
currentSelection = lambda: builtins.currentSelection() if isfunction(builtins.currentSelection) else builtins.currentSelection
currentLocation = lambda: builtins.currentLocation() if isfunction(builtins.currentLocation) else builtins.currentLocation
currentHighlight = lambda: builtins.currentHighlight() if isfunction(builtins.currentHighlight) else builtins.currentHighlight

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(obj, filepath):
    os.umask(0o022)
    ensure_dir_of(filepath)
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def maybe_do(outfile, work_fn):
    if os.path.exists(outfile):
        print(f'Output file {os.path.abspath(outfile)} already exists, skipping')
        return

    ensure_dir_of(outfile)
    ret = work_fn()
    save_json(ret, outfile)
    print(f'Output written to {os.path.abspath(outfile)}')

def get_compute_fns():
    all_fns = list(currentProgram().getFunctionManager().getFunctions(True))
    compute_fns = []
    for _i, f in enumerate(all_fns):
        if f.getName().endswith('_compute_'):  # TVM
            callees = list(f.getCalledFunctions(DUMMY_MONITOR))
            if not callees:
                compute_fns.append(f)
            else:
                compute_fns.extend([x for x in callees if x.getName().startswith('FUN_')])
        elif f.getName().startswith('libjit_'):  # Glow
            compute_fns.append(f)
    return compute_fns

def get_insts_in_range(first_addr, last_addr) -> List[Instruction]:
    ret = []
    inst = getInstructionAt(first_addr)
    while inst and inst.getMinAddress() <= last_addr:
        ret.append(inst)
        inst = inst.getNext()
    return ret

def get_insts_in_fn(f) -> List[Instruction]:
    # Functions are not always continuous
    ret = []
    for block in f.getBody():
        ret.extend(get_insts_in_range(block.getMinAddress(), block.getMaxAddress()))
    return ret

def get_byte(addr):
    return getByte(addr) & 0xff

def get_bytes(addr, length):
    return [x & 0xff for x in getBytes(addr, length)]

def set_byte(addr, byte):
    setByte(addr, p2jb(byte))

def j2pb(x):
    '''Converts Java (signed) byte to Python (unsigned) byte'''
    return x if x >= 0 else x + 256

def p2jb(x):
    '''Converts Python (unsigned) byte to Java (signed) byte'''
    return x if x < 128 else x - 256

def friendly_hex(x):
    return '_'.join([f'{j2pb(x):02x}' for x in x])
