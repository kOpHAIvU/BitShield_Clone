import os
import json
import sys

try:  # https://pypi.org/project/ghidra-stubs/
    import ghidra.ghidra_builtins  # noqa: F401
    from ghidra.ghidra_builtins import *  # noqa: F401,F403
except Exception:
    # In Ghidra environment, functions are normally available in global namespace.
    pass

from ghidra.program.model.listing import Instruction
from ghidra.util.task import TaskMonitor

DUMMY_MONITOR = TaskMonitor.DUMMY

def _call_if_callable(obj):
    try:
        return obj() if callable(obj) else obj
    except TypeError:
        return obj

def get_current_program():
    """
    Returns the current program in both Ghidra GUI and headless environments.
    Handles cases where currentProgram is a callable or an object.
    """
    # 1. Check caller / __main__ namespace (most common in Ghidra scripts)
    main_module = sys.modules.get('__main__')
    main_ns = getattr(main_module, '__dict__', {}) if main_module else {}
    for name in ('currentProgram', 'getCurrentProgram'):
        if name in main_ns:
            return _call_if_callable(main_ns[name])

    # 2. Check utils module globals (if functions imported via ghidra_builtins)
    module_globals = globals()
    for name in ('currentProgram', 'getCurrentProgram'):
        if name in module_globals:
            return _call_if_callable(module_globals[name])

    # 3. As a fallback, attempt to import from ghidra.ghidra_builtins
    try:
        from ghidra.ghidra_builtins import currentProgram as builtin_current_program
        return _call_if_callable(builtin_current_program)
    except Exception:
        pass

    # 4. Final fallback: try state variable provided by headless analyzer (if available)
    if 'state' in main_ns:
        try:
            return main_ns['state'].getCurrentProgram()
        except Exception:
            pass

    raise RuntimeError('Unable to determine current program (currentProgram not available)')

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
        print('Output file {} already exists, skipping'.format(os.path.abspath(outfile)))
        return

    ensure_dir_of(outfile)
    ret = work_fn()
    save_json(ret, outfile)
    print('Output written to {}'.format(os.path.abspath(outfile)))

def get_compute_fns():
    prog = get_current_program()
    all_fns = list(prog.getFunctionManager().getFunctions(True))
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

def get_insts_in_range(first_addr, last_addr):
    prog = get_current_program()
    listing = prog.getListing()
    ret = []
    inst = listing.getInstructionAt(first_addr)
    while inst and inst.getMinAddress() <= last_addr:
        ret.append(inst)
        inst = listing.getInstructionAfter(inst.getMinAddress())
    return ret

def get_insts_in_fn(f):
    # Functions are not always continuous
    ret = []
    for block in f.getBody():
        ret.extend(get_insts_in_range(block.getMinAddress(), block.getMaxAddress()))
    return ret

def _get_memory():
    return get_current_program().getMemory()

def get_byte(addr):
    mem = _get_memory()
    return mem.getByte(addr) & 0xff

def get_bytes(addr, length):
    mem = _get_memory()
    buf = bytearray(length)
    read = mem.getBytes(addr, buf)
    if read == -1:
        # Older Ghidra versions return -1; data placed in buf
        read = length
    elif read is None:
        read = length
    if read < len(buf):
        buf = buf[:read]
    return [b & 0xff for b in buf]

def set_byte(addr, byte):
    mem = _get_memory()
    mem.setByte(addr, p2jb(byte))

def j2pb(x):
    '''Converts Java (signed) byte to Python (unsigned) byte'''
    return x if x >= 0 else x + 256

def p2jb(x):
    '''Converts Python (unsigned) byte to Java (signed) byte'''
    return x if x < 128 else x - 256

def friendly_hex(x):
    return '_'.join(['{:02x}'.format(j2pb(x)) for x in x])
