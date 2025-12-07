# Ghidrathon script, python 3
#@menupath Tools.Export Analysis

import os
import struct
import utils

# Get current program name (works in both GUI and headless)
prog = utils.get_current_program()

# Check if output directory is specified via environment variable (from Python code)
# Otherwise, use current working directory or project root
if 'GHIDRA_ANALYSIS_OUTPUT_DIR' in os.environ:
    analysis_dir = os.environ['GHIDRA_ANALYSIS_OUTPUT_DIR']
else:
    # Fallback: use current working directory (cwd) for output path
    # When run from temp directory (as in one_shot_ghidra_analyse), cwd will be the temp dir
    cwd = os.getcwd()
    analysis_dir = os.path.join(cwd, 'ghidra', 'analysis')

outfile = os.path.join(analysis_dir, '{}-analysis.json'.format(prog.getName()))

def export_fn(f):
    return {
        'base10_offset': f.getEntryPoint().getOffset(),
        'is_compute_fn': f in utils.get_compute_fns(),
        # NOTE: Since functions aren't necessarily continuous, we use this to get the
        # approx. (can be a bit larger) range.
        'size': f.getBody().getMaxAddress().getOffset() - f.getBody().getMinAddress().getOffset() + 1,
        'called_by': [x.getName() for x in f.getCallingFunctions(utils.DUMMY_MONITOR)],
        'insts': [
            {
                'base10_offset': inst.getAddress().getOffset(),
                'asm': str(inst),
                'nbytes': inst.getLength(),
                'bytes': utils.friendly_hex(inst.getBytes()),
                'imask': utils.friendly_hex(inst.getPrototype().getInstructionMask().getBytes()),
                'omasks': [
                    utils.friendly_hex(inst.getPrototype().getOperandValueMask(i).getBytes())
                    for i in range(inst.getNumOperands())
                ],
            }
            for inst in utils.get_insts_in_fn(f)
        ]
    }

def export_memory_map():
    return {
        blk.getName(): {
            'base10_start': blk.getStart().getOffset(),
            'size': blk.getSize(),
            'base10_end': blk.getEnd().getOffset(),
        }
        for blk in getMemoryBlocks()
    }

def export_dev_mblob_info():
    sym = getSymbol('__tvm_dev_mblob', None)
    if not sym:
        return None
    addr = sym.getAddress()
    raw = utils.get_bytes(addr, 8)
    if len(raw) != 8:
        # If we cannot read header properly, skip dev mblob info
        print('Warning: Unable to read __tvm_dev_mblob header, skipping dev_mblob info')
        return None
    header = struct.unpack('<Q', bytearray(raw))[0]
    return {
        'base10_offset': addr.getOffset(),
        'header_nbytes': 8,
        'body_nbytes': header,
    }

def export_analysis():
    utils.ensure_dir_of(outfile)

    # Get current program - try multiple ways
    prog = utils.get_current_program()

    ret = {
        'memory_map': export_memory_map(),
        'dev_mblob': export_dev_mblob_info(),
        'fns': {
            f.getName(): export_fn(f) for f in prog.getFunctionManager().getFunctions(True)
        }
    }

    utils.save_json(ret, outfile)
    print('Output written to {}'.format(os.path.abspath(outfile)))

export_analysis()
