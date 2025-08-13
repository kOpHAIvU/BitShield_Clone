# Ghidrathon script, python 3
#@menupath Tools.Export Analysis

import os
import struct
import utils

outfile = f'./ghidra/analysis/{utils.currentProgram().getName()}-analysis.json'

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
    return {
        'base10_offset': sym.getAddress().getOffset(),
        'header_nbytes': 8,  # The header consists of a 64-bit body length
        'body_nbytes': struct.unpack('<Q', bytes(utils.get_bytes(sym.getAddress(), 8)))[0]
    }

def export_analysis():
    utils.ensure_dir_of(outfile)

    ret = {
        'memory_map': export_memory_map(),
        'dev_mblob': export_dev_mblob_info(),
        'fns': {
            f.getName(): export_fn(f) for f in utils.currentProgram().getFunctionManager().getFunctions(True)
        }
    }

    utils.save_json(ret, outfile)
    print(f'Output written to {os.path.abspath(outfile)}')

export_analysis()
