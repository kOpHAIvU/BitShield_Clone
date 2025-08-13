# Ghidrathon script, python 3
#@menupath Tools.Find CIG Spots

import utils

def find_cig_spots_in_fn(f):
    cig_spots = []
    insts = utils.get_insts_in_fn(f)
    for i, inst in enumerate(insts):
        offset = inst.getAddress().getOffset()
        asm = str(inst).split()
        if len(asm) != 2 or asm[0] not in {'CALL', 'JMP'}:
            continue
        callee_addr = toAddr(asm[1])
        if not callee_addr or getSymbolAt(callee_addr).getName(True) != '<EXTERNAL>::rand':
            continue

        inst_len = inst.getLength()
        need_ret = False
        # Recognise call-rets that are optimised to jumps, so we can patch
        # away the jump and add back the ret later when prepatching. This only
        # applies when CIG spots are inserted after the call to launch the
        # compute function.
        if asm[0] == 'JMP' and i == len(insts) - 1:
            need_ret = True
            inst_len -= 1

        if not cig_spots or \
            cig_spots[-1]['base10_offset'] + cig_spots[-1]['nbytes'] != offset or \
            cig_spots[-1]['need_ret']:
            cig_spots.append({
                'base10_offset': offset,
                'nbytes': inst_len,
                'fn': f.getName(),
                'need_ret': need_ret,
            })
            continue

        cig_spots[-1]['nbytes'] += inst_len

    # If CIG spots are inserted after the call to launch the compute function,
    # we want the prepatching pass to also remove the JNZ in the 'TEST EAX,EAX;
    # JNZ' check that follows the call and lies right before the spots.
    # Otherwise, the CIGs may be skipped with an easy flip in the JNZ.
    for inst in insts[:-1]:
        asm = str(inst).split()
        if len(asm) != 2 or asm[0] != 'JNZ':
            continue
        rip = inst.getAddress().getOffset() + inst.getLength()
        jmp_addr = toAddr(asm[1]).getOffset()
        jmp_offset = jmp_addr - rip
        for cig_spot in cig_spots:
            if cig_spot['base10_offset'] == rip and cig_spot['nbytes'] == jmp_offset:
                cig_spot['base10_offset'] -= inst.getLength()
                cig_spot['nbytes'] += inst.getLength()

    return cig_spots

def find_cig_spots():
    '''Finds CIG spots in the given function. Each CIG dict looks like:
        {
            'base10_offset': 1234567,
            'nbytes': 123,
            'fn': 'foo',
            'need_ret': True,
        }'''

    cig_spots = []
    for f in utils.currentProgram().getFunctionManager().getFunctions(True):
        cig_spots.extend(find_cig_spots_in_fn(f))

    return cig_spots

def main():
    spots = find_cig_spots()
    utils.save_json(spots, f'cig-spots.json')

main()
