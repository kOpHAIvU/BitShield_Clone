import struct
from dataclasses import dataclass
from typing import Any, List
from contextlib import contextmanager

OFFSET_MAGIC_BASE = 0x15C9B862_00_00_00_00
LEN_MAGIC_BASE = 0x2319CBC2_00_00_00_00
# GROUND_TRUTH_MAGIC_BASE = 0x3A7A3A30

curr_ophack_ctx = None

@dataclass
class Instrumentation:
    data: Any
    weight: Any
    # Placeholder for args: offset, len
    offset_magic: int
    len_magic: int
    # ground_truth_magic: int

class OpHackContext:
    def __init__(self, coop_cig_ver):
        self.coop_cig_ver = coop_cig_ver
        self.instrumentations: List[Instrumentation] = []
        self.next_inst_idx = 0
        # Two auxiliary variables for Co-op CIG v2
        self.total_nspots = 0
        self.curr_spot_idx = 0

    def new_instrumentation(self, data, weight):
        inst = Instrumentation(
            data=data,
            weight=weight,
            offset_magic=OFFSET_MAGIC_BASE + self.next_inst_idx,
            len_magic=LEN_MAGIC_BASE + self.next_inst_idx,
            # ground_truth_magic=GROUND_TRUTH_MAGIC_BASE + self.next_inst_idx,
        )
        self.instrumentations.append(inst)
        self.next_inst_idx += 1
        return inst

@contextmanager
def use_op_hack(coop_cig_ver=1):
    # Import module to replace default functions with hacks
    from . import strategy as _
    global curr_ophack_ctx
    assert curr_ophack_ctx is None
    curr_ophack_ctx = OpHackContext(coop_cig_ver)
    try:
        yield curr_ophack_ctx
    finally:
        assert curr_ophack_ctx is not None
        curr_ophack_ctx = None
