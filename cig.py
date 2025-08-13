import os
import subprocess
import tempfile
import shutil
from typing import Dict, List, Union, Iterable, NamedTuple
import zlib
from functional import seq
import struct
import peachpy
from peachpy.x86_64.instructions import BranchInstruction, Instruction
from peachpy.x86_64 import operand
import peachpy.x86_64 as asm
from zlib import adler32
import mmap
import re
from ophack.ctx import use_op_hack

import tvm
from tvm import tir

import cfg
import utils
import cig

rip = asm.registers.rip

# TODO: Support other compilers

def SpacemakerPass():

    def preorder_mutator(op):
        return tir.SeqStmt(
            [op] +
            [tir.Evaluate(tir.call_extern(None, 'rand')) for _ in range(50)]
        )

    @tir.transform.prim_func_pass(opt_level=0)
    def spacemaker_pass(f, mod, ctx):

        is_fn_mutated = False
        def preorder_mutate_once(op):
            nonlocal is_fn_mutated
            if is_fn_mutated:
                return op
            is_fn_mutated = True
            return preorder_mutator(op)

        return f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, preorder_mutate_once, None))

    return spacemaker_pass

def prepatch(fname, out_fname, out_spots_fname):
    '''Discovers the CIG spots made by spacemaker in the given function,
    replaces them with NOP, and saves the spots info to a JSON.'''

    with tempfile.TemporaryDirectory() as tmpdir:
        # TODO: We assume the user is already running in docker?
        subprocess.run([
            f'{cfg.ghidra_dir}/import-run-script-once.sh', fname, 'find-cig-spots.py'
        ], check=True, cwd=tmpdir, stdout=subprocess.DEVNULL)
        utils.ensure_dir_of(out_spots_fname)
        shutil.copy(f'{tmpdir}/cig-spots.json', out_spots_fname)

    # NOP the CIG spots
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(fname, f'{tmpdir}/work.so')
        with open(f'{tmpdir}/work.so', 'rb+') as f:
            for spot in utils.load_json(out_spots_fname):
                nops = [0x90] * spot['nbytes']
                # Since Ghidra stops disassembling after 16 repeated bytes, we
                # insert some 0x66 prefixes.
                if len(nops) > 15:
                    for i in range(0, len(nops), 16):
                        nops[i] = 0x66
                    # Make sure the last byte is not a prefix
                    if nops[-1] == 0x66:
                        nops[-2:] = [0x66, 0x90]
                # Add back a required ret if instructed
                if spot['need_ret']:
                    nops.append(0xC3)
                f.seek(spot['base10_offset'] - cfg.GHIDRA_BASE_ADDR)
                f.write(bytes(nops))
        utils.ensure_dir_of(out_fname)
        shutil.move(f'{tmpdir}/work.so', out_fname)


class CodeIntegrityGuard:
    '''RIP-relative addressing is assumed.'''

    def __init__(self, checked_offset, patch_offset=None) -> None:
        self.checked_offset = checked_offset
        self.patch_offset = patch_offset
        self.encoded_bytes: Union[bytes, bytearray] = None

    @property
    def size(self):
        return len(self.encoded_bytes)

    @property
    def checked_code_len(self):
        raise NotImplementedError

    def encode(self):
        '''Encodes the CIG into self.encoded_bytes. Users may need to set
        self.patch_offset first.'''
        raise NotImplementedError

    def get_est_size(self):
        '''Returns the estimated size of the encoded CIG.'''
        raise NotImplementedError

    def maybe_encode_at(self, patch_offset, max_size):
        '''Try encoding the CIG at the given patch offset. If the CIG can be
        encoded within the given size limit, returns True and preserves the
        result. Otherwise, returns False and reverts the state.'''
        orig_patch_offset = self.patch_offset
        orig_encoded_bytes = self.encoded_bytes
        self.patch_offset = patch_offset
        self.encode()
        if self.size > max_size:
            self.patch_offset = orig_patch_offset
            self.encoded_bytes = orig_encoded_bytes
            return False
        return True

    def __str__(self) -> str:
        return f'{self.__class__.__name__}@{self.patch_offset}->{self.checked_offset}'

    def __repr__(self) -> str:
        return str(self)

class BasicCIG(CodeIntegrityGuard):
    def __init__(self, checked_offset, ground_truth: bytes):
        super().__init__(checked_offset)
        assert len(ground_truth) in {1, 2, 4, 8}
        self.ground_truth = ground_truth
        self.truth_val = int(ground_truth[::-1].hex(), 16)  # Little endian

    @property
    def checked_code_len(self):
        return len(self.ground_truth)

    def _get_asmf_8(self):
        mov_insn_len = len(asm.MOV(asm.rax, [rip+0]).encode())
        with asm.Function('foo', ()) as asmf:
            reloff = self.checked_offset - (self.patch_offset + mov_insn_len)
            asm.MOV(asm.rax, [rip + reloff])
            asm.MOV(asm.rdx, self.truth_val)
            asm.CMP(asm.rax, asm.rdx)
            good_label = asm.Label('good')
            asm.JE(good_label)
            asm.INT(3)
            asm.LABEL(good_label)
            asm.RET()
        return asmf

    def encode(self):
        checked_len = len(self.ground_truth)
        if checked_len == 8:
            asmf = self._get_asmf_8()
        else:
            dummy_opnd = asm.operand.MemoryOperand(rip+0x1234, size=checked_len)
            cmp_insn_len = len(asm.CMP(dummy_opnd, self.truth_val).encode())
            with asm.Function('foo', ()) as asmf:
                reloff = self.checked_offset - (self.patch_offset + cmp_insn_len)
                asm.CMP(
                    asm.operand.MemoryOperand(rip + reloff, size=checked_len),
                    self.truth_val
                )
                good_label = asm.Label('good')
                asm.JE(good_label)
                asm.INT(3)
                asm.LABEL(good_label)
                asm.RET()
        encf = asmf.finalize(asm.abi.system_v_x86_64_abi).encode()
        self.encoded_bytes = encf.code_section.content[:-1]  # Remove the RET

    def get_est_size(self):
        return {1: 10, 2: 12, 4: 13, 8: 23}[len(self.ground_truth)]

    def __str__(self) -> str:
        return super().__str__() + f'({len(self.ground_truth)})'

class Adler32CIG(CodeIntegrityGuard):

    # Since PeachPy has some unsupported instructions, we hack them together
    # here. Note that they won't support all PeachPy features, e.g., auto
    # register allocation.

    class LODSB(Instruction):
        def __init__(self, *args, **kwargs):
            origin = kwargs.get("origin")
            prototype = kwargs.get("prototype")
            super().__init__("LODSB", origin=origin, prototype=prototype)
            self.operands = tuple(map(operand.check_operand, args))
            assert len(self.operands) == 0
            self.encodings.append((0x00, lambda op: bytearray([0xAC])))
            if peachpy.stream.active_stream is not None:
                peachpy.stream.active_stream.add_instruction(self)

    class LOOP(BranchInstruction):
        def __init__(self, *args, **kwargs):
            origin = kwargs.get("origin")
            prototype = kwargs.get("prototype")
            super().__init__("LOOP", origin=origin, prototype=prototype)
            self.operands = tuple(map(operand.check_operand, args))
            assert len(self.operands) == 1
            if operand.is_rel8(self.operands[0]):
                self.encodings.append((0x00, lambda op: bytearray([0xE2, op[0].offset & 0xFF])))
            elif operand.is_label(self.operands[0]):
                self.encodings.append((0x04, lambda off: bytearray([0xE2, off & 0xFF])))
            else:
                assert False
            if peachpy.stream.active_stream is not None:
                peachpy.stream.active_stream.add_instruction(self)

    @property
    def checked_code_len(self):
        return len(self.checked_bytes)

    def __init__(self, checked_offset, checked_bytes: bytes) -> None:
        super().__init__(checked_offset)
        self.checked_bytes = checked_bytes
        self.ground_truth = zlib.adler32(checked_bytes)

    def encode(self):
        checked_len = len(self.checked_bytes)
        lea_insn_len = len(asm.LEA(asm.rsi, [rip+0x1234]).encode())
        with asm.Function('foo', ()) as asmf:
            # Using golfed_adler32_amd64_v3 from Peter Cordes
            # Args: len in rcx, const char *buf in rsi; result in eax
            reloff = self.checked_offset - (self.patch_offset + lea_insn_len)
            asm.LEA(asm.rsi, [rip + reloff])
            asm.MOV(asm.ecx, checked_len)
            # ---
            asm.XOR(asm.eax, asm.eax)
            asm.CDQ()
            asm.LEA(asm.edi, [asm.rdx+1])
            with asm.Loop('byteloop') as byteloop:
                self.LODSB()
                asm.ADD(asm.edi, asm.eax)
                asm.ADD(asm.edx, asm.edi)
                self.LOOP(byteloop.begin)
            asm.MOV(asm.cx, 0xfff1)
            asm.XCHG(asm.eax, asm.edx)
            asm.CDQ()
            asm.DIV(asm.ecx)
            asm.PUSH(asm.rdx)
            asm.XCHG(asm.eax, asm.edi)
            asm.CDQ()
            asm.DIV(asm.ecx)
            asm.PUSH(asm.dx)
            asm.POP(asm.rax)
            asm.POP(asm.dx)
            # ---
            asm.CMP(asm.eax, self.ground_truth)
            good_label = asm.Label('good')
            asm.JE(good_label)
            asm.INT(3)
            asm.LABEL(good_label)
            asm.RET()
        encf = asmf.finalize(asm.abi.system_v_x86_64_abi).encode()
        self.encoded_bytes = encf.code_section.content[:-1]  # Remove the RET

    def get_est_size(self):
        return 51

    def __str__(self) -> str:
        return super().__str__() + f'({len(self.checked_bytes)})'

class CIGPlanner:
    def __init__(self):
        pass

    def get_plan(
        self, text: bytes, vuln_byteoffs: Union[None, Iterable[int]],
        csmgr: 'CIGSpotsManager', fn_ranges: Dict[str, Dict]
    ) -> List[CodeIntegrityGuard]:
        '''Returns a list of *encoded* CIGs.
        Note that vuln_byteoffs may be None, because not all planners need it,
        and the sweep file may not be available in all cases.'''
        raise NotImplementedError

class GreedyCIGPlanner(CIGPlanner):
    def __init__(self, **kwargs):
        self.cscig_min_len = 9
        self.cscig_max_len = 512
        self.cscig_max_gap = 32
        self.cscig_min_protected_bytes = 3
        self.__dict__.update(kwargs)

        # For debugging
        self._b_assignments = None
        self._cs_assignments = None

    def _get_assignments(self, vbytes_offsets):
        b_bytes = []
        b_assignments = []  # List of lists
        cs_assignments = []  # List of lists

        curr_cs_bytes = []

        def commit_curr_cs(curr_byte):
            # Commit assignment for current bytes
            nonlocal curr_cs_bytes
            if curr_cs_bytes[-1] - curr_cs_bytes[0] >= self.cscig_min_len and \
                len(curr_cs_bytes) >= self.cscig_min_protected_bytes:
                cs_assignments.append(curr_cs_bytes)
            else:
                b_bytes.extend(curr_cs_bytes)
            curr_cs_bytes = [curr_byte]

        for vbo in vbytes_offsets:
            if not len(curr_cs_bytes):
                curr_cs_bytes.append(vbo)
                continue
            if vbo - curr_cs_bytes[-1] - 1 > self.cscig_max_gap or \
                vbo - curr_cs_bytes[0] >= self.cscig_max_len:
                commit_curr_cs(vbo)
                continue
            curr_cs_bytes.append(vbo)

        if len(curr_cs_bytes):
            commit_curr_cs(vbo)

        # Merge nearby basic assignments if possible
        bcig_max_size = self.cscig_min_len - 1
        for vbo in b_bytes:
            if not len(b_assignments):
                b_assignments.append([vbo])
                continue
            if vbo - b_assignments[-1][0] + 1 <= bcig_max_size:
                b_assignments[-1].append(vbo)
                continue
            b_assignments.append([vbo])

        self._b_assignments, self._cs_assignments = b_assignments, cs_assignments
        return b_assignments, cs_assignments

    @staticmethod
    def _get_covered_fns(fn_ranges, cig: CodeIntegrityGuard):
        covered_first = cig.checked_offset
        covered_last = cig.checked_offset + cig.checked_code_len - 1
        return {
            fn_name for fn_name, fn_range in fn_ranges.items()
            if (covered_first <= fn_range['start'] <= covered_last) or \
                (fn_range['start'] <= covered_first <= fn_range['last'])
        }

    def get_plan(
        self, text: bytes, vuln_byteoffs: Union[None, Iterable[int]],
        csmgr: 'CIGSpotsManager', fn_ranges: Dict[str, Dict]
    ) -> List[CodeIntegrityGuard]:
        # TODO: We exclude bytes in CIG spots for now
        vuln_byteoffs = [x for x in vuln_byteoffs if not csmgr.is_in_spots(x)]
        b_assigns, cs_assigns = self._get_assignments(vuln_byteoffs)

        bcigs = []
        bcig_size_map = [None, 1, 2, 4, 4, 8, 8, 8, 8]
        for offs in b_assigns:
            start = offs[0]
            cig_size = bcig_size_map[offs[-1] - start + 1]
            bcigs.append(BasicCIG(start, text[start:start+cig_size]))
        cscigs = []
        for offs in cs_assigns:
            start, end = offs[0], offs[-1]
            cscigs.append(Adler32CIG(start, text[start:end+1]))

        cigs = bcigs + cscigs
        for cig in cigs[::-1]:  # Plan in reverse order
            covered_fns = self._get_covered_fns(fn_ranges, cig)
            patch_offset = csmgr.alloc(cig.get_est_size(), avoid_fns=covered_fns)
            assert patch_offset is not None, 'Not enough CIG spots'
            assert cig.maybe_encode_at(patch_offset, cig.get_est_size()), 'CIG size unexpected'

        return cigs

class ScanAllCIGPlanner(GreedyCIGPlanner):
    '''This is a baseline planner that tries to use CIGs to scan all bytes. It
    is a special case of GreedyCIGPlanner where (almost) every byte is
    considered vulnerable.'''

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
            cscig_min_len=0, cscig_max_gap=0, cscig_min_protected_bytes=0,
            cscig_max_len=4096
        )

    def get_plan(
        self, text: bytes, vuln_byteoffs: Union[None, Iterable[int]],
        csmgr: 'CIGSpotsManager', fn_ranges: Dict[str, Dict]
    ) -> List[CodeIntegrityGuard]:
        vuln_byteoffs = range(len(text))
        return super().get_plan(text, vuln_byteoffs, csmgr, fn_ranges)

def get_planner_cls(codename):
    return {
        'h1': GreedyCIGPlanner,
        'sa': ScanAllCIGPlanner
    }[codename]

class CIGSpotsManager:

    class CIGSpot:
        def __init__(self, base_offset, size, fn_name=None) -> None:
            self.base_offset = base_offset
            self.size = size
            self.fn_name = fn_name

            # We assume the bytes in the spot is used contiguously for simplicity
            self.first_free, self.last_free = 0, size - 1

        @property
        def free_size(self):
            return self.last_free - self.first_free + 1

        def alloc(self, size):
            '''Allocates the given number of bytes in the spot, starting from
            the end. Returns the start offset of the allocated bytes or None if
            not enough space is available.'''
            if self.free_size < size:
                return None
            self.last_free -= size
            return self.last_free + 1

    def __init__(self, raw_cig_spots: List[Dict]) -> None:
        self.spots = [
            self.CIGSpot(spot['base10_offset'], spot['nbytes'], fn_name=spot['fn'])
            for spot in raw_cig_spots
        ]
        self.next_spot_idx = 0

    def is_in_spots(self, offset):
        for spot in self.spots:
            if spot.base_offset <= offset < spot.base_offset + spot.size:
                return True
        return False

    @property
    def free_size(self):
        return sum(spot.free_size for spot in self.spots)

    @property
    def total_size(self):
        return sum(spot.size for spot in self.spots)

    def alloc(self, size, avoid_fns: Iterable[str] = ()):
        '''Allocates the given number of bytes in the first available spot.
        Avoids allocating in the given functions if instructed. The available
        CIG spots are used in a round-robin fashion. Returns the start offset
        of the allocated bytes or None if not enough space is available.'''
        for _ in range(len(self.spots)):
            spot = self.spots[self.next_spot_idx]
            self.next_spot_idx = (self.next_spot_idx + 1) % len(self.spots)
            if spot.fn_name in avoid_fns:
                continue
            ret = spot.alloc(size)
            if ret is not None:
                return spot.base_offset + ret
        return None

class CIGPatcher:
    def __init__(
        self, prepatched_bi: utils.BinaryInfo, sweepfile=None,
        so_path=None, analysis=None, cig_spots=None
    ):
        from analysis import extract_dfs

        assert prepatched_bi or (so_path and analysis and cig_spots)

        self.bi = prepatched_bi
        self.so_path = so_path or f'{cfg.built_dir}/{prepatched_bi.fname}'
        self.analysis = analysis or prepatched_bi.get_analysis()

        # Users of this class should use 0-based offsets, mapped to the start
        # of .text in Ghidra (text_start_g)
        self.text_start_g = self.analysis['memory_map']['.text']['base10_start']

        # The offset of .text in the file
        self.text_seek_offset = self.text_start_g - cfg.GHIDRA_BASE_ADDR

        self.graph_json = utils.extract_graph_json(self.so_path)
        # Get the functions actually referenced by the graph - inserting CIGs
        # into functions not actually used is useless.
        self.referenced_fns = (
            seq(self.graph_json['nodes'])
            .map(lambda node: node.get('attrs', {}).get('func_name', None))
            .filter(lambda x: x not in {None, '__nop'})
            .to_set()
        )

        raw_cig_spots = cig_spots or prepatched_bi.get_cig_spots()
        self.cig_spots = [
            # Use 0-based offsets
            {**spot, 'base10_offset': spot['base10_offset'] - self.text_start_g}
            for spot in raw_cig_spots
            if spot['fn'].replace('_compute_', '') in self.referenced_fns
        ]

        self.fn_ranges = {
            name: {
                # Use 0-based offsets, both ends inclusive
                'start': fn['base10_offset'] - self.text_start_g,
                'last': fn['base10_offset'] - self.text_start_g + fn['size'] - 1
            }
            for name, fn in self.analysis['fns'].items()
        }

        self.sweep_df = None
        if not so_path:
            # If so_path is set, the user may be using a custom binary, so it's
            # better not to guess the sweep file
            if not sweepfile:
                sweepfile = f'{cfg.sweep_dir}/{prepatched_bi.fname.replace(".so", "-sweep.pkl")}'
            if os.path.exists(sweepfile):
                dfs = extract_dfs(sweepfile)
                assert len(dfs) == 1
                self.sweep_df = dfs[0]

        self.text: bytearray = None
        self.csmgr: CIGSpotsManager = None
        self.planned_cigs: List[CodeIntegrityGuard] = []

    def plan(
        self, planner: CIGPlanner,
        min_bit_strength=40, sus_score_range=None
    ):
        # TODO: Set default min_bit_strength to 0

        with open(self.so_path, 'rb') as f:
            f.seek(self.text_seek_offset)
            self.text = bytearray(f.read(self.analysis['memory_map']['.text']['size']))

        self.csmgr = CIGSpotsManager(self.cig_spots)

        # Get vulnerable bits undetected by DIG

        vuln_byteoffs = None
        if self.sweep_df is not None and self.bi.has_dig:
            if not sus_score_range:
                from eval import evalutils
                sus_score_range = evalutils.get_sus_score_range(self.bi)

            df = self.sweep_df
            df = df[
                (df['bit_strength'] > min_bit_strength) &
                (df['sus_score'] >= sus_score_range[0]) &
                (df['sus_score'] <= sus_score_range[1])
            ]
            df = df.groupby('g_offset').size().reset_index(name='counts')
            # Use 0-based offsets
            df['byteidx'] = df['g_offset'].apply(lambda x: int(x, 16) - self.text_start_g)
            vuln_byteoffs = df['byteidx']

        # TODO: Add known common vulnerable bits

        self.planned_cigs = planner.get_plan(
            self.text, vuln_byteoffs, self.csmgr, self.fn_ranges
        )

    def _do_patch(self):
        for cig in self.planned_cigs:
            # Since we use 90h and 6690h as NOPs, we need to make sure the byte
            # before the patch offset is not 66h by moving the NOP back by one
            # byte.
            po = cig.patch_offset
            self.text[po:po+cig.size] = cig.encoded_bytes
            if po > 0 and self.text[po-1] == 0x66 and self.csmgr.is_in_spots(po-1):
                self.text[po-1] = 0x90
                if po > 1 and self.csmgr.is_in_spots(po-2):
                    self.text[po-2] = 0x66

    def patch(self, outfile):
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(self.so_path, f'{tmpdir}/work.so')
            with open(f'{tmpdir}/work.so', 'rb+') as f:
                self._do_patch()
                f.seek(self.text_seek_offset)
                f.write(self.text)
            utils.ensure_dir_of(outfile)
            shutil.move(f'{tmpdir}/work.so', outfile)

class CoopCIGV1ModuleBuilder:

    @staticmethod
    def get_gefmod_exportfn(orig_ll_outfile=None):
        def gefmod_exportfn(gefmod, fpath):
            if orig_ll_outfile:
                utils.ensure_dir_of(orig_ll_outfile)
                gefmod.module.imported_modules[0].save(orig_ll_outfile)
            gefmod.export_library(fpath, fcompile=None, options=[
                f'{cfg.unmasker_dir}/unmasker.o', '-g',
                # TODO: Automate building the unmasker
            ])
        return gefmod_exportfn

    @staticmethod
    def patch_inst_args(mm: mmap.mmap, inst, code_offset, code_len):
        mm.seek(0)
        offset_spot = mm.find(struct.pack('<Q', inst.offset_magic))
        mm.seek(offset_spot)
        mm.write(struct.pack('<q', code_offset))
        len_spot = mm.find(struct.pack('<Q', inst.len_magic))
        mm.seek(len_spot)
        mm.write(struct.pack('<Q', code_len))

    @staticmethod
    def NCXn2NC_shape(shape):
        return (shape[0] * shape[2], shape[1])

    @staticmethod
    def float_xor_uint(f, u):
        bytes_xor = lambda ba1, ba2: bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])
        b = struct.pack('<f', f)
        b = bytes_xor(b, struct.pack('<I', u))
        return struct.unpack('<f', b)[0]

    @staticmethod
    def masked_state_dict(state_dict, insts, ground_truth):
        import torch
        state_dict = state_dict.copy()
        for inst in insts:
            packw_shape = utils.get_tvm_shape(inst.weight)
            orig_shape = CoopCIGV1ModuleBuilder.NCXn2NC_shape(packw_shape)
            origw_kvs = (
                seq(state_dict.items())
                .filter(lambda x: x[1].shape == orig_shape)
                .to_list()  # -> [(name, tensor), ...]
            )
            assert len(origw_kvs) == 1
            origw_name, origw = origw_kvs[0]
            # For now, we xor each item with ground truth
            new_elems = [
                CoopCIGV1ModuleBuilder.float_xor_uint(x, ground_truth)
                for x in origw.flatten().tolist()
            ]
            neww = torch.tensor(new_elems).reshape(orig_shape)
            state_dict[origw_name] = neww
        return state_dict

    def build(self, bi: utils.BinaryInfo, irmod, params, outfile):
        import modman
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpmod_path = f'{tmpdir}/mod.so'

            with use_op_hack() as hctx:
                gemod, gefmod = modman.build_module(
                    irmod, params,
                    export_path=tmpmod_path, cig_make_space=False,
                    exportfn=self.get_gefmod_exportfn(),
                )

            # Get analysis data
            ana = utils.one_shot_ghidra_analyse(tmpmod_path)
            tgt_insn_bytes_pat = re.compile(r'48_8d_.._00_00_00_00')  # lea *, [rip]
            rip_at_load = (
                seq(ana['fns'].values())
                .filter(lambda f: f['insts'])  # filter out empty functions
                .flat_map(lambda f: f['insts'])  # mapcat all instructions
                .filter(lambda i: tgt_insn_bytes_pat.match(i['bytes']))
                .map(lambda i: i['base10_offset'] + i['nbytes'])  # get the address of the next instruction
                .first()
            )
            text_start = ana['memory_map']['.text']['base10_start']
            code_len = ana['memory_map']['.text']['size']
            rip_code_offset = text_start - rip_at_load  # rip + offset = text_start
            mblob = ana['dev_mblob']

            # Get ground truth
            # TODO: We may want to support cascading later
            with open(tmpmod_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                mm.seek(text_start - cfg.GHIDRA_BASE_ADDR)
                code = mm.read(code_len)
                mm.close()
                ground_truth = adler32(code)

            # Decide which layer(s) to instrument
            # We only use one instrumentation for now
            enabled_insts = []
            if bi.dig == 'nd':
                enabled_insts.append(hctx.instrumentations[-1])
            elif bi.dig == 'gn1':
                # This should be for the last forward dense. -1 would be the backward
                # dense's weight, which is an intermediate tensor.
                enabled_insts.append(hctx.instrumentations[-2])
            else:
                raise NotImplementedError

            # Patch args
            with open(tmpmod_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                for inst in hctx.instrumentations:
                    if inst in enabled_insts:
                        self.patch_inst_args(mm, inst, rip_code_offset, code_len)
                    else:
                        self.patch_inst_args(mm, inst, 0, 0)
                mm.close()

            # Get the new params (TODO: Operate directly on tvm ndarray?)
            torch_model = modman.get_torch_mod(bi.model_name, bi.dataset)
            state_dict = torch_model.state_dict()
            torch_model.load_state_dict(
                self.masked_state_dict(state_dict, enabled_insts, ground_truth)
            )
            scripted_model = torch.jit.trace(torch_model, torch.randn(bi.input_shape)).eval()
            _, new_params = tvm.relay.frontend.from_pytorch(scripted_model, [('input0', bi.input_shape)])

            # Build an otherwise identical module with the new weights - here we
            # use the same irmod so we don't need to do another DIG
            # instrumentation (if there was any).
            new_tmpmod_path = f'{tmpdir}/new.so'
            with use_op_hack() as _:
                modman.build_module(
                    irmod, new_params,
                    export_path=new_tmpmod_path, cig_make_space=False
                )
            new_ana = utils.one_shot_ghidra_analyse(new_tmpmod_path)
            new_mblob = new_ana['dev_mblob']

            assert new_mblob == mblob, f'{new_mblob=} {mblob=}'
            assert utils.extract_graph_json(new_tmpmod_path) == utils.extract_graph_json(tmpmod_path)

            # Graft new mblob into old mod
            with open(new_tmpmod_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                mm.seek(new_mblob['base10_offset'] - cfg.GHIDRA_BASE_ADDR)
                new_mblob_bytes = mm.read(new_mblob['header_nbytes'] + new_mblob['body_nbytes'])
                mm.close()
            with open(tmpmod_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                mm.seek(mblob['base10_offset'] - cfg.GHIDRA_BASE_ADDR)
                mm.write(new_mblob_bytes)
                mm.close()

            shutil.copyfile(tmpmod_path, outfile)

class CoopCIGV2ModuleBuilder:

    # Matches the magic in getkey.cc
    GROUND_TRUTH_MAGIC = 0x1DDECC41629D2D4B

    @staticmethod
    def get_gefmod_exportfn(orig_ll_outfile=None):
        def gefmod_exportfn(gefmod, fpath):
            if orig_ll_outfile:
                utils.ensure_dir_of(orig_ll_outfile)
                gefmod.module.imported_modules[0].save(orig_ll_outfile)
            gefmod.export_library(fpath, fcompile=None, options=[
                f'{cfg.unmasker_dir}/unmasker.o', f'{cfg.unmasker_dir}/getkey.o', '-g',
                # TODO: Automate building the unmasker
            ])
        return gefmod_exportfn

    @staticmethod
    def patch_inst_args(mm: mmap.mmap, inst, code_offset, code_len):
        return CoopCIGV1ModuleBuilder.patch_inst_args(mm, inst, code_offset, code_len)

    def build(self, bi: utils.BinaryInfo, irmod, params, outfile, add_sa=True):
        '''add_sa: Whether to also add SA CIGs to the built binary.'''

        import modman

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpmod_path = f'{tmpdir}/mod.so'

            with use_op_hack(coop_cig_ver=2) as hctx:
                _gemod, _gefmod = modman.build_module(
                    irmod, params,
                    export_path=tmpmod_path, cig_make_space=add_sa,
                    exportfn=self.get_gefmod_exportfn(),
                )

            if add_sa:
                cig_spots_fpath = f'{tmpdir}/cig-spots.json'
                cig.prepatch(tmpmod_path, tmpmod_path, cig_spots_fpath)

            # Get analysis data
            ana = utils.one_shot_ghidra_analyse(tmpmod_path)
            tgt_insn_bytes_pat = re.compile(r'48_8d_.._00_00_00_00')  # lea *, [rip]
            rip_at_load = (
                seq(ana['fns'].values())
                .filter(lambda f: f['insts'])  # filter out empty functions
                .flat_map(lambda f: f['insts'])  # mapcat all instructions
                .filter(lambda i: tgt_insn_bytes_pat.match(i['bytes']))
                .map(lambda i: i['base10_offset'] + i['nbytes'])  # get the address of the next instruction
                .first()
            )
            text_start = ana['memory_map']['.text']['base10_start']
            code_len = ana['memory_map']['.text']['size']
            rip_code_offset = text_start - rip_at_load  # rip + offset = text_start

            if add_sa:
                patcher = cig.CIGPatcher(
                    None, so_path=tmpmod_path, analysis=ana,
                    cig_spots=utils.load_json(cig_spots_fpath)
                )
                planner = cig.ScanAllCIGPlanner()
                patcher.plan(planner)
                patcher.patch(tmpmod_path)

            # Get ground truth
            # TODO: We may want to support cascading later
            with open(tmpmod_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                mm.seek(text_start - cfg.GHIDRA_BASE_ADDR)
                code = mm.read(code_len)
                mm.close()
                ground_truth = adler32(code)

            enabled_insts = []
            if bi.dig == 'gn1':
                enabled_insts.append(hctx.instrumentations[-1])
            else:
                raise NotImplementedError

            # Patch args and ground truth
            with open(tmpmod_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                for inst in hctx.instrumentations:
                    if inst in enabled_insts:
                        self.patch_inst_args(mm, inst, rip_code_offset, code_len)
                    else:
                        self.patch_inst_args(mm, inst, 0, 0)
                mm.seek(0)
                ground_truth_spot = mm.find(struct.pack('<Q', self.GROUND_TRUTH_MAGIC))
                mm.seek(ground_truth_spot)
                mm.write(struct.pack('<LL', ground_truth, 0))
                mm.close()

            shutil.copyfile(tmpmod_path, outfile)
