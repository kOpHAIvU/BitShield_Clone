# Ref: python/tvm/topi/x86/dense.py

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas
from tvm.contrib import mkl
from tvm.contrib import dnnl

from tvm.topi.x86.utils import get_simd_32bit_lanes
from tvm.topi import generic, tag
from tvm.topi.utils import traverse_inline, get_const_tuple
from tvm.topi.x86.tensor_intrin import dot_16x1x16_uint8_int8_int32_cascadelake

from . import ctx

import utils

# Remove the default task first
del autotvm.task.task.TASK_TABLE["dense_pack.x86"]

@autotvm.register_topi_compute("dense_pack.x86")
def dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense with transformed weight."""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if len(weight.shape) == 3:
        N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
        N = N * packw_bn
    else:
        N, _ = get_const_tuple(weight.shape)  # out_dim
    # create tuning space
    cfg.define_split(
        "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=3
    )
    cfg.define_split(
        "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=3
    )
    cfg.define_split(
        "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
    )
    cfg.define_split(
        "tile_inner",
        32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
        num_outputs=2,
        filter=lambda y: y.size[-1] <= 16,
    )
    if cfg.is_fallback:
        tvm.topi.x86.dense._default_dense_pack_config(cfg, M, N, K)

    if len(weight.shape) == 2:
        packw_bn = cfg["tile_x"].size[-1]
        packw_shape = (N // packw_bn, K, packw_bn)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            packw = tvm.te.placeholder(packw_shape, weight.dtype, name="packed_weight")
        else:
            packw = te.compute(
                packw_shape, lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight"
            )
    else:
        packw = weight

    def unmasked_packw(inst):
        return te.extern(
            packw.shape[:],
            [packw],
            lambda ins, outs: tvm.tir.call_cpacked(
                "unmask_weights",
                ins[0], outs[0],
                *(tvm.tir.const(x, 'int64') for x in [inst.offset_magic, inst.len_magic]),
                0
            ),
            name="C",
        )

    if ctx.curr_ophack_ctx is not None:
        hctx = ctx.curr_ophack_ctx
        if packw.name != 'packed_weight':
            # ^ Hack: It seems this function is invoked twice in the pipeline, once
            # during alter op and once during constant folding. Instrumenting in
            # the former phase seems to do nothing. Since the weights tensor seems
            # to have the name 'packed_weight' instead of 'placeholder' at that
            # time, we use this info to avoid calling new_instrumentation() twice.
            if hctx.coop_cig_ver == 1:
                inst = hctx.new_instrumentation(data, packw)
                packw = unmasked_packw(inst)
            elif hctx.coop_cig_ver == 2:
                # We only instrument the dense of the GN, which should be the
                # last dense.
                if hctx.curr_spot_idx == hctx.total_nspots - 1:
                    # We first mask the weights. Since we only work on the
                    # dense for DIG in v2, getkey doesn't need args.
                    inst = hctx.new_instrumentation(data, packw)
                    packw = te.compute(
                        packw.shape,
                        lambda x, y, z: utils.thread_last(
                            packw[x, y, z],
                            (tvm.tir.call_intrin, 'uint32', 'tir.reinterpret'),
                            (tvm.tir.call_intrin, 'uint32', 'tir.bitwise_xor', tvm.tir.call_extern(
                                'uint32', 'getkey',
                            )),
                            (tvm.tir.call_intrin, 'float', 'tir.reinterpret'),
                        )
                    )
                    packw = unmasked_packw(inst)
                hctx.curr_spot_idx += 1
        else:
            # In Co-op CIG v2, we use the first phase to count the total number
            # of denses.
            hctx.total_nspots += 1

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C
