# Ref: python/tvm/relay/op/strategy/x86.py

from tvm import tir, topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.meta_schedule import is_meta_schedule_enabled
from tvm.relay.ty import is_dynamic
from tvm.target import Target
from tvm.te import SpecializedCondition
from tvm.topi.x86.utils import target_has_vnni

from tvm.relay import op as _op
from tvm.relay.op.strategy.generic import *

from . import topi as mytopi  # Hacked topi

@dense_strategy.register("cpu")
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense x86 strategy"""
    strategy = _op.OpStrategy()
    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
    strategy.add_implementation(
        wrap_compute_dense(topi.x86.dense_nopack),
        wrap_topi_schedule(topi.x86.schedule_dense_nopack),
        name="dense_nopack.x86",
        plevel=5,
    )

    strategy.add_implementation(
        wrap_compute_dense(mytopi.dense_pack),
        wrap_topi_schedule(topi.x86.schedule_dense_pack),
        name="dense_pack.x86",
        plevel=10,
    )

    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()

    if need_auto_scheduler_layout or need_meta_schedule_layout:
        strategy.add_implementation(
            wrap_compute_dense(
                topi.nn.dense,
                need_auto_scheduler_layout=need_auto_scheduler_layout,
                need_meta_schedule_layout=need_meta_schedule_layout,
            ),
            naive_schedule,
            name="dense.generic",
            plevel=11,
        )

    if "cblas" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_cblas),
                wrap_topi_schedule(topi.x86.schedule_dense_cblas),
                name="dense_cblas.x86",
                plevel=13,
            )
    if "mkl" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_mkl),
                wrap_topi_schedule(topi.x86.schedule_dense_mkl),
                name="dense_mkl.x86",
                plevel=14,
            )
    if "dnnl" in target.libs:
        with SpecializedCondition(same_type and dtype == "float32"):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_dnnl),
                wrap_topi_schedule(topi.x86.schedule_dense_dnnl),
                name="dense_dnnl.x86",
                plevel=15,
            )
    return strategy

@dense_pack_strategy.register("cpu")
def dense_pack_strategy_cpu(attrs, inputs, out_type, target):
    """dense_pack x86 strategy"""
    strategy = _op.OpStrategy()

    if (
        inputs[0].dtype == "uint8"
        and inputs[1].dtype == "int8"
        and out_type.dtype == "int32"
        and attrs["weight_layout"] == "NC16n4c"
    ):
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_vnni),
            wrap_topi_schedule(topi.x86.schedule_dense_vnni),
            name="dense_vnni.x86",
            plevel=12,
        )
    else:
        strategy.add_implementation(
            wrap_compute_dense(mytopi.dense_pack),
            wrap_topi_schedule(topi.x86.schedule_dense_pack),
            name="dense_pack.x86",
            plevel=10,
        )

    return strategy
