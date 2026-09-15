# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kunlunxin (XPU) override of mish / mish_.
#
# The generic `flag_gems.ops.mish` uses pointwise_dynamic without an explicit
# CodeGenConfig, so on XPU it specializes the kernel per input shape
# (per-shape recompile) and runs with the default codegen knobs -> ~0.03x gems
# speedup (41.5ms vs 1.45ms torch for [4096,4096] fp16; 167ms vs 5.75ms for
# [1024,65536]). Following the established Kunlunxin unary recipe (exp /
# log1p / silu / mish_backward), the same kernel body is recompiled with an
# explicit bounded 1D-tile CodeGenConfig: kunlunAutoGrid=True +
# prefer_1d_tile + unroll_num=8 + buffer_size_limit=4096 +
# isCloseVectorization=True. isCloseVectorization must stay CLOSED: with
# vectorization OPEN the vectorized `log` MISCOMPILES bf16 (~1.6% of elements
# off by exactly +ln(2)=0.6931, see log1p.py); CLOSED fixes it at negligible
# perf cost.
#
# Kernel body / math are unchanged from the generic implementation
# (zero correctness risk): `x * tanh(log(1 + exp(x)))` staged in fp32 with
# `tl_extra_shim.tanh` (the XPU libdevice tanhf/htanh) + triton-core
# `tl.exp`/`tl.log`. Probe (2026-09-14) confirms on the mish-relevant domain
# (|x| <= 7, softplus in [0, 7.001]) that tl.exp / tl.log match IEEE and
# tanh max abs err is 1.8e-7; unlike tl.log2/exp2 (natural logs on this
# backend), tl.exp/tl.log are genuine. Note: tanh(100)=NaN on this libdevice
# (>=88.7 with the x>20 guard absent) - outside the test domain (randn).
import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_tanh = tl_extra_shim.tanh

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def mish_func(x):
    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    x_fp32 = x.to(tl.float32)
    return (x_fp32 * _tanh(tl.log(1 + tl.exp(x_fp32)))).to(x.dtype)


def mish(A):
    logger.debug("GEMS_KUNLUNXIN MISH")
    return mish_func(A)


def mish_(A):
    logger.debug("GEMS_KUNLUNXIN MISH_")
    return mish_func(A, out0=A)