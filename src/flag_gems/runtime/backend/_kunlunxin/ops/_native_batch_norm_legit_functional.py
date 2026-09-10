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

# NOTE (kunlunxin / XPU rewrite of _native_batch_norm_legit_functional):
# The generic src/flag_gems/ops/_native_batch_norm_legit_functional.py uses the fused
# batch_norm_forward_kernel with a 2D [BLOCK_M, BLOCK_N] tile on grid=(feat_dim,). On XPU
# that 2D-tile kernel fails to compile for all non-trivial shapes (verified on the test
# matrix: only the [16, 1] tile of shape (16, 3) compiles; (32,32,32) and up die with
# 'triton_xpu.convert_layout' op requires the same shape for all operands and results /
# TritonXPUUnrollControl / OutOfResources: uni_sram). This is the same 2D-tile compile
# wall the vendor batch_norm (runtime/backend/_kunlunxin/ops/batch_norm.py) already hit;
# it was solved there with the 3-stage contiguous path (stats kernel over [N*C] slices +
# per-channel combine kernel + normalize kernel over [N*C] slices), which compiles fine on
# XPU. We reuse those kernels and add a combine variant with SEPARATE in/out running-stat
# pointers, because _native_batch_norm_legit_functional is FUNCTIONAL: torch (verified on
# this device) returns the updated running stats as NEW outputs and does not mutate the
# input running_mean / running_var in place (the generic implementation does mutate).

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

from .batch_norm import (
    batch_norm_normalize_kernel,
    batch_norm_stats_kernel,
    make_3d_for_bn,
)

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


@libentry()
@triton.jit
def nbnlf_combine_kernel(
    part_sum_pointer,  # [N*C] f32, layout [n, c]
    part_sqsum_pointer,  # [N*C] f32, layout [n, c]
    mean_pointer,  # [C] f32 out (save_mean, fp32)
    inv_std_pointer,  # [C] f32 out (save_var = 1/sqrt(var+eps), fp32)
    run_mean_in_pointer,  # [C] input dtype (NOT mutated)
    run_var_in_pointer,  # [C] input dtype (NOT mutated)
    run_mean_out_pointer,  # [C] input dtype out
    run_var_out_pointer,  # [C] input dtype out
    batch_dim,
    feat_dim,
    count,  # batch_dim * spatial_dim
    momentum,
    eps,
    TILE_N: tl.constexpr,
):
    # One program per channel. Reduce the batch_dim partial (sum, sqsum) values for this
    # channel (strided by feat_dim), then compute mean / inv_std and the updated running
    # statistics into the OUT pointers (functional; the IN tensors stay untouched).
    c = tl.program_id(axis=0)
    idx = tl.arange(0, TILE_N)
    # XPU pitfall: mask= on a strided (discrete) access is silently dropped, so a
    # partial tail tile (batch_dim not a power of two) reads out of bounds. Clamp the
    # address into the valid range and zero the out-of-range contribution instead.
    idx_c = tl.minimum(idx, batch_dim - 1)
    mask = idx < batch_dim
    part_sum = tl.load(part_sum_pointer + c + idx_c * feat_dim)
    part_sqsum = tl.load(part_sqsum_pointer + c + idx_c * feat_dim)
    part_sum = tl.where(mask, part_sum, 0.0)
    part_sqsum = tl.where(mask, part_sqsum, 0.0)
    ssum = tl.sum(part_sum)
    sqsum = tl.sum(part_sqsum)
    mean = ssum / count
    var = sqsum / count - mean * mean
    inv_std = rsqrt(var + eps)
    tl.store(mean_pointer + c, mean)
    tl.store(inv_std_pointer + c, inv_std)

    run_mean_in = tl.load(run_mean_in_pointer + c).to(tl.float32)
    run_var_in = tl.load(run_var_in_pointer + c).to(tl.float32)
    unbiased_var = var * count / (count - 1)
    tl.store(
        run_mean_out_pointer + c,
        ((1 - momentum) * run_mean_in + momentum * mean).to(
            run_mean_out_pointer.dtype.element_ty
        ),
    )
    tl.store(
        run_var_out_pointer + c,
        ((1 - momentum) * run_var_in + momentum * unbiased_var).to(
            run_var_out_pointer.dtype.element_ty
        ),
    )


def _native_batch_norm_legit_functional(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    """Functional batch normalization returning (out, save_mean, save_var, running_mean_out,
    running_var_out). save_var is inv_std = 1/sqrt(var + eps), matching torch. The running
    statistics are returned as NEW tensors; the input running_mean / running_var are not
    mutated (verified torch semantics on this device)."""
    logger.debug("GEMS_KUNLUNXIN _NATIVE_BATCH_NORM_LEGIT_FUNCTIONAL")

    input_3d = make_3d_for_bn(input).contiguous()  # [N, C, S] contiguous
    batch_dim, feat_dim, spatial_dim = input_3d.shape
    n_slices = batch_dim * feat_dim
    count = batch_dim * spatial_dim

    output = torch.empty_like(input_3d)
    input_flat = input_3d.reshape(-1)
    output_flat = output.reshape(-1)

    tile_s = min(triton.next_power_of_2(spatial_dim), 4096) if spatial_dim > 0 else 1

    mean_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
    inv_std_f = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

    if training:
        # Stage 1: per-(n, c) partial sum / sum-of-squares over contiguous spatial run.
        part_sum = torch.empty(n_slices, device=input.device, dtype=torch.float32)
        part_sqsum = torch.empty(n_slices, device=input.device, dtype=torch.float32)
        # Updated running statistics go to fresh output tensors (functional semantics).
        run_mean_out = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        run_var_out = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
        with torch_device_fn.device(input.device):
            batch_norm_stats_kernel[(n_slices,)](
                input_flat, part_sum, part_sqsum, spatial_dim, TILE_S=tile_s
            )
            # Stage 2: combine batch partials -> per-channel mean / inv_std AND write the
            # updated running stats into the out pointers in a single launch.
            nbnlf_combine_kernel[(feat_dim,)](
                part_sum,
                part_sqsum,
                mean_f,
                inv_std_f,
                running_mean,
                running_var,
                run_mean_out,
                run_var_out,
                batch_dim,
                feat_dim,
                count,
                momentum,
                eps,
                TILE_N=triton.next_power_of_2(batch_dim),
            )
        # torch returns all outputs in the input dtype (verified on this device).
        save_mean = mean_f.to(input.dtype)
        save_var = inv_std_f.to(input.dtype)
    else:
        # Eval: use the running statistics as-is; torch returns EMPTY save_mean / save_var
        # and the (unchanged) running_mean / running_var as the last two outputs.
        mean_f = running_mean.to(torch.float32)
        inv_std_f = torch.rsqrt(running_var.to(torch.float32) + eps)
        run_mean_out = running_mean
        run_var_out = running_var
        save_mean = torch.empty((0,), dtype=input.dtype, device=input.device)
        save_var = torch.empty((0,), dtype=input.dtype, device=input.device)

    has_weight = weight is not None
    has_bias = bias is not None
    with torch_device_fn.device(input.device):
        batch_norm_normalize_kernel[(n_slices,)](
            input_flat,
            output_flat,
            mean_f.contiguous(),
            inv_std_f.contiguous(),
            weight if has_weight else input_flat,
            bias if has_bias else input_flat,
            feat_dim,
            spatial_dim,
            HAS_WEIGHT=has_weight,
            HAS_BIAS=has_bias,
            TILE_S=tile_s,
        )

    return (
        output.view_as(input),
        save_mean,
        save_var,
        run_mean_out,
        run_var_out,
    )