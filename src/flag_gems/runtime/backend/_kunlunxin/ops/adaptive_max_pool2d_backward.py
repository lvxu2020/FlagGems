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
#
# Kunlunxin(XPU) vendor override for adaptive_max_pool2d_backward.
#
# The generic `src/flag_gems/ops/adaptive_max_pool2d_backward.py` scatters with
# `tl.atomic_add(grad_input_ptr + indices_val, ...)`, i.e. a data-dependent
# (discrete) address with a mask. On triton-XPU:
#   * `tl.atomic_add` is NOT atomic across programs and its mask/other
#     handling on data-dependent addresses is not reliable, and
#   * adjacent adaptive windows overlap whenever out_size does not exactly
#     divide in_size, so the same input position can be the argmax of several
#     output positions and MUST be accumulated (the generic scatter would lose
#     those updates).
#
# This implementation instead uses the proven gather pattern of
# `_kunlunxin/ops/max_pool2d_with_indices.py` (no atomics at all): every
# program owns a 1D block of grad_input positions; for each position (n, c,
# ih, iw) the set of output rows `oh` whose adaptive window contains `ih` is
# exactly
#     oh in [ floor(ih*out_h/in_h), ceil((ih+1)*out_h/in_h) - 1 ]
# (same for columns).  The kernel iterates that bounded candidate set, loads
# `indices[oh, ow]` and `grad_output[oh, ow]` at CLAMPED (always in-bounds)
# addresses, and accumulates `grad_output` when `indices == ih*in_w + iw`.
# All stores are contiguous 1D; all addresses are clamped, so no OOB access is
# possible even when callers feed non-canonical indices (e.g. the torch-XPU
# native forward of this op yields uninitialized index buffers).

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def adaptive_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_h,
    in_w,
    out_h,
    out_w,
    in_numel,
    in_numel_per_nc,
    out_stride_nc,
    # Number of candidate output rows/cols per input position (host-computed
    # upper bound, exact for the common cases). Iterations whose candidate falls
    # outside the window are masked to zero contributions.
    N_DH: tl.constexpr,
    N_DW: tl.constexpr,
    # Tiling
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a block of grad_input elements (flattened over
    n*c*in_h*in_w). For each input element, iterate over the (bounded) set of
    output positions whose adaptive window contains it; accumulate
    grad_output[oh, ow] whenever indices[oh, ow] marks this input element as
    the local argmax.
    """
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    in_bounds = offsets < in_numel
    # Tail lanes (offsets >= in_numel) must not be used to build any load
    # address: clamp them to element 0 (the store below masks them out), so
    # every index/grad load stays inside the tensor even for huge
    # out-of-range nc values.
    safe_offsets = tl.where(in_bounds, offsets, 0)

    # Decompose flat index -> (nc, ih, iw)
    nc = safe_offsets // in_numel_per_nc
    rem = safe_offsets - nc * in_numel_per_nc
    ih = rem // in_w
    iw = rem - ih * in_w

    # Candidate output rows/cols whose adaptive window contains (ih, iw):
    #   oh in [floor(ih*out_h/in_h), ceil((ih+1)*out_h/in_h) - 1]
    oh_lo = (ih * out_h) // in_h
    oh_hi = ((ih + 1) * out_h + in_h - 1) // in_h - 1
    ow_lo = (iw * out_w) // in_w
    ow_hi = ((iw + 1) * out_w + in_w - 1) // in_w - 1

    target = ih * in_w + iw
    out_base = nc * out_stride_nc

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # `tl.static_range` with the *exact* candidate count N_DH*N_DW (see
    # `_candidate_bound`): the loop is fully unrolled, which is what
    # triton-XPU's UnrollControl pass (and the scf.for pipeline in general)
    # handles reliably. Do NOT switch to `tl.range`: the runtime-loop variant
    # crashes TritonXPUUnrollControl for this loop body, and a 4x4=16-way
    # unroll was observed to fault the hardware (see README) -- the exact
    # bound keeps every shape at <= 3x3=9 unrolled iterations.
    for dh in tl.static_range(0, N_DH):
        oh = oh_lo + dh
        row_ok = dh <= (oh_hi - oh_lo)
        for dw in tl.static_range(0, N_DW):
            ow = ow_lo + dw
            cand_ok = row_ok & (dw <= (ow_hi - ow_lo))
            # Clamp to a legal address (0,0) of this plane; the value is
            # zeroed below via cand_ok, so the clamped lane contributes 0.
            safe_oh = tl.where(cand_ok, oh, 0)
            safe_ow = tl.where(cand_ok, ow, 0)
            ofs = out_base + safe_oh * out_w + safe_ow
            idx = tl.load(indices_ptr + ofs)
            gval = tl.load(grad_output_ptr + ofs)
            match = cand_ok & (idx == target)
            acc += tl.where(match, gval.to(tl.float32), 0.0)

    # Store on the clamped address (masked-off tail lanes would otherwise
    # point past the buffer end); the mask guarantees no value is written.
    tl.store(grad_input_ptr + safe_offsets, acc, mask=in_bounds)


@libentry()
@triton.jit
def adaptive_max_pool2d_backward_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_numel,
    in_numel_per_nc,
    out_stride_nc,
    out_numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast path for non-overlapping windows (in % out == 0 in both dims): every
    output position maps to a distinct input position, so a plain (non-atomic)
    scatter is correct. One program per block of output elements; O(out_numel)
    memory traffic instead of the gather's O(N_DH*N_DW * in_numel).

    The loaded index is used as a store target, so it must be range-checked:
    a caller may legitimately pass non-canonical indices (e.g. the torch-XPU
    native forward of this op returns uninitialized index buffers), and a
    garbage idx would produce a multi-petabyte out-of-bounds store. The store
    address is therefore CLAMPED to [0, in_numel-1] (never just masked):
    on triton-XPU a masked store still issues the access with whatever address
    the mask produced, so only a physically clamped address is safe; the mask
    (plus the idx range test) decides whether the clamped lane actually writes.
    """
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    in_bounds = offsets < out_numel
    # Clamp tail lanes to element 0 (mask guarantees no write); the loaded
    # idx is only used as a store target, so it must come from a real lane.
    safe = tl.where(in_bounds, offsets, 0)
    gval = tl.load(grad_output_ptr + safe)
    idx = tl.load(indices_ptr + safe)
    nc = safe // out_stride_nc
    gaddr = nc * in_numel_per_nc + idx
    # Physical clamp of the store address; the mask below is a write-enable
    # only (garbage idx lanes must not write, but even if the mask is dropped
    # by the backend the address itself stays inside the tensor).
    gaddr_clamped = tl.minimum(tl.maximum(gaddr, 0), in_numel - 1)
    idx_ok = (idx >= 0) & (idx < in_numel_per_nc)
    tl.store(grad_input_ptr + gaddr_clamped, gval, mask=in_bounds & idx_ok)


def _candidate_bound(in_size: int, out_size: int) -> int:
    """Exact max number of output windows (along one dim) containing one input
    position.

    count(i) = ceil((i+1)*O/I) - floor(i*O/I). With q = O//I and r = O%I:
    count(i) = q + g(i) where g(i) = ceil((i+1)*r/I) - floor(i*r/I) and
    g(i) in {0, 1, 2}. max g = 2 iff (i+1)*r mod I lands in [1, r-1] for
    some i -- solvable iff gcd(r, I) < r (the multiplicative structure of
    the residues); otherwise max g = 1. Exhaustively validated on I,O <= 64
    and 20000 random pairs.

    Why it must be exact: the gather kernel is fully unrolled via
    `tl.static_range` with N_DH*N_DW iterations, and triton-XPU's hardware
    faulted on the 16-way (4x4) unrolled form (see file header / README);
    keeping the exact count caps every shape at 3x3 = 9 iterations, matching
    the known-good shapes.
    """
    if in_size % out_size == 0:
        return 1
    if out_size % in_size == 0:
        return out_size // in_size
    r = out_size % in_size
    g = math.gcd(r, in_size)
    return out_size // in_size + (2 if g < r else 1)


def adaptive_max_pool2d_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for adaptive_max_pool2d.

    Args:
        grad_output: Gradient from the output, shape (N, C, out_H, out_W)
        self: The original input tensor, shape (N, C, in_H, in_W)
        indices: Indices of max values from forward pass, shape (N, C, out_H, out_W)

    Returns:
        grad_input: Gradient with respect to input, shape (N, C, in_H, in_W)
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL2D_BACKWARD")

    input_is_3d = self.dim() == 3
    if input_is_3d:
        self = self.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)
        indices = indices.unsqueeze(0)

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    in_n, in_c, in_h, in_w = self.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_input = torch.zeros(
        (in_n, in_c, in_h, in_w),
        device=grad_output.device,
        dtype=torch.float32,
    )

    in_numel = grad_input.numel()
    if in_numel == 0 or grad_output.numel() == 0:
        # Empty input or empty output: the kernel must not run (it would load
        # from an empty indices/grad_output buffer).
        result = grad_input.to(grad_output.dtype)
        return result.squeeze(0) if input_is_3d else result

    in_numel_per_nc = in_h * in_w
    out_stride_nc = out_h * out_w
    block_size = 1024

    # Non-overlapping windows (in % out == 0 in both dims) mean every input
    # position is written at most once -> a plain (non-atomic) scatter of
    # O(out_numel) traffic is correct and far faster than the gather path
    # (whose per-input-element candidate scan dominates for large downsampling
    # ratios). Anything else (window overlap, upsampling) goes through the
    # gather kernel which accumulates and needs no atomics either.
    no_overlap = (in_h % out_h == 0) and (in_w % out_w == 0)

    with torch_device_fn.device(grad_input.device):
        if no_overlap:
            out_numel = grad_output.numel()
            grid = (triton.cdiv(out_numel, block_size),)
            adaptive_max_pool2d_backward_scatter_kernel[grid](
                grad_output,
                indices,
                grad_input,
                in_numel,
                in_numel_per_nc,
                out_stride_nc,
                out_numel,
                block_size,
            )
        else:
            n_dh = _candidate_bound(in_h, out_h)
            n_dw = _candidate_bound(in_w, out_w)
            grid = (triton.cdiv(in_numel, block_size),)
            adaptive_max_pool2d_backward_kernel[grid](
                grad_output,
                indices,
                grad_input,
                in_h,
                in_w,
                out_h,
                out_w,
                in_numel,
                in_numel_per_nc,
                out_stride_nc,
                n_dh,
                n_dw,
                block_size,
            )

    result = grad_input.to(grad_output.dtype)
    return result.squeeze(0) if input_is_3d else result