# Kunlunxin (XPU) override of pairwise_distance.
#
# torch.nn.functional.pairwise_distance(x1, x2, p, eps, keepdim) computes
#   out[i] = ( sum_j |x1[i,j] - x2[i,j] + eps|^p ) ^ (1/p)      (p != 0)
#   p == 0:     count_j ( |x1[i,j] - x2[i,j] + eps| != 0 )
#   p == inf:   max_j |x1[i,j] - x2[i,j] + eps|
#   p == -inf:  min_j |x1[i,j] - x2[i,j] + eps|
# with x1/x2 broadcastable, batched over all leading dims (N = numel // D).
#
# Why an XPU-specific implementation (the generic op is wrong AND slow on XPU):
#
# 1) Masked loads are UNCONDITIONAL on triton_xpu.  tl.load(ptr, mask, other) is
#    lowered to an unmasked load (the mask/other operands are dropped) and
#    tl.where(mask, val, 0.0) is fused into the load's mask/other operands, so
#    masked-off lanes read real (cross-row / OOB) memory and the select cannot
#    correct them.  Every partial tail of a D-tile then injects garbage into the
#    reduction -- (1, 10000000) p=0 sums to 2442*4096 = 10002432, p=1/2/1.5 are
#    off by 2432 / 0.54 / nan; bf16 (2,3,4) with autotuned BLOCK_D=256 reads the
#    next rows for the tail lanes.  Fix: keep the memory completely free of
#    masked loads and of data-dependent address math on the hot path -- see 3).
#
# 2) Runtime control flow on this XPU backend is fragile: a kernel containing an
#    `if` (scf.if) together with a tt.reduce fails to legalize
#    ("failed to legalize operation 'tt.reduce' that was explicitly marked
#    illegal"), and a 1-D grid (grid=(N,)) with a reduce crashes the LLVM
#    backend.  Fix: p is a tl.constexpr (static ifs only; one cached compile per
#    p value) and every launch uses a 2-D grid (N, x) with 1-D [BLOCK_SIZE]
#    tiles (2-D [BM, BD] tiles + axis reduce also hit "out of resource:
#    uni_sram", see _euclidean_dist.py).
#
# 3) Bandwidth: tl.minimum(offset, D-1) (address clamp) and any masked load
#    prevent the vectorized contiguous load -- the same kernel goes 559 GB/s
#    (plain load) -> 249 GB/s (masked load) -> 36 GB/s (clamped).  So the hot
#    path is a FULL-CHUNK kernel whose pointer arithmetic is provably in-bounds
#    (pid_d < D // BLOCK_SIZE  =>  pid_d*BLOCK_SIZE + BLOCK_SIZE - 1 < D): no
#    mask, no clamp, just load + reduce.  Only the (rare, small) tail chunk and
#    the partial-reduction finalize use the clamped slow path.
#
# 4) No @libtuner: PAIRWISE_DISTANCE_CONFIGS has no kunlunxin entry so the
#    generic op could only see the nvidia autotune table (22 configs re-benched
#    per new D, and the "best" config may have BLOCK_D > D on XPU).  Fixed,
#    bounded 1-D tiles (BLOCK_SIZE = 4096 on the hot path).
#
# Accumulation is fp32 (fp16/bf16 inputs are upcast; XPU has fp64_enabled=False).
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# x ** p is decomposed as exp2(p * log2(x)).  Use the triton math builtins
# (math.exp2 / math.log2 -> LLVM exp2/log2): the xpu libdevice externs
# (triton.language.extra.xpu.libdevice.exp2f/log2f) return wrong results when
# applied to the 0-d scalar left by a tl.sum reduction (a 1e7-D p=1.5 run
# summed 1/8 of the partials), while the builtins are exact on the same input.
exp2 = tl.exp2
log2 = tl.log2

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# Width of the 1-D D-tile on the hot (full-chunk) path.  4096 keeps 2 x
# [4096] f32 tiles inside the XPU unified-buffer budget and stays ~560 GB/s.
_BLOCK_SIZE = 4096


@libentry()
@triton.jit
def pairwise_distance_d1_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    eps,
    P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """D == 1: the p-distance of a scalar pair is just the value (no reduction);
    batch many rows per program so small rows are not launch-bound. Grid (cdiv(N, BLOCK_N), 1)."""
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    safe = tl.minimum(rows, N - 1)
    a = tl.load(x1_ptr + safe).to(acc_dtype)
    b = tl.load(x2_ptr + safe).to(acc_dtype)
    diff = tl.abs(a - b + eps)
    if P == 0.0:
        res = (diff != 0).to(acc_dtype)
    else:
        # (|d|^P)^(1/P) == |d| within a couple ulp; the reference computes the
        # same double pow in fp64, so the exact fp32 value matches.
        res = diff
    tl.store(out_ptr + rows, res, row_mask)


@libentry()
@triton.jit
def pairwise_distance_single_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    D,
    eps,
    P: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """D <= BLOCK_SIZE: one program per row, whole row in one (clamped) tile. Grid (N, 1)."""
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < D
    # Clamped unmasked load (see header); D is small here so the clamp cost is
    # immaterial, and the explicit tl.where keeps the tail lanes at zero.
    safe = tl.minimum(offset, D - 1)
    a = tl.load(x1_ptr + pid_n * D + safe).to(acc_dtype)
    b = tl.load(x2_ptr + pid_n * D + safe).to(acc_dtype)
    diff = tl.abs(a - b + eps)

    if P == 2.0:
        part = tl.sum(tl.where(mask, diff * diff, 0.0))
        res = tl.sqrt(part)
    elif P == 1.0:
        part = tl.sum(tl.where(mask, diff, 0.0))
        res = part
    elif P == 0.0:
        part = tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0))
        res = part
    elif P == float("inf"):
        res = tl.max(tl.where(mask, diff, -float("inf")))
    elif P == float("-inf"):
        res = tl.min(tl.where(mask, diff, float("inf")))
    else:
        part = tl.sum(tl.where(mask, exp2(P * log2(diff)), 0.0))
        res = exp2((1.0 / P) * log2(part))
    tl.store(out_ptr + pid_n, res)


@libentry()
@triton.jit
def pairwise_distance_full_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    D,
    eps,
    P: tl.constexpr,
    FULL,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per (row, full chunk); FULL*BLOCK_SIZE <= D, so all loads are
    provably in-bounds: NO mask, NO address clamp -> contiguous vectorized loads."""
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(x1_ptr + pid_n * D + offset).to(acc_dtype)
    b = tl.load(x2_ptr + pid_n * D + offset).to(acc_dtype)
    diff = tl.abs(a - b + eps)

    if P == 2.0:
        part = tl.sum(diff * diff)
    elif P == 1.0:
        part = tl.sum(diff)
    elif P == 0.0:
        part = tl.sum((diff != 0).to(acc_dtype))
    elif P == float("inf"):
        part = tl.max(diff)
    elif P == float("-inf"):
        part = tl.min(diff)
    else:
        part = tl.sum(exp2(P * log2(diff)))
    tl.store(mid_ptr + pid_n * FULL + pid_d, part)


@libentry()
@triton.jit
def pairwise_distance_tail_kernel(
    x1_ptr,
    x2_ptr,
    mid_ptr,
    D,
    eps,
    P: tl.constexpr,
    TAIL_START,
    MID_SIZE,
    FULL,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row for the last (partial) chunk [D % BLOCK_SIZE, D).
    The clamped slow path, but it only touches BLOCK_SIZE elements per row."""
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    offset = TAIL_START + tl.arange(0, BLOCK_SIZE)
    mask = offset < D
    safe = tl.minimum(offset, D - 1)
    a = tl.load(x1_ptr + pid_n * D + safe).to(acc_dtype)
    b = tl.load(x2_ptr + pid_n * D + safe).to(acc_dtype)
    diff = tl.abs(a - b + eps)

    if P == 2.0:
        part = tl.sum(tl.where(mask, diff * diff, 0.0))
    elif P == 1.0:
        part = tl.sum(tl.where(mask, diff, 0.0))
    elif P == 0.0:
        part = tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0))
    elif P == float("inf"):
        part = tl.max(tl.where(mask, diff, -float("inf")))
    elif P == float("-inf"):
        part = tl.min(tl.where(mask, diff, float("inf")))
    else:
        part = tl.sum(tl.where(mask, exp2(P * log2(diff)), 0.0))
    # The tail partial is the LAST of the MID_SIZE partials: index FULL.
    tl.store(mid_ptr + pid_n * MID_SIZE + FULL, part)


@libentry()
@triton.jit
def pairwise_distance_finalize_kernel(
    mid_ptr,
    out_ptr,
    P: tl.constexpr,
    MID_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce one row's partials (a few KB) and apply the final op. Grid (N, 1)."""
    acc_dtype = tl.float64 if mid_ptr.type.element_ty == tl.float64 else tl.float32
    pid_n = tl.program_id(0)
    if P == float("inf"):
        acc = tl.full([BLOCK_SIZE], -float("inf"), acc_dtype)
        for start in range(0, MID_SIZE, BLOCK_SIZE):
            offset = start + tl.arange(0, BLOCK_SIZE)
            m = tl.load(mid_ptr + pid_n * MID_SIZE + tl.minimum(offset, MID_SIZE - 1))
            acc = tl.maximum(acc, tl.where(offset < MID_SIZE, m, -float("inf")))
        res = tl.max(acc)
    elif P == float("-inf"):
        acc = tl.full([BLOCK_SIZE], float("inf"), acc_dtype)
        for start in range(0, MID_SIZE, BLOCK_SIZE):
            offset = start + tl.arange(0, BLOCK_SIZE)
            m = tl.load(mid_ptr + pid_n * MID_SIZE + tl.minimum(offset, MID_SIZE - 1))
            acc = tl.minimum(acc, tl.where(offset < MID_SIZE, m, float("inf")))
        res = tl.min(acc)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=acc_dtype)
        for start in range(0, MID_SIZE, BLOCK_SIZE):
            offset = start + tl.arange(0, BLOCK_SIZE)
            m = tl.load(mid_ptr + pid_n * MID_SIZE + tl.minimum(offset, MID_SIZE - 1))
            acc += tl.where(offset < MID_SIZE, m, 0.0)
        s = tl.sum(acc)
        if P == 2.0:
            res = tl.sqrt(s)
        elif (P == 1.0) | (P == 0.0):
            res = s
        else:
            res = exp2((1.0 / P) * log2(s))
    tl.store(out_ptr + pid_n, res)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN PAIRWISE_DISTANCE")
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]

    # Empty feature dim: torch returns 0 for finite p; inf/-inf have no identity
    # element over an empty reduction and torch raises. Short-circuit here.
    if D == 0:
        if p == float("inf") or p == float("-inf"):
            raise RuntimeError(
                "pairwise_distance cannot compute the inf/-inf norm on an empty "
                "reduction dimension (no identity element)"
            )
        out = torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        if keepdim:
            out = out.unsqueeze(-1)
        return out

    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)
    mid_dtype = torch.float64 if x1.dtype == torch.float64 else torch.float32

    with torch_device_fn.device(x1.device):
        if D == 1:
            # Row-batched, no reduction; a 1-elem row does not amortize a
            # per-row program.
            pairwise_distance_d1_kernel[(triton.cdiv(N, 512), 1)](
                x1, x2, out, N, eps, p, 512
            )
        elif D < _BLOCK_SIZE:
            # Whole row fits one clamped tile; data is small, single launch.
            BLOCK_SIZE = triton.next_power_of_2(D)
            pairwise_distance_single_kernel[(N, 1)](
                x1, x2, out, D, eps, p, BLOCK_SIZE
            )
        else:
            # Fast path: full BLOCK_SIZE-wide chunks with in-bounds (unmasked,
            # unclamped) loads; a small clamped tail kernel covers the
            # remainder; a finalize kernel reduces the partials.  For tiny
            # chunk grids (few rows * few chunks) fall back to 1024-wide
            # chunks so the launch count stays high enough to saturate.
            if N * triton.cdiv(D, _BLOCK_SIZE) >= 128:
                BLOCK_SIZE = _BLOCK_SIZE
            else:
                BLOCK_SIZE = min(1024, triton.next_power_of_2(D))
            full = D // BLOCK_SIZE
            has_tail = (D % BLOCK_SIZE) != 0
            MID_SIZE = full + (1 if has_tail else 0)
            mid = torch.empty((N, MID_SIZE), device=x1.device, dtype=mid_dtype)
            pairwise_distance_full_kernel[(N, full)](
                x1, x2, mid, D, eps, p, full, BLOCK_SIZE
            )
            if has_tail:
                pairwise_distance_tail_kernel[(N, 1)](
                    x1, x2, mid, D, eps, p, full * BLOCK_SIZE, MID_SIZE, full,
                    BLOCK_SIZE
                )
            pairwise_distance_finalize_kernel[(N, 1)](
                mid, out, p, MID_SIZE,
                min(triton.next_power_of_2(MID_SIZE), _BLOCK_SIZE),
            )

    return out