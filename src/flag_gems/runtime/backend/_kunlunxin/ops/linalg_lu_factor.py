import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _lu_swap_rows_kernel(
    LU, PIVOTS, M, N, K, J, BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Swap rows J and PIVOTS[batch, J] - 1 of each batch matrix.

    J is a runtime scalar (never baked in): on TritonXPU every vector index
    must keep the form ``scalar-total + arange * stride`` and every mask the
    pure-tail form ``arange < tail``, otherwise the store is miscompiled or
    the device raises a kernel exception.  (The same three kernels are shared
    with linalg_slogdet, which calls them with a per-launch constant J.)
    """
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    columns = block * BLOCK_N + tl.arange(0, BLOCK_N)
    pivot_row = tl.load(PIVOTS + batch * K + J).to(tl.int64) - 1
    base = LU + batch * M * N
    ntail = N - (block * BLOCK_N)
    msk = tl.arange(0, BLOCK_N) < ntail
    current = tl.load(base + J * N + columns, mask=msk, other=0.0)
    pivot = tl.load(base + pivot_row * N + columns, mask=msk, other=0.0)
    tl.store(base + J * N + columns, pivot, mask=msk)
    tl.store(base + pivot_row * N + columns, current, mask=msk)


@triton.jit
def _lu_scale_column_kernel(
    LU, M, N, J, BLOCKS: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    batch = pid // BLOCKS
    block = pid % BLOCKS
    rbase = (J + 1 + block * BLOCK_M) * N
    ar = tl.arange(0, BLOCK_M)
    base = LU + batch * M * N
    pivot = tl.load(base + J * N + J)
    tail = M - (J + 1 + block * BLOCK_M)
    msk = ar < tail
    values = tl.load(base + rbase + ar * N + J, mask=msk, other=0.0)
    tl.store(base + rbase + ar * N + J, values / pivot, mask=msk)


@triton.jit
def _lu_update_trailing_kernel(
    LU,
    M,
    N,
    J,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // (ROWS * BLOCKS)
    row = J + 1 + (pid // BLOCKS) % ROWS
    block = pid % BLOCKS
    cbase = J + 1 + block * BLOCK_N
    ar = tl.arange(0, BLOCK_N)
    base = LU + batch * M * N
    ntail = N - cbase
    msk = ar < ntail
    multiplier = tl.load(base + row * N + J)
    urow = tl.load(base + J * N + cbase + ar, mask=msk, other=0.0)
    offsets = base + row * N + cbase + ar
    values = tl.load(offsets, mask=msk, other=0.0)
    tl.store(offsets, values - multiplier * urow, mask=msk)


@triton.jit
def _lu_find_pivot_kernel(
    LU,
    PIVOTS,
    M,
    N,
    K,
    J,
    M2: tl.constexpr,
    N2: tl.constexpr,
):
    """Pivot row for column J: max |LU[r, J]| over r in [J, M), one program
    per batch element.

    The whole column is handled in one full M2-lane vector (M2 = next_pow2(M)
    is a per-shape constant), so the two reduces (tl.max / tl.min) stay at
    kernel top level - tt.reduce is explicitly illegal inside runtime loops on
    this backend.  No tl.argmax either (index-reduce is not legalized, see
    linalg_slogdet); the pivot row is the smallest row achieving the maximum
    (LAPACK first-tie semantics) via min(where(cands == mx, rows, M2)).

    Padded rows [M, M2) of the workspace always hold 0.0 (see the caller), so
    their candidates are 0.0 and they lose every tie against a real row; the
    (rows < M) guard makes them -1.0 anyway.
    """
    batch = tl.program_id(0)
    rows = tl.arange(0, M2)
    vals = tl.load(LU + batch * M2 * N2 + rows * N2 + J)
    av = tl.abs(vals)
    cands = tl.where((rows >= J) & (rows < M), av, -1.0)
    mx = tl.max(cands, axis=0)
    first = tl.min(tl.where(cands == mx, rows, M2), axis=0)
    row = tl.where(first == M2, J, first)
    tl.store(PIVOTS + batch * K + J, (row + 1).to(tl.int32))


def _check_linalg_lu_factor(input, pivot):
    if input.dim() < 2:
        raise RuntimeError(
            "torch.linalg.lu_factor: Expected input to have at least 2 dimensions, "
            f"got {input.dim()}"
        )
    if input.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently supports float32 and float64 only, "
            f"got {input.dtype}"
        )
    if input.shape[-2] == 0 or input.shape[-1] == 0:
        raise NotImplementedError(
            "FlagGems linalg_lu_factor currently does not support empty matrices"
        )
    if not isinstance(pivot, bool):
        raise TypeError(f"pivot must be a bool, got {type(pivot)}")


def _linalg_lu_factor(input, pivot):
    _check_linalg_lu_factor(input, pivot)
    if not pivot:
        raise NotImplementedError(
            "Kunlunxin linalg_lu_factor does not support pivot=False: "
            "the vendor lu_factor_ex primitive rejects it and no XPU-safe "
            "no-pivot kernel is available"
        )

    x = input.contiguous()
    m, n = x.shape[-2:]
    k = min(m, n)
    batch = x.numel() // (m * n)
    x2 = x.reshape(batch, m, n)
    # Workspace in per-shape power-of-two dims so every kernel can take full
    # vectors; padded rows [M, M2) are zeroed (they must lose every pivot
    # tie), the linear index stays within int32 for the supported shapes.
    m2 = max(2, triton.next_power_of_2(m))
    n2 = max(2, triton.next_power_of_2(n))
    work = torch.empty(batch, m2, n2, dtype=x.dtype, device=x.device)
    work[:, :m, :n].copy_(x2)
    work[:, m:, :].zero_()
    pivots = torch.empty(batch, k, dtype=torch.int32, device=x.device)

    with torch_device_fn.device(x.device):
        for j in range(k):
            _lu_find_pivot_kernel[(batch,)](
                work,
                pivots,
                m,
                n,
                k,
                j,
                m2,
                n2,
                num_warps=4,
            )
            swap_blocks = triton.cdiv(n2, 64)
            _lu_swap_rows_kernel[(batch * swap_blocks,)](
                work,
                pivots,
                m2,
                n2,
                k,
                j,
                BLOCKS=swap_blocks,
                BLOCK_N=64,
                num_warps=4,
            )
            scale_blocks = triton.cdiv(m2 - j - 1, 64)
            _lu_scale_column_kernel[(batch * scale_blocks,)](
                work,
                m2,
                n2,
                j,
                BLOCKS=scale_blocks,
                BLOCK_M=64,
                num_warps=4,
            )
            trailing_blocks = triton.cdiv(n2 - j - 1, 128)
            _lu_update_trailing_kernel[(batch * (m2 - j - 1) * trailing_blocks,)](
                work,
                m2,
                n2,
                j,
                ROWS=m2 - j - 1,
                BLOCKS=trailing_blocks,
                BLOCK_N=128,
                num_warps=4,
            )
    lu = work[:, :m, :n].contiguous().reshape(x.shape)
    return lu, pivots.reshape(x.shape[:-2] + (k,))


def linalg_lu_factor(input, *, pivot=True):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR")
    return _linalg_lu_factor(input, pivot)


def _resolve_linalg_lu_factor_out_args(input, LU, pivots):
    if LU is None or pivots is None:
        raise TypeError(
            "linalg_lu_factor(): LU and pivots must both be provided " "for out variant"
        )
    if LU.device != input.device or pivots.device != input.device:
        raise RuntimeError("linalg_lu_factor(): out tensors must be on input's device")
    if LU.dtype != input.dtype:
        raise RuntimeError("linalg_lu_factor(): LU out tensor must match input dtype")
    if pivots.dtype != torch.int32:
        raise RuntimeError(
            "linalg_lu_factor(): pivots out tensor must have dtype int32"
        )
    return LU, pivots


def linalg_lu_factor_out(input, *, pivot=True, LU=None, pivots=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_LU_FACTOR_OUT")
    lu_out, pivots_out = _resolve_linalg_lu_factor_out_args(input, LU, pivots)
    lu, pivots_result = _linalg_lu_factor(input, pivot)
    lu_out.resize_(lu.shape)
    pivots_out.resize_(pivots_result.shape)
    lu_out.copy_(lu)
    pivots_out.copy_(pivots_result)
    return lu_out, pivots_out