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

"""Kunlunxin(XPU) implementation of torch.cholesky_inverse.

The generic implementation (src/flag_gems/ops/cholesky_inverse.py) launches
ONE program per batch element and runs the O(N^3) triangular substitution plus
O(N^3) symmetric product fully serially in that single program, which on XPU
collapses to one core (~4.4 s at N=256 vs 6.1 ms torch -> 0.001x).

This vendor version parallelizes both phases with n-fold program parallelism
and 1-D [BLOCK_N] register vectors:

1. _cholesky_tri_inv_kernel: grid = (batch, n). Program (b, i) computes row i
   of L_inv by solving L^T y = e_i with back substitution, holding the whole
   row in a [BLOCK_N] register vector y_vec; each step is one masked column
   load of L, one 1-D dot (tl.sum) and one select.  n programs in flight,
   O(n^2) vector work per program.

2. _cholesky_sym_kernel: grid = (batch, n). Program (b, i) computes row i of
   A_inv = L_inv^T @ L_inv as a [BLOCK_N] register reduction over k
   (row-parallel GEMV).  The k-loop starts at k = i because L_inv[k, i] = 0
   for k < i (L_inv is lower triangular), halving the work for free.

All addresses are affine in (program_id, loop variables, tl.arange): the
vector loads/stores use only the masked, contiguous-BLOCK_N pattern that the
XPU backend handles, and there are no data-dependent (gather/scatter)
addresses, so no physical address clamping is needed.  Index arithmetic is
kept in int64 (pid/j/k cast explicitly) so batched matrices with
numel > 2^31 do not silently wrap.

upper=True is remapped to the lower path by transposing the input:
  cholesky_inverse(U, upper=True) computes (U^T U)^{-1} which equals
  cholesky_inverse(U^T, upper=False) (A = U^T U = L L^T with L = U^T).
This keeps a single code path and avoids strided column stores in the
triangular phase.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _cholesky_tri_inv_kernel(
    L_ptr,
    L_inv_ptr,
    N,
    batch_stride,
    stride_row,
    stride_col,
    BLOCK_N: tl.constexpr,
):
    """Inverse of a lower-triangular matrix, one row per program.

    Program (pid, pid_i) computes row pid_i of L_inv, i.e. solves L^T y = e_i
    by back substitution: with y in the register vector y_vec,
        y_j = (delta_{ij} - sum_{k>j} L[k, j] y_k) / L[j, j],  j = pid_i .. 0
    (y_j = 0 for j > pid_i since L_inv is lower triangular).
    """
    pid = tl.program_id(0)
    pid_i = tl.program_id(1)
    base = pid.to(tl.int64) * batch_stride

    offs = tl.arange(0, BLOCK_N)
    offs64 = offs.to(tl.int64)
    col_mask = offs < N

    y_vec = tl.zeros((BLOCK_N,), dtype=L_ptr.dtype.element_ty)
    for t in range(pid_i + 1):
        j = pid_i - t
        j64 = j.to(tl.int64)
        # column j of L (padding lanes masked off)
        col_j = tl.load(
            L_ptr + base + offs64 * stride_row + j64 * stride_col,
            mask=col_mask,
            other=0.0,
        )
        s = tl.sum(col_j * y_vec)
        l_jj = tl.load(L_ptr + base + j64 * stride_row + j64 * stride_col)
        val = tl.where(j == pid_i, 1.0, -s) / l_jj
        y_vec = tl.where(offs == j, val, y_vec)

    tl.store(
        L_inv_ptr
        + base
        + pid_i.to(tl.int64) * stride_row
        + offs64 * stride_col,
        y_vec,
        mask=col_mask,
    )


@libentry()
@triton.jit
def _cholesky_sym_kernel(
    L_inv_ptr,
    Out_ptr,
    N,
    batch_stride,
    stride_row,
    stride_col,
    out_stride_row,
    out_stride_col,
    BLOCK_N: tl.constexpr,
):
    """A_inv = L_inv^T @ L_inv, one output row per program.

    Program (pid, pid_i) computes row pid_i:
        Out[i, :] = sum_{k>=i} L_inv[k, i] * L_inv[k, :]
    as a [BLOCK_N] register vector load/fma per k (contiguous masked loads).
    The k-loop starts at k = pid_i because L_inv[k, i] = 0 for k < i.
    """
    pid = tl.program_id(0)
    pid_i = tl.program_id(1)
    base = pid.to(tl.int64) * batch_stride
    pid_i64 = pid_i.to(tl.int64)

    offs = tl.arange(0, BLOCK_N)
    offs64 = offs.to(tl.int64)
    col_mask = offs < N

    acc = tl.zeros((BLOCK_N,), dtype=L_inv_ptr.dtype.element_ty)
    for k in range(pid_i, N):
        k64 = k.to(tl.int64)
        l_ki = tl.load(
            L_inv_ptr + base + k64 * stride_row + pid_i64 * stride_col
        )
        row_k = tl.load(
            L_inv_ptr + base + k64 * stride_row + offs64 * stride_col,
            mask=col_mask,
            other=0.0,
        )
        acc += l_ki * row_k

    tl.store(
        Out_ptr
        + base
        + pid_i64 * out_stride_row
        + offs64 * out_stride_col,
        acc,
        mask=col_mask,
    )


def cholesky_inverse(L, upper=False):
    """Compute the inverse of a symmetric positive-definite matrix from its
    Cholesky decomposition (XPU vendor implementation).

    Given the Cholesky factor L (lower) where A = L @ L^T, computes A^{-1}.
    For upper=True, given U where A = U^T @ U, computes A^{-1}
    (remapped to the lower path on L = U^T).
    """
    logger.debug("GEMS CHOLESKY_INVERSE (kunlunxin)")
    assert L.dtype in (
        torch.float32,
        torch.float64,
    ), "cholesky_inverse only supports float32 and float64"

    if L.numel() == 0:
        return L

    shape = L.shape
    if len(shape) < 2:
        raise ValueError("Input must be at least 2D")

    n = shape[-1]
    m = shape[-2]
    if n != m:
        raise ValueError("Input must be a square matrix")

    # Flatten batch dims
    if len(shape) == 2:
        batch_size = 1
        L_flat = L.unsqueeze(0).contiguous()
    else:
        batch_size = 1
        for dim in shape[:-2]:
            batch_size *= dim
        L_flat = L.reshape(batch_size, n, n).contiguous()

    if upper:
        # A = U^T @ U  ==  (U^T) @ (U^T)^T  with the lower factor L = U^T.
        L_flat = L_flat.transpose(-2, -1).contiguous()

    batch_stride = L_flat.stride(0)
    stride_row = L_flat.stride(1)
    stride_col = L_flat.stride(2)
    BLOCK_N = triton.next_power_of_2(n)

    # Step 1: triangular inverse (row-parallel)
    L_inv = torch.zeros_like(L_flat)
    grid = (batch_size, n)

    with torch.no_grad():
        with torch_device_fn.device(L_flat.device):
            _cholesky_tri_inv_kernel[grid](
                L_flat,
                L_inv,
                n,
                batch_stride,
                stride_row,
                stride_col,
                BLOCK_N=BLOCK_N,
                isCloseUnrollControl=True,
            )

            # Step 2: symmetric product (row-parallel)
            output = torch.empty_like(L_flat)
            _cholesky_sym_kernel[grid](
                L_inv,
                output,
                n,
                batch_stride,
                stride_row,
                stride_col,
                output.stride(1),  # out_stride_row
                output.stride(2),  # out_stride_col
                BLOCK_N=BLOCK_N,
                isCloseUnrollControl=True,
            )

    # Reshape to original shape
    if len(shape) == 2:
        output = output.squeeze(0)
    else:
        output = output.reshape(shape)

    return output