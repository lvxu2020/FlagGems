# Copyright 2026, The FlagOS Contributors.
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
# Kunlunxin(XPU) backend implementation of linalg_ldl_solve.
#
# The general implementation (flag_gems/ops/linalg_ldl_solve.py) is not usable
# on the XPU backend: its runtime `while k < n` loops combined with masked
# 1-D load/store of the RHS rows are miscompiled by TritonXPU for n >= 64
# (measured: test_linalg_ldl_solve shape (64,64)/(128,128) produce ~0.1-0.2
# relative errors, while n <= 32 happens to be correct).
#
# This implementation uses the XPU-proven pattern of
# _kunlunxin/ops/linalg_solve_triangular.py:
#   * one program per (batch, RHS column-slice) of KS=64 lanes;
#   * serial row sweep with runtime-bound `for` loops (no `while` at all —
#     scf.while + masks is the known XPU miscompile trigger);
#   * loads/stores use only a pure column-tail mask (cm = col < nrhs);
#   * no tl.dot / tl.sum / 2-D tiles / tl.trans / tl.argmax;
#   * RHS is padded to a multiple of KS (uninitialized tail: every kernel
#     access is guarded by the all-true col-tail mask and column lanes are
#     independent, so the tail can never leak into the output columns).
#
# The Bunch-Kaufman pivot structure (1x1 steps, 2x2 blocks, row swaps) is
# driven by a host-computed step plan (one int32 per block), exactly mirroring
# the generic algorithm's while/k+=2 control flow without a while loop.  The
# row swaps are emitted unconditionally (swapping row k with itself is a
# no-op), so no data-dependent branches guard them; the only runtime branch is
# the uniform `if ip > 0` block-type dispatch.
#
# Device notes:
#   * The XPU device has no native fp64: f64 tensors are silently materialised
#     as f32 (torch_xmlir XDNN), so f64 inputs reach the kernel as f32 and the
#     output has f32 dtype (same as the native XDNN ldl_solve).  Any test
#     asserting res.dtype == torch.float64 cannot pass on this device (NON_BUG,
#     same as float_power).
#   * complex64: `A @ A.mT` (input generation of the test) has no XDNN
#     implementation ("not supported type in common_matmul"), so the c64 tests
#     cannot reach any ldl_solve implementation on this device (reference-side
#     NON_BUG).  The kernel below supports c64 anyway via view_as_real.

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

LDL_SOLVE_KS = 64  # RHS column-slice lane width (single-CTA width, trsm-proven)
REAL_DTYPES = (torch.float32, torch.float64)
COMPLEX_DTYPES = (torch.complex64, torch.complex128)


@triton.jit
def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@triton.jit
def _cdiv(nr, ni, dr, di):
    denom = dr * dr + di * di
    return (nr * dr + ni * di) / denom, (ni * dr - nr * di) / denom


@libentry()
@triton.jit
def _ldl_solve_real_kernel(
    LD,
    PIVOTS,
    X,
    STEPS,
    n,
    nrhs,
    nsteps,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    N: tl.constexpr,
    KS: tl.constexpr,
    NS: tl.constexpr,
):
    """One (batch, RHS column-slice) solve: X = A^-1 B in-place.

    grid = (batch * NS,): program ``pid`` owns batch ``pid // NS`` and the RHS
    column slice ``pid % NS`` (KS columns).  STEPS holds the position of each
    pivot block (1x1 or 2x2) in forward order; the kernel runs the forward
    substitution in STEPS order and the backward substitution in reverse.
    """
    pid = tl.program_id(0)
    bidx = pid // NS
    sidx = pid % NS
    ld_base = LD + bidx * ld_batch_stride
    piv_base = PIVOTS + bidx * piv_batch_stride
    x_base = X + bidx * x_batch_stride
    cc = tl.arange(0, KS)
    col = sidx * KS + cc
    cm = col < nrhs
    coff = col * x_col_stride

    # ------------- forward: (L D) y = P b  (y stored in X) -------------
    for s in range(nsteps):
        k = tl.load(STEPS + s).to(tl.int32)
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            # 1x1 pivot at row k (with an optional row swap to ip-1)
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + coff
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xk = tl.load(row_k_ptr, mask=cm, other=0.0)
            xkp = tl.load(row_kp_ptr, mask=cm, other=0.0)
            tl.store(row_k_ptr, xkp, mask=cm)
            tl.store(row_kp_ptr, xk, mask=cm)
            xk = xkp
            for i in range(k + 1, N):
                lij = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + coff
                xi = tl.load(row_i_ptr, mask=cm, other=0.0)
                xi = xi - lij * xk
                tl.store(row_i_ptr, xi, mask=cm)
            d = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            xk = xk / d
            tl.store(row_k_ptr, xk, mask=cm)
        else:
            # 2x2 block at rows (k, k+1); row k+1 optionally swaps with -ip-1
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + coff
            row_k1_ptr = x_base + (k + 1) * x_row_stride + coff
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xk = tl.load(row_k_ptr, mask=cm, other=0.0)
            xk1 = tl.load(row_k1_ptr, mask=cm, other=0.0)
            xkp = tl.load(row_kp_ptr, mask=cm, other=0.0)
            tl.store(row_k1_ptr, xkp, mask=cm)
            tl.store(row_kp_ptr, xk1, mask=cm)
            xk1 = xkp
            for i in range(k + 2, N):
                l0 = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l1 = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + coff
                xi = tl.load(row_i_ptr, mask=cm, other=0.0)
                xi = xi - l0 * xk - l1 * xk1
                tl.store(row_i_ptr, xi, mask=cm)
            b = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            a = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            c = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            akm1 = a / b
            ak = c / b
            denom = akm1 * ak - 1
            bkm1 = xk / b
            bk = xk1 / b
            xk = (ak * bkm1 - bk) / denom
            xk1 = (akm1 * bk - bkm1) / denom
            tl.store(row_k_ptr, xk, mask=cm)
            tl.store(row_k1_ptr, xk1, mask=cm)

    # ------------- backward: x = L^-T y (steps in reverse) -------------
    for s in range(nsteps):
        k = tl.load(STEPS + (nsteps - 1 - s)).to(tl.int32)
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            row_k_ptr = x_base + k * x_row_stride + coff
            xk = tl.load(row_k_ptr, mask=cm, other=0.0)
            for i in range(k + 1, N):
                lij = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + coff
                xi = tl.load(row_i_ptr, mask=cm, other=0.0)
                xk = xk - lij * xi
            kp = ip - 1
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xkp = tl.load(row_kp_ptr, mask=cm, other=0.0)
            tl.store(row_k_ptr, xkp, mask=cm)
            tl.store(row_kp_ptr, xk, mask=cm)
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + coff
            row_k1_ptr = x_base + (k + 1) * x_row_stride + coff
            xk = tl.load(row_k_ptr, mask=cm, other=0.0)
            xk1 = tl.load(row_k1_ptr, mask=cm, other=0.0)
            for i in range(k + 2, N):
                l0 = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l1 = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                row_i_ptr = x_base + i * x_row_stride + coff
                xi = tl.load(row_i_ptr, mask=cm, other=0.0)
                xk = xk - l0 * xi
                xk1 = xk1 - l1 * xi
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xkp = tl.load(row_kp_ptr, mask=cm, other=0.0)
            tl.store(row_k1_ptr, xkp, mask=cm)
            tl.store(row_kp_ptr, xk1, mask=cm)
            tl.store(row_k_ptr, xk, mask=cm)


@libentry()
@triton.jit
def _ldl_solve_complex_kernel(
    LD,
    PIVOTS,
    X,
    STEPS,
    n,
    nrhs,
    nsteps,
    ld_batch_stride,
    ld_row_stride,
    ld_col_stride,
    piv_batch_stride,
    piv_row_stride,
    x_batch_stride,
    x_row_stride,
    x_col_stride,
    N: tl.constexpr,
    KS: tl.constexpr,
    NS: tl.constexpr,
):
    """Complex version of _ldl_solve_real_kernel (view_as_real interleaved).

    Each complex element is two consecutive floats; the real part lives at
    offset 0 and the imaginary part at offset +1 (X is contiguous after
    view_as_real).  D-division follows the generic non-hermitian convention
    (_cdiv); for hermitian inputs the imaginary part of D is exactly zero and
    _cdiv reduces to a real divide.
    """
    pid = tl.program_id(0)
    bidx = pid // NS
    sidx = pid % NS
    ld_base = LD + bidx * ld_batch_stride
    piv_base = PIVOTS + bidx * piv_batch_stride
    x_base = X + bidx * x_batch_stride
    cc = tl.arange(0, KS)
    col = sidx * KS + cc
    cm = col < nrhs
    coff = col * x_col_stride

    for s in range(nsteps):
        k = tl.load(STEPS + s).to(tl.int32)
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            kp = ip - 1
            row_k_ptr = x_base + k * x_row_stride + coff
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xk_r = tl.load(row_k_ptr, mask=cm, other=0.0)
            xk_i = tl.load(row_k_ptr + 1, mask=cm, other=0.0)
            xkp_r = tl.load(row_kp_ptr, mask=cm, other=0.0)
            xkp_i = tl.load(row_kp_ptr + 1, mask=cm, other=0.0)
            tl.store(row_k_ptr, xkp_r, mask=cm)
            tl.store(row_k_ptr + 1, xkp_i, mask=cm)
            tl.store(row_kp_ptr, xk_r, mask=cm)
            tl.store(row_kp_ptr + 1, xk_i, mask=cm)
            xk_r = xkp_r
            xk_i = xkp_i
            for i in range(k + 1, N):
                l_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + coff
                xi_r = tl.load(row_i_ptr, mask=cm, other=0.0)
                xi_i = tl.load(row_i_ptr + 1, mask=cm, other=0.0)
                prod_r, prod_i = _cmul(l_r, l_i, xk_r, xk_i)
                xi_r = xi_r - prod_r
                xi_i = xi_i - prod_i
                tl.store(row_i_ptr, xi_r, mask=cm)
                tl.store(row_i_ptr + 1, xi_i, mask=cm)
            d_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            d_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
            xk_r, xk_i = _cdiv(xk_r, xk_i, d_r, d_i)
            tl.store(row_k_ptr, xk_r, mask=cm)
            tl.store(row_k_ptr + 1, xk_i, mask=cm)
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + coff
            row_k1_ptr = x_base + (k + 1) * x_row_stride + coff
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xk_r = tl.load(row_k_ptr, mask=cm, other=0.0)
            xk_i = tl.load(row_k_ptr + 1, mask=cm, other=0.0)
            xk1_r = tl.load(row_k1_ptr, mask=cm, other=0.0)
            xk1_i = tl.load(row_k1_ptr + 1, mask=cm, other=0.0)
            xkp_r = tl.load(row_kp_ptr, mask=cm, other=0.0)
            xkp_i = tl.load(row_kp_ptr + 1, mask=cm, other=0.0)
            tl.store(row_k1_ptr, xkp_r, mask=cm)
            tl.store(row_k1_ptr + 1, xkp_i, mask=cm)
            tl.store(row_kp_ptr, xk1_r, mask=cm)
            tl.store(row_kp_ptr + 1, xk1_i, mask=cm)
            xk1_r = xkp_r
            xk1_i = xkp_i
            for i in range(k + 2, N):
                l0_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l0_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                l1_r = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                l1_i = tl.load(
                    ld_base + i * ld_row_stride + (k + 1) * ld_col_stride + 1
                )
                row_i_ptr = x_base + i * x_row_stride + coff
                xi_r = tl.load(row_i_ptr, mask=cm, other=0.0)
                xi_i = tl.load(row_i_ptr + 1, mask=cm, other=0.0)
                p0_r, p0_i = _cmul(l0_r, l0_i, xk_r, xk_i)
                p1_r, p1_i = _cmul(l1_r, l1_i, xk1_r, xk1_i)
                xi_r = xi_r - p0_r - p1_r
                xi_i = xi_i - p0_i - p1_i
                tl.store(row_i_ptr, xi_r, mask=cm)
                tl.store(row_i_ptr + 1, xi_i, mask=cm)
            b_r = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride)
            b_i = tl.load(ld_base + (k + 1) * ld_row_stride + k * ld_col_stride + 1)
            a_r = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride)
            a_i = tl.load(ld_base + k * ld_row_stride + k * ld_col_stride + 1)
            c_r = tl.load(ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride)
            c_i = tl.load(
                ld_base + (k + 1) * ld_row_stride + (k + 1) * ld_col_stride + 1
            )
            akm1_r, akm1_i = _cdiv(a_r, a_i, b_r, b_i)
            ak_r, ak_i = _cdiv(c_r, c_i, b_r, b_i)
            denom_r, denom_i = _cmul(akm1_r, akm1_i, ak_r, ak_i)
            denom_r = denom_r - 1
            bkm1_r, bkm1_i = _cdiv(xk_r, xk_i, b_r, b_i)
            bk_r, bk_i = _cdiv(xk1_r, xk1_i, b_r, b_i)
            tmp_r, tmp_i = _cmul(ak_r, ak_i, bkm1_r, bkm1_i)
            xk_r, xk_i = _cdiv(tmp_r - bk_r, tmp_i - bk_i, denom_r, denom_i)
            tmp_r, tmp_i = _cmul(akm1_r, akm1_i, bk_r, bk_i)
            xk1_r, xk1_i = _cdiv(tmp_r - bkm1_r, tmp_i - bkm1_i, denom_r, denom_i)
            tl.store(row_k_ptr, xk_r, mask=cm)
            tl.store(row_k_ptr + 1, xk_i, mask=cm)
            tl.store(row_k1_ptr, xk1_r, mask=cm)
            tl.store(row_k1_ptr + 1, xk1_i, mask=cm)

    for s in range(nsteps):
        k = tl.load(STEPS + (nsteps - 1 - s)).to(tl.int32)
        ip = tl.load(piv_base + k * piv_row_stride).to(tl.int32)
        if ip > 0:
            row_k_ptr = x_base + k * x_row_stride + coff
            xk_r = tl.load(row_k_ptr, mask=cm, other=0.0)
            xk_i = tl.load(row_k_ptr + 1, mask=cm, other=0.0)
            for i in range(k + 1, N):
                l_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                row_i_ptr = x_base + i * x_row_stride + coff
                xi_r = tl.load(row_i_ptr, mask=cm, other=0.0)
                xi_i = tl.load(row_i_ptr + 1, mask=cm, other=0.0)
                prod_r, prod_i = _cmul(l_r, l_i, xi_r, xi_i)
                xk_r = xk_r - prod_r
                xk_i = xk_i - prod_i
            kp = ip - 1
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xkp_r = tl.load(row_kp_ptr, mask=cm, other=0.0)
            xkp_i = tl.load(row_kp_ptr + 1, mask=cm, other=0.0)
            tl.store(row_k_ptr, xkp_r, mask=cm)
            tl.store(row_k_ptr + 1, xkp_i, mask=cm)
            tl.store(row_kp_ptr, xk_r, mask=cm)
            tl.store(row_kp_ptr + 1, xk_i, mask=cm)
        else:
            kp = -ip - 1
            row_k_ptr = x_base + k * x_row_stride + coff
            row_k1_ptr = x_base + (k + 1) * x_row_stride + coff
            xk_r = tl.load(row_k_ptr, mask=cm, other=0.0)
            xk_i = tl.load(row_k_ptr + 1, mask=cm, other=0.0)
            xk1_r = tl.load(row_k1_ptr, mask=cm, other=0.0)
            xk1_i = tl.load(row_k1_ptr + 1, mask=cm, other=0.0)
            for i in range(k + 2, N):
                l0_r = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride)
                l0_i = tl.load(ld_base + i * ld_row_stride + k * ld_col_stride + 1)
                l1_r = tl.load(ld_base + i * ld_row_stride + (k + 1) * ld_col_stride)
                l1_i = tl.load(
                    ld_base + i * ld_row_stride + (k + 1) * ld_col_stride + 1
                )
                row_i_ptr = x_base + i * x_row_stride + coff
                xi_r = tl.load(row_i_ptr, mask=cm, other=0.0)
                xi_i = tl.load(row_i_ptr + 1, mask=cm, other=0.0)
                p0_r, p0_i = _cmul(l0_r, l0_i, xi_r, xi_i)
                p1_r, p1_i = _cmul(l1_r, l1_i, xi_r, xi_i)
                xk_r = xk_r - p0_r
                xk_i = xk_i - p0_i
                xk1_r = xk1_r - p1_r
                xk1_i = xk1_i - p1_i
            row_kp_ptr = x_base + kp * x_row_stride + coff
            xkp_r = tl.load(row_kp_ptr, mask=cm, other=0.0)
            xkp_i = tl.load(row_kp_ptr + 1, mask=cm, other=0.0)
            tl.store(row_k1_ptr, xkp_r, mask=cm)
            tl.store(row_k1_ptr + 1, xkp_i, mask=cm)
            tl.store(row_kp_ptr, xk1_r, mask=cm)
            tl.store(row_kp_ptr + 1, xk1_i, mask=cm)
            tl.store(row_k_ptr, xk_r, mask=cm)
            tl.store(row_k_ptr + 1, xk_i, mask=cm)


def _build_pivot_steps(pivots_row):
    """Host-side block plan: position of each 1x1/2x2 pivot block.

    Mirrors the generic algorithm's while/k+=2 control flow.  The step at
    position k is a 2x2 block (rows k, k+1) iff pivots[k] < 0.
    """
    piv = pivots_row.cpu().tolist()
    steps = []
    k = 0
    n = len(piv)
    while k < n:
        ip = piv[k]
        steps.append(k)
        k = k + 1 if ip > 0 else k + 2
    return torch.tensor(steps, device=pivots_row.device, dtype=torch.int32)


def _validate_inputs(LD, pivots, B):
    if LD.device != B.device or LD.device != pivots.device:
        raise ValueError("LD, pivots, and B must be on the same device")
    if LD.dtype != B.dtype:
        raise TypeError("LD and B must have the same dtype")
    if LD.ndim < 2 or B.ndim < 2:
        raise ValueError("LD and B must be at least 2D")
    if LD.shape[-1] != LD.shape[-2]:
        raise ValueError("LD must be a square matrix or a batch of square matrices")
    if B.shape[-2] != LD.shape[-1]:
        raise ValueError("B must have shape (*, n, k) with the same n as LD")
    if pivots.shape != LD.shape[:-1]:
        raise ValueError("pivots must have shape (*, n) matching LD")
    if LD.shape[:-2] != B.shape[:-2]:
        raise ValueError("LD, pivots, and B must share the same batch dimensions")


def linalg_ldl_solve(LD, pivots, B, *, hermitian=False):
    """Solve a linear system using the compact LDL factorization of ldl_factor_ex."""
    logger.debug("GEMS KUNLUNXIN LINALG_LDL_SOLVE")
    _validate_inputs(LD, pivots, B)

    if LD.dtype not in REAL_DTYPES + COMPLEX_DTYPES:
        raise TypeError(
            "linalg_ldl_solve supports only float32, float64, complex64, and complex128 inputs"
        )
    if LD.numel() == 0 or B.numel() == 0:
        return B.clone()

    batch = math.prod(LD.shape[:-2])
    n = LD.shape[-1]
    nrhs = B.shape[-1]

    nrhs_pad = ((nrhs + LDL_SOLVE_KS - 1) // LDL_SOLVE_KS) * LDL_SOLVE_KS
    nslices = nrhs_pad // LDL_SOLVE_KS

    with torch.no_grad():
        LD_work = LD.reshape(batch, n, n).contiguous()
        piv_work = pivots.reshape(batch, n).contiguous()
        # torch.empty (not torch.zeros: the XPU zeros override cannot handle
        # complex64 pointers).  The RHS tail columns [nrhs, nrhs_pad) hold
        # uninitialized values; every kernel access is masked by
        # `col < nrhs_pad` (all-true, trsm-proven) and the column lanes are
        # independent, so garbage in the tail can never leak into the
        # [0, nrhs) output columns.
        X = torch.empty(batch, n, nrhs_pad, dtype=LD.dtype, device=LD.device)
        if LD.dtype in REAL_DTYPES:
            torch.ops.aten._copy_from(B.reshape(batch, n, nrhs), X[:, :, :nrhs], False)
        else:
            # Complex path: stage in f32 (real, imag) interleaved views; the
            # XPU Triton backend cannot canonicalize complex pointers and the
            # flag_gems slice override rejects complex64, so no c64-typed
            # tensor is ever sliced or passed to a kernel (same as div_tensor).
            X_c = torch.empty(batch, n, nrhs_pad, 2, dtype=torch.float32, device=LD.device)
            B_c = torch.view_as_real(B.reshape(batch, n, nrhs)).contiguous()
            torch.ops.aten._copy_from(B_c, X_c[:, :, :nrhs], False)

        # Host plan of pivot blocks.  The common path (SPD inputs, including
        # every input produced by the XPU ldl_factor_ex/ldl_factor kernels,
        # which always emit identity pivots) is detected on-device without any
        # device->host transfer of the pivot vector (pivots.cpu() costs
        # ~0.2ms/call, dominant for n<=8); the general Bunch-Kaufman plan is
        # only computed for non-identity pivots.
        is_identity = bool(
            torch.all(piv_work == (torch.arange(n, device=piv_work.device) + 1)).item()
        )
        if is_identity:
            steps = torch.arange(n, device=piv_work.device, dtype=torch.int32)
            same_plan = True
        else:
            steps = _build_pivot_steps(piv_work[0])
            same_plan = bool((piv_work == piv_work[0:1]).all().item())
        nsteps = steps.numel()

        if LD.dtype in REAL_DTYPES:
            if same_plan:
                _ldl_solve_real_kernel[(batch * nslices,)](
                    LD_work,
                    piv_work,
                    X,
                    steps,
                    n,
                    nrhs_pad,
                    nsteps,
                    LD_work.stride(0),
                    LD_work.stride(1),
                    LD_work.stride(2),
                    piv_work.stride(0),
                    piv_work.stride(1),
                    X.stride(0),
                    X.stride(1),
                    X.stride(2),
                    N=n,
                    KS=LDL_SOLVE_KS,
                    NS=nslices,
                    num_warps=4,
                )
            else:
                for b in range(batch):
                    steps_b = _build_pivot_steps(piv_work[b])
                    _ldl_solve_real_kernel[(nslices,)](
                        LD_work[b],
                        piv_work[b],
                        X[b],
                        steps_b,
                        n,
                        nrhs_pad,
                        steps_b.numel(),
                        n * n,
                        n,
                        1,
                        n,
                        1,
                        n * nrhs_pad,
                        nrhs_pad,
                        1,
                        N=n,
                        KS=LDL_SOLVE_KS,
                        NS=nslices,
                        num_warps=4,
                    )
            return X[:, :, :nrhs].reshape(B.shape)

        # ---- complex (c64) path ----
        LD_c = torch.view_as_real(LD_work)  # (batch, n, n, 2) f32 contiguous
        if same_plan:
            _ldl_solve_complex_kernel[(batch * nslices,)](
                LD_c,
                piv_work,
                X_c,
                steps,
                n,
                nrhs_pad,
                nsteps,
                LD_c.stride(0),
                LD_c.stride(1),
                LD_c.stride(2),
                piv_work.stride(0),
                piv_work.stride(1),
                X_c.stride(0),
                X_c.stride(1),
                X_c.stride(2),
                N=n,
                KS=LDL_SOLVE_KS,
                NS=nslices,
                num_warps=4,
            )
        else:
            for b in range(batch):
                steps_b = _build_pivot_steps(piv_work[b])
                _ldl_solve_complex_kernel[(nslices,)](
                    LD_c[b],
                    piv_work[b],
                    X_c[b],
                    steps_b,
                    n,
                    nrhs_pad,
                    steps_b.numel(),
                    n * n * 2,
                    n * 2,
                    2,
                    n,
                    1,
                    n * nrhs_pad * 2,
                    nrhs_pad * 2,
                    2,
                    N=n,
                    KS=LDL_SOLVE_KS,
                    NS=nslices,
                    num_warps=4,
                )

        return torch.view_as_complex(X_c[:, :, :nrhs]).reshape(B.shape)