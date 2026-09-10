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

"""Kunlunxin/XPU ``lu_unpack`` / ``lu_unpack_out``.

The generic ``flag_gems.ops.lu_unpack`` implementation (re-exported below)
is correct and fast on this backend: the f32 test channel is fully clean and
the benchmark dtype-equal-weight Gems Speedup is ~10.7x (see
``harness/solution/lu_unpack/README.md``).  The one functional gap found by
the semantic sweep is that empty inputs (``m == 0`` or ``n == 0``, which
``torch.linalg.lu_factor`` happily produces and torch's ``lu_unpack``
handles) crash the generic kernels at ``tl.arange(0, BLOCK_M)`` with
``BLOCK_M`` reaching 0.  A host-side guard below reproduces torch's exact
empty-input semantics (P = eye(m) when ``unpack_pivots``, L = (m, 0),
U = (0, n) when ``unpack_data``, 0-element tensors otherwise) without
touching the proven kernel path.

XPU-specific kernels were attempted for P (the generic's m > 512 path is a
per-row serial k-chain) and for L/U (mask-free full blocks + clamped tails).
All three candidate designs were rejected with evidence, because the
unmasked / loop-carried-register-tensor patterns are unreliable on this
device:

- a ``[BLOCK_M]`` register vector (``perm``) carried through the dynamic
  k-loop miscompiles at some (m, k) combinations (e.g. (512, 512) hard
  fault, (1024, 1024) silently wrong or faulting) — the same class as the
  linalg_householder_product "register tensor through scf.for" finding;
- unmasked 4/8-wide loads with the column-major (strided) LU layout that
  ``torch.linalg.lu_factor`` returns on this device (``stride == (1, m)``)
  are silently wrong at (4, 4) / (8, 8);
- the XPU masked-memory path (``mask=`` + ``other=0``), used by the generic,
  is the only one verified correct across the full shape matrix.

Per the harness rule "only candidates strictly better than the baseline are
kept", the working generic implementation is kept unchanged.
"""

import torch

from flag_gems.ops.lu_unpack import lu_unpack as _lu_unpack_generic


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """Unpacks the LU decomposition into P, L, U.

    Matches ``torch.ops.aten.lu_unpack`` semantics; for empty inputs
    (``k = min(m, n) == 0``) returns torch's exact shapes/values (P = eye(m)
    when ``unpack_pivots``), which the generic kernel path cannot handle
    (``tl.arange`` of size 0).
    """
    lu_shape = LU_data.shape
    m, n = lu_shape[-2], lu_shape[-1]
    dtype = LU_data.dtype
    device = LU_data.device
    batch_dims = lu_shape[:-2]

    if m == 0 or n == 0:
        if unpack_pivots:
            P = (
                torch.eye(m, device=device, dtype=dtype)
                .expand(*batch_dims, m, m)
                .contiguous()
            )
        else:
            P = torch.empty(0, device=device, dtype=dtype)
        if unpack_data:
            L = torch.empty(*batch_dims, m, 0, device=device, dtype=dtype)
            U = torch.empty(*batch_dims, 0, n, device=device, dtype=dtype)
        else:
            L = torch.empty(0, device=device, dtype=dtype)
            U = torch.empty(0, device=device, dtype=dtype)
        return (P, L, U)

    return _lu_unpack_generic(LU_data, LU_pivots, unpack_data, unpack_pivots)


def lu_unpack_out(
    LU_data, LU_pivots, unpack_data=True, unpack_pivots=True, *, P=None, L=None, U=None
):
    """Out variant (see ``lu_unpack``); identical results, copied into the
    provided outputs (or freshly allocated ones when ``None``)."""
    P_result, L_result, U_result = lu_unpack(
        LU_data, LU_pivots, unpack_data, unpack_pivots
    )

    if P is not None and P_result.numel() > 0:
        P.copy_(P_result)
    else:
        P = P_result

    if L is not None and L_result.numel() > 0:
        L.copy_(L_result)
    else:
        L = L_result

    if U is not None and U_result.numel() > 0:
        U.copy_(U_result)
    else:
        U = U_result

    return (P, L, U)