"""
Portable RNG: PCG64 (64-bit LCG + XSH-RR 64/32) and standard normal via Marsaglia polar.
Same algorithm can be implemented in C++/Java for identical streams.
Reference: https://www.pcg-random.org/
Requires numba.
"""

import math
import numpy as np
from numba import njit

# PCG64 64/32 constants (single 64-bit seed -> same stream everywhere)
PCG64_MULT = 6364136223846793005
PCG64_INC = 1442695040888963407
TWO32 = 2**32

_MULT = np.uint64(6364136223846793005)
_INC = np.uint64(1442695040888963407)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _pcg64_next(state: int) -> tuple[int, int]:
    """One step: (state, output_32bit). State and output are unsigned 64/32."""
    state = (state * PCG64_MULT + PCG64_INC) & 0xFFFFFFFFFFFFFFFF
    x = (((state >> 18) ^ state) >> 27) & 0xFFFFFFFF
    rot = state >> 59
    out32 = ((x >> rot) | (x << ((32 - rot) & 31))) & 0xFFFFFFFF
    return state, out32


def next_seed(seed: int | None) -> int:
    """Derive a new 64-bit seed from an optional parent seed or from time (when seed is None/0).

    Uses two PCG64 outputs to fill 64 bits, consistent with the rest of this module.
    """
    state = (int(seed) & 0xFFFFFFFFFFFFFFFF) if seed is not None else 0
    if state == 0:
        import time
        state = int(time.perf_counter_ns()) & 0xFFFFFFFFFFFFFFFF
    state, lo = _pcg64_next(state)
    state, hi = _pcg64_next(state)
    return int(((hi << 32) | lo) & 0xFFFFFFFFFFFFFFFF)


@njit(cache=True)
def _pcg64_next_numba(s):
    s = (s * _MULT + _INC) & _MASK64
    x = ((s >> np.uint64(18)) ^ s) >> np.uint64(27)
    rot = int(s >> np.uint64(59)) & 31
    x_i = int(x) & 0xFFFFFFFF
    out32 = np.uint64(((x_i >> rot) | (x_i << ((32 - rot) & 31))) & 0xFFFFFFFF)
    return s, out32


@njit(cache=True)
def _fill_standard_normal_impl(seed, out):
    # Marsaglia polar: U1,U2 uniform in (0,1] -> V1,V2 = 2*U-1 in (-1,1]; keep (V1,V2) with S = V1^2+V2^2 < 1; then X = V*sqrt(-2*ln(S)/S) is N(0,1)
    state = np.uint64(seed) & _MASK64
    n = out.size
    i = 0
    inv_2p32 = 1.0 / 4294967296.0
    while i < n:
        state, u1_32 = _pcg64_next_numba(state)
        state, u2_32 = _pcg64_next_numba(state)
        v1 = 2.0 * (float(u1_32) + 1.0) * inv_2p32 - 1.0
        v2 = 2.0 * (float(u2_32) + 1.0) * inv_2p32 - 1.0
        s = v1 * v1 + v2 * v2
        if 0.0 < s < 1.0:
            f = math.sqrt(-2.0 * math.log(s) / s)
            out[i] = v1 * f
            i += 1
            if i < n:
                out[i] = v2 * f
                i += 1


def fill_standard_normal(seed: int, out: np.ndarray) -> None:
    """Fill out (float64 or float32) with standard normals."""
    seed_u64 = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    _fill_standard_normal_impl(seed_u64, out.ravel())


def standard_normal(seed: int, size: int | tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """Portable standard normal array. Seed + size must reproduce in C++/Java."""
    out = np.empty(size, dtype=dtype)
    if out.size == 0:
        return out
    fill_standard_normal(seed, out.ravel())
    return out
