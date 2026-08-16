"""THRandom — exact Python port of Torch7's Mersenne Twister RNG.

Source of truth: untouched `torch/torch7 @ 814ea4a`, `lib/TH/THRandom.c`
(SHA-256 5752496219762bfc70c359c8b3e3c96eae19ac56478a38efd4190e7e4128ee36)
and the `torch.random(a,b)` wrapper mapping `TensorMath.lua:773`
`(THRandom_random(gen) % (b+1-a)) + a`.

Contract (the certified Phase-2A / HUNL_DATAGEN_SOURCE_AUDIT §3 scheme):
  random_u32()      -> one tempered MT19937 draw (THRandom_random);
  rand_float(n)     -> n float32 uniforms; each = f32(u32 * 2^-32)
                       (Torch7 __uniform__ double, FloatTensor store);
                       consumes exactly n state draws;
  random_range(a,b) -> inclusive integer (u32 % (b+1-a)) + a;
                       consumes exactly 1 state draw.
All derived draws route through random_u32() so draw-counting subclasses
(CountingTHRandom) observe every state draw.

Seeding is THRandom_manualSeed = standard MT19937 init_genrand
(state[0]=seed&0xffffffff; Knuth 1812433253 recurrence), left=1 so the
first draw triggers nextState — identical to Tammelin's rng.c stream,
which is why the frozen `certification/oracles/rng/` anchors apply.

BIT_EXACT certification of this restoration:
`certification/hunl_g1/thrandom_oracle/run_thrandom_cert.py`.
"""
from __future__ import annotations

import numpy as np

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UMASK = 0x80000000
_LMASK = 0x7FFFFFFF


class THRandom:
    """Exact THRandom.c state machine (LP64 build semantics)."""

    def __init__(self, seed: int):
        self.manual_seed(seed)

    def manual_seed(self, seed: int) -> None:
        state = [0] * _N
        s = seed & 0xFFFFFFFF
        state[0] = s
        for j in range(1, _N):
            s = (1812433253 * (s ^ (s >> 30)) + j) & 0xFFFFFFFF
            state[j] = s
        self.state = state
        self.left = 1
        self.next = 0
        self.the_initial_seed = seed

    def _next_state(self) -> None:
        st = self.state
        self.left = _N
        self.next = 0
        for p in range(_N - _M):            # for(j=n-m+1; --j; p++)
            v = st[p + 1]
            st[p] = st[p + _M] ^ (
                (((st[p] & _UMASK) | (v & _LMASK)) >> 1)
                ^ (_MATRIX_A if v & 1 else 0))
        for p in range(_N - _M, _N - 1):    # for(j=m; --j; p++)
            v = st[p + 1]
            st[p] = st[p + _M - _N] ^ (
                (((st[p] & _UMASK) | (v & _LMASK)) >> 1)
                ^ (_MATRIX_A if v & 1 else 0))
        v = st[0]                           # wrap term
        st[_N - 1] = st[_M - 1] ^ (
            (((st[_N - 1] & _UMASK) | (v & _LMASK)) >> 1)
            ^ (_MATRIX_A if v & 1 else 0))

    def random_u32(self) -> int:
        self.left -= 1
        if self.left == 0:
            self._next_state()
        y = self.state[self.next]
        self.next += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    def rand_float(self, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float32)
        for i in range(count):
            out[i] = np.float32(self.random_u32() * (1.0 / 4294967296.0))
        return out

    def random_range(self, a: int, b: int) -> int:
        return (self.random_u32() % (b + 1 - a)) + a
