# NEON Softmax Kernel

**English** | [简体中文](README.md)

![C11](https://img.shields.io/badge/C-C11-blue)
![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C)
![tests](https://img.shields.io/badge/checks-6421%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> A hand-written softmax kernel for aarch64 NEON: vectorised max reduction →
> polynomial `exp` approximation → vectorised normalisation, with a scalar reference
> implementation kept for correctness checks and non-ARM fallback.

## Why a dedicated softmax

Softmax runs in every attention layer and classifier head. It looks like `O(n)` adds, but three
things dominate:

| Step | Naive | Cost |
| --- | --- | --- |
| max reduction (numerical stability) | full pass | memory-bound but reducible |
| `exp(x - max)` | one `expf` per element | **libm call, tens of cycles** — the real bottleneck |
| normalisation | one divide, `n` multiplies | replace divide with reciprocal |

So the win is not loop unrolling, it is **replacing `expf`**: range-reduce with
`k = round(x·log₂e)`, evaluate a 5th-order Taylor polynomial on `r = x − k·ln2 ∈ [−0.35, 0.35]`,
then scale by `2^k`. Error stays around `1e-6` (see tests) while the library call disappears.

## Layout

| File | Content |
| --- | --- |
| `include/softmax.h` | Public API; `SOFTMAX_HALF_T` is `__fp16` when available, else a `uint16_t` holding IEEE754 binary16 |
| `src/softmax_scalar_f32.c` | Reference: `expf` + **double accumulation** |
| `src/softmax_neon_f32.c` | NEON: dual-accumulator 8-wide max reduction, 8-element exp blocks, vector reciprocal |
| `src/softmax_neon_f16.c` | fp16 entry point with full half↔float bit-level conversion |
| `src/exp_poly.h` | `exp_approx_float()`: range reduction + degree-5 polynomial |
| `tests/test_softmax.c` | **6421 assertions**, returns non-zero on failure |
| `bench/bench_softmax.c` | reports `ns/elem`, accepts n and iteration count |

## Build & test

```bash
cmake -S . -B out -DCMAKE_BUILD_TYPE=Release
cmake --build out
ctest --test-dir out --output-on-failure
./out/bench_softmax 16384 200
```

On aarch64 CMake enables `-DUSE_AARCH64_NEON` automatically; elsewhere the NEON entry point
falls back to the scalar path — so the same test suite validates both.

## What the tests pin down

`null`/`n<=0` safety · **tail boundaries `n = 7,8,9,1023,1024,1025`** · error vs a `double`
reference (`<1e-6` scalar, `<1e-4` NEON) · shift invariance `softmax(x+c) == softmax(x)` (this
directly guards the max-subtraction) · uniform input · `±100` magnitudes must stay finite ·
in-place operation · fp16 ordering and sum.

## Measured (x86 scalar baseline, not the target platform)

```
AMD Ryzen 5 7500F, gcc 14.2.0 -O2
n=1024     2.06 ns/elem
n=16384    2.19 ns/elem
n=262144   2.25 ns/elem
```

On x86 `f32_neon` and `f32_scalar` are identical because the NEON branch falls back. No
cross-architecture speed-up is claimed here.

## Known limitations

1. **The exp inside the NEON loop is still scalar.** Every 8 elements are stored to a stack array,
   passed one by one to `exp_approx_float()` (which itself calls `ldexpf`), then reloaded —
   giving back most of the vectorisation win. This is the biggest gap.
2. **The fp16 path loses extra precision**: `exp` values are quantised to half *before*
   normalisation, contradicting the promise in `softmax.h`.
3. `src/exp_poly.c` is an empty translation unit.
4. `extern float ldexpf(float, int);` is declared inside a function body.
5. Only two accumulators for an 8-wide unroll; no FMA.
6. **CI covers the x86 scalar path only** — the NEON branch needs an aarch64 runner.

## Roadmap

- [ ] Implement exp polynomial and `2^k` scaling with NEON intrinsics
- [ ] Normalise in FP32, quantise to fp16 last
- [ ] FMA + multi-accumulator pipelining
- [ ] Add a `macos-14` (arm64) CI job so NEON is actually executed
- [ ] Compare `vdivq` against reciprocal approximation
- [ ] bfloat16 entry point

## License

[MIT](LICENSE) © 2026 DOUFUQAQ
