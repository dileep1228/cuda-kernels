# SAXPY: y = a·x + y

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements, a = 2.5, NVIDIA A10 / A10G.

Single-precision A·X Plus Y, one of the oldest BLAS (Basic Linear Algebra Subprograms)
routines. The first kernel in this repo with real maths: one multiply and one add per
element, done as a single FMA (fused multiply-add) instruction.

**Result:** 1.63 ms, **~495 GB/s (gigabytes per second) = 82.5% of the 600 GB/s spec**,
DRAM (dynamic random-access memory) 91% busy. **Limited by DRAM bandwidth.** The extra
multiply costs nothing measurable: saxpy moves the same 12 bytes per element as
[vector add](../01_vector_add/) and runs at the same speed (1.63 vs 1.61 ms).
**Correctness needed care:** a plain CPU reference disagrees with the GPU on 19% of
elements, because the two round differently. Using `std::fma` on the CPU gives 0 mismatches.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA A10 (72 SMs (streaming multiprocessors)) or A10G (80 SMs), both 600 GB/s; Modal assigns either, see [hardware.md](../hardware.md) |
| Launch | 262,144 blocks × 256 threads, grid-stride loop, 24 registers per thread |
| Build | `nvcc -O3 -arch=sm_86 -lineinfo` (CUDA 12.8) |
| Timing | 3 warm-up launches, then 20 launches timed with CUDA events; median reported |
| Input | random floats in [0, 1) for x and y |

```bash
modal run tools/run.py --file 03_saxpy/saxpy.cu                                  # benchmark
modal run tools/run.py --file 03_saxpy/saxpy.cu --mode ncu --skip 3 --count 1    # Nsight Compute
modal run tools/run.py --file 03_saxpy/saxpy.cu --mode sanitize                  # compute-sanitizer
```

## Prediction vs measurement

The prediction was written and committed before any code existed
(commit [`7229fc9`](https://github.com/dileep1228/cuda-kernels/commit/7229fc9)).

| | Predicted | Measured |
| --- | --- | --- |
| Bytes moved per element | 12 (read x, read y, write y); `a` is one value kept in a register | — |
| Total bytes moved | 805 MB (megabytes) (805,306,368 bytes) | — |
| FLOPs (floating-point operations) per element | 2 (multiply + add, one FMA instruction) | — |
| Arithmetic intensity | 1/6 ≈ 0.17 FLOP/byte, ~300× below the ~52 needed to be compute-bound | — |
| Bottleneck | memory (DRAM bandwidth) | memory: DRAM 91.0% busy, SMs 13.6% (ncu) |
| Ideal kernel time at 600 GB/s | 1.342 ms, identical to `vector_add` (same bytes) | — |
| Kernel time | ~1.60 ms | **1.625–1.635 ms** median |
| Fraction of 600 GB/s | ~84% | **82.1–82.6%** |
| vs `vector_add` (1.61 ms) | about the same | **about the same** (~1% slower, within run-to-run noise) |

The prediction was ~2% fast. The reasoning held: each warp waits hundreds of cycles for
memory, and the one FMA (~4 cycles) runs while other warps are still waiting.

## Results

### Kernel

| Card | N | Fastest | Median | Slowest | Bandwidth | % of 600 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| A10 | 2²⁶ | 1.618 ms | 1.627 ms | 1.635 ms | 495.1 GB/s | 82.5% |
| A10 | 2²⁶ | 1.612 ms | 1.625 ms | 1.630 ms | 495.7 GB/s | 82.6% |
| A10G | 2²⁶ | 1.624 ms | 1.635 ms | 1.645 ms | 492.4 GB/s | 82.1% |
| A10G | 2²⁶ + 3 | 1.622 ms | 1.630 ms | 1.640 ms | 494.0 GB/s | 82.3% |

Same speed on both cards, as expected for a memory-bound kernel (same memory, more SMs
on the A10G only add compute).

### Nsight Compute (Speed of Light section, A10)

| Metric | vector add | vector copy | **saxpy** |
| --- | --- | --- | --- |
| Duration | 1.61 ms | 1.08 ms | **1.63 ms** |
| DRAM throughput | 91.8–93.0% | 91.8% | **91.0%** |
| Compute (SM) throughput | 12.6% | 17.4% | **13.6%** |
| L2 cache throughput | ~20% | 20.0% | **20.0%** |
| L1/TEX cache throughput | ~10% | 13.0% | **11.2%** |

Compute throughput matches vector add: same bytes per element, so the same number of
elements per second, and the same instruction count per element (one `FFMA` instead of
one `FADD`).

### Copies over PCIe (Peripheral Component Interconnect Express), pageable memory

| H2D (x + y, 537 MB) | D2H (y, 268 MB) | Copies ÷ kernel |
| --- | --- | --- |
| 41.0–67.2 ms, 8.0–13.1 GB/s | 25.8–26.7 ms, 10.1–10.4 GB/s | 41–58× |

H2D = host to device, D2H = device to host.

## Limiting resource

**DRAM bandwidth.** 805 MB per call at ~91% DRAM busy, the same ceiling as vector add
and vector copy. All three kernels stream at ~495–500 GB/s whatever the arithmetic in
the middle, as long as there is this little of it.

## Correctness

Two traps, both specific to an in-place kernel with real maths.

**1. `y` is updated in place.** 3 warm-ups + 20 timed launches apply `y = a·x + y`
23 times, so the final `y` is not one saxpy of the input. The program keeps a host copy
of the original `y`, copies it back to the GPU after timing, runs the kernel once more,
and checks that single result.

**2. CPU and GPU round differently.** The GPU computes `a * x[i] + y[i]` as one FMA:
the exact result, rounded once. On the CPU, `a * x[i] + y[i]` rounds after the multiply
and again after the add. Measured with the plain CPU expression as the reference:

| CPU reference | Mismatches |
| --- | --- |
| `a * h_x[i] + h_y[i]` (two roundings) | **12,705,635 of 67,108,864 (18.9%)** |
| `std::fma(a, h_x[i], h_y[i])` (one rounding, like the GPU) | **0** |

The mismatches are one unit in the last place (ULP), a relative difference around 10⁻⁷:
both answers are correct to FP32 precision. Matching the GPU's rounding makes an exact
check possible here. Kernels that add numbers in a different order than the CPU
(reductions, matrix multiply) cannot match bit for bit, and will need a tolerance.

- `compute-sanitizer --tool memcheck`: **0 errors**.
- **Awkward size, N = 2²⁶ + 3 = 67,108,867:** 262,145 blocks × 256 threads, 253 surplus
  threads in the last block. 0 mismatches and 0 sanitizer errors, both runs printing
  `N = 67108867 elements`.

## What I learned

_To be written._
