# ReLU and square: y = max(x, 0), y = x²

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements, input random in [-1, 1),
NVIDIA A10 / A10G. Two elementwise kernels in one file, each reading `x` and writing a
separate `y`.

**Result:** both run in **1.08–1.09 ms, ~495 GB/s (gigabytes per second) = 82–83% of the
600 GB/s spec**, DRAM (dynamic random-access memory) 91% busy. **Limited by DRAM
bandwidth.** They match [vector copy](../02_vector_copy/) (1.08 ms), which moves the same
8 bytes per element: whether the one instruction in the middle is nothing, a `max` or a
multiply makes no measurable difference.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA A10 (72 SMs (streaming multiprocessors)) or A10G (80 SMs), both 600 GB/s; Modal assigns either, see [hardware.md](../hardware.md) |
| Launch | 262,144 blocks × 256 threads, grid-stride loop, 28 registers per thread |
| Build | `nvcc -O3 -arch=sm_86 -lineinfo` (CUDA 12.8) |
| Timing | per kernel: 3 warm-up launches, then 20 launches timed with CUDA events; median reported |
| Input | random floats in [-1, 1): about half are negative, so ReLU really changes them |

```bash
modal run tools/run.py --file 04_relu_square/relu_square.cu                                   # benchmark both
modal run tools/run.py --file 04_relu_square/relu_square.cu --mode ncu --skip 3 --count 1     # profile relu
modal run tools/run.py --file 04_relu_square/relu_square.cu --mode ncu --skip 26 --count 1    # profile square
modal run tools/run.py --file 04_relu_square/relu_square.cu --mode sanitize                   # compute-sanitizer
```

**Choosing `--skip`:** ncu (Nsight Compute) numbers kernel launches from the start of the
program. ReLU launches 3 warm-ups (0–2) and 20 timed runs (3–22), then square launches
3 warm-ups (23–25) and 20 timed runs (26–45). `--skip 3` profiles ReLU's first warmed-up
launch; `--skip 26` profiles square's.

## Prediction vs measurement

The prediction was written and committed before any code existed
(commit [`7b3cd87`](https://github.com/dileep1228/cuda-kernels/commit/7b3cd87)).

| | ReLU predicted | ReLU measured | Square predicted | Square measured |
| --- | --- | --- | --- | --- |
| Bytes per element | 8 | — | 8 | — |
| FLOPs (floating-point operations) per element | 0 (one `FMNMX` instruction) | — | 1 | — |
| Arithmetic intensity | 0 | — | 1/8 | — |
| Ideal time at 600 GB/s | 0.895 ms | — | 0.895 ms | — |
| Kernel time | ~1.08 ms | **1.081–1.094 ms** | ~1.10 ms | **1.082–1.090 ms** |
| Fraction of 600 GB/s | ~83% | **81.8–82.7%** | ~81% | **82.1–82.7%** |
| Should match | vector copy | ✓ | vector copy | ✓ |
| CPU vs GPU rounding | exact | **0 mismatches** | exact (one rounding) | **0 mismatches** with plain `x * x` |

Both predictions were within 2%. The method (ideal time ÷ the ~82% fraction measured on
earlier kernels) now predicts streaming kernels reliably.

## Results

### Kernel (A10, median of 20)

| Kernel | Fastest | Median | Slowest | Bandwidth | % of 600 GB/s |
| --- | --- | --- | --- | --- | --- |
| ReLU | 1.078 ms | 1.081 ms | 1.094 ms | 496.5 GB/s | 82.7% |
| Square | 1.078 ms | 1.082 ms | 1.096 ms | 496.3 GB/s | 82.7% |

### Nsight Compute (Speed of Light section, A10)

| Metric | vector copy | **ReLU** | **Square** |
| --- | --- | --- | --- |
| Duration | 1.08 ms | **1.08 ms** | **1.09 ms** |
| DRAM throughput | 91.8% | **91.3%** | **91.2%** |
| Compute (SM) throughput | 17.4% | **20.0%** | **19.9%** |
| L2 cache throughput | 20.0% | **20.1%** | **20.0%** |
| L1/TEX cache throughput | 13.0% | **14.7%** | **14.6%** |

Compute throughput is a little above copy's: each element costs one more instruction
(`FMNMX` or `FMUL`). It is still far from mattering; the SMs mostly wait on memory.
A second ReLU profile that landed on an A10G gave the same 1.09 ms and 91.3% DRAM busy.

Reports: [`profiles/relu.ncu-rep`](profiles/relu.ncu-rep), [`profiles/square.ncu-rep`](profiles/square.ncu-rep).

**Measurement note:** while ncu is attached, every launch is slowed, not only the profiled
one. In the square profiling run, ReLU's (unprofiled) median was 1.336 ms. Speed comes only
from normal runs.

## Limiting resource

**DRAM bandwidth.** 537 MB per call at ~91% DRAM busy: the same ceiling as copy, add and
saxpy. Four kernels, four different operations, one speed for a given number of bytes.

## Correctness

- **ReLU** checked exactly against `max(h_x[i], 0.0f)` on the CPU. The input range
  [-1, 1) matters: with the first draft's [0, 1) input, every element was positive, ReLU
  behaved exactly like a copy, and the 0-mismatch check proved nothing.
- **Square** checked exactly against plain `h_x[i] * h_x[i]`. One multiply rounds once
  on both CPU and GPU, so (unlike saxpy) no `std::fma` is needed: **0 mismatches**.
- `compute-sanitizer --tool memcheck`: **0 errors** at N = 2²⁶ and at the awkward size
  N = 2²⁶ + 3 = 67,108,867 (253 surplus threads in the last block), both kernels
  0 mismatches.

**Compiler errors met on the way**, each read from the log and fixed:
`max(x[i], 0)` is ambiguous (float and int: which `max`?), so it needs `0.0f`;
declaring the timing variables a second time for square ("already declared in the
current scope"); and `h_x[i] * h_x` (a number times a pointer).

## What I learned

_To be written._
