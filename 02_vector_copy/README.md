# Vector copy: B = A

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements, NVIDIA A10 / A10G.

The simplest memory kernel: no maths, one read and one write per element. It sets the
practical speed limit for any kernel that streams through memory once.

**Result:** 1.08 ms, **~495 GB/s (gigabytes per second) = 82.6% of the 600 GB/s spec**,
with DRAM (dynamic random-access memory) 91.8% busy according to Nsight Compute.
**Limited by DRAM bandwidth**, the same wall as [vector add](../01_vector_add/), at the
same fraction of peak. Kernel time scales with bytes moved: 8 bytes per element instead
of 12 gives ⅔ of vector add's time (1.08 vs 1.61 ms).

## Setup

| | |
| --- | --- |
| GPU | NVIDIA A10 (72 SMs (streaming multiprocessors)) or A10G (80 SMs), both 600 GB/s; Modal assigns either, see [hardware.md](../hardware.md) |
| Launch | 262,144 blocks × 256 threads, grid-stride loop, 22 registers per thread |
| Build | `nvcc -O3 -arch=sm_86 -lineinfo` (CUDA 12.8) |
| Timing | 3 warm-up launches, then 20 launches timed with CUDA events; median reported |
| Input | random floats in [0, 1) |

```bash
modal run tools/run.py --file 02_vector_copy/vector_copy.cu                                  # benchmark
modal run tools/run.py --file 02_vector_copy/vector_copy.cu --mode ncu --skip 3 --count 1    # Nsight Compute
modal run tools/run.py --file 02_vector_copy/vector_copy.cu --mode sanitize                  # compute-sanitizer
```

## Prediction vs measurement

The prediction was written and committed before any code existed
(commit [`35b587f`](https://github.com/dileep1228/cuda-kernels/commit/35b587f)).

| | Predicted | Measured |
| --- | --- | --- |
| Bytes moved per element | 8 (read A, write B) | — |
| Total bytes moved | 537 MB (megabytes) (536,870,912 bytes) | — |
| FLOPs (floating-point operations) per element | 0 | — |
| Arithmetic intensity | 0 FLOP/byte: purely memory-bound | — |
| Bottleneck | memory (DRAM bandwidth) | memory: DRAM 91.8% busy (ncu) |
| Ideal kernel time at 600 GB/s | 0.895 ms | — |
| Kernel time | ~1.05 ms (milliseconds) | **1.084 ms** median (A10) |
| Fraction of 600 GB/s | ~85%, a little above `vector_add`'s 83% | **82.6%**, about the same as `vector_add` |

The prediction was 3% fast. Copy and add land at the same fraction of peak: neither does
any meaningful maths, so both run into the same DRAM ceiling.

## Results

### Kernel

| Card | N | Fastest | Median | Slowest | Bandwidth | % of 600 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| A10 | 2²⁶ | 1.078 ms | 1.084 ms | 1.093 ms | 495.3 GB/s | 82.6% |
| A10G | 2²⁶ + 3 | 1.085 ms | 1.095 ms | 1.106 ms | 490.2 GB/s | 81.7% |

The two cards differ by 8 SMs but run this kernel at the same speed. That is what
[hardware.md](../hardware.md) predicts for a memory-bound kernel: both cards have the same
384-bit memory bus and memory clock, and the extra SMs only add compute, which this kernel
does not use.

### Nsight Compute (Speed of Light section, A10G)

| Metric | vector copy | vector add, for comparison |
| --- | --- | --- |
| DRAM throughput | 91.8% | 91.8–93.0% |
| Compute (SM) throughput | 17.4% | 12.6% |
| L2 cache throughput | 20.0% | ~20% |
| L1/TEX cache throughput | 13.0% | ~10% |
| Duration | 1.08 ms | 1.60–1.62 ms |

Compute throughput is *higher* for copy, though it has less work per element. At this
level it mostly counts instructions issued (index maths, loads, stores), not FLOPs. Copy
moves 8 bytes per element instead of 12, so the same DRAM bandwidth pushes 1.5× more
elements per second through the SMs: 12.6% × 1.5 ≈ 19%, close to the 17.4% measured.
The SMs are still mostly idle, waiting on memory.

**Measurement note:** timings printed by the program during `ncu` and `compute-sanitizer`
runs are not valid. Under `ncu` the profiled launch is replayed 40 times (one "launch"
measured 2,731 ms); under the sanitizer every memory access is checked and the kernel
runs ~80× slower (85–89 ms). Speed comes only from normal runs.

### Copies over PCIe (Peripheral Component Interconnect Express), pageable memory

| H2D (A, 268 MB) | D2H (B, 268 MB) | Copies ÷ kernel |
| --- | --- | --- |
| 20.5–39.4 ms, 6.8–13.1 GB/s | 26.4–71.2 ms, 3.8–10.2 GB/s | 39–102× |

H2D = host to device, D2H = device to host. As with vector add, moving the data over PCIe
costs far more than the kernel itself, and pageable copies vary 2–3× from run to run.

## Limiting resource

**DRAM bandwidth.** The kernel moves 537 MB and keeps DRAM ~92% busy, the same ceiling
vector add hits. A copy cannot move fewer bytes than it copies, so ~495 GB/s is the
practical streaming limit on this card: the baseline every later memory-bound kernel in
this repo is compared against.

## Correctness

- Every element compared exactly against the input (a copy changes no bits):
  **0 mismatches** of 67,108,864.
- `compute-sanitizer --tool memcheck`: **0 errors**.
- **Awkward size, N = 2²⁶ + 3 = 67,108,867:** 262,145 blocks × 256 threads, so the last
  block has 3 threads with work and **253 surplus threads**. With the `i < N` guard:
  0 mismatches and 0 sanitizer errors. The program prints N at start-up, so the run
  itself records the size it tested (`N = 67108867 elements`).

## What I learned

_To be written._
