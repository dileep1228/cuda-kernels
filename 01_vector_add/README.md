# Vector add: C = A + B

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements, NVIDIA A10.

**Result:** 1.61 ms, **~500 GB/s (gigabytes per second) = 83% of the 600 GB/s spec**,
and 92% of what the memory can actually sustain according to Nsight Compute.
**Limited by DRAM (dynamic random-access memory) bandwidth.** The math units sit
~87% idle. Getting the data to and from the GPU (graphics processing unit) over
PCIe (Peripheral Component Interconnect Express) costs **20–60× the kernel itself**.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA A10 (Ampere, sm_86, 72 SMs (streaming multiprocessors), 24 GB, 600 GB/s spec), on [Modal](https://modal.com) |
| Launch | 262,144 blocks × 256 threads, grid-stride loop |
| Build | `nvcc -O3 -arch=sm_86 -lineinfo` (CUDA 12.8) |
| Timing | 3 warm-up launches, then 20 launches timed with CUDA events; median reported |
| Input | random floats in [0, 1) |

```bash
modal run tools/run.py --file 01_vector_add/vector_add.cu                 # pageable host memory
modal run tools/run.py --file 01_vector_add/vector_add.cu --args pinned   # pinned host memory
modal run tools/run.py --file 01_vector_add/vector_add.cu --mode ncu      # Nsight Compute
modal run tools/run.py --file 01_vector_add/vector_add.cu --mode sanitize # compute-sanitizer
```

## Prediction vs measurement

The prediction was written and committed before any timing code existed
(commit [`ce6389f`](https://github.com/dileep1228/cuda-kernels/commit/ce6389f)).

| | Predicted | Measured |
| --- | --- | --- |
| Bytes moved per element | 12 (read A, read B, write C) | — |
| Total bytes moved | 805 MB (megabytes) | — |
| FLOPs (floating-point operations) per element | 1 | — |
| Arithmetic intensity | 1/12 ≈ 0.083 FLOP/byte (A10 needs ~52 to be compute-bound) | — |
| Bottleneck | memory | memory: DRAM 92% busy, SMs 12.6% (ncu) |
| Ideal kernel time at 600 GB/s | 1.342 ms | — |
| Kernel time | ~1.6 ms (at ~84% of peak) | **1.605–1.616 ms** median |
| Fraction of 600 GB/s | ~84% | **83.1–83.6%** |
| Copies, pageable `malloc` memory | ~15 GB/s, ~54 ms, ~34× the kernel | **8.6–13.1 GB/s** H2D, **4.6–10.0 GB/s** D2H, ~57–62× |
| Copies, pinned memory (not predicted) | (25 GB/s from PCIe 4.0 × 16 arithmetic) | **25.2 GB/s** H2D, **26.3 GB/s** D2H, ~20× |

H2D = host to device, D2H = device to host.

**How honest the prediction was.** The kernel-time prediction was *not* blind: before
writing it I had already seen two measurements of the same maths on the A10
(89% at N = 2²⁰, 84% at N = 2²⁶). My first blind guess was 40–60% of peak, which
would have meant 2.2–3.4 ms, off by 1.4–2×. The copy-time prediction was blind.

## Results

### Kernel

Median of 20 runs, across repeated runs on Modal:

| | Fastest | Median | Slowest | Bandwidth | % of 600 GB/s |
| --- | --- | --- | --- | --- | --- |
| Typical run | 1.599 ms | 1.605 ms | 1.617 ms | 501.9 GB/s | 83.6% |

The spread within a run is about 1%, so the median is stable. Seven of eight runs
landed at 1.60–1.62 ms. One run measured 1.430 ms (563 GB/s, 94%) with no code
change, most likely a different physical A10: Modal can assign a different card
each run, which is why `tools/run.py` now prints the GPU's UUID (universally unique
identifier) with every result.

### Nsight Compute (Speed of Light section)

| Metric | Value |
| --- | --- |
| DRAM throughput | 91.8–93.0% |
| Compute (SM) throughput | 12.6% |
| L2 cache throughput | ~20% |
| L1/TEX cache throughput | ~10% |
| Duration | 1.60–1.62 ms (matches the CUDA-event timing) |
| SM active cycles / elapsed cycles | ~99.7% |

502 GB/s measured ÷ 0.92 busy ≈ 545 GB/s: roughly the ceiling the DRAM can
realistically sustain. So about half of the gap to the 600 GB/s spec is unreachable
by any code (refresh, read/write turnaround, ECC (error-correcting code) is on), and
only ~8% is left on the table. Compute throughput is 12.6% rather than ~0% because
it counts load/store and integer index instructions, not only the one FP32 add.

### Copies over PCIe

| Host memory | H2D (A + B, 537 MB) | D2H (C, 268 MB) | Copies ÷ kernel |
| --- | --- | --- | --- |
| Pageable (`malloc`) | 41–62 ms, 8.6–13.1 GB/s | 27–59 ms, 4.6–10.0 GB/s | 57–62× |
| Pinned (`cudaMallocHost`) | 21.3 ms, 25.2 GB/s | 10.2 ms, 26.3 GB/s | ~20× |

Pinned copies hit the PCIe 4.0 × 16 practical limit (~25 GB/s: 16 lanes × 1.97 GB/s
= 31.5 GB/s theoretical, minus protocol overhead) and are identical run to run.
Pageable copies go through the driver's pinned staging buffer (a CPU (central
processing unit) copy, then the DMA (direct memory access) transfer), are 2–5×
slower, and vary a lot between runs.

**Hypothesis tested and ruled out:** an early pageable run had D2H 3× slower than H2D.
I guessed first-touch page faults on the never-written `h_C`. Pre-touching `h_C` with
`memset` changed nothing (4.7 → 4.6 GB/s), and later runs showed the asymmetry itself
was run-to-run noise.

## Limiting resource

**DRAM bandwidth.** Every element needs 12 bytes of memory traffic for one FLOP, and the
kernel keeps DRAM ~92% busy. It cannot move fewer than 805 MB, so the only way to be
faster is to not make the trip at all: fuse this add into the kernel that produces A or
consumes C.

## Correctness

- Every element compared exactly against the CPU result (a single FP32 add is
  bit-identical on CPU and GPU): **0 mismatches** of 67,108,864.
- `compute-sanitizer --tool memcheck`: **0 errors**.
- **Awkward size, N = 2²⁶ + 3 = 67,108,867:** 262,145 blocks × 256 = 67,109,120 threads,
  so the last block has 3 threads with work and **253 surplus threads**. With the `i < N`
  guard: 0 mismatches and 0 sanitizer errors. N = 2²⁶ alone divides evenly by 256, so it
  never exercises the guard.
- **Off-by-one check:** changing the guard to `i <= N` lets exactly one surplus thread
  through, thread (3,0,0) of block 262,144, whose index is N. `compute-sanitizer` reports
  an invalid 4-byte read at `vector_add.cu:29`, just past the end of `A`. Run normally,
  the same broken kernel gives **0 mismatches, normal speed and no CUDA error**: the
  correctness check never looks at element N, and the access lands in padding inside the
  allocation. Only the awkward size plus the sanitizer together catch this bug.

## What I learned

- How to increase the compute power of GPUs: add threads, blocks and grids to increase the compute speed.
- Multiple CUDA commands.
- How data travels between the GPU and the CPU, and how the GPU processes it in parallel.
- The latency of copying from the CPU.
- Warming up GPUs before timing.
- Although I am confused while calculating some predictions and numbers, I am sure I will slowly get used to it.
