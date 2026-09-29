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
- **Not yet tested:** a size that isn't a multiple of 256. N = 2²⁶ divides evenly,
  so the `i < N` guard is never exercised by these runs.

## What I learned

- **More threads help only until the GPU hits its real limit.** Going from one thread to
  one block to blocks on all 72 SMs made the kernel much faster, but for vector add the
  real limit is memory bandwidth, not compute. Once all SMs were busy, the math units
  still sat ~87% idle, waiting for data.
- **The prediction method works.** Bytes moved ÷ bandwidth gave an ideal time of 1.34 ms;
  the measured 1.61 ms is 83% of that. Counting bytes and FLOPs before writing code tells
  you what the kernel is limited by.
- **The CUDA basics:** separate CPU and GPU memory (`cudaMalloc`, `cudaMemcpy`, the
  destination comes first), CUDA events for timing GPU work, and warm-up launches,
  because the first launches are slower while clocks ramp up and one-off setup happens.
- **Failures can be silent.** A launch with 2048 threads per block (the limit is 1024)
  printed nothing and returned garbage until I added `cudaGetLastError()`. Error checking
  and a correctness check catch different failures, so every kernel needs both.
- **Moving data to the GPU costs more than the work on it.** The copies took 20–60× longer
  than the kernel. Pinned memory (`cudaMallocHost`) made them 2–5× faster and steady at
  the PCIe limit; `malloc` memory was slow and varied a lot between runs.
- **The spec is not reachable.** 600 GB/s is the spec, but the memory can realistically
  sustain about 545 GB/s, so 83% of spec is really about 92% of what is possible.
- **One measurement proves nothing.** A slow D2H copy I tried to explain turned out to be
  run-to-run noise, and a fast 1.43 ms run turned out to be a different card.
- **Still getting comfortable with** unit conversions in predictions (bytes → GB,
  ms → s, GB/s vs TFLOP/s). They get easier each time I do them.
