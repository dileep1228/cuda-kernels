# cuda-kernels

CUDA and Triton kernels written from scratch, each one profiled and explained.

The goal is not a collection of fast kernels. It is one habit, applied to every
kernel in this repo:

> predict → implement → measure → profile → explain the gap → change one thing → repeat

Every kernel directory records the prediction made *before* the first run, the
benchmark, the Nsight Compute evidence, and a short write-up of what limited it
and why.

## Platform

Every number in this repo comes from one GPU, so results are comparable.

| | |
| --- | --- |
| GPU | NVIDIA A10 or A10G (Ampere, sm_86, 24 GB GDDR6, ECC on); Modal assigns either, see [hardware.md](hardware.md) | 
| Host | [Modal](https://modal.com), container `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Toolkit | CUDA 12.8, driver 580 |
| Profilers | Nsight Compute, Nsight Systems, compute-sanitizer |

Two platform notes that affect reproduction:

- **`ncu --clock-control none`** — Modal does not allow Nsight Compute to lock
  GPU clocks, so profiles run at the GPU's own clocks. Run-to-run variation is
  slightly higher; benchmarks report medians.
- **`nsys profile --trace=cuda,nvtx`** — the default trace set records no
  kernels in this environment.
- **A10 or A10G** — `gpu="A10"` on Modal lands on either card: 72 or 80 SMs, same
  600 GB/s memory. Every run prints which one it got. Details in [hardware.md](hardware.md).

`tools/env_check.py` verifies the whole toolchain on a fresh GPU:

```bash
modal run tools/env_check.py
```

`tools/run.py` compiles and runs any single `.cu` file on the A10:

```bash
modal run tools/run.py --file path/to/kernel.cu                 # compile + run
modal run tools/run.py --file path/to/kernel.cu --mode ncu      # Nsight Compute, saves profiles/*.ncu-rep
modal run tools/run.py --file path/to/kernel.cu --mode nsys     # Nsight Systems timeline
modal run tools/run.py --file path/to/kernel.cu --mode sanitize # compute-sanitizer memcheck
```

## What "done" means for a kernel

- a written prediction: bytes moved, FLOPs, arithmetic intensity, expected bottleneck, expected time
- a correctness test against a reference, including sizes that are not a multiple of the block size
- a median benchmark, with the GPU named
- Nsight Compute numbers, with the `.ncu-rep` committed: time, bandwidth, DRAM and SM throughput, registers, occupancy
- an Nsight Systems timeline, with the `.nsys-rep` committed
- one sentence naming the limiting resource
- a clean `compute-sanitizer` run
- the write-up, committed the same day the kernel works

## Kernels

| Kernel | CUDA | Triton | Result | Limited by |
| --- | --- | --- | --- | --- |
| [Vector add](01_vector_add/) | ✅ | | 1.61 ms, ~500 GB/s, 83% of spec (92% DRAM busy) | DRAM bandwidth |
| [Vector copy](02_vector_copy/) | ✅ | | 1.08 ms, ~495 GB/s, 83% of spec (92% DRAM busy) | DRAM bandwidth |
| [SAXPY](03_saxpy/) | ✅ | | 1.63 ms, ~495 GB/s, 82.5% of spec (91% DRAM busy) | DRAM bandwidth |
| [ReLU + square](04_relu_square/) | ✅ | | 1.08 ms each, ~495 GB/s, 82.7% of spec (91% DRAM busy) | DRAM bandwidth |

## Profile summary

The six numbers for every kernel, from Nsight Compute (one warmed-up launch each):

| Kernel | Time | Bandwidth | DRAM busy | SM busy | Registers | Occupancy (theoretical → achieved) | FLOP/byte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vector add | 1.62 ms | ~500 GB/s | 91.7% | 14.1% | 30 | 100% → 77.9% | 0.083 |
| Vector copy | 1.08 ms | ~495 GB/s | 91.5% | 19.6% | 22 | 100% → 80.1% | 0 |
| SAXPY | 1.62 ms | ~495 GB/s | 91.6% | 13.6% | 24 | 100% → 82.2% | 0.17 |
| ReLU | 1.09 ms | ~495 GB/s | 91.2% | 20.1% | 28 | 100% → 79.5% | 0 |
| Square | 1.09 ms | ~495 GB/s | 91.2% | 17.6% | 28 | 100% → 78.9% | 0.125 |

**Roofline.** The A10's ridge point is 31.2 TFLOP/s ÷ 600 GB/s ≈ 52 FLOP/byte. Every kernel
here is at 0.17 FLOP/byte or less, 300× or more to the left of it, and sits just under the
memory roof: DRAM ~91% busy, SMs ~80% idle. Time is set by bytes moved, not by the
arithmetic: 8 bytes per element takes ~1.08 ms whatever the operation, and 12 bytes takes ~1.62 ms.

**Nsight Systems.** In every program, PCIe copies cost 1.7–2.5× the total of all 23–46
kernel launches, and the first CUDA call spends ~130–150 ms creating the CUDA context.
