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
| GPU | NVIDIA A10 (Ampere, sm_86, 24 GB GDDR6, ECC on) |
| Host | [Modal](https://modal.com), container `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| Toolkit | CUDA 12.8, driver 580 |
| Profilers | Nsight Compute, Nsight Systems, compute-sanitizer |

Two platform notes that affect reproduction:

- **`ncu --clock-control none`** — Modal does not allow Nsight Compute to lock
  GPU clocks, so profiles run at the GPU's own clocks. Run-to-run variation is
  slightly higher; benchmarks report medians.
- **`nsys profile --trace=cuda,nvtx`** — the default trace set records no
  kernels in this environment.

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
- Nsight Compute numbers, with the `.ncu-rep` committed
- one sentence naming the limiting resource
- a clean `compute-sanitizer` run
- the write-up, committed the same day the kernel works

## Kernels

| Kernel | CUDA | Triton | Limited by |
| --- | --- | --- | --- |
| *(first one coming)* | | | |
