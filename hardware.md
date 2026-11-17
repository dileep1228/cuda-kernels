# Hardware: what the GPU says about itself

Measured with [`00_device_query/device_query.cu`](00_device_query/device_query.cu),
which asks the driver for the GPU's properties (`cudaGetDeviceProperties`). No kernel runs.

```bash
modal run tools/run.py --file 00_device_query/device_query.cu
```

## Modal's "A10" is two different cards

Asking Modal for `gpu="A10"` returned an **NVIDIA A10** on some runs and an **NVIDIA A10G**
(the AWS (Amazon Web Services) variant) on others. Same chip (GA102, sm_86), different
number of SMs (streaming multiprocessors) switched on:

| | A10 | A10G |
| --- | --- | --- |
| SMs | **72** | **80** |
| Max SM clock | 1,695 MHz | 1,710 MHz |
| Max threads per SM | 1,536 | 1,536 |
| Max blocks per SM | 16 | 16 |
| Registers per SM | 65,536 | 65,536 |
| Shared memory per SM | 102,400 bytes (100 KB) | 102,400 bytes (100 KB) |
| L2 cache | 6,291,456 bytes (6 MB) | 6,291,456 bytes (6 MB) |
| Warp size | 32 | 32 |
| Memory bus width | 384 bits | 384 bits |
| Memory clock | 6,251 MHz | 6,251 MHz |

**What this means for results in this repo:**

- **Memory-bound kernels** (vector add, copy, reductions) should be close on both cards:
  same bus, same memory clock, same 600 GB/s.
- **Compute-bound kernels** (matrix multiply) can be ~12% faster on the A10G, from 8 more SMs.
- So every result names the card it ran on. `tools/run.py` prints the GPU name and UUID
  (universally unique identifier) before every run.

## Derived numbers

### Peak memory bandwidth: 600 GB/s

```
6,251,000 kHz × 1000      = 6.251 × 10⁹ clock ticks per second
            × 2           = 12.5 × 10⁹ transfers per second (two per clock)
            × 384 / 8     = 48 bytes per transfer
                          = 600.1 × 10⁹ bytes per second = 600.1 GB/s
```

The 600 GB/s (gigabytes per second) spec is this product. No kernel reaches it:
`vector_add` measured ~92% of the DRAM (dynamic random-access memory) busy at ~502 GB/s.

### Peak FP32 compute: 31.2 / 35.0 TFLOP/s

Each SM has 128 FP32 (32-bit floating point) units. Each does one fused multiply-add
per cycle = 2 FLOPs (floating-point operations).

```
A10:  72 SMs × 128 × 2 × 1.695 × 10⁹ = 31.2 TFLOP/s  (matches the spec sheet)
A10G: 80 SMs × 128 × 2 × 1.710 × 10⁹ = 35.0 TFLOP/s
```

### Compute-bound threshold: ~52 FLOP/byte

Peak compute ÷ peak bandwidth: **31.2 TFLOP/s ÷ 600 GB/s ≈ 52 FLOP/byte** on the A10
(≈ 58 on the A10G). A kernel doing fewer FLOPs per byte than this is limited by memory.
Vector add does 1/12.

### Blocks per SM for 256-thread blocks: 6

Each SM has fixed budgets. The one that runs out first sets the limit:

| Limit | Per SM | Blocks of 256 threads that fit |
| --- | --- | --- |
| Threads | 1,536 | 1,536 ÷ 256 = **6** ← runs out first |
| Blocks | 16 | 16 |
| Registers | 65,536 | 6 blocks need ≤ ~40 registers per thread (65,536 ÷ 1,536 ≈ 42) |
| Shared memory | 100 KB | no limit if the kernel uses none |

6 blocks × 8 warps = 48 warps, the SM maximum (1,536 ÷ 32): **100% occupancy**.

### Block size vs occupancy (24 registers per thread)

Blocks that fit per SM for each limit. The smallest wins; round down, since a block fits whole or not at all.
Every block also costs 1 KB of driver-reserved shared memory and one block slot, whatever its size.

| Block size | Warps per block | Block slots (16) | Warps (48) | Registers (65,536) | Shared memory | Winner | Active warps | Theoretical occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1 | 16 | 48 | 84 | 16 (16 KB config) | slots / shared memory | 16 | **33.3%** |
| 256 | 8 | 16 | **6** | 10 | 8 (8 KB config) | warps | 48 | **100%** |
| 1,024 | 32 | 16 | **1** | 2 | 8 | warps | 32 | **66.7%** |

- 1,024-thread blocks: a second block would need 64 warps; the last 16 warp slots stay empty.
- 32-thread blocks: per-block costs run out long before warps do. On paper (8 KB config) only 8 blocks
  fit, giving 16.7%; the driver chose a 16 KB config instead, so 16 fit. Measure, don't assume.

Measured with `stride.cu`, stride 1, A10 (ncu):

| Block size | Achieved occupancy | DRAM throughput | Duration |
| --- | --- | --- | --- |
| 256 | 79.9% | 90.4% | 0.272 ms |
| 32 | 24.3% | 63.7% | 0.399 ms (+47%) |

Same bytes moved (~75 MB read), so the slowdown is idle DRAM time: too few warps to keep
enough loads in flight (Little's Law: bandwidth = bytes in flight ÷ latency). At stride 8 the
32-thread version matched the 256-thread one (1.17 ms), because each request asks for 32 sectors
instead of 4, so fewer warps still keep the memory busy. Occupancy matters most when each warp
has little in flight.

### L2 cache: 6 MB

Small next to vector add's 805 MB of traffic (~130× larger), so that kernel gets no
help from the cache: every byte comes from DRAM.
