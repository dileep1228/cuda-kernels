# SAXPY: y = a·x + y

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements.

Single-precision A·X Plus Y, one of the oldest BLAS (Basic Linear Algebra Subprograms)
routines. The first kernel in this repo with real maths: one multiply and one add per
element, done as a single FMA (fused multiply-add) instruction.

## Prediction

Written before any code existed.

| | Predicted |
| --- | --- |
| Bytes moved per element | 12 (read x, read y, write y); `a` is one value kept in a register |
| Total bytes moved | 805 MB (megabytes) (805,306,368 bytes) |
| FLOPs (floating-point operations) per element | 2 (multiply + add, one FMA instruction) |
| Arithmetic intensity | 2 ÷ 12 = 1/6 ≈ 0.17 FLOP/byte, ~300× below the ~52 needed to be compute-bound |
| Bottleneck | memory (DRAM (dynamic random-access memory) bandwidth) |
| Ideal kernel time at 600 GB/s | 1.342 ms, identical to `vector_add` (same bytes) |
| Kernel time | ~1.60 ms (milliseconds) |
| Fraction of 600 GB/s | ~84% |
| vs `vector_add` (1.61 ms) | about the same: the extra FMA (~4 cycles) hides behind hundreds of cycles of memory wait |
