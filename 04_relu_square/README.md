# ReLU and square: y = max(x, 0), y = x²

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements, input random in [-1, 1).
Two elementwise kernels in one file, each writing to a separate output array.

## Prediction

Written before any code existed. The fraction of peak comes from the three kernels
already measured (add, copy and saxpy all reached 82–83% of 600 GB/s).

| | ReLU | Square |
| --- | --- | --- |
| Bytes moved per element | 8 (read x, write y) | 8 (read x, write y) |
| FLOPs (floating-point operations) per element | 0 (a comparison, one `FMNMX` instruction) | 1 (one multiply) |
| Arithmetic intensity | 0 FLOP/byte | 1/8 FLOP/byte |
| Bottleneck | memory (DRAM (dynamic random-access memory) bandwidth) | memory (DRAM bandwidth) |
| Ideal kernel time at 600 GB/s | 0.895 ms | 0.895 ms |
| Kernel time | ~1.08 ms (0.895 ÷ 0.83) | ~1.10 ms (0.895 ÷ 0.81) |
| Should match | `vector_copy` (1.08 ms): same bytes | `vector_copy` (1.08 ms): same bytes |
| CPU vs GPU rounding | exact (no arithmetic) | exact: one multiply rounds once on both |
