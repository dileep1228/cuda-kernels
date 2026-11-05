# Vector copy: B = A

FP32 (32-bit floating point), N = 2²⁶ = 67,108,864 elements.

The simplest memory kernel: no maths, one read and one write per element. It sets the
practical speed limit for any kernel that streams through memory once.

## Prediction

Written before any code existed.

| | Predicted |
| --- | --- |
| Bytes moved per element | 8 (read A, write B) |
| Total bytes moved | 537 MB (megabytes) (536,870,912 bytes) |
| FLOPs (floating-point operations) per element | 0 |
| Arithmetic intensity | 0 FLOP/byte: purely memory-bound |
| Bottleneck | memory (DRAM (dynamic random-access memory) bandwidth) |
| Ideal kernel time at 600 GB/s | 0.895 ms |
| Kernel time | ~1.05 ms (milliseconds) |
| Fraction of 600 GB/s | ~85%, a little above `vector_add`'s 83% |
