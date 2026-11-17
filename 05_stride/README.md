# Stride benchmark: out[i] = in[i * stride]

FP32 (32-bit floating point), N = 2²⁴ = 16,777,216 elements, NVIDIA A10 / A10G.

A copy where only the **read** is spread out: thread `i` reads `in[i * stride]` and writes
`out[i]`, for stride = 1, 2, 4, 8, 16, 32. The write stays coalesced; the read wastes more
of every 32-byte sector as the stride grows.

## Prediction

Written before any code existed.

**Model:** memory delivers whole 32-byte sectors (8 floats). A warp of 32 threads reading
4 bytes each needs 128 useful bytes. At stride *s* the warp touches min(*s*, 8) × 4
sectors, so each element costs 4 × *s* bytes of read traffic, capped at one full sector
(32 bytes) from stride 8 on. The write always costs 4 bytes.

| Stride | Read bytes / element | Write | Total bytes / element | Predicted time (÷ 495 GB/s) | vs stride 1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 | 4 | 8 | 0.27 ms (134 MB) | 1× |
| 2 | 8 | 4 | 12 | 0.41 ms | 1.5× |
| 4 | 16 | 4 | 20 | 0.68 ms | 2.5× |
| 8 | 32 | 4 | 36 | 1.22 ms (604 MB) | 4.5× |
| 16 | 32 | 4 | 36 | 1.22 ms | 4.5× |
| 32 | 32 | 4 | 36 | 1.22 ms | 4.5× |

**Expected shape:** time rises with the stride up to 8, then stays flat. Useful bandwidth
(8 bytes × N ÷ time) falls from ~495 GB/s to ~110 GB/s.

This is a model to test, not a law: the L2 cache or the DRAM's own access granularity
may make the large strides cost more or less than one sector per element.
