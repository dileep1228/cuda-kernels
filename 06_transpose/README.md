# Matrix transpose: out[c][r] = in[r][c]

FP32 (32-bit floating point), N × N = 8,192 × 8,192 = 67,108,864 elements, 268 MB per
matrix, NVIDIA A10 / A10G.

Row `r`, column `c` of the input becomes row `c`, column `r` of the output. Unlike the
stride benchmark, the bad stride here is unavoidable: if the reads walk along a row
(coalesced), the writes walk down a column (strided by N), or the other way round.

| Version | Kernel | Idea |
| --- | --- | --- |
| v0 | `copy2d` | `out[r][c] = in[r][c]`, same 2D layout: the speed limit |
| v1 | `transpose_naive` | `out[c][r] = in[r][c]`: coalesced read, strided write |
| v2 | (step 3) | shared-memory tile: coalesced read **and** write |
| v3 | (step 4) | v2 with padding to remove shared-memory bank conflicts |

Launch: `dim3 threads(32, 8)`, `dim3 blocks(N / 32, N / 8)`, one thread per element.
`threadIdx.x` walks along a row (column index `c`), `threadIdx.y` picks the row `r`.
Address of element `[row][col]` = `row × N + col`.

## Prediction (v0, v1)

Written before any code existed.

**Model (from the stride benchmark):** memory moves whole 32-byte sectors (8 floats).
What matters is the **bytes moved per element**: time = bytes per element × elements ÷
bandwidth, and that number is the only part the code controls.

**Read `in[r][c..c+31]`** (both kernels): one warp's 32 floats sit side by side in one row.
32 × 4 bytes = 128 bytes = 4 sectors per request, every byte used: **4 bytes per element**.

**Write:**
- v0 `out[r][c..c+31]`: same as the read, 4 sectors per request: **4 bytes per element**.
- v1 `out[c..c+31][r]`: neighbouring threads write to neighbouring **rows**, 8,192 floats
  (32 KB) apart: stride 8,192. Every write lands in its own sector, 32 sectors per request,
  4 of 32 bytes used: **32 bytes per element** (cost stops growing from stride 8 on).

| | Requests (reads) | Sectors / read request | Sectors / write request | Bytes / element | Bytes moved | Predicted time (÷ 495 GB/s) | vs v0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v0 copy | 2,097,152 | 4 | 4 | 4 + 4 = 8 | 537 MB | 1.08 ms | 1× |
| v1 naive | 2,097,152 | 4 | 32 | 4 + 32 = 36 | 2,416 MB | 4.88 ms | 4.5× |

Read sectors, both kernels: 2,097,152 requests × 4 = 8,388,608 sectors = 268 MB, exactly
one matrix.

**Rule of thumb:** the index multiplied by the row length must not be the one that changes
with `threadIdx.x`. In the read it is `r` (fixed across the warp): good. In v1's write it is
`c` (changes across the warp): bad.

**Goal for v2:** bring the write back from 32 to 4 bytes per element: 36 → 8 bytes,
~4.9 ms → ~1.1 ms.

This is a model to test. The stride benchmark showed DRAM over-fetching at large strides,
and writes may behave differently from reads (the L2 cache can collect partial-sector
writes before they reach DRAM), so v1 could land above or below 4.88 ms.
