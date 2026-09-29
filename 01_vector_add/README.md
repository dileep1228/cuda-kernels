# Vector add: C = A + B (FP32) == (32-bit floating point)

## Prediction (written before the first run)

N = 2^26 = 67,108,864 elements

- Bytes moved per element: ___  (which reads? which writes?)
    Which of A[i], B[i] and C[i] are read from memory? for this its 4*2 elements(a,b) = 8
    Which are written back? only c need to be written back so 4 (c)
    so for a 4, b 4 while reading and while writing c 4.
- Total bytes moved: ___ MB
    so total would be 12 bytes * 2^26 ≈ 805 MB.
- FLOPs (floating-point operation) per element: ___   → total FLOPs: ___
    C = A + B (one operation thats +)
    1 * 2^26
- Arithmetic intensity: ___ FLOPs (floating-point operations) per element  / bytes per element
    1 / 12
- A10 memory bandwidth (spec): 600 GB/s
- Expected bottleneck: memory or compute? ___ because ___ 
    I think memory is the bottleneck, math units mostly sit idle waiting. This is based on the calculations below.
    - Expected bottleneck: memory.
    The math takes 0.002 ms but moving the bytes takes 1.34 ms (~600× longer), so the math units mostly sit idle waiting for data.
- Predicted kernel time: ___ ms
    A kernel has two costs, moving the bytes and doing the math, and the slower of the two sets the time. So you work out both lower bounds and take the larger.

    1. Memory time: how long just moving the bytes takes
        memory time = total bytes ÷ memory bandwidth
                    = 12 bytes * 2^26 / 600 GB/s = 805 mb / 600 GB/s = 805,306,368 / 600 000 000 000 
                    = 0.001342 seconds = 1.342 ms

    2. Compute time: how long just doing the math takes
        compute time = total FLOPs (floating-point operations) ÷ compute throughput
                     = 1 * 2^26 / 31.2 TFLOP/s = 67,108,864 / 31,200,000,000,000 FLOPs per second
                     = 0.00000215 seconds
                     = 0.00215 ms

    Predicted ideal time = the larger of the two. = max (1.342 ms, 0.00215 ms) = 1.342 ms.

- Predicted fraction of peak I'll actually reach: ___ %  (hint: you measured this once already)
    My first guess: 40% to 60%, reasoning only from the ideal time.

    Note: this is not a blind prediction. Before writing it, I had already seen two
    measurements of the same maths on this same A10:
      - 89% at N = 2^20 (1M elements): my prefetch run of the "Even Easier
        Introduction to CUDA" kernel, 0.0235 ms → 535 GB/s.
      - 84% at N = 2^26: the environment check's test kernel, same size as this
        one, 1.603 ms → 502 GB/s.
    So the honest expectation is ~84%, i.e. a kernel time of about 1.342 / 0.84 ≈ 1.6 ms.

    Why a 64× bigger array could differ from the 1M run:
      - up: the kernel runs 64× longer, so start-up and wind-down (when not every
        SM (streaming multiprocessor) is busy) are a smaller share of the time.
      - down: each array is 268 MB, far bigger than the 6 MB L2 cache (level-2 cache),
        so every byte really comes from VRAM (video random-access memory); at 1M
        elements (4 MB per array) some may have been served from cache.
      - DRAM (dynamic random-access memory) never delivers 100% of its spec: it
        spends time refreshing and switching between reads and writes, and ECC
        (error-correcting code) is enabled on this A10.
        
- Predicted time for the H2D + D2H copies over PCIe (~25 GB/s): ___ ms
    (H2D = host to device, D2H = device to host, PCIe = Peripheral Component Interconnect Express)
    H2D: A + B = 2 × 268 MB = 537 MB
    D2H: C     = 1 × 268 MB = 268 MB
    total      = 805 MB
    805 MB ÷ 25 GB/s = 0.0322 s = 32.2 ms  →  ~20× the kernel time (~1.6 ms)
    Will the copies actually reach 25 GB/s? Not sure. Guess: lower, ~15 GB/s → 805 MB ÷ 15 GB/s ≈ 54 ms
    (~34× the kernel time), because 25 GB/s is itself a best case:
      - The A10 link is PCIe 4.0 × 16 lanes (nvidia-smi: gen 4, width 16).
        One lane = 16 GT/s (gigatransfers per second) × 128/130 encoding ÷ 8 ≈ 1.97 GB/s per direction;
        × 16 lanes ≈ 31.5 GB/s theoretical. Protocol overhead (packet headers, flow control)
        brings the best real-world figure to ~25 GB/s.
      - Every best case so far has come in lower when measured (VRAM: 600 GB/s spec, ~84–89% reached),
        so I expect the copies to fall short of 25 GB/s too.