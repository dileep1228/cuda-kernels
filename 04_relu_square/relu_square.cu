#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

std::random_device rd;                                    // seed source
std::mt19937 gen(rd());                                   // Mersenne Twister engine
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  // [-1, 1): half the inputs are negative,
                                                          // so relu really has work to do

// Two elementwise kernels, each reading x and writing y: 8 bytes per element.
// Grid-stride loop: each thread starts at its global index and jumps by the
// total number of threads in the grid, so any launch size covers all N elements.

// ReLU (rectified linear unit): y = max(x, 0). 0.0f, not 0: with an int 0 the
// compiler can't choose between max(int, int) and max(float, float).
__global__ void relu(const float* __restrict__ x, float* __restrict__ y, int N) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = index; i < N; i += stride) {
        y[i] = max(x[i], 0.0f);
    }
}

// Square: y = x * x. One multiply, one rounding, the same on CPU and GPU.
__global__ void square(const float* __restrict__ x, float* __restrict__ y, int N) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = index; i < N; i += stride) {
        y[i] = x[i] * x[i];
    }
}

int main(int argc, char** argv) {
    const int N = (1 << 26);                  // 2^26 = 67,108,864 elements
    const size_t bytes = N * sizeof(float);   // size of ONE array in bytes

    // Host memory kind, chosen on the command line:
    //   ./relu_square          pageable (malloc): copies go through the driver's staging buffer
    //   ./relu_square pinned   pinned (cudaMallocHost): the copy engine reads/writes our arrays directly
    const bool pinned = argc > 1 && strcmp(argv[1], "pinned") == 0;
    printf("host memory: %s \n", pinned ? "pinned (cudaMallocHost)" : "pageable (malloc)");
    printf("N = %d elements\n", N);

    // 1. Host (CPU (central processing unit)) memory: x is the input, y receives each
    //    kernel's result back from the GPU (relu's first, then square's)
    float *h_x, *h_y;
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_x, bytes));
        CUDA_CHECK(cudaMallocHost(&h_y, bytes));
    } else {
        h_x = (float*)malloc(bytes);
        h_y = (float*)malloc(bytes);
    }
    // Touch every page of h_y before the D2H copy, as the earlier kernels did,
    // so copy timings are measured the same way.
    memset(h_y, 0, bytes);

    for (int i = 0; i < N; i++) {
        h_x[i] = dist(gen);
    }

    // 2. Device (GPU (graphics processing unit)) memory
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // 3. Events: GPU-side timestamps, created once and reused for every measurement below
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 4. Copy the input host → device (H2D), timed. Only x: 1 array over PCIe.
    //    Both kernels read this same d_x, so it is copied once.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));   // (destination, source, bytes, direction)
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    // Launch shape and timing buffer, shared by both kernels
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    const int RUNS = 20;
    float times[RUNS];   // one kernel time per run, in ms (milliseconds); reused by square

    // ===================================== ReLU =====================================

    printf("\n--- relu: y = max(x, 0) ---\n");

    // 5. Warm-up: run 3 times, don't time these (clocks ramp up, first-launch setup)
    for (int w = 0; w < 3; w++) {
        relu<<<blocks, threads>>>(d_x, d_y, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());   // wait until all 3 have finished

    // 6. Kernel timing: 20 runs
    for (int r = 0; r < RUNS; r++) {
        CUDA_CHECK(cudaEventRecord(start));                        // (a)
        relu<<<blocks, threads>>>(d_x, d_y, N);                    // (b)
        CUDA_CHECK(cudaEventRecord(stop));                         // (c)
        CUDA_CHECK(cudaEventSynchronize(stop));                    // (d)
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));  // (e)
    }
    CUDA_CHECK(cudaGetLastError());

    // 7. Copy the result device → host (D2H), timed. Only y: 1 array over PCIe.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // 8. Check against the CPU, exactly (max only selects a value, no rounding)
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (h_y[i] != max(h_x[i], 0.0f)) {
            mismatches++;
        }
    }
    printf("total number of mismatches: %d \n", mismatches);

    // 9. Results
    // Median: sort the 20 times (in place, on the CPU), then average the two
    // middle values, times[9] and times[10], since 20 has no single middle.
    // The median ignores the odd slow or fast outlier; min and max show the spread.
    std::sort(times, times + RUNS);
    float median_ms = (times[RUNS/2 - 1] + times[RUNS/2]) / 2.0f;
    printf("Fastest: %.3f ms, median: %.3f ms and slowest: %.3f ms \n", times[0], median_ms, times[RUNS-1]);

    // Achieved bandwidth = bytes moved / time taken, in GB/s (gigabytes per second).
    //   bytes            = size of ONE array = N * 4 bytes = 268 MB (megabytes)
    //   2.0 * bytes      = the kernel reads x and writes y: 2 arrays = 537 MB
    //                      (same as 8 bytes per element * N). 2.0, not 2, keeps the maths in double.
    //   median_ms / 1000 = milliseconds -> seconds
    //   / 1e9            = bytes per second -> gigabytes per second (1e9 = 10^9; ^ in C++ is XOR, not power)
    // Example: 536,870,912 bytes / 0.00108 s / 1e9 = 497 GB/s
    float achieved_bandwidth = ((2.0 * bytes) / (median_ms / 1000)) / 1e9;
    printf("Achieved bandwidth in GB/s: %.3f \n", achieved_bandwidth);

    // Percent of peak = achieved / A10 spec bandwidth (600 GB/s) * 100.
    // Example: 497 / 600 * 100 = 83%. The spec is a theoretical maximum; no kernel reaches 100%.
    double percent_of_peak = achieved_bandwidth / 600.0 * 100.0;
    printf("fraction of peak : %.3f \n", percent_of_peak);

    // Copies over PCIe (Peripheral Component Interconnect Express), same bandwidth formula:
    //   H2D moved x = 1.0 * bytes, D2H moved y = 1.0 * bytes.
    printf("H2D: %.2f ms, %.1f GB/s \n", h2d_ms, (1.0 * bytes) / (h2d_ms / 1000) / 1e9);
    printf("D2H: %.2f ms, %.1f GB/s \n", d2h_ms, (1.0 * bytes) / (d2h_ms / 1000) / 1e9);
    printf("copies / kernel: %.1fx \n", (h2d_ms + d2h_ms) / median_ms);

    // ==================================== Square ====================================
    // Same steps 5–9, reusing d_y, h_y, times and the result variables.
    // ReLU's result has already been checked, so overwriting y is safe.

    printf("\n--- square: y = x * x ---\n");

    // 5. Warm-up
    for (int w = 0; w < 3; w++) {
        square<<<blocks, threads>>>(d_x, d_y, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Kernel timing: 20 runs, overwriting relu's times
    for (int r = 0; r < RUNS; r++) {
        CUDA_CHECK(cudaEventRecord(start));                        // (a)
        square<<<blocks, threads>>>(d_x, d_y, N);                  // (b)
        CUDA_CHECK(cudaEventRecord(stop));                         // (c)
        CUDA_CHECK(cudaEventSynchronize(stop));                    // (d)
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));  // (e)
    }
    CUDA_CHECK(cudaGetLastError());

    // 7. Copy the result device → host (D2H), timed
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // 8. Check against the CPU, exactly: one multiply rounds once on both sides,
    //    so no std::fma is needed (unlike saxpy)
    mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (h_y[i] != h_x[i] * h_x[i]) {
            mismatches++;
        }
    }
    printf("total number of mismatches: %d \n", mismatches);

    // 9. Results: same formulas as relu (8 bytes per element)
    std::sort(times, times + RUNS);
    median_ms = (times[RUNS/2 - 1] + times[RUNS/2]) / 2.0f;
    printf("Fastest: %.3f ms, median: %.3f ms and slowest: %.3f ms \n", times[0], median_ms, times[RUNS-1]);

    achieved_bandwidth = ((2.0 * bytes) / (median_ms / 1000)) / 1e9;
    printf("Achieved bandwidth in GB/s: %.3f \n", achieved_bandwidth);

    percent_of_peak = achieved_bandwidth / 600.0 * 100.0;
    printf("fraction of peak : %.3f \n", percent_of_peak);

    // Square reuses the x already on the GPU, so it has no H2D copy of its own:
    // only its D2H is reported, and copies / kernel counts that one copy.
    printf("D2H: %.2f ms, %.1f GB/s \n", d2h_ms, (1.0 * bytes) / (d2h_ms / 1000) / 1e9);
    printf("copies / kernel: %.1fx (D2H only) \n", d2h_ms / median_ms);

    // 10. Free both sides
    // Whatever allocated it frees it: cudaMallocHost → cudaFreeHost, malloc → free
    if (pinned) {
        CUDA_CHECK(cudaFreeHost(h_x));
        CUDA_CHECK(cudaFreeHost(h_y));
    } else {
        free(h_x);
        free(h_y);
    }
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
