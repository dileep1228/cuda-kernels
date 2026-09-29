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

std::random_device rd;                          // seed source
std::mt19937 gen(rd());                         // Mersenne Twister engine
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

// Grid-stride loop: each thread starts at its global index and jumps by the
// total number of threads in the grid, so any launch size covers all N elements.
__global__ void vector_add(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = index; i<N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    const int N = 1 << 26;                        // 2^26 = 67,108,864 elements
    const size_t bytes = N * sizeof(float);       // size of ONE array in bytes

    // Host memory kind, chosen on the command line:
    //   ./vector_add          pageable (malloc): copies go through the driver's staging buffer
    //   ./vector_add pinned   pinned (cudaMallocHost): the copy engine reads/writes our arrays directly
    const bool pinned = argc > 1 && strcmp(argv[1], "pinned") == 0;
    printf("host memory: %s \n", pinned ? "pinned (cudaMallocHost)" : "pageable (malloc)");

    // 1. Host (CPU (central processing unit)) memory, filled with random floats
    float *h_A, *h_B, *h_C;
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_A, bytes));
        CUDA_CHECK(cudaMallocHost(&h_B, bytes));
        CUDA_CHECK(cudaMallocHost(&h_C, bytes));
    } else {
        h_A = (float*)malloc(bytes);
        h_B = (float*)malloc(bytes);
        h_C = (float*)malloc(bytes);
    }
    // Experiment: touch every page of h_C before the D2H copy, to test whether
    // first-touch page faults explain the slow D2H. Result: no — D2H stayed at
    // ~4.6 GB/s with or without this line (see README).
    memset(h_C, 0, bytes);

    for(int i = 0; i< N; i++) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // 2. Device (GPU (graphics processing unit)) memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 3. Events: GPU-side timestamps, created once and reused for every measurement below
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 4. Copy inputs host → device (H2D), timed. A and B = 2 arrays over PCIe.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));   // (destination, source, bytes, direction)
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    // 5. Warm-up: run 3 times, don't time these (clocks ramp up, first-launch setup)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int w = 0; w < 3; w++) {
        vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());   // wait until all 3 have finished

    // 6. Kernel timing: 20 runs
    const int RUNS = 20;
    float times[RUNS];         // one kernel time per run, in ms (milliseconds)

    for (int r = 0; r < RUNS; r++) {
        CUDA_CHECK(cudaEventRecord(start));                        // (a)
        vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);         // (b)
        CUDA_CHECK(cudaEventRecord(stop));                         // (c)
        CUDA_CHECK(cudaEventSynchronize(stop));                    // (d)
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));  // (e)
    }
    CUDA_CHECK(cudaGetLastError());

    // 7. Copy the result device → host (D2H), timed. C = 1 array over PCIe.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // 8. Check against the CPU
    int mismatches = 0;

    for(int i = 0; i< N; i++) {
        if( h_C[i] != h_A[i] + h_B[i] ) {
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
    //   3.0 * bytes      = the kernel reads A, reads B and writes C: 3 arrays = 805 MB
    //                      (same as 12 bytes per element * N). 3.0, not 3, keeps the maths in double.
    //   median_ms / 1000 = milliseconds -> seconds
    //   / 1e9            = bytes per second -> gigabytes per second (1e9 = 10^9; ^ in C++ is XOR, not power)
    // Example: 805,306,368 bytes / 0.0016 s / 1e9 = 503 GB/s
    float achieved_bandwidth = ((3.0 * bytes) / (median_ms / 1000)) / 1e9 ;

    printf("Achieved bandwidth in GB/s: %.3f \n", achieved_bandwidth);

    // Percent of peak = achieved / A10 spec bandwidth (600 GB/s) * 100.
    // Example: 503 / 600 * 100 = 84%. The spec is a theoretical maximum; no kernel reaches 100%.
    double percent_of_peak = achieved_bandwidth / 600.0 * 100.0;

    printf("fraction of peak : %.3f \n", percent_of_peak);

    // Copies over PCIe (Peripheral Component Interconnect Express), same bandwidth formula:
    //   H2D moved A + B = 2.0 * bytes, D2H moved C = 1.0 * bytes.
    //   Prediction: ~15 GB/s, ~54 ms total, ~34x the kernel time.
    printf("H2D: %.2f ms, %.1f GB/s \n", h2d_ms, (2.0 * bytes) / (h2d_ms / 1000) / 1e9);
    printf("D2H: %.2f ms, %.1f GB/s \n", d2h_ms, (1.0 * bytes) / (d2h_ms / 1000) / 1e9);
    printf("copies / kernel: %.1fx \n", (h2d_ms + d2h_ms) / median_ms);

    // 10. Free both sides
    // Whatever allocated it frees it: cudaMallocHost → cudaFreeHost, malloc → free
    if (pinned) {
        CUDA_CHECK(cudaFreeHost(h_A));
        CUDA_CHECK(cudaFreeHost(h_B));
        CUDA_CHECK(cudaFreeHost(h_C));
    } else {
        free(h_A);
        free(h_B);
        free(h_C);
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
