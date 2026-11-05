#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <cmath> // for std::fma(a, x[i], y[i])

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

// SAXPY: y = a*x + y (Single-precision A*X Plus Y). One multiply and one add per
// element, done as a single FMA (fused multiply-add) instruction. 12 bytes moved
// for 2 FLOPs (floating-point operations): still memory-bound.
//
// Grid-stride loop: each thread starts at its global index and jumps by the
// total number of threads in the grid, so any launch size covers all N elements.
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y, 
                                    const float a, int N) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = index; i<N; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    const int N = (1 << 26);    // 2^26 = 67,108,864 elements
    const size_t bytes = N * sizeof(float);       // size of ONE array in bytes
    printf("N = %d elements\n", N);              // every run records its size

    // Host memory kind, chosen on the command line:
    //   ./saxpy               pageable (malloc): copies go through the driver's staging buffer
    //   ./saxpy pinned        pinned (cudaMallocHost): the copy engine reads/writes our arrays directly
    const bool pinned = argc > 1 && strcmp(argv[1], "pinned") == 0;
    printf("host memory: %s \n", pinned ? "pinned (cudaMallocHost)" : "pageable (malloc)");

    // 1. Host (CPU (central processing unit)) memory: x and y are inputs (random floats);
    //    y also receives the result back from the GPU. h_y_copy keeps the original y,
    //    for resetting the GPU's y and for the CPU reference.
    float *h_x, *h_y, *h_y_copy;
    const float a = 2.5f;
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_x, bytes));
        CUDA_CHECK(cudaMallocHost(&h_y, bytes));
    } else {
        h_x = (float*)malloc(bytes);
        h_y = (float*)malloc(bytes);
    }
    h_y_copy = (float*)malloc(bytes);

    for(int i = 0; i< N; i++) {
        h_x[i] = dist(gen);
        h_y[i] = dist(gen);
        h_y_copy[i] = h_y[i];
    }

    // 2. Device (GPU (graphics processing unit)) memory
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // 3. Events: GPU-side timestamps, created once and reused for every measurement below
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 4. Copy the inputs host → device (H2D), timed. x and y: 2 arrays over PCIe.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));   // (destination, source, bytes, direction)
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    // 5. Warm-up: run 3 times, don't time these (clocks ramp up, first-launch setup)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int w = 0; w < 3; w++) {
        saxpy<<<blocks, threads>>>(d_x, d_y, a, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());   // wait until all 3 have finished

    // 6. Kernel timing: 20 runs
    const int RUNS = 20;
    float times[RUNS];         // one kernel time per run, in ms (milliseconds)

    for (int r = 0; r < RUNS; r++) {
        CUDA_CHECK(cudaEventRecord(start));                        // (a)
        saxpy<<<blocks, threads>>>(d_x, d_y, a, N);             // (b)
        CUDA_CHECK(cudaEventRecord(stop));                         // (c)
        CUDA_CHECK(cudaEventSynchronize(stop));                    // (d)
        CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));  // (e)
    }
    CUDA_CHECK(cudaGetLastError());

    // Reset y to its original values and run once, so the check sees exactly one saxpy.
    // (The 23 launches above each updated y again: y = a*x + y, 23 times over.)
    CUDA_CHECK(cudaMemcpy(d_y, h_y_copy, bytes, cudaMemcpyHostToDevice));
    saxpy<<<blocks, threads>>>(d_x, d_y, a, N);
    CUDA_CHECK(cudaGetLastError());

    // 7. Copy the result device → host (D2H), timed. Only y: 1 array over PCIe.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    // 8. Check against the CPU, exactly. The GPU computes a*x + y as one FMA, which
    //    rounds once; plain a*x[i] + y[i] on the CPU rounds twice (after the multiply
    //    and after the add). Measured: 12,705,635 of 67,108,864 elements (19%) then
    //    differ by one last bit. std::fma rounds once, like the GPU: 0 mismatches.
    int mismatches = 0;

    for(int i = 0; i< N; i++) {
        float expected = std::fma(a, h_x[i], h_y_copy[i]);
        if( h_y[i] != expected ) {
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
    //   3.0 * bytes      = the kernel reads x, reads y and writes y: 3 arrays = 805 MB
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
    //   H2D moved x and y = 2.0 * bytes, D2H moved y = 1.0 * bytes.
    printf("H2D: %.2f ms, %.1f GB/s \n", h2d_ms, (2.0 * bytes) / (h2d_ms / 1000) / 1e9);
    printf("D2H: %.2f ms, %.1f GB/s \n", d2h_ms, (1.0 * bytes) / (d2h_ms / 1000) / 1e9);
    printf("copies / kernel: %.1fx \n", (h2d_ms + d2h_ms) / median_ms);

    // 10. Free both sides
    // Whatever allocated it frees it: cudaMallocHost → cudaFreeHost, malloc → free
    if (pinned) {
        CUDA_CHECK(cudaFreeHost(h_x));
        CUDA_CHECK(cudaFreeHost(h_y));
    } else {
        free(h_x);
        free(h_y);
    }
    free(h_y_copy);   // always malloc, in both modes
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
