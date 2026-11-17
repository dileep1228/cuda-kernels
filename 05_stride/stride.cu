#include <cstddef>
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

// Vector copy: B = A. No maths, one read and one write per element:
// the practical speed limit for any kernel that streams through memory once.
//
// Grid-stride loop: each thread starts at its global index and jumps by the
// total number of threads in the grid, so any launch size covers all N elements.
__global__ void strided_copy(const float* __restrict__ A, float* __restrict__ B, int N, int s) {
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = index; i<N; i += stride) {
        B[i] = A[i * s];
    }
}

int main(int argc, char** argv) {
    const size_t N = (1 << 24);    // 2^26 = 67,108,864 elements
    const size_t in_bytes = N * 32 * sizeof(float);   // input: room for the largest stride (32)
    const size_t bytes = N * sizeof(float);       // size of ONE array in bytes
    printf("N = %zu elements\n", N);              // every run records its size

    // Host memory kind, chosen on the command line:
    //   ./strided_copy         pageable (malloc): copies go through the driver's staging buffer
    //   ./strided_copy pinned  pinned (cudaMallocHost): the copy engine reads/writes our arrays directly
    const bool pinned = argc > 1 && strcmp(argv[1], "pinned") == 0;
    printf("host memory: %s \n", pinned ? "pinned (cudaMallocHost)" : "pageable (malloc)");

    // 1. Host (CPU (central processing unit)) memory: A is the input (random floats),
    //    B receives the copy back from the GPU
    float *h_A, *h_B;
    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_A, in_bytes));
        CUDA_CHECK(cudaMallocHost(&h_B, bytes));
    } else {
        h_A = (float*)malloc(in_bytes);
        h_B = (float*)malloc(bytes);
    }
    // Touch every page of h_B before the D2H copy, as vector_add did for h_C,
    // so the copy timings of the two kernels are measured the same way.
    memset(h_B, 0, bytes);

    for(size_t i = 0; i< N*32; i++) {
        h_A[i] = dist(gen);
    }

    // 2. Device (GPU (graphics processing unit)) memory
    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));

    // 3. Events: GPU-side timestamps, created once and reused for every measurement below
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 4. Copy the input host → device (H2D), timed. Only A: 1 array over PCIe.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, in_bytes, cudaMemcpyHostToDevice));   // (destination, source, bytes, direction)
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

     int stride[6] = {1, 2, 4, 8, 16, 32};

    for(int i=0; i < 6 ; i++) {

        // 5. Warm-up: run 3 times, don't time these (clocks ramp up, first-launch setup)
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        for (int w = 0; w < 3; w++) {
            strided_copy<<<blocks, threads>>>(d_A, d_B, N, stride[i]);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());   // wait until all 3 have finished

        // 6. Kernel timing: 20 runs
        const int RUNS = 20;
        float times[RUNS];         // one kernel time per run, in ms (milliseconds)
    

        for (int r = 0; r < RUNS; r++) {
            CUDA_CHECK(cudaEventRecord(start));                        // (a)
            strided_copy<<<blocks, threads>>>(d_A, d_B, N, stride[i]);             // (b)
            CUDA_CHECK(cudaEventRecord(stop));                         // (c)
            CUDA_CHECK(cudaEventSynchronize(stop));                    // (d)
            CUDA_CHECK(cudaEventElapsedTime(&times[r], start, stop));  // (e)
        }
        CUDA_CHECK(cudaGetLastError());

        // 7. Copy the result device → host (D2H), timed. B = 1 array over PCIe.
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float d2h_ms;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

        // 8. Check: every element of B must equal A exactly (a copy changes no bits)
        int mismatches = 0;

        for(int ii = 0; ii < N; ii++) {
            if( h_B[ii] != h_A[ii * stride[i]] ) {
                mismatches++;
            }
        }

        printf("total number of mismatches: %d \n", mismatches);

        std::sort(times, times + RUNS);
        float median_ms = (times[RUNS/2 - 1] + times[RUNS/2]) / 2.0f;

        float bandwidth = ((8 * N) / median_ms) / 1e6;

        printf("stride: %d, Fastest: %.3f ms, median: %.3f ms and slowest: %.3f ms, bandwidth: %.3f GB/s \n",stride[i], times[0], median_ms, times[RUNS-1], bandwidth);

    }

    // 10. Free both sides
    // Whatever allocated it frees it: cudaMallocHost → cudaFreeHost, malloc → free
    if (pinned) {
        CUDA_CHECK(cudaFreeHost(h_A));
        CUDA_CHECK(cudaFreeHost(h_B));
    } else {
        free(h_A);
        free(h_B);
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
