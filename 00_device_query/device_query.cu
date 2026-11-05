// Ask the GPU about itself: the per-SM limits that decide how many blocks fit,
// and the memory clock and bus width that set peak bandwidth.
// No kernel runs here; the driver fills in the struct on the CPU side.
#include <cstdio>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

int main() {
    // 1. Ask the GPU for its properties (device 0 = the first GPU)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // 2. Print each field
    printf("Name: %s\n", prop.name);              // %s for text
    printf("SMs: %d\n", prop.multiProcessorCount); // %d for whole numbers
    printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("maxBlocksPerMultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("sharedMemPerMultiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("l2CacheSize: %d\n", prop.l2CacheSize);
    printf("warpSize: %d\n", prop.warpSize);
    printf("memoryBusWidth: %d\n", prop.memoryBusWidth);

    // 3. Memory clock, asked for separately (in kHz (kilohertz))
    int clk_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, 0));
    printf("clk_khz: %d\n", clk_khz);

    // 4. Peak memory bandwidth in GB/s (gigabytes per second):
    //    clock (kHz → Hz: × 1000.0) × 2 transfers per clock × bus width (bits ÷ 8 = bytes) ÷ 1e9
    //    1000.0, not 1000: keeps the maths in double, so clk_khz * 2 * 1000 can't overflow an int

    double bandwidth = clk_khz * 2 * 1000.0 * prop.memoryBusWidth / 8 / 1e9;

    printf("bandwidth: %.1f GB/s\n", bandwidth);

    return 0;
}