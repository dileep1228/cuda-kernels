# Week 0 environment check: does the full toolchain work on a Modal A10?
#
#   modal run env_check.py
#
# Answers one question before any real work: can ncu read the hardware
# performance counters here? If it prints ERR_NVGPUCTRPERM, this provider is
# out and we move to Lambda. The kernel is a throwaway smoke test, not the 3A
# vector_add - that one gets written from scratch.

import subprocess

import modal

GPU = "A10"

image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
).apt_install("cuda-nsight-systems-12-8")
app = modal.App("phase3-env-check", image=image)

KERNEL = r"""
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void smoke_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : (1 << 26);
    size_t bytes = (size_t)n * sizeof(float);
    float *a, *b, *c;
    CHECK(cudaMalloc(&a, bytes)); CHECK(cudaMalloc(&b, bytes)); CHECK(cudaMalloc(&c, bytes));
    CHECK(cudaMemset(a, 0, bytes)); CHECK(cudaMemset(b, 0, bytes));

    int block = 256, grid = (n + block - 1) / block;
    for (int i = 0; i < 3; i++) smoke_add<<<grid, block>>>(a, b, c, n);   // warm-up
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    const int reps = 20;
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < reps; i++) smoke_add<<<grid, block>>>(a, b, c, n);
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); ms /= reps;

    printf("n=%d  %.3f ms  %.1f GB/s effective\n", n, ms, 3.0 * bytes / ms / 1e6);
    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(c));
    return 0;
}
"""


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip(), flush=True)


@app.function(gpu=GPU, timeout=600)
def check():
    sh("nvidia-smi")
    sh("nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,"
       "pcie.link.gen.max,pcie.link.width.max,ecc.mode.current --format=csv")
    sh("cat /proc/driver/nvidia/params 2>/dev/null | grep -i RestrictProfiling || echo 'params not readable'")
    sh("id -u")

    sh("nvcc --version | tail -2")
    sh("which ncu nsys compute-sanitizer; ls /usr/local/cuda/bin | grep -Ei 'ncu|nsys|sanitizer'")

    with open("/root/smoke.cu", "w") as f:
        f.write(KERNEL)
    sh("cd /root && nvcc -O3 -arch=sm_86 -Xptxas -v smoke.cu -o smoke")
    sh("/root/smoke")

    # The decisive one. Modal refuses ncu's default clock locking, so profile
    # at the GPU's own clocks; numbers will vary a little more run to run.
    sh("ncu --clock-control none --metrics gpu__time_duration.sum,"
       "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
       "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
       "launch__registers_per_thread,"
       "sm__warps_active.avg.pct_of_peak_sustained_active "
       "--launch-count 1 /root/smoke")

    sh("compute-sanitizer --tool memcheck /root/smoke 1048576 | tail -3")
    sh("nsys --version")
    sh("cd /root && nsys profile --trace=cuda,nvtx -o smoke_trace --force-overwrite true ./smoke 2>&1 "
       "| grep -v '^\\[1/1\\]' && nsys stats --report cuda_gpu_kern_sum smoke_trace.nsys-rep 2>&1 | tail -8")


@app.local_entrypoint()
def main():
    check.remote()
