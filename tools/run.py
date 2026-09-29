# Compile and run a single .cu file on the A10, from the Mac.
#
#   modal run tools/run.py --file path/to/kernel.cu
#   modal run tools/run.py --file path/to/kernel.cu --mode ncu
#   modal run tools/run.py --file path/to/kernel.cu --mode sanitize
#   modal run tools/run.py --file path/to/kernel.cu --mode nsys
#   modal run tools/run.py --file path/to/kernel.cu --args "1048576"
#   modal run tools/run.py --file path/to/kernel.cu --mode ncu --skip 3 --count 1
#
# Modes:
#   run       compile and run
#   ncu       Nsight Compute: why is this kernel slow? Saves profiles/<name>.ncu-rep
#             Profiles --count launches (default 1) after skipping --skip (default 0),
#             e.g. --skip 3 to step over 3 warm-up launches.
#   nsys      Nsight Systems: where did the time go? Saves profiles/<name>.nsys-rep
#   sanitize  compute-sanitizer memcheck: out-of-bounds and misaligned accesses
#
# Reports are saved next to the source file, in a profiles/ directory.

import subprocess
from pathlib import Path

import modal

image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
).apt_install("cuda-nsight-systems-12-8")
app = modal.App("cuda-kernels-run", image=image)

WORK = Path("/root/work")


def sh(cmd: str) -> int:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=WORK).returncode


@app.function(gpu="A10", timeout=900)
def remote(name: str, source: str, mode: str, args: str, skip: int, count: int) -> bytes | None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"{name}.cu").write_text(source)

    # Which physical GPU did this run land on? Modal may hand out a different card
    # each run, and two "A10"s can differ by ~10%, so every result records its GPU.
    sh("nvidia-smi --query-gpu=name,uuid,clocks.max.sm,clocks.max.mem,temperature.gpu "
       "--format=csv,noheader")

    # -lineinfo lets ncu map metrics back to source lines; it does not slow the code.
    if sh(f"nvcc -O3 -arch=sm_86 -lineinfo -Xptxas -v {name}.cu -o {name}") != 0:
        return None

    if mode == "run":
        sh(f"./{name} {args}")
        return None
    if mode == "sanitize":
        sh(f"compute-sanitizer --tool memcheck ./{name} {args}")
        return None
    if mode == "ncu":
        # Modal does not allow ncu to lock GPU clocks.
        sh(f"ncu --clock-control none --set full --launch-skip {skip} --launch-count {count} "
           f"-o {name} -f ./{name} {args}")
        sh(f"ncu --import {name}.ncu-rep --page details --section SpeedOfLight")
        report = WORK / f"{name}.ncu-rep"
    elif mode == "nsys":
        # The default trace set records no kernels here; ask for CUDA explicitly.
        sh(f"nsys profile --trace=cuda,nvtx -o {name} -f true ./{name} {args}")
        sh(f"nsys stats --report cuda_gpu_kern_sum,cuda_api_sum {name}.nsys-rep")
        report = WORK / f"{name}.nsys-rep"
    else:
        raise ValueError(f"unknown mode {mode!r}: use run, ncu, nsys or sanitize")

    return report.read_bytes() if report.exists() else None


@app.local_entrypoint()
def main(file: str, mode: str = "run", args: str = "", skip: int = 0, count: int = 1):
    path = Path(file)
    report = remote.remote(path.stem, path.read_text(), mode, args, skip, count)
    if report:
        out = path.parent / "profiles" / (f"{path.stem}.ncu-rep" if mode == "ncu" else f"{path.stem}.nsys-rep")
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(report)
        print(f"\nsaved {out}  (open it in Nsight {'Compute' if mode == 'ncu' else 'Systems'} on the Mac)")
