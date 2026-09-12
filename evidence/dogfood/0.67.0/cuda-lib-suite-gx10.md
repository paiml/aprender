# 67-B1 receipt — `cargo test -p aprender-gpu --features cuda --lib --release` on gx10

- date: 2026-09-10T13:41Z · commit `bd05c8161` (branch PMAT-1095-cuda-rust-fleet, PR #3068)
- host: gx10 (aarch64) · GPU 0: NVIDIA GB10 · driver 590.48.01
- container: `sovereign-gpu-runner:2.337.0` (the disposable GitHub-runner image, infra#500) run with `--runtime nvidia --gpus all`, the host toolkit mounted read-only at `/usr/local/cuda` (`cuda-13.3`) and `LD_LIBRARY_PATH=/usr/local/cuda/lib64`; clone `--depth 1`, `CARGO_INCREMENTAL=0`
- result: `test result: ok. 2601 passed; 0 failed; 12 ignored; 0 measured; 0 filtered out; finished in 22.01s` · `rc=0`
- ignored (0): -
- prior run at `bb0b32e83` (same image, same mounts): 6 FAILED — 5× CUDA_ERROR_STREAM_CAPTURE_IMPLICIT (906) in cublas_tests + CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901) in test_cuda_graph_with_kernel; fixed by `bd05c8161` (GPU-ORD-5 non-blocking capture streams; fleet-neutral VRAM floor)
- log: `~/eph-work/b1/aprender-gpu-cuda-lib.log` on gx10 (not committed; 10 summary lines captured here)
