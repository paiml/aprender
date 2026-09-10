# 67-B1 receipt — `cargo test -p aprender-gpu --features cuda --lib --release` on yoga

- date: 2026-09-10T13:41Z · commit `bd05c8161` (branch PMAT-1095-cuda-rust-fleet, PR #3068)
- host: yoga (x86_64) · GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU · driver 595.91.07
- container: `sovereign-gpu-runner:2.337.0` (the disposable GitHub-runner image, infra#500) run with `--runtime nvidia --gpus all`, the host toolkit mounted read-only at `/usr/local/cuda` (`cuda-13.3`) and `LD_LIBRARY_PATH=/usr/local/cuda/lib64`; clone `--depth 1`, `CARGO_INCREMENTAL=0`
- result: `test result: ok. 2601 passed; 0 failed; 12 ignored; 0 measured; 0 filtered out; finished in 8.56s` · `rc=0`
- ignored (0): -
- prior run at `bb0b32e83` (same image, same mounts): 2 FAILED — test_context_memory_info / test_context_total_memory asserted ">20GB VRAM" on an 8 GB RTX 4060 Laptop; fixed by `bd05c8161` (GPU-ORD-5 non-blocking capture streams; fleet-neutral VRAM floor)
- log: `~/eph-work/b1/aprender-gpu-cuda-lib.log` on yoga (not committed; 10 summary lines captured here)
