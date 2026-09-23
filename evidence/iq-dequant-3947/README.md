# #3947: aprender-core IQ dequant, oracle evidence

Host: lambda. Models: `~/models/`. sha256 prefixes: IQ3_M `0b2e17471ccc784d`,
IQ4_XS `df178ccd68e24ce0`, UD-Q4_K_XL `b252c5610a42ca82`.
Oracle: **gguf-py 0.17.1** `gguf.quants.dequantize` (numpy 2.3.5).

| file | what it is |
|---|---|
| `census.py`, `census.txt` | per-tensor type census (gguf-py `GGUFReader`) |
| `compare.py`, `compare.txt`, `compare_report.json`, `harness-serve/` | **before the port**: serve's decoders vs gguf-py, every IQ tensor (harness sha256 `33afc4ad00957378`) |
| `compare_core.py`, `compare_core.txt`, `compare_core_report.json`, `harness-core/` | **the port**: aprender-core `GgufReader::get_tensor_f32` @ 18f10a78e vs gguf-py, every IQ tensor, shapes included (harness sha256 `8fc0195c7a79114e`) |
| `qa/before-*.json` / `.rc` | `apr qa` @ 8963f91a3 (sha256 `53c3e843baa966d7`) |
| `qa/after-*.json` / `.rc` | `apr qa` @ 18f10a78e (sha256 `f9afa3419bef75db`) |
| `qa/gate-control.json` | gate positive control: a copy of the IQ3_M file with an f16 NaN block scale planted in `blk.0.ffn_gate` (IQ4_NL) and `blk.11.ffn_down` (IQ3_S) |

Both `qa` runs are CPU-only (`CUDA_VISIBLE_DEVICES=`), with the same flags:
`--json --skip-throughput --skip-ollama --skip-gpu-speedup --skip-ptx-parity --skip-gpu-state --skip-capability --skip-format-parity`.
The gate control also passes `--skip-golden --skip-metadata`.

Positive controls: every comparison first flips bit 0 of the first quant byte of block 0 and requires exactly one mismatch (RED). Only after that does a green count. The harness-core control flips the bit inside a *copy of the GGUF file*, so the flip travels through core's real offset and dispatch path.

Scope: this covers the CPU dequant (core's `get_tensor_f32`; serve's `iq_dispatch` uses the same block decoders). The CUDA IQ4 GEMV kernels (`iq4_xs_gemv_warp_reduce`, `iq4_nl_gemv_warp_reduce`) are separate code and are **not** measured here.

Reproduce: build each harness (`cargo build --release` in a copy of `harness-*/`, with the `path =` in its Cargo.toml pointed at your checkout), copy the binary next to the script as `iqdec.bin` / `coredec.bin`, then run `python3 compare*.py`. Each script exits non-zero if any control fails to go RED or any cell is RED.

Note: `qa/after-Qwen2.5-0.5B-*.rc` were transcribed from that run's terminal line (`rc=0`), not written by the run itself. Their JSON shows zero failing gates, which agrees. Every other `.rc` file was written by the run.
