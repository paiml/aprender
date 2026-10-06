# E4 census: refusals and CPU fallbacks on a WGPU or Metal run (landing-map row 9)

Read at origin/main 11f844a772 by `git show` and `git grep`. No build, no GPU.
Paths are under `crates/`. Kind: R = refused, F = falls back to the CPU,
H = runs on wgpu with part of the step on the host. S = silent: the whole run is on the CPU while it reports wgpu.

E4 says the refusals are removed. This sheet lists them so that E4 can be checked:
each TARGET row names the bundle that strikes it and the falsifier that proves it is
gone. GUARD rows stay. They make a fallback loud, which is what E4 asks of the
fallbacks that remain, so their falsifier proves they still fire.

## Metal
There is no native Metal backend. `BackendKind::Metal` is always unavailable
(`aprender-compute/src/registry/mod.rs:568-573` @11f844a772: "no native Metal backend
in 0.66 (a Metal adapter appears under wgpu)"), and a forced `--backend metal` is
refused (row G1). Apple GPUs are reached only through `--backend wgpu` on the Metal
transport, so every wgpu row below also holds on C5b (mini-metal). E1's "Metal" leg is
therefore the wgpu leg on Apple hardware. Whether E1 means that or a native backend is
an open question (RQ-10 below).

## Targets: E4 is met when each is gone
| id | kind | trigger | where @11f844a772 | strikes it | falsifier: gone when |
|---|---|---|---|---|---|
| T1 | R→F | a Q8_0 or Q4_0 tensor on a wgpu load | `aprender-serve/src/gpu/adapters/wgpu_adapter.rs:338-340` ("Unsupported quantization type {} for WGPU dequant") | P4 item 4 | a forced wgpu run of a Q8_0 and a Q4_0 GGUF exits 0 with used_gpu=true |
| T2 | F | a legacy-quant model never tries wgpu | `aprender-serve/src/infer/gguf_gpu_generate.rs:493`, `:535`; `aprender-serve/src/infer/inference_result.rs:703-716` | P4 item 4 | the same runs print no "Q4_0 format - GPU Q4_K kernels incompatible" line |
| T3 | H | every wgpu load dequantizes to F32 on the host, then uploads | `wgpu_adapter.rs:301-312` (Q5_K at :312) | P4 (WGSL GEMV), row 19 for Q5_K | the trace names a quantized WGSL kernel per matmul; no host dequant span |
| T4 | F | the wgpu parity probe or init fails, and the run goes on on the CPU | `gguf_gpu_generate.rs:270-307`, `:767-806`; the CPU line (`:543`, `:606`, `:611`) is `--verbose` only | P4 (cosine 0.955 → ≥ 0.995) | on C1, the probe passes on every E1 model; on a planted bad kernel a forced run exits 14, never 0 |
| T5 | H | each decode step: embed, output norm, LM head, argmax on the host | `gguf_gpu_generate.rs:334`, `:343-358` | row 12 | until then RQ-4 holds: E2 and E6 report the cell HYBRID |
| T6 | H | `apr serve --backend wgpu`: host LM head and argmax | `apr-cli/src/commands/serve/handlers.rs:254`, `:317`, `:775-785` | row 12 | as T5, for the serve route |
| T7 | H/F | batch inference: host LM head; the CPU when wgpu init fails | `aprender-serve/src/infer/batch_wgpu.rs:24`, `:95`; `aprender-serve/src/infer/batch.rs:349-356` | row 12 | as T5; the init failure is loud (G2) |
| T8 | F | a sampled request (temperature > 0, top-k != 1) | `gguf_gpu_generate.rs:62-74` (notice), call sites `:536`, `:600` | row 17 (#3760) | a sampled forced wgpu run prints no WGPU_SAMPLING_NOTICE and has used_gpu=true |
| T9 | F, R if forced | a MoE model: run dispatch is CUDA or CPU | `aprender-serve/src/infer/inference_result.rs:356-364`; `aprender-serve/src/infer/qwen3_moe_dispatch.rs:36` | P5, its wgpu half after P4 | a forced wgpu run of the E3 MoE model exits 0 with used_gpu=true |
| T10 | R | the wgpu MoE forward is a stub | `aprender-serve/src/gguf/wgpu_backend/mod.rs:197-206` | P5 | as T9; the stub's UnsupportedOperation is unreachable |
| T11 | R | the MoE guard in `run_gguf_generate` | `gguf_gpu_generate.rs:474-490` | P5 | as T9 |
| T12 | F, R if forced | qwen35: the GPU forward is CUDA only | `aprender-serve/src/infer/inference_result.rs:365-385`; `apr-cli/src/commands/chat_load_tokenizers.rs:201` | row 6 (RQ-8: stays at 6) | a forced wgpu run of Qwen3.5-4B exits 0 with used_gpu=true |
| T13 | S | no working GPU driver: `GpuDevice::new()` takes a software Vulkan adapter (`DeviceType::Cpu`) that the registry probe refuses | `aprender-compute/src/backends/gpu/device/mod.rs:125-129`; probe `registry/wgpu_probe.rs:58-70` | row 13 (`R13-intel-amd-wgpu.md`, R13-1) | a lavapipe-only run exits non-zero naming the adapter |

## Guards: they stay, and their falsifier proves they fire
| id | what | where @11f844a772 | falsifier |
|---|---|---|---|
| G1 | a forced backend that is unavailable or not compiled is refused, never downgraded | `apr-cli/src/registry.rs:94-115`; Metal at `:77`, `:266-280` | a forced `--backend metal` exits non-zero |
| G2 | a forced accelerator whose generation ran on the CPU is refused, exit 14 | `apr-cli/src/registry.rs:418-431`; `apr-cli/src/commands/run_entry.rs:331-337`; `apr-cli/src/commands/chat_generate_session.rs:121-126`; `apr-cli/src/error.rs:181` | T4's planted bad kernel exits 14 |
| G3 | `--backend cuda` without CUDA is refused rather than run on wgpu or the CPU | `apr-cli/src/dispatch.rs:198-210` | exits non-zero on a no-cuda build |
| G4 | a default (not forced) selection that fell back says so | `apr-cli/src/registry.rs:433-436` ("selected: cpu (fallback: ...") | the line is printed without `--verbose` |
| G5 | a bare `apr run` does not try wgpu | `gguf_gpu_generate.rs:93-99`; `apr-cli/src/commands/inference_output.rs:344-347` | by design: E1 legs force the backend |

## New findings (not on the landing map)
- **N1, silent, E4 defect.** `apr serve --backend wgpu` on a build without the
  `wgpu` feature returns `Ok(false)` and goes on to the normal path
  (`apr-cli/src/commands/serve/handlers.rs:857-868`: "WGPU feature not enabled. Build
  with --features wgpu"). That is a forced backend downgraded, the case G1 and G3
  refuse for run and cuda. Candidate side fix: refuse it like G3. Falsifier: a forced
  `apr serve --backend wgpu` on a no-wgpu build exits non-zero. Not cargo-checked.
- **N2 [U, inferred].** `apr serve --backend wgpu` has no architecture check before it
  dequantizes (`handlers.rs:905-937`), so a qwen35 or MoE model may fail inside
  dequant or forward rather than with a clean refusal. Needs a run to confirm.

## Open
- **RQ-10:** does E1's "Metal" leg mean wgpu on the Metal transport (today's only
  route), or a native Metal backend? Default: wgpu on Metal, which needs no new row.
  A native backend would be a new bundle and is not in 0.73 scope.
  A (quorum, 2026-10-05, degraded: same-family, agy 503): (a) wgpu on Metal, 2-0
  (sonnet, haiku; planted false claim caught by both). Receipt: cop handoff
  `quorum-0.73-rq10.md`.
- Not read: the parity and bench code paths beyond the cites above.
