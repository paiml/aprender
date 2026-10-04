# R4 — MoE on GPU backends: wiring scope (0.73, L3 draft, 2026-09-27)

All citations are at origin/main aca6f2d7f6, read with `git show origin/main:<path>`. [V] means the line was read; [A] means it was inferred.

## H4 status: mostly overtaken
The roadmap row reads "the CUDA forward exists; the gap is wiring." On main, most of that wiring has already landed:
- [V] `run_qwen3_moe_generate_dispatch` (`infer/qwen3_moe_dispatch.rs:24`, #3714) is one entry point for all these callers. It tries CUDA first unless `--no-gpu` is set or the build lacks `cuda`, and checks the result against the CPU forward on the real prompt. If the GPU cannot serve, it prints `QWEN3MOE_GPU_FALLBACK_PREFIX` unconditionally and runs the CPU chain.
- [V] Callers: `apr run` (`infer/inference_result.rs:357`), `apr chat` (`apr-cli chat_generate_session_02.rs:330`), `serve` non-streaming chat (`api/cuda_chat_backend.rs:1266`, #3987), and completions (`api/realize_handlers_embed_completion.rs:908`, in `try_quantized_completions`).
- A first Explore pass read the local checkout, which is on a stale branch. That pass reported "`apr run` always runs MoE on the CPU", which is false on main. See K16.

## Remaining items
1. **Streaming serve is CPU-only.** [V at 316dee2cd4] `cuda_chat_backend.rs:1176` (in `moe_stream_cpu`, :1155) calls `run_qwen3_moe_generate_streaming` directly and never goes through the dispatch. On a CUDA server, a `stream: true` request runs on the CPU while a non-streaming request runs on the GPU. The fix is a streaming variant of the dispatch that puts the same fallback line on the stream path.
   - Falsifier FALSIFY-R4-001: on C0, the same prompt with `stream: true` and with `stream: false` must report the same `used_gpu`.
2. **The wgpu MoE forward is a stub.** [V] `gguf/wgpu_backend/mod.rs:197` always returns `UnsupportedOperation` (M-GPU-MOE-2.0). The planned stages, 2.1 per-expert dispatch, 2.2 full forward and 2.3 CPU parity, are in its doc comment at `:107-126` ([A], line numbers from the stale tree). C1–C3 E3 depends on this, and R2 must land first because the MoE forward reuses the dense attention.
   - Falsifier FALSIFY-R4-002: a wgpu MoE run with no fallback line must pass leg A ≥ 0.995 (BPM gate).
3. **Receipt proof of the forward used.** [V] Today the only evidence is `used_gpu` plus the fallback prefix line. A receipt can already assert "`used_gpu = true` AND no line starts with `QWEN3MOE_GPU_FALLBACK_PREFIX`", which meets verification rule 2 for CUDA. [A] The trace JSON has no forward name, which only matters once wgpu and CUDA both serve MoE.
   - Falsifier FALSIFY-R4-003: a planted run with `--no-gpu` must be refused as a C0 E3 GPU receipt.
   - [V] The dispatch also prints an unconditional banner, `Backend: GPU (CUDA, <device>, <MB> VRAM) [qwen3moe routed-expert forward, #3714 ...]` (`qwen3_moe_dispatch.rs`). That banner is the self-reported backend line BPM `backend_is_self_reported` needs; a receipt should require it.
4. **Decode cost — settled [V]** (`infer/qwen3_moe_dispatch.rs:100-190` @aca6f2d7f6). The CUDA decode is KV-cached: `new_state()` then `forward_single(token, &mut state, pos)` per token. But **prefill is token by token**: the prompt loop calls `forward_single` once per prompt position, and no batched prefill exists. See K18.

## Exit
E3 PASS on C0 and C5 with the streaming path included (item 1). E3 on C1–C3 after item 2. C4 is CPU only.

## Risks
- **K16 (new, M):** research agents read the working tree, and this checkout is on a stale branch. Every citation must come from `git show origin/main:`, or from a worktree at origin/main, such as `la-73/wt-r3`.
- **K17 (L):** the dispatch's CPU-parity pre-check doubles first-token latency. E2 prefill timing must say whether the check was included.
- **K18 (M, E2):** MoE prefill on CUDA runs one `forward_single` per prompt token, while llama.cpp prefills in a batch. The E2 prefill ratio for MoE will probably miss 0.5 on long prompts [A, unmeasured]. Report decode and prefill separately, and treat batched MoE prefill as a candidate R4 item 5.
- **K19 (H, E1):** leg A's CPU reference is ambiguous. The dispatch's own comment (`cpu_reference`, #3714) cites a measurement on Qwen3-Coder-30B-A3B: CUDA vs FP32-activation CPU gave cosine 1.000000 at all 65 positions, and CUDA vs the **production Q8_K** CPU path gave **0.985**. Leg A in BPM is `cos(apr_backend, apr_cpu)`. If `apr_cpu` means the production path, CUDA MoE fails E1 by construction, at 0.985 < 0.995, and the failure measures the CPU's quantization rather than the GPU. BPM must pin the leg-A reference, and leg B must use the same CPU path so the composed bound still holds. The figures are quoted from a code comment and not re-measured (R-1).

## L25 review (2026-10-03, la-73, at origin/main 316dee2cd4)
Each check below could pass without measuring anything. [V] means the line was read at 316dee2cd4.

| # | Check | Vacuous pass | Fix |
|---|---|---|---|
| 1 | FALSIFY-R4-001 (stream and non-stream report the same `used_gpu`) | Both are `false` on a build without `cuda`, with `--no-gpu`, or once both fall back; `false = false` passes. If the streaming path does not report `used_gpu` at all, absent = absent passes too. | Require `used_gpu = true` on BOTH runs, the CUDA `Backend: GPU (CUDA, …)` banner on both, and the build's `cuda` feature in the receipt. Mutation: route streaming to the CPU; the test must turn RED. |
| 2 | FALSIFY-R4-002 ("no fallback line" on wgpu MoE) | `QWEN3MOE_GPU_FALLBACK_PREFIX` is printed only inside `#[cfg(feature = "cuda")]` (`qwen3_moe_dispatch.rs:31-36` [V]). A wgpu run never prints it, so "no fallback line" holds when wgpu never ran, and leg A is then CPU against CPU at 1.0. | Require positive evidence: a wgpu MoE banner (to be added with item 2), BPM `device_type ≠ Cpu`, and a backend `kernel_id` that differs from the reference's on every quantized matmul tensor (BPM-012 extended to wgpu; per tensor since 2026-10-03). |
| 3 | FALSIFY-R4-003 (a planted `--no-gpu` run is refused) | A checker that refuses every receipt passes. | Add a positive control: a real C0 GPU receipt must be ACCEPTED. Both must hold. |
| 4 | Item 3, "no line starts with the fallback prefix" | If stderr was not captured, it has no lines, so the check passes. | Absence counts only when the banner line is present in the same captured stderr. |
| 5 | The in-process F2 guard (`f2_validate_qwen3_moe`, [V] `qwen3_moe_dispatch.rs:275-316`) | `SKIP_PARITY_GATE=1` returns `Ok` and prints "nothing was compared". A prompt under 2 tokens returns `Ok` with "nothing was judged". In both cases the GPU then serves as validated. | FALSIFY-BPM-017: refuse E3 GPU evidence unless the "F2 guard: GPU matches … min cosine" line is present and neither skip line is. |
| 6 | E2 decode tok/s taken from the `qwen3moe CUDA: … tok/s` line | That line is "(prompt + generated) / time, including the token-by-token prefill" [V]. A long prompt inflates the figure as decode speed. | E2 reads decode and prefill from the harness's own timers (BPM `speed_ratio_ci`), never from this line. |

- **K17 update [V]:** `mark_generation_start()` runs after the F2 guard (#3981), so generation timing already excludes the parity pre-check. Time to first token from the client side still includes it. E2 prefill must say which clock it used.
- **K19** is now carried by BPM `cpu_ref_path` (FALSIFY-BPM-008). RQ-5 is ruled `fp32_act` (cop, 2026-09-27 20:12Z; dense parity and parity-moe share the reference), which parity_moe.rs:110 already uses. CPU MoE experts run only Q4_K and Q6_K (`SUPPORTED_EXPERT_QTYPES`, qwen3_moe_load.rs:84), both on f32 activations inside the scope. An expert tensor of any other qtype refuses on the CPU reference [V at 316dee2cd4].
