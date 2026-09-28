# R4 — MoE on GPU backends: wiring scope (0.73, L3 draft, 2026-09-27)

All citations are at origin/main aca6f2d7f6, read with `git show origin/main:<path>`. [V] means the line was read; [A] means it was inferred.

## H4 status: mostly overtaken
The roadmap row reads "the CUDA forward exists; the gap is wiring." On main, most of that wiring has already landed:
- [V] `run_qwen3_moe_generate_dispatch` (`infer/qwen3_moe_dispatch.rs:24`, #3714) is one entry point for all these callers. It tries CUDA first unless `--no-gpu` is set or the build lacks `cuda`, and checks the result against the CPU forward on the real prompt. If the GPU cannot serve, it prints `QWEN3MOE_GPU_FALLBACK_PREFIX` unconditionally and runs the CPU chain.
- [V] Callers: `apr run` (`infer/inference_result.rs:357`), `apr chat` (`apr-cli chat_generate_session_02.rs:330`), `serve` non-streaming chat (`api/cuda_chat_backend.rs:1266`, #3987), and completions (`api/realize_handlers_embed_completion.rs:838`).
- A first Explore pass read the local checkout, which is on a stale branch. That pass reported "`apr run` always runs MoE on the CPU", which is false on main. See K16.

## Remaining items
1. **Streaming serve is CPU-only.** [V] `cuda_chat_backend.rs:1167` calls `run_qwen3_moe_generate_streaming` directly and never goes through the dispatch. On a CUDA server, a `stream: true` request runs on the CPU while a non-streaming request runs on the GPU. The fix is a streaming variant of the dispatch that puts the same fallback line on the stream path.
   - Falsifier FALSIFY-R4-001: on C0, the same prompt with `stream: true` and with `stream: false` must report the same `used_gpu`.
2. **The wgpu MoE forward is a stub.** [V] `gguf/wgpu_backend/mod.rs:196` always returns `UnsupportedOperation` (M-GPU-MOE-2.0). The planned stages, 2.1 per-expert dispatch, 2.2 full forward and 2.3 CPU parity, are in its doc comment at `:107-126` ([A], line numbers from the stale tree). C1–C3 E3 depends on this, and R2 must land first because the MoE forward reuses the dense attention.
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
