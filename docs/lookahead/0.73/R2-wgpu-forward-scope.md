# R2 — wgpu forward scope (0.73, L3 draft, 2026-09-27)
Source: R5 census F-R5-3/F-R5-4. Tags: [V] = verified in the tree, [A] = asserted and needs a re-read before coding.

## Goal
Measured wgpu decode cosine is 0.955 against a floor of 0.995 (E1 per leg). R2 closes that gap. R1's FALSIFY-BPM-007 is the control that shows the gap today, BPM-006 marks a wgpu cell with no wgpu forward line as NotRun, and BPM-016 labels a cell whose GEMVs fall back to the CPU per op as HYBRID.

## Items (ordered: correctness first, coverage second)
1. **rope_theta from GGUF metadata.** It is hardcoded as `pow(1000000.0, …)` in **two** WGSL RoPE shaders, `crates/aprender-compute/src/backends/gpu/device/linalg/wgsl_forward.rs:242` and `:278` [V at main 316dee2cd4, 2026-10-03; :229/:265 at 00052c0128; the earlier `:943` cite was stale]. That is wrong for every model whose theta is not 1e6. Falsifier: a theta = 1e4 fixture must give a different RoPE output than 1e6 in both shaders, at position >= 1 (at position 0 RoPE is the identity for any theta).
2. **head_dim from metadata**, not hidden/heads (`gguf_gpu_generate.rs:185` [A]). Qwen3 has head_dim ≠ hidden/heads on some sizes. Falsifier: a synthetic config with head_dim ≠ hidden/heads must match the CPU forward.
3. **Qwen3 q_norm/k_norm in WGSL.** Today they are absent on the wgpu path [A]. This is the likely main source of the 0.955 cosine; confirm with `apr trace` layer diff before coding.
4. **Q6_K, Q8_0 and Q4_0 WGSL GEMV.** Only Q4_K and F32 exist [V]. This removes the `wgpu_adapter.rs:339` refusal and the CPU fallback (F-R5-2). Note [V at 00052c0128]: Q6_K and Q5_K are not refused. `dequant_tensor_public` widens them to F32 on the host at load, and the device runs an F32 GEMV. op_placement reads `device` for that, so the receipt needs a per-tensor `device_qtype` to tell a widen from a native GEMV (wgpu-forward-v1 FALSIFY-WGF-004).
5. **Move attention, RoPE, LM head and argmax onto the device** (F-R5-4). Until then, every wgpu receipt is `hybrid: true` (R1 §10; RQ-4 provisional default: E1 may pass hybrid, and E2/E6 name it).
6. **gated-delta WGSL route** for qwen35 (`forward_qwen35.rs:1341`, the `delta_rule_head` call in `deltanet_mix_rows` [V at 316dee2cd4]). Needed for E1 and E4 on C1–C3, not E3: Qwen3.5-4B is E1's model, and E3's is the MoE. It may slip to 0.74. The landing map ranks it 6 (§Ranking, rows 6 to 20).

## Exit
E1 ≥ 0.995 per leg on C1, C2 and C3 for Qwen3 Q4_K_M, with op_placement all-device for items 1–5.

## Risks
- **K13:** fixing theta or head_dim may *lower* cosine on models that happened to match 1e6. Measure every model in the ledger, not one.
- **K14:** item 3's attribution is unmeasured (R-1). Run the layer-diff trace first.

## L25 review (2026-10-03, la-73)
The contract already blocked the theta-blind positions (0 and pos_in_head 0) and identity-like norm fixtures. The new holes are:

| # | Check | Vacuous pass | Fix |
|---|---|---|---|
| 1 | Ledger leg of rope_theta and head_dim | Qwen2/Qwen3 GGUFs use freq_base 1e6, so an all-Qwen ledger passes with the hardcoded value. The same holds for head_dim when key_length = hidden / n_heads on every model. | The ledger needs one model per equation that discriminates, e.g. a Llama-family freq_base (5e5 or 1e4 [A]) and a decoupled head_dim (Qwen3 sizes [A]). Without one: NOT_MEASURED (WGF-007). |
| 2 | qk_norm_applied | Vacuous on a fixture with no q_norm tensors; a NaN layer cosine is dropped by `f32::min`. | Assert the tensors are present; non-finite is FAIL. |
| 3 | qtype_coverage | "device GEMV or named refusal" holds when every qtype is refused. | Q4_K and Q6_K are required: Q4_K_M stores some tensors as Q6_K (WGF-008). |
| 4 | Hybrid flag (WGF-005) | A map that omits ops has no host ops, so `hybrid` derives false. | op_placement keys must equal the full decode op set (WGF-009). |
| 5 | Never-worse sweep (WGF-006) | Item 4 makes refusal legitimate; a model refused after the change leaves the sweep, which then passes. | A refusal after a prior measurement is a regression (WGF-010). |

Contract: obligations 6 -> 10.
