# R2 — wgpu forward scope (0.73, L3 draft, 2026-09-27)
Source: R5 census F-R5-3/F-R5-4. Tags: [V] = verified in the tree, [A] = asserted and needs a re-read before coding.

## Goal
Measured wgpu decode cosine is 0.955 against a floor of 0.995 (E1 per leg). R2 closes that gap. R1's FALSIFY-BPM-007 is the control that shows the gap today, and BPM-006 refuses silent CPU fallback.

## Items (ordered: correctness first, coverage second)
1. **rope_theta from GGUF metadata.** It is hardcoded to 1e6 at `wgsl_forward.rs:943` [V]. That is wrong for every model whose theta is not 1e6. Falsifier: a theta = 1e4 fixture must give a different RoPE output than 1e6.
2. **head_dim from metadata**, not hidden/heads (`gguf_gpu_generate.rs:185` [A]). Qwen3 has head_dim ≠ hidden/heads on some sizes. Falsifier: a synthetic config with head_dim ≠ hidden/heads must match the CPU forward.
3. **Qwen3 q_norm/k_norm in WGSL.** Today they are absent on the wgpu path [A]. This is the likely main source of the 0.955 cosine; confirm with `apr trace` layer diff before coding.
4. **Q6_K, Q8_0 and Q4_0 WGSL GEMV.** Only Q4_K and F32 exist [V]. This removes the `wgpu_adapter.rs:339` refusal and the CPU fallback (F-R5-2).
5. **Move attention, RoPE, LM head and argmax onto the device** (F-R5-4). Until then, every wgpu receipt is `hybrid: true` (R1 §10, RQ-4).
6. **gated-delta WGSL route** for qwen35 (`forward_qwen35.rs:1351` [A]). Needed for E3 on C1–C3; it may slip to 0.74.

## Exit
E1 ≥ 0.995 per leg on C1, C2 and C3 for Qwen3 Q4_K_M, with op_placement all-device for items 1–5.

## Risks
- **K13:** fixing theta or head_dim may *lower* cosine on models that happened to match 1e6. Measure every model in the ledger, not one.
- **K14:** item 3's attribution is unmeasured (R-1). Run the layer-diff trace first.
