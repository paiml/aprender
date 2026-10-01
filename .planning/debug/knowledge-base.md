# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## laya-rescore-drift — Rust Laya/ModernBERT re-score drifts past the 1e-5 parity bar on real checkpoints
- **Date:** 2026-09-26
- **Error patterns:** RescoreDrift, pack_rescore_probs_abs, logits_abs, laya-pack exit 2, zero_shot row 49, max_abs 1.028e-5, fine_tuned row 59 4.667e-5, parity drift, torch vs Rust fp32, RoPE, inv_freq, sin/cos ULP, __sincosf_stret, fp32 noise floor, float64 reference
- **Root cause(s):** 1e-5 prob / 1e-4 logit bar sits at or below the fp32 noise floor of real Laya checkpoints (torch's own fp32 output is 3.7e-5 from a float64 forward on fixed_epochs; the bar came from 14 spike rows on the base model); rope_inv_freq single-rounded 1/theta^(2p/hd) from f64 while transformers double-rounds in f32 (1-ULP off at 8/32 and 10/32 frequencies, up to ~32 ULP in sin/cos); 71e2306e5's RopeTable hoist made LLVM stop fusing sin+cos into Apple's __sincosf_stret (1 ULP on ~0.6% of q/k), which alone flipped zero-shot row 49 across the bar
- **Fix:** rope_inv_freq now computes f32(1 / f32(theta^e)), reproducing transformers 5.17's inv_freq bits 32/32 (commit 8e55e0bed). Bar kept at 1e-5 by user decision (option A): early_stopping -> GateFailed[ece_post] exit 3 (rescore 6.7e-6, zs 6.8e-6); fixed_epochs stays a fail-closed RescoreDrift refusal (exit 2, 4.667e-5). The noise-floor cause is not fixable in fp32 Rust; bar fragility (~1.5x headroom) is queued on the calibration spike todo
- **Files changed:** crates/aprender-core/src/models/modernbert/layer.rs
- **Why not caught:** tiny-fixture parity tests (1e-7, short rows) cannot see a pos x 1-ULP RoPE angle error, no test pinned inv_freq bits, and the refactor's bit-identity proof compared Rust against Rust; for the bar, no gate existed: 08-09 was the first full 280-row re-score of a fine-tuned checkpoint and nothing measured either side against a float64 reference
- **Recurrence guard:** crates/aprender-core/src/models/modernbert/layer.rs::tests::rope_inv_freq_is_torch_bitwise (torch bit patterns, theta 160000/10000 x hd 64/16; cargo-mutants 12/12 caught); lesson for future parity drift: measure torch32 and rust32 separately against a float64 forward before blaming Rust, and diff libm imports (`nm`) across builds when a pure refactor moves results by 1 ULP
---

