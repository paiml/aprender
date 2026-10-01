---
type: spike
severity: high
created: 2026-09-26
found_in: phase-08-plan-08-08
resolves_phase:
---

# Spike: can the Laya stance demo pass the ECE ceiling at all? (calibration slice size × temperature cap)

## Why

Plan 08-08's TweetEval stance demo failed the pre-declared `laya-finetune-gate-v1`
gate twice. Both times accuracy passed and calibration did not. The user chose
**option 3** (2026-09-25/26): both failing runs become fail-closed test vectors, the
D-18 live stance deploy waits for a passing run, and this spike runs **before** any
third attempt is declared, so that attempt isn't declared blind.

| | fixed 12 epochs (`d0f4e40d…`) | early stop on cal slice (`3d4b91da…`) |
|---|---|---|
| macro-F1 margin | 0.130 ✓ (≥0.05) | 0.106 ✓ |
| ECE post-calibration | 0.377 ✗ (≤0.10) | 0.222 ✗ |
| fitted T | 5.0 (clamped) | 3.14 |

Evidence (gitignored): `models/decide/tweet-stance-16-fixed-epochs/gate-report.json`,
`models/decide/tweet-stance-16/gate-report.json`; `08-08-SUMMARY.md`.

## Two causes, and neither lever alone was enough on the checkpoint measured

1. **Laya's T ≤ 5.0 cap.** Using the eval labels to pick the best T inside [0.5, 5.0]
   still only reaches ECE 0.1126. ECE falls to 0.0515 at T = 7.5 and 0.0389 at T = 10.
2. **The calibration slice is 12 rows.** The standard error of the stopping signal is 0.22,
   against about 0.07 between candidate epochs. The slice is also easier than eval
   (accuracy 0.67 vs 0.47), so it fits T = 3.14 when the eval-optimal T is about 7.5.

## Questions to answer (with measurements, not declared gate runs)

- With the committed `s64-seed13` selection (64 shots per class, 48-row calibration slice),
  what ECE does post-calibration reach at T ≤ 5 and at T ≤ 10, under both the fixed-epoch
  recipe and the early-stopping recipe? Use seeds 13/17/23 so this is not a single-seed result.
- Does a larger slice move the fitted T toward the eval-optimal T?
- Is there any recipe at 16 shots that passes at T ≤ 5, or is the cap binding everywhere?
- If a cap change is needed, what is the smallest one? How does upstream Laya justify 5.0,
  and would a change be acceptable upstream?

## Added 2026-09-26: is the 1e-5 re-score bar above fp32 noise for real checkpoints?

The debug session `laya-rescore-drift` (`.planning/debug/resolved/`) found that
`pack_rescore_probs_abs` 1e-5 (and `logits_abs` 1e-4), fitted from spike 025's 14 base-model
rows, sits at the fp32 rounding-noise floor on real fine-tuned checkpoints:

- Against a float64 reference of the same model, torch's own stored fp32 probabilities are
  up to 3.71e-5 off on the fixed-epochs checkpoint (13 rows > 1e-5; 52 rows > 1e-4 in logits).
  Rust's error is statistically the same size as torch's, layer by layer.
- After the RoPE `inv_freq` fix, early_stopping and zero-shot pass with only about 1.5x headroom
  (torch is 7.9e-6 to 9.5e-6 from exact). A one-ULP codegen change once flipped zero-shot row 49.
- x86_64 has not been measured.
- The fixed-epochs vector is refused on `RescoreDrift` (user option A), not on the gate.

This matters here because this spike exists to produce a gate-passing run, and pack must then
accept that run. Measure, on each candidate checkpoint, the Rust-vs-torch and
torch-vs-float64 re-score tails over the full eval set, on aarch64 and on x86_64. If the bar would
refuse a passing model, bring a contract amendment to the user, for example a bar defined as a
multiple of the per-checkpoint torch-fp32-vs-float64 error. It must be declared before any gate
run is read. It is a tolerance change, and the user has not yet approved one.

## Constraints

- A spike is not a gate run. Declare whatever it recommends in `laya-finetune-gate-v1`
  **before** the next gate run is read (D-07). Do not move `gate_max_ece`; the user
  ranked that option "not recommended".
- If this leads to raising the cap, it changes `laya-parity-v1` and the Rust clamp
  (`calibration_temp_max`), and it diverges from upstream Laya. That is an
  architectural decision for the user.
- A 64-shot cell changes the D-19 demo data, so D-19 needs an amendment.

Run it with `/gsd-spike`.
