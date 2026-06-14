# Decode sampling subsystem — adversarial bug-hunt (2026-06-14)

Adversarial bug-hunt over `crates/aprender-serve/src/generate/` (the decode sampling path):
6 finder dimensions (temperature, top-k, top-p, penalties, chain-order, RNG/draw) → each
finding pipelined into a default-refute skeptic verifier → confirmed-only. **28 findings
verified → 13 REAL, 15 refuted.** Two of the 13 are "confirmed CORRECT" (the skeptic upheld
the code). All findings were re-checked by hand against the actual source before action —
the skeptic verdicts are good but not infallible (see item 10).

## FIXED this PR (PMAT-757 — sampling degenerate-input robustness)

- **[1] NaN temperature not rejected** (`generate/mod.rs:apply_temperature`). `f32::NAN <= 0.0`
  is false (IEEE-754 unordered) → NaN passed the guard → `x / NaN = NaN` for every logit →
  NaN softmax → `rng_value < cumsum` never fires → silent biased fallback to the last token,
  no error. Fix: `!temperature.is_finite() || temperature <= 0.0`. Falsifier
  `test_apply_temperature_rejects_non_finite` (mutation-verified).
- **[11/13] RNG draw can equal 1.0** (`generate/sampler_logit_chain.rs`). Integer math
  `(state>>33)/2^31` is mathematically [0,1), but `2^31-1 as f32` rounds UP to `2^31.0`
  → `rng_value == 1.0` → `rng_value < cumsum` never matches → biased last-token fallback.
  Fix: extracted `lcg_state_to_unit_f32` using the f32-safe 24-bit/2^24 construction
  (numerator exact in f32 → strictly < 1.0). Falsifier
  `test_lcg_state_to_unit_f32_in_half_open_unit_interval` (mutation-verified at u64::MAX).
- **[+] Same f32 LCG bug in a second live sampling loop** (`layers/model_model.rs:157`) —
  fixed to reuse the shared helper (Toyota Way: root-cause all instances). The f64 LCG
  copies (`cli/display_utils.rs`, `cli/benchmark.rs`) are CORRECT — f64 represents 2^31-1
  exactly, no rounding to 1.0 — left unchanged.

Co-evolution: contract `apr-cli-sampling-v1` → 1.1.0 (temperature-finite + RNG-[0,1)
invariants, FALSIFY-SAMP-007/008, 2 proof obligations).

## Confirmed CORRECT (no action — documents that these are NOT bugs)

- **[7] Top-p nucleus cutoff** (`build_nucleus`): pushes the token THEN checks `cumsum >= p`,
  so the boundary token that crosses p is INCLUDED — matches HF/llama.cpp. Correct.
- **[9] Repetition-penalty sign** (`sampler.rs`): asymmetric divide-if-positive-else-multiply
  rule, matches HF `RepetitionPenaltyLogitsProcessor`. Correct.

## FALSE POSITIVE (skeptic erred; rejected on hand re-check)

- **[10] "Top-k computes softmax on the subset, not full vocab"** — NOT a bug. Softmax over
  the kept set is mathematically identical to full-vocab softmax then renormalizing the kept
  set: `e^{x_i}/Σ_{j∈S} e^{x_j}` either way. No distribution error.

## DEFERRED (real but separate scope — tracked follow-ups)

- **[2] temperature==0 = error vs greedy** (`apply_temperature` rejects ≤0). Several backends
  already special-case `temperature==0` via `top_k=1` BEFORE calling apply_temperature
  (cuda/quantized configs); the registry/CPU path does not. Multi-backend semantics
  reconciliation — own PR. (PMAT-757's finite-check is orthogonal: it keeps rejecting 0,
  which should be routed to greedy upstream.)
- **[4/5] TopKSampler.apply() vs sample_top_k() k=0 / k>=vocab divergence**
  (`sampler_topk.rs`): TopKSampler.apply() no-ops on k=0 (≈ "disabled", the llama.cpp
  convention) while sample_top_k() errors. Needs a convention decision (k=0 = disabled?) +
  alignment — own PR.
- **[6/8] TopPSampler::new accepts unvalidated p** (p>1 / p≤0): constructor validation gap —
  own PR (bundle with [4/5] as "sampler-struct input validation").
- **[3] No API-layer temperature bound check** (`api/types.rs`): bare f32 → generic 500
  instead of 400 on bad input. Defense-in-depth at the HTTP boundary; [1] already fixes the
  core NaN propagation. Own PR (bundle with serve-API param-plumbing audit items 7-10).

## Method
Parallel finders → skeptic verification (default-refute). Several findings cite the
codebase's OWN reference (HF asymmetric penalty; llama.cpp temp/greedy; f64 LCG copies).
Each FIXED item ships a unit falsifier + mutation verification (restore the bug, confirm the
test fails) + a contract obligation.
