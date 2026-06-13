# BEAT falsifiers — adversarial mutation verification (2026-06-14)

**Purpose.** A "beat" is only credible if it actually FAILS when its property breaks.
A green-by-construction test that can never fail is theater. This verifies the shipped
beat gates are REAL falsifiers by injecting a regression into each beat's underlying
property (in a throwaway git worktree, auto-reverted) and confirming the beat FAILS.

**Method.** 3 worktree-isolated agents, one mutation each, run the beat, confirm FAIL.
Production code untouched (worktrees discarded). The 4th beat
(`beat_pytorch_autograd_grad`) is a real falsifier by construction — it compares to
pinned PyTorch gradients at 1e-4 tolerance when the measured Δ is 5e-7, so any autograd
impl change shifts the gradients past tolerance and fails.

## Results — all mutated beats FAILED as expected (✓ real falsifiers)

| Beat | Injected regression | Outcome |
|------|---------------------|---------|
| `beat_nf4_bitsandbytes_equivalence` (PMAT-745) | `NF4_LUT[8]` 0.0795… → 0.20 (corrupt codebook) | **FAILED** — max\|Δrecon\|=0.192672 ≫ 1e-3 |
| `beat_lora_merge_forward_equivalence` (PMAT-747) | merge rank index `lora_a[col*r+k]` → `…+((k+1)%r)` (transpose bug) | **FAILED** — max\|y_merged−y_factored\|=0.276500 ≫ 1e-4 (confirms it is NOT tautological) |
| `beat_fail_closed_garbage` (PMAT-744) | `validate_weight` density gate `>80.0` → `>200.0` (disabled) | **FAILED** — apr accepted `all_zero_weight` (96.9% zeros); *and* the embedding + no-false-positive tests still passed, so the falsifier isolates the mutated gate |

## Conclusion
The four-pillar BEAT gates are adversarially-verified real falsifiers: each catches a
regression of its property at the pinned tolerance, and the fail-closed falsifier is
isolated (a single weakened gate fails exactly one assertion, not the suite). The
campaign's "beat = falsifiable benchmark" discipline holds under mutation, not just
green-on-main. See docs/BEATS.md and the per-beat contracts under contracts/beat-*.yaml
/ apr-*-beat-v1.yaml.
