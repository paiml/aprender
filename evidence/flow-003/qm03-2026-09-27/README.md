# FLOW-003 QM-03: release-PR retries, Corollary 3 at the measured inputs (#4516)

## Decision

**The release PR runs with `r = 0` (nextest `[profile.ci] retries = 0`).** Corollary 3 does not
supply the basis for this. FLAKE-0 does: the operator ruled zero tolerance for flaky tests. A retry
turns a test that fails and then passes into a green run, so the flake goes unreported. Corollary 3
leaves out the value of that signal.

At the measured inputs, Corollary 3 favours `r = 1`, and the gain it prices is small: **0.75 to
0.82 min per fold** at k = 1..8, which is **1.0% to 1.6% of E[T_fold]**. That is the whole price of
choosing `r = 0` before the flaky tests are gone. Once they are gone, φ is below φ\* and `r = 0`
also wins on Corollary 3 itself.

## Inputs

All inputs come from QM-01 (#4513): `evidence/flow-003/qm01-2026-09-27/queue-inputs.json` on branch
`89/4513-qm01-inputs` @ `10c39f45b0`. The window is 2026-09-20T07:09:01Z to 2026-09-27T07:09:01Z
(168 h), and the verdict is `GREEN` with 0 `[U]` inputs.

| Input | Value | n | Source |
|---|---|---|---|
| φ (flake rate per run) | 0.0192 | 156 | 3 runs: `falsify_h3_attention_correctness` ×2, `g9_roofline_efficiency` ×1 |
| C (release-PR cycle) | 43.57 min | 95 | median CI created→completed per release-PR push |
| ρ_rel (retry cost) | 0.1259 min | 83 | mean nextest `TRY n>=2` attempt on release-PR runs |
| E[M] | 0.140 (k=1) … 0.701 (k=8) | — | Lemma 1 at q = 0.140, f = 1.0 |

## Corollary 3 evaluated

The break-even is φ(C − ρ)/(1 − φ²) > ρ·E[M], which favours r = 1.

- LHS = 0.0192 × 43.44 / (1 − 0.00037) = **0.834 min**
- RHS = 0.1259 × E[M] = **0.018 (k=1) … 0.088 (k=8) min**

Result: LHS > RHS at every k, so **r = 1 wins before the disable list**. The QM-01 xtask
independently gives `r_best = 1` for every k = 1..8.

The break-even flake rate is **φ\* = 0.0004 (k=1) … 0.0020 (k=8)**. Each value is the positive root
of the Corollary 3 equality. Both the QM-01 jq and an independent python evaluation reproduce it.

## Correction to FLOW-003 §7

§7 gives the basis for the release-PR retries row as "Cor. 3 (φ\* = 0.075 at C = 80; 0.25 at C = 30)".
Those thresholds assumed that a retry costs a large share of a cycle. The measured ρ_rel is a
**single-test** re-run of about 8 s, because nextest retries one test, not the cycle. That makes φ\*
**about 40× smaller** than §7 states, so the direction of the argument flips: retries are cheap, and
Corollary 3 alone would keep them.

The §7 row stays `r = 0`. Its basis should read "FLAKE-0 policy; Cor. 3 cost of r = 0 ≤ 0.82 min per
fold at QM-01 inputs, and 0 once φ < φ\* ≈ 0.002". That is a spec edit for the FLOW-003 owner and is
not made here.

## Gate: the disable list must cover every harvested flake

`r = 0` goes onto the release PR only after every test that flaked in the window is fixed or disabled,
and never mid-cut (S-1).

| Flaky test | Handled by |
|---|---|
| `aprender-serve falsify_h3_attention_correctness` | a2/flake0-retries0 @a0b704a013, 0d/flake0-serve-h3 |
| `aprender-core g9_roofline_efficiency` | **this branch.** It was not on any FLAKE-0 branch. It was a single wall-clock ratio (t(128) > 1.5·t(32)) and is now a deterministic value check: every cell of ones(n)² equals n, at n = 32, 64, 128 |

## What reverses the decision

Measured post-disable φ > φ\* ≈ 0.002 on the release PR (about 1 flaky run in 500) makes r = 1 cheaper
by up to 0.8 min per fold. Under FLAKE-0 that is still handled by disabling the flaky test, never by
re-arming retries.

With n = 156, a zero post-disable count bounds φ ≤ 3/156 = 0.019 (rule of three, 95%). That cannot
show φ < φ\*, which would need about 7,500 runs. It does not need to: the cost of being wrong is ≤ 0.82
min per fold.

## S-4 carried forward, not resolved here

QM-01 measured an identification rate of **0.287** against the 0.9 floor. 62 of 87 red release cycles
name no failing test. Theorem 2's one-fold-per-push premise is weakened, and the bisect upper bound
reaches 142 min at k = 8, against 77 min without bisection. Re-deriving the fold policy belongs to
QM-07 and is out of scope for this branch.
