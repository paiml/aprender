# Phase 5 Plan 03 — Which lower bound binds the production epsilon

**Purpose:** the measured input to the D-04 human checkpoint. This file DECIDES NOTHING. It
derives the candidate lower bounds from the 12 banked calibration passes, states what each
candidate gives up, and refutes the ones that are refutable by arithmetic. The selection is the
human's.

**Status of everything below:** produced by RUNNING the shipped combine over the committed
store, never by hand arithmetic (threat T-05-03-05). The invocation and its status are quoted in
the next section so a reader can reproduce every number.

**Branch:** `gsd/phase-2-contract-gate` (02-01 policy; no PR).

---

## How these numbers were produced

```bash
CARGO_INCREMENTAL=0 \
  APRENDER_CALIBRATION_STORE=.planning/phases/05-benchmark-and-claims-gate/calibration-store \
  APRENDER_CALIBRATION_COMBINE="s8:13,s8:31,s8:53,s64:13" \
  SETFIT_PRODUCTION_CALIBRATION_REPORT=/tmp/05-03-candidates.txt \
  cargo test --release -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture > /tmp/combine.log 2>&1
rc=$?
```

```text
COMBINE_RC=101
```

Status captured on its own line, never through a pipe (CLAUDE.md verification rule 1).

**`rc=101` is the EXPECTED state, and it is the point.** Plan 05-14 turned the window rule into
one fail-closed site, so the same four-cell combine that 05-01 recorded exiting `rc=0` with five
`EMPTY` windows now refuses. The report is still written before the refusal panics (05-14's
"write the report first" property), which is what makes the candidate tables below readable at
all. `PASSES RUN: 12`, four cells loaded, no SKIP in either direction.

The candidate columns are NOT open-coded arithmetic. `epsilon_basis` was widened to take a named
`LowerBound` rule, and each candidate's per-class status is obtained by CALLING it with that
rule. The non-comment count of `lower < upper` in `evidence.rs` is still exactly **1**, and
`grep -c 'fn epsilon_basis'` still returns **1** — 05-14's structural invariant is held across
both plans rather than only at 05-14's end.

**Coverage.** `Derived from 4 of 6 boundary cells. MISSING: s64:31, s64:53.` Every table below
inherits that asterisk.

---

## L1 — the D-03 rule: `lower = 10 x max(worst_ctrl, worst_nnull)`

Printed as the baseline being REPLACED, not as a live option.

| class | lower | upper | width (upper/lower) | window |
|---|---|---|---|---|
| `embedding` | 1.553e-3 | 1.813e-4 | 0.12 | **EMPTY 8.56x** |
| `layer_norm_weight` | 4.802e-6 | 1.891e-5 | 3.94 | EXISTS |
| `layer_norm_bias` | 9.844e-4 | 7.112e-5 | 0.07 | **EMPTY 13.84x** |
| `projection_weight` | 1.051e-3 | 1.231e-4 | 0.12 | **EMPTY 8.54x** |
| `projection_bias` | 1.101e-3 | 3.447e-5 | 0.03 | **EMPTY 31.94x** |
| `attention_key_bias` | 9.278e-8 | 1.714e-8 | 0.18 | **EMPTY 5.41x** |

Five of six classes have no legal epsilon. The exceed factors reproduce 05-01's FINDING 1 to the
digit.

**The cause is the near-null leg, not the control.** `worst_ctrl` is `0.000e0` for every class in
every measured cell — the 1e-30 control writes back bit-identical weights even at 1536 steps (101
rows, zero moved) — so the near-null leg alone sets every lower bound.

**And it is not an artifact of cross-cell mixing.** Within `s64:13` alone,
`best_real / worst_nnull = 3.683e-7 / 9.278e-9 = 39.7`, against a rule that needs `> 100`. Mixing
cells makes it worse (the smallest `best_real` comes from s8's 24 steps while the largest
`worst_nnull` comes from s64's 1536), but the collapse survives without the mixing.

---

## CANDIDATE LOWER BOUNDS

All candidates share the contracted upper edge `best_real / 10`. Nothing 05-01 measured
challenges the upper factor, so only the lower edge is in question.

### L2-bare — `lower = noise_floor` (the bare contracted clearance condition)

| class | lower | upper | width | window |
|---|---|---|---|---|
| `embedding` | 1.779e-6 | 1.813e-4 | **101.94** | EXISTS |
| `layer_norm_weight` | 5.960e-8 | 1.891e-5 | **317.24** | EXISTS |
| `layer_norm_bias` | 5.960e-8 | 7.112e-5 | **1193.23** | EXISTS |
| `projection_weight` | 5.960e-8 | 1.231e-4 | **2065.56** | EXISTS |
| `projection_bias` | 5.960e-8 | 3.447e-5 | **578.27** | EXISTS |
| `attention_key_bias` | 1.133e-9 | 1.714e-8 | **15.12** | EXISTS |

### L2-10x — `lower = 10 x noise_floor` (a CHOSEN safety factor)

| class | lower | upper | width | window |
|---|---|---|---|---|
| `embedding` | 1.779e-5 | 1.813e-4 | **10.19** | EXISTS |
| `layer_norm_weight` | 5.960e-7 | 1.891e-5 | **31.72** | EXISTS |
| `layer_norm_bias` | 5.960e-7 | 7.112e-5 | **119.32** | EXISTS |
| `projection_weight` | 5.960e-7 | 1.231e-4 | **206.56** | EXISTS |
| `projection_bias` | 5.960e-7 | 3.447e-5 | **57.83** | EXISTS |
| `attention_key_bias` | 1.133e-8 | 1.714e-8 | **1.51** | EXISTS |

**The two columns differ by exactly ten, as they must** — the same upper edge over a lower edge
ten times larger. The L2-10x width is the already-reported `eps/noise` column divided by ten,
because that column IS `upper / noise_floor`.

**These were re-derived from the four-cell run, not carried forward from the s8 half.** They
happen to COINCIDE with the s8-half figures, and that coincidence is itself a measured result
worth stating: `s64:13` did not raise any class's noise floor. The worst embedding floor is
`1.779e-6` from `s8:53` against `s64:13`'s `1.741e-6`, and the five dense classes sit at
`5.960e-8` in every cell. So folding the s64 cell in moved `best_real` (upward, in every class)
but left `lower` alone. The distinction matters because the reverse would have been invisible if
the s8 numbers had simply been copied over.

### L3 — relaxing the near-null bound's safety factors

Largest admissible values, per class, from the run:

| class | `k_low_max = best_real / (10 x worst_nnull)` | `product_max = best_real / worst_nnull` |
|---|---|---|
| `embedding` | 1.168 | 11.678 |
| `layer_norm_weight` | 39.379 | 393.795 |
| `layer_norm_bias` | 0.723 | 7.225 |
| `projection_weight` | 1.171 | 11.713 |
| `projection_bias` | **0.313** | **3.131** |
| `attention_key_bias` | 1.847 | 18.471 |

**Binding class: `projection_bias`.** The largest admissible near-null factor is **0.313** and the
largest admissible factor PRODUCT is **3.131**, against the contracted 100.

**This is not a reduced margin — it is an inverted one.** A factor below 1 means epsilon would
have to sit BELOW the near-null delta it is supposed to exceed, so a 2000x-underpowered run would
PASS the gate. L3 is refuted by arithmetic on the record, not by taste. It is also forbidden
outright by 05-01's closeout instruction: do not pick a number to make the table close.

---

## What L2 gives up, at full strength

**Under L2 the gate no longer refuses a genuinely-executed but 2000x-underpowered contrastive
run.** That is the whole of it, stated without softening. The 1e-8 near-null condition exists
precisely to bound epsilon from below with something a not-really-training run actually produces,
and L2 stops using it for that. A run at `encoder_lr = 1e-8` moves `projection_bias` by
`1.101e-4` at s64, and a frozen epsilon of `3.4e-5` sits below that, so such a run passes.

**Why that is not the contract's central claim.** The contract states its claim and names its
adversaries explicitly: "the encoder demonstrably moved", against a frozen encoder, a centroid
baseline, and a 1e-30-learning-rate run. All three produce exactly `0.000e0` at 1536 steps —
measured, `ctrl_max = 0.000e0` for all six classes in every cell — so all three remain refused
with the widest possible margin under L2. The 1e-8 near-null run is **not** one of the named
adversaries; it was introduced by plan 03-05 to obtain a non-degenerate lower bound, because the
1e-30 control bounds epsilon from above only. L2 supplies a different non-degenerate lower bound,
one whose CONDITION the contract already requires.

**That is a narrowing of the claim, and the contract must record it as one.** "The encoder moved
beyond f32 rounding noise, and a run that does not train produces exactly zero" is a weaker
statement than "the encoder moved more than a 2000x-underpowered run would". Substituting one
bound for the other silently would be exactly the repudiation T-05-03-06 names.

### The precedent claim, narrowed to what is true

The noise floor is **already a contracted lower-bound CONDITION**: the invariant "EVERY FROZEN
EPSILON CLEARS f32 ROUNDING NOISE" requires that no parameter can satisfy its threshold by
rounding alone.

It was **already the OPERATIVE lower bound for one class**. The contract's own fixture derivation
table records `layer_norm_weight` with `worst 1e-8` = `0.000e0` and `lower edge` = `0.000e0`. The
near-null leg was degenerate there, so the clearance condition was the only thing doing the work.

**Three things this does NOT license, stated because each is an easy over-read:**

1. It does **not** mean the whole rule pre-exists. The contract states a CONDITION and records
   49x–315x as the clearance the fixture epsilons HAPPENED TO HAVE. There is no
   `lower = k x floor` formula anywhere in it.
2. The **multiplier is therefore CHOSEN, not inherited**. Any factor on the noise floor is this
   plan's choice and must be recorded in the contract as chosen.
3. L2 is a **WEAKENING** of what the gate claims however the factor is set. Being able to cite
   the bound does not make the substitution claim-preserving.

**The recommendation uses `10x`, and that is a choice.** The reasons: it mirrors the near-null
leg's own factor, so the two bounds are stated at comparable strength; and it is strictly
stricter than bare clearance, so it cannot be weaker than what the contract literally requires.
Both are good reasons. Neither is a citation.

### Frozen epsilons under the recommended option, and their clearance

The upper edge is unchanged by the choice of lower bound, so the frozen values are the same
numbers L1 would have produced had its windows existed — the upper edge rounded DOWN to two
significant figures, exactly as the fixture derivation does.

| class | upper edge | frozen eps | over L2-10x lower | over the bare noise floor |
|---|---|---|---|---|
| `embedding` | 1.813e-4 | **1.8e-4** | 10.1x | 101x |
| `layer_norm_weight` | 1.891e-5 | **1.8e-5** | 30.2x | 302x |
| `layer_norm_bias` | 7.112e-5 | **7.1e-5** | 119x | 1191x |
| `projection_weight` | 1.231e-4 | **1.2e-4** | 201x | 2013x |
| `projection_bias` | 3.447e-5 | **3.4e-5** | 57.0x | 570x |
| `attention_key_bias` | 1.714e-8 | **null (ungated)** | — | — |

Rounding DOWN rather than to nearest, per the contract: rounding up would erode the
false-rejection margin that protects legitimate runs. Every rounded value still clears its
L2-10x lower bound, so the rounding does not push any epsilon out of its own window.

The bare-floor clearance range **101x–2013x** is comparable to the fixture's recorded 49x–315x,
which is the sense in which the contracted clearance condition is satisfied at least as well here
as it was there.

**Run-level `embedding_delta_floor`: 2.6e-4.** Derived by the contract's own rule — the smallest
embedding-class MEDIAN across measured cells divided by ten, rounded down. Read off the run
(`EMBEDDING DELTA MEDIAN MIN across measured cells: 2.611e-3`), not recomputed by hand from the
per-cell table. It sits 14.6x above `10 x noise_floor` for the sparse class.

---

## `attention_key_bias` — the disposition, with numbers

**Three quantities, kept apart, because conflating them credits the epsilon with a margin it does
not have:**

| quantity | value | the other five classes |
|---|---|---|
| RAW class separation `best_real / noise_floor` | **151x** (1.714e-7 / 1.133e-9) | 1019x – 20654x |
| the frozen epsilon's clearance `eps/noise` = `upper / noise_floor` | **15.12x** | 101x – 2065x |
| the L2-10x window WIDTH `upper / lower` | **1.51x** | 10.19x – 206.56x |

Only the second is the epsilon's margin — the contracted clearance condition is about the
epsilon, not about `best_real`. The third is how much room a frozen value has inside its window.

**The refuted justification, and what survives.** 05-01 measured `grad_norm_max = 8.084e-10` (s8)
and `7.298e-10` (s64) on the binding key-bias row, and the delta SCALES WITH THE LEARNING RATE
(2e-5 -> 1.714e-7; 1e-8 -> 3.105e-11). A parameter receiving no gradient cannot do that. So the
sentence "these deltas are f32 cancellation residue" and the unqualified "makes dL/db_k exactly
zero" are refuted. What survives, and must be kept, is `dL/db_k = 0` **in EXACT arithmetic** — the
encoder executes f32, where softmax shift-invariance is only approximate, so a ~1e-10 residual
gradient is the expected numerical consequence rather than a bug.

**Option (i) — GATE it at `1.7e-8`** (the window's upper edge rounded down). Legal under L2, and
razor thin: a **1.51x** window. A 1.51x window cannot honestly be frozen from a PROVISIONAL basis,
because two unmeasured cells could close it outright (see the coverage arithmetic below). Choosing
this coherently means ordering option B first, or selecting the bare factor, under which the class
has a 15.12x window instead.

**Option (ii) — UNGATE it, with a MEASURED justification of the margin kind.** Not "it cannot
move" — that claim is dead. The justification is that its usable margin is an order of magnitude
narrower than any other class's on every one of the three quantities above, so freezing a value
there buys a discriminator whose false-rejection behaviour is dominated by rounding rather than by
training. This mirrors the methodology of the contract's own `WHY NO EPSILON` invariant, which
also argued from a measured margin rather than from physics. The class is **not load-bearing**:
the five gated classes already refuse all three named adversaries with `ctrl_max = 0.000e0`, and
an empty gated set is separately unpassable (`NoTestifyingParameters`).

**On D-17.** D-17 closed BOTH options **under L1** — ungated had lost its stated reason, and gated
had no window. Reopening (ii) under a different lower bound is not an override of D-17; it is
precisely what "05-03 must choose a replacement rule on this evidence" means. The refuted sentences
still go, and the exact-arithmetic sentence still stays, whichever option is selected.

### Disarming an apparent D-17 collision

A reader who knows D-17 will stop on this: D-17 forbids carrying forward `eps/noise = 15.1`, and
the L2-10x width for this class is derived as `15.12 / 10 = 1.51`. **That is a different use, and
a legal one.**

What D-17 prohibits is reading `upper / noise_floor` as a MARGIN for an epsilon that has no legal
window — dividing an illegal `upper` by the noise floor and reporting the quotient as headroom.
Under L1 the window is empty, so `upper` is not an epsilon and the quotient means nothing. Under
L2 the window is NON-EMPTY, so `upper` IS a legal epsilon, and `upper / lower` is a width: pure
arithmetic on two measured quantities, both legal. The quantity changed meaning because the
window did.

The number was also recomputed from the four-cell noise floors read out of this run rather than
inherited from 05-01's table, which is what makes it this plan's own.

---

## Coverage arithmetic for `s64:31` and `s64:53`

Under L2 a window closes when `lower` reaches `upper`. Per class, the factor by which its noise
floor would have to RISE, or its `best_real` FALL, is exactly the L2-10x width:

| class | factor that would close the window |
|---|---|
| `attention_key_bias` | **1.51x** |
| `embedding` | 10.19x |
| `layer_norm_weight` | 31.72x |
| `projection_bias` | 57.83x |
| `layer_norm_bias` | 119.32x |
| `projection_weight` | 206.56x |

**Only `attention_key_bias` is plausibly within reach of two more cells.** Embedding is next at
10.19x, and the rest need 31x–207x. Noise floors have been strikingly stable — `5.960e-8` in
every cell for all five dense classes, and `1.741e-6`–`1.779e-6` for embedding across four cells
including the s64 one — so a 10x rise is not a live risk on this evidence. A 1.51x rise is.

**Why it is legitimate to re-ask for the compute now.** 05-01 stopped those six passes because
they could not change the verdict UNDER L1: within a fixed rule `lower` is monotone
non-decreasing and `upper` monotone non-increasing in the measured cells, so a collapse observed
on a subset can never be cleared by measuring more. Under L2 the lower bound is set by the worst
noise floor instead, and the unmeasured cells CAN move it. This is a new reason, not a revision
of the old one.

---

## Options for the D-04 checkpoint

**TWO SEPARATE KNOBS.** (i) WHICH bound binds — contracted either way. (ii) WHAT MULTIPLIER sits
on it — NOT contracted for the noise floor. Options A/B/C are written at the 10x factor; any of
them may be selected with `lower = noise_floor` instead (ten times wider windows), and both
columns are printed above so that substitution needs no re-derivation. **Say which factor you are
selecting.**

| # | Option | What it costs |
|---|---|---|
| **A** | L2 at 10x, `attention_key_bias` UNGATED with a measured-margin justification, frozen from the four-cell PROVISIONAL basis | No new compute. Five gated classes at widths 10.19x–206.56x. All three named adversaries still refused at `0.000e0`. Gives up the near-null claim; commits to a chosen multiplier; freezes with 2 of 6 boundary cells unmeasured. |
| **B** | Option A, but measure `s64:31` and `s64:53` first | ~5.6 h (6 passes at a measured 55.3–55.8 min each, plus ~2 min `cargo test` overhead per pass). Retryable at one-pass granularity under D'; disk preflight `MIN_FREE_GIB = 10` before each pass. Removes the PROVISIONAL asterisk from the number every benchmark cell is judged against. |
| **C** | L2 at 10x with `attention_key_bias` GATED at `1.7e-8` | Keeps all six classes gated. Requires option B's compute first — a 1.51x window cannot honestly be frozen from a provisional basis. At the bare factor its window is 15.12x instead, which is the coherent way to gate it without a razor-thin margin, at the cost of a lower bound with no safety factor at all. |
| D | Relax the near-null safety factors | **REFUTED BY ARITHMETIC** — largest admissible factor 0.313, largest admissible product 3.131 against 100 (`projection_bias` binding). An inverted margin, under which a near-null run would PASS. Also forbidden by 05-01's closeout. |
| E | Narrow the regime envelope to the measured cells | The benchmark's s16/s32/s64 cells would hit `UncalibratedRegime` mid-matrix, which is what D-02 exists to prevent. Not viable for this milestone. |
| F | HALT — land only what is true regardless of the rule | The `sole()` migration, the D-17 correction and the L1 refutation land; no epsilon is frozen. F-10 stays blocked and 05-07, 05-12, 05-09, 05-11, 05-13 stay blocked with it. |

### Recommendation — NON-BINDING

**Option A, at the 10x factor.**

Reasoning, stated so it can be disagreed with: the five gated classes carry 10.19x–206.56x
windows and clear the bare noise floor by 101x–2013x, comfortably inside the band the fixture
epsilons occupy (49x–315x); the noise floors that set the lower bound have not moved across four
cells spanning a 64x range in optimizer steps, so the two unmeasured cells are unlikely to move
them by the 10x the thinnest gated class would need; and `attention_key_bias` — the one class
where the provisionality genuinely bites at 1.51x — is excluded from the verdict rather than
frozen at a margin the basis cannot support.

**The strongest argument against it**, which the human should weigh directly: freezing from 4 of 6
boundary cells means the number every Phase 5 benchmark cell is judged against carries an asterisk
that ~5.6 h of already-understood, retryable compute would remove. If that asterisk is
unacceptable in the milestone's claims layer, option B is the honest answer and the cost is known.

**A note on what the choice is between.** Options A/B/C all narrow the gate's claim in the same
way; they differ only in how much evidence backs the number and whether the thinnest class is
gated. Option F is the only one that preserves the claim, and it preserves it by shipping nothing.
