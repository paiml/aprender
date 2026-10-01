---
phase: 03-faithful-two-stage-trainer-and-head
plan: 07
subsystem: training
tags: [setfit, head, trn-05, d-08, typestate, encode-once, in-band-negative, no-grad, lambda]

requires:
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 06
    provides: "SetFitRun<EncoderTuned> + tune::PassedEvidence, the armed evidence gate, tune::selection_texts"
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 05
    provides: "autograd::graph_tape_len(), the baseline_encode no_grad precedent, the calibration fixture"
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 04
    provides: "MultinomialLogisticRegression, HeadFitReport, HeadFitError, Regularization::resolve_lambda"
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 03
    provides: "SetFitRun/LifecycleState, HeadRegularization, ResolvedSetFitConfig, SetFitTrainError"
provides:
  - "train::setfit::head_input::head_dataset — the encode-once, eval-mode, no_grad, detached head input with a call-site encode ledger"
  - "train::setfit::head_input::EncodeWitness — training/requires-grad/tape observations taken INSIDE the encode and CHECKED on the shipped path"
  - "train::setfit::head_input::resolve_lambda — THE one lambda resolution, against the unique row count"
  - "train::setfit::head_input::fit_on_selection — stage two without the typestate wrapper, shared by fit_head and the adversary"
  - "SetFitRun<EncoderTuned>::fit_head() -> SetFitRun<HeadFitted> — zero non-self parameters"
  - "HeadFittedEvidence — a named struct with seven private fields and seven read-only accessors (B-3)"
  - "SetFitTrainError::{HeadFit, HeadEncodeNotIsolated, HeadEncodeBatchSizeZero}"
  - "train::setfit::negative — the in-band pair-weighted fitter: negative + control + mirror at ONE shared lambda"
  - "NoEvidence — the named unit that makes the tuple-evidence guard non-vacuous"
affects: [03-08, 03-09, 03-10, 04, 05]

tech-stack:
  patterns:
    - "Write the evidence ledger from the SAME window slice the consumer is handed, never rebuild it from the source afterwards — a ledger derived from the selection agrees with the selection by construction"
    - "CHECK the witness on the shipped path; an observation nothing acts on is a comment with a struct around it"
    - "Cut the test module off before a source-scan guard reads its own file — every needle is otherwise satisfied by the assertion that spells it"
    - "Strip comments before a structural source scan; module prose legitimately quotes the shapes the guard forbids"
    - "Give the adversary a plain f64 where the trusted path takes a policy object, so the control cannot drift into a second variable"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/head_input.rs
    - crates/aprender-train/src/train/setfit/negative.rs
  modified:
    - crates/aprender-train/src/train/setfit/mod.rs

key-decisions:
  - "03-07: the encode ledger is written from the window slice that is about to be encoded, not rebuilt from selection.ordered_ids(). A ledger derived from the selection would agree with the selection by construction and could not see the duplicate-plus-omission defect it exists for"
  - "03-07: EncodeWitness is CHECKED, not merely recorded. head_dataset refuses its own output with SetFitTrainError::HeadEncodeNotIsolated if training mode, a live gradient or tape growth was observed inside the encode. Found by Task 3's clippy gate, which reported the witness as dead code in the lib build — an accurate description of an observation nothing acted on"
  - "03-07: the no-grad proof reads autograd::get_grad(param.id()), NOT param.grad(). Gradients live in the graph's registry copies (tune.rs module doc), so param.grad() is structurally None on this path and the plan's literal assertion would have compared None against None"
  - "03-07: HeadFittedEvidence carries a SEVENTH field, effective_lambda. Without it nothing downstream can state which objective produced the weights, and 03-08's bundle would have to re-derive the one number TRN-05 is about"
  - "03-07: the plan's end-to-end pair-budget weight-invariance test is FALSE and was not written. The budget is stage ONE's input: two budgets take different numbers of optimizer steps and hand stage two different encoders. The true claim — stage two never reads the budget — is asserted against fit_on_selection with the encoder held fixed, plus an end-to-end assertion that the RECORDED lambda is budget-independent"
  - "03-07: Prepared's evidence is the named `NoEvidence` rather than a bare `()`. The B-3 guard is meaningless otherwise: `type Evidence = ();` matches any scan for a tuple evidence type, so the guard can never distinguish the deliberate empty case from an accidental (A, B)"
  - "03-07: the adversarial control uses uniform multiplicity TWO, not one, and observes distance exactly 0e0. That is expected rather than suspicious — doubling a sum is exact in binary floating point and the 1/(2n) mean divides it back, so the 48-row objective is bitwise identical to the 24-row one and L-BFGS from a fixed zero start walks the same trajectory"

patterns-established:
  - "A guard that reads the file it lives in must cut its own test module off first; the first draft of this plan's structural guard reported a count of 2 for a declaration that appears once"
  - "When a plan's acceptance criterion is a grep, run it against the real file before trusting it — three of this plan's five greps count comment and test lines and are unsatisfiable as written"
  - "Prove a count-based gate blind before claiming the multiset gate is necessary: the induced duplicate-plus-omission left encode_call_count, two-build determinism and the row count all GREEN"

requirements-completed: [TRN-05]

duration: ~1h15m
completed: 2026-08-10
---

# Phase 3 Plan 07: The Encode-Once Head and the Inexpressible Multiplicity Summary

**Stage two now consumes `SetFitRun<EncoderTuned>` with zero non-self parameters and fits its head on an encode-once, eval-mode, graph-free matrix whose exactly-once property is a MEASURED multiset — an induced duplicate-plus-omission left the row count, the encode-call count and two-build bitwise determinism all green and was caught only by the ledger — while an in-band pair-weighted fitter proves at an identical lambda of 1/48 that multiplicity WOULD move the coefficients by 8.5e-2 if the trusted surface could express it.**

## Performance

- **Duration:** ~1h15m · **Tasks:** 3 of 3 · **Files created:** 2 · **Files modified:** 1

## Task Commits

| Task | Name | Commit | Type |
|---|---|---|---|
| 1 | RED — behaviour tests against a skeleton with the shape and none of the evidence | `9b28fb35e` | test |
| 1 | GREEN — encode-once head input with a call-site encode ledger | `2fc17e000` | feat |
| 2 | `fit_head` transition, `HeadFittedEvidence`, one lambda resolution | `6f7d352fc` | feat |
| 3 | `negative.rs` — the pair-weighted fitter (+ the witness gate it surfaced) | `0f698b5e0` | test |

---

## What the evidence actually says

### The exactly-once proof, and the proof that a count could not have given it

The plan's central review fix was that a row count cannot see the defect it is meant to exclude.
That was **measured, not argued**. With the module green, one line was induced in `encode_once`:

```rust
rows[4] = rows[2];   // duplicate row 2, omit row 4 — count unchanged, order unchanged
```

| Assertion | Under the mutation |
|---|---|
| `encode_ledger` multiset == `selection.ordered_ids()` multiset | **RED** — `train:0-12` twice, `train:0-6` absent |
| ledger order == selection order | **RED** |
| distinct texts produce distinct embeddings | **RED** |
| `encode_call_count == ceil(n / batch)` | GREEN |
| two builds are bitwise identical | GREEN |
| row count == `selection.len()` | GREEN |

Three of the six gates are blind to it, including the two the review named. The mutation was
reverted and the suite re-run green (13/13).

The ledger is only capable of that because of **where it is written**. Ids and texts travel in one
ordered `Vec<(&str, &str)>`; each window is a slice of that vector, and the ledger entry and the
encoder's input come from the same slice. A ledger rebuilt afterwards from `selection.ordered_ids()`
would have agreed with the selection by construction and stayed green under the mutation above.

### No-grad and eval mode, proven engaged

| Mechanism | How it is proven |
|---|---|
| `no_grad` | Tape length sampled inside the encode loop; a test seeds a real forward/backward first so the baseline is non-empty (`tape_before > 0`, observed 236 in the RED run) and then asserts zero growth |
| gradients untouched | Every trainable parameter's gradient snapshotted **through `autograd::get_grad(param.id())`**, byte-compared across the call, with a vacuity guard that at least one is `Some` |
| `detach` | `requires_grad_enabled()` of every stored tensor, OR-ed inside the loop |
| eval mode | `encoder.training()` sampled inside every window; a training-mode encoder handed in produces embeddings **bitwise identical** to an eval-mode one, which the slice's 0.1 dropout probability makes discriminating |

The RED commit is the two-sided control for all four: with the mechanisms absent the tape grew
236 → 341, `requires_grad_enabled()` was true, and `training()` was true inside the window.

**The witness is now CHECKED, not just recorded.** `head_dataset` refuses its own output with
`SetFitTrainError::HeadEncodeNotIsolated { training_observed, requires_grad_observed, tape_growth }`
if any of the three failed, and a both-ways case table exercises the gate on each defect
individually plus the clean case. See Deviation 5.

### Label order

Sourced from `PreparedDataset::label_names()` and pinned to the fixture's declared
`["alpha", "beta", "gamma"]`, cross-checked against `dataset.label_names()` and against the fitted
head's own `labels()`. Class indices are range-checked against that map, with
`SelectionLabelOutOfRange` as the typed refusal.

### The lambda

One resolution site in the whole module directory (`grep -rho 'fn resolve_lambda' | wc -l` → **1**).
It owns only the choice of `n`; the `1/(2*C*n)` arithmetic stays in `aprender-core`'s
`Regularization::resolve_lambda`, so the half-constant has exactly one home. The resolved value is
handed to the head as `Regularization::Lambda`, so the head does not re-resolve it against its own
row count.

- `resolve_lambda(SklearnEquivalentC { c: 1.0 }, 24)` == **exactly `1.0/48.0`**.
- Pinned through the pipeline against `run.selection().len()`, not against a literal.
- Discriminating: the same reference `C` against the fixture's pair budget resolves to a
  different number, asserted `assert_ne!`.

### The adversarial set — three elements, all green in every `cargo test`

| Element | Measured |
|---|---|
| **Mirror** | adversary at multiplicity 1 reproduces `fit_on_selection`'s stored `f32` weights and intercepts **bitwise** |
| **Control** | uniform multiplicity 2 (48 rows vs 24), same lambda → max coefficient distance **`0e0`** |
| **Negative** | real endpoint frequencies over the 64-pair stream → max coefficient distance **`8.507794141769409e-2`** at the identical lambda `1/48` |

Negative skew, as reported by its own message: per-row multiplicity **1..11 (11.000x)**, per-class
endpoint totals **[45, 39, 45]**, worst class **`alpha`** (index 0) at **1.154x**, 129 weighted rows
from 24 unique. Vacuity guards run **before** the coefficient claim: the test refuses to conclude
anything unless `hi >= 2*lo` and the per-class ratio exceeds 1.

The control's exact zero is **not** the trivially-green control the plan warned about. The two calls
are handed 48 and 24 rows respectively (asserted), and the same `fit_at_lambda` returns 8.5e-2 in the
negative. The zero has a reason: doubling every row doubles the NLL sum exactly — scaling by a power
of two is exact in binary floating point and commutes with rounding — and the `1/(2n)` mean divides
it back, so the objective and its analytic gradient are bitwise identical to the 24-row problem, and
L-BFGS from a fixed zero start walks the same trajectory. A `1e-6` tolerance is kept rather than
asserting `== 0.0`, because that argument assumes no subnormal or overflow in the accumulation.

### The expressible-surface analysis (the D-08 claim, made falsifiable)

`fit_head` takes `self` and nothing else. Everything it fits on is reached from the consumed run,
and it builds its input through `head_dataset`, whose only count is a batch size. There is no
parameter, no field and no builder on that path that can say "weight this row twice", so the
adversary **cannot** be routed through the transition. It is assembled from the surface that does
remain expressible — raw embedding rows plus the public `MultinomialLogisticRegression` — exactly as
Phase 2's `negative_leaky.rs` had to poison an untrusted DTO rather than a `LabeledPair`. Had the
attack been routable through a `SetFitRun` door, that would have been the finding instead of this
negative. The analysis lives in `negative.rs`'s module doc and a test asserts the doc discusses
`fit_head`, `lambda` and `MultinomialLogisticRegression`.

`negative.rs` is `#[cfg(test)]` — **not** `#[doc(hidden)]`. A pair-weighted head fitter must not be
reachable from a shipped build by any door, and 03-10's acceptance criteria reject a doc-hidden
test-support door on the shipped surface.

---

## Deviations from Plan

### DEVIATION 1 [Rule 1 — Bug] The plan's end-to-end pair-budget invariance test is FALSE

**Found during:** Task 2.

The plan asked for a test asserting "varying the PAIR BUDGET … leaves the resolved lambda and the
fitted weights bitwise unchanged". End to end that statement is **untrue**: the budget is stage
ONE's input, so two budgets take different numbers of optimizer steps and hand stage two two
different encoders. The head's weights then legitimately differ for a reason that has nothing to do
with TRN-05, and writing the assertion as specified would have asserted something false.

**What was written instead**, both halves in one test with the reasoning in its doc comment:

1. **Stage-two form (the real claim):** with the encoder held fixed, two `ResolvedSetFitConfig`s
   differing only in budget (12 vs 20, asserted different) produce identical lambda, **bitwise
   identical weights and intercepts**, and identical ledgers — asserted against `fit_on_selection`,
   the very body `fit_head` runs.
2. **End-to-end form (what IS true across the whole pipeline):** two complete
   `prepare → tune_encoder → fit_head` runs at budgets 12 and 20 record the **same
   `effective_lambda` (1/48)** and the same encode ledger, even though their encoders differ.

`fit_on_selection` was extracted for this reason and is `pub(crate)`: `fit_head` consumes its run, so
an invariance claim of the form "changing knob X leaves the head unchanged" is not expressible
against the transition at all. Sharing one body means the property is asserted about the code the
transition runs, not about a second implementation written for the test.

### DEVIATION 2 [Rule 1 — Bug] Two source guards were self-satisfying; both fixed

**Found during:** Task 2.

The first draft of `fit_head_signature_and_evidence_shape_are_pinned` read the whole of `mod.rs` —
the file it lives in. Every needle was therefore matched by the assertion that spells it:
`assert!(mod_rs.contains("pub fn fit_head(self) -> …"))` would have stayed green with the transition
deleted, and `matches("pub struct HeadFittedEvidence").count()` came back as **2** for a declaration
that appears once. `head_input`'s mechanism scan had the same defect
(`text.contains("no_grad")` satisfied by its own assertion).

Both now cut the test module off before scanning (`\nmod tests {`, assembled at runtime), and the
`mod.rs` scan additionally strips comments — the module's prose legitimately quotes the shapes the
guard forbids, and a guard that is red for writing its own explanation is a guard nobody keeps. This
is Phase 2's `negative_leaky.rs` `split("//")` discipline.

### DEVIATION 3 [Rule 1 — Bug] `param.grad()` cannot testify on this path

**Found during:** Task 1.

The plan asked that "every trainable parameter's `grad()` is unchanged". `tune.rs`'s module doc
records that `ComputationGraph::backward` writes gradients into the graph's own registry copies, so
`param.grad()` on an encoder parameter is **structurally `None`** even immediately after a
successful backward. Asserting it unchanged would have compared `None` against `None` — the exact
"PF-001 in a new costume" the same doc warns about.

The test reads `autograd::get_grad(param.id())` instead, seeds real gradients with a forward and
backward first, and carries a vacuity guard that at least one gradient is `Some` before the
comparison is made.

### DEVIATION 4 [Rule 2 — Missing critical functionality] `HeadFittedEvidence` carries `effective_lambda`

**Found during:** Task 2.

The plan enumerated six fields. A seventh was added: the L2 coefficient the fit actually minimized
under. Without it nothing downstream can state which objective produced these weights, and 03-08's
bundle would have to **re-derive** the one number TRN-05 is about — precisely the recompute-shaped
false green that `PassedEvidence`'s own doc argues against. It is also what makes the end-to-end half
of Deviation 1 assertable through the shipped surface. The plan's six are all present; the awk
criterion shows all seven.

### DEVIATION 5 [Rule 2 — Missing critical functionality] The encode witness is now enforced

**Found during:** Task 3, by its own clippy gate, which reported
`method 'witness' is never used` / `field 'witness' is never read` in the lib build. That is an
accurate description of an observation nothing acted upon: the three mechanisms were measured and
then only ever inspected by tests.

`EncodeWitness::require_isolated` is now called by `head_dataset` before the input is returned, and
refuses with `SetFitTrainError::HeadEncodeNotIsolated { training_observed, requires_grad_observed,
tape_growth }`. The shipped encode cannot produce any of the three, so the gate would only ever be
seen passing — `head_input_the_encode_witness_refuses_a_non_isolated_encode` runs the case table
both ways: the clean witness is accepted, and each of training-mode / live-gradients / tape-growth is
individually refused with the growth reported. This makes T-3-24's mitigation a runtime property
rather than a test-only one.

### DEVIATION 6 [Rule 3 — Blocking] `Prepared`'s evidence is the named `NoEvidence`

**Found during:** Task 2. See Falsified Criteria below — this is what makes the B-3 guard mean one
thing. `pub type NoEvidence = ();` is a transparent alias, so no construction site changed.

---

## Falsified acceptance criteria (measured, not assumed)

Three of the plan's five grep criteria are unsatisfiable as literally written, because `grep -c`
counts **lines** across the whole file including doc comments and the test module. Each was run
against the real file; the shipped-code equivalent is enforced by
`fit_head_signature_and_evidence_shape_are_pinned`.

| Criterion as written | Measured | Shipped-code form (enforced by test) |
|---|---|---|
| `grep -c 'Evidence = (' mod.rs` → 0 | **5** | `type Evidence = (` in comment-stripped shipped source → **0**. All 5 raw hits are prose or test literals; the two `type Evidence = (` hits (lines 121, 1296) are both comments |
| `grep -c 'struct HeadFittedEvidence' mod.rs` → 1 | **4** | `pub struct HeadFittedEvidence` in shipped source → **1** (line 146); the other 3 are a doc line and two test literals |
| `grep -c 'Pair' head_input.rs` → 0 | **0** | met exactly, whole file |
| `grep -rho 'fn resolve_lambda' \| wc -l` → 1 | **1** | met exactly |
| `grep -rc 'Option<HeadFittedEvidence>' \| grep -v ':0$'` → nothing | **nothing** | met exactly |

**The `Evidence = (` criterion is unsatisfiable for a second, structural reason** and this is why
`NoEvidence` exists. The needle straddles a word boundary: it matches `pub type NoEvidence = ();`
just as it matches `type Evidence = ();`. So no spelling of "Prepared has no evidence" can drive the
raw count to zero, and the loose needle can never distinguish the deliberate empty case from an
accidental `(A, B)`. The enforced form is the associated-type declaration `type Evidence = (`, which
is **two-sided**: the test also asserts the needle FIRES on the forbidden shape, so a count of 0 for
the wrong reason is caught.

The awk criterion is met verbatim:

```
$ awk '/struct HeadFittedEvidence/,/^}/' crates/aprender-train/src/train/setfit/mod.rs
pub struct HeadFittedEvidence {
    passed: tune::PassedEvidence,
    head: MultinomialLogisticRegression,
    report: HeadFitReport,
    effective_lambda: f64,
    ordered_labels: Vec<String>,
    encode_ledger: Vec<String>,
    encode_call_count: usize,
}
```

All seven accessors return shared borrows or `Copy` scalars; `awk '/impl HeadFittedEvidence/,/^}/'
… | grep -c '&mut'` → **0**.

---

## Verification

| Command | Result |
|---|---|
| `cargo test -p aprender-train --lib --features setfit head_input_` | **rc 0 — 13 passed** (criterion: ≥ 8) |
| `cargo test -p aprender-train --lib --features setfit fit_head` | **rc 0 — 7 passed** |
| `cargo test -p aprender-train --lib --features setfit pair_weight` | **rc 0 — exactly 3 passed** (negative, control, mirror) |
| `cargo test -p aprender-train --lib --features setfit train::setfit` | **rc 0 — 145 passed, 1 ignored** |
| `cargo check -p aprender-train --features setfit` | **rc 0** |
| clippy findings under `crates/aprender-train/src` (`--all-targets`) | **0** |

The 4th `negative.rs` test (`adversary_fitter_takes_a_bare_lambda_not_a_regularization`) is
deliberately **not** named `pair_weight*`, so the plan's "exactly 3 tests" criterion stays exact
while the source assertion still runs in the full suite.

### Pre-existing red, re-measured with two-sided controls

| Command | rc | Attribution |
|---|---|---|
| `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | 101 | `aprender-compute` 19 errors, `aprender-present-terminal` 1. **0 diagnostics cite `crates/aprender-train/`.** Control: the identical command WITHOUT `--features setfit` (this plan's modules are not compiled at all) is red with the same 22 findings. Matches **D-ITEM-02** exactly |
| `cargo check -p aprender-train --no-default-features --features setfit` | 101 | 8 errors, all in `src/monitor/tui/{app,dashboard}.rs` (`presentar_terminal`). Control: without `--features setfit`, identical. Matches **D-ITEM-05** exactly |
| `cargo test -p aprender-train --lib --features setfit` (full) | 101 | 7754 passed, **24 failed** — 21 `gpu::*` + 3 `prune::snapshot_tests::*`, name-for-name the phase's `known-red-baseline.md`. Control: both groups fail identically without `--features setfit` (196/21 and 14/3) |

No new deferred item: every red above is already an entry in the phase's `deferred-items.md` or
`known-red-baseline.md`, and this plan's measurements confirm them unchanged rather than extending
them.

---

## Threat register disposition

| Threat ID | Disposition | What discharges it |
|---|---|---|
| T-3-23 | mitigated | `fit_head(self)` has no pair-shaped input (source-asserted); the in-band adversary is red at a shared lambda with distance 8.5e-2 |
| T-3-24 | mitigated | eval mode + `no_grad` + `detach`, each observed inside the loop AND **refused at runtime** if absent (`HeadEncodeNotIsolated`); bitwise re-encode on pinned windows |
| T-3-25 | mitigated | one `resolve_lambda` (grep → 1) against `input.n()`; exact `1/48` pin; budget-invariance at fixed encoder; the adversary takes a plain `f64` |
| T-3-44 | mitigated | the ledger multiset assertion, proven load-bearing by an induced duplicate-plus-omission that three other gates missed |
| T-3-58 | mitigated | label order from `label_names()`, pinned against the fixture's declared labels and cross-checked against the fitted head's own `labels()` |
| T-3-SC | accepted | no package installs in this plan |

## Threat Flags

None. This plan opens no network endpoint, no auth path, no schema at a trust boundary, and no file
access outside `#[cfg(test)]` reads of the crate's own source for structural guards.

## Known Stubs

None. `head_dataset`, `fit_on_selection` and `fit_head` are complete; there is no placeholder value,
no hardcoded empty collection and no TODO in either new file.

---

## Notes for 03-08

- Every accessor 03-08 needs resolves to a field on `HeadFittedEvidence`: `head()` (predict on probe
  texts, serialize `weights()`/`intercepts()`), `ordered_labels()`, `report()`, `effective_lambda()`,
  `encode_ledger()`, `encode_call_count()`, and `passed()` for the whole stage-one chain. None
  returns `&mut`.
- The probe-prediction pattern 03-08's verify needs is already demonstrated in
  `fit_head_completes_the_pipeline_and_probabilities_sum_to_one`: encode through `run.encoder()`,
  chunk by `shape()[1]`, `predict_proba`.
- 03-10's cross-process exactly-once assertion is served by `encode_ledger()` + `encode_call_count()`
  surviving the transition intact — asserted here by
  `fit_head_evidence_carries_the_ledger_the_labels_and_the_passed_chain`.

## Self-Check: PASSED
