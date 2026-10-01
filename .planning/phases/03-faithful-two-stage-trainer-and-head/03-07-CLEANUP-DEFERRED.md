# 03-07 cleanup: deferred findings

Produced by a 4-angle cleanup review (reuse / simplification / efficiency / altitude) of the
wave-4 diff `542c89a96..HEAD` over `crates/aprender-train/src/train/setfit/`.

Findings that were APPLIED are in the commit that accompanies this file. The findings below
were deliberately NOT applied, with the reason. They are recorded because several were raised
independently by more than one reviewer and are real; they were out of scope for a cleanup
pass, not wrong.

## Deferred — cross-plan refactors (touch 03-05 / 03-06 / 03-04 code)

### D-1. `encode_once` is the THIRD copy of the eval-mode + `no_grad` + chunked-encode block
Raised by: reuse (high), altitude (high).

Three open-coded copies now exist in one directory:
- `head_input.rs::encode_once` (new, this plan)
- `tune.rs:446 baseline_encode`
- `baseline.rs:145-162 FrozenProbeRun::fit`

All the new hardening landed only in the new copy: only `head_input` detaches, only it
validates the `[B,H]` shape via `push_rows`, only it refuses a non-isolated encode. The other
two still do `embedded.shape()[1]` unchecked. `SetFitTrainError::Encoder { reason:
e.to_string() }` is spelled three times.

**Why deferred:** the fix is a shared windowed-eval-encode seam called by all three, which
edits `tune.rs` and `baseline.rs` — landed 03-05/03-06 code whose measured calibration
thresholds are frozen and armed by 03-06's identity gate. A change to how those two encode
risks shifting bitwise results the frozen thresholds were measured against. That is a phase
decision, not a cleanup edit.

**Recommended owner:** a follow-up plan, with the 03-06 gate re-measured deliberately.

### D-2. `HeadFitReport` does not report the lambda the fit resolved
Raised by: altitude (medium).

`validate_fit_inputs` computes `regularization.resolve_lambda(features.len())` at
`multinomial.rs:742` and drops it. Because core does not report it, `aprender-train` must
pre-resolve, which is what forces `head_input::resolve_lambda` and `core_regularization` to
exist, which is what forces the two source-scanning guards that police "exactly one
resolution site" — an invariant the compiler would give for free if core resolved and
reported.

**Why deferred:** adds a field to a landed `aprender-core` public struct (03-04) with pinned
tests. Cross-crate, and the current arrangement is correct, merely over-guarded.

### D-3. `config.batch_size()` throws away a proof it already has
Raised by: altitude (high).

`config.rs:450` validates `batch_size` non-zero, then `batch_size()` returns plain `u32`,
discarding the proof. That is why `HeadEncodeBatchSizeZero` had to be invented. Returning
`NonZeroU32` would make `chunks(0)` inexpressible at every call site at once and delete the
variant, its `Display` arm and its test — and would also cover `tune.rs:455`, which today has
no guard at all.

**Why deferred:** changes a widely-used config accessor signature; blast radius well outside
the reviewed diff.

### D-4. `baseline.rs:168` still flattens `HeadFitError` to a string
Raised by: altitude (medium).

The new `SetFitTrainError::HeadFit(HeadFitError)` variant was added at the new call site while
the identical failure at the existing call site is still
`SetFitTrainError::Evidence { reason: e.to_string() }`. One failure mode, two error shapes in
one module.

**Why deferred:** one-line change in 03-06 code, but 03-06 tests may match on the current
variant. Correct fix, wrong pass.

## Deferred — would change intended behaviour or a pinned contract

### D-5. Dropping `detach()` from the encode loop
Raised by: efficiency (medium). One full `B x H` f32 alloc + memcpy per window purely to read
one flag; `push_rows` copies the rows again anyway, so the storage copy *is* the detach.

**Why deferred:** `detach` is a named acceptance criterion of plan 03-07 ("head_input.rs
contains `no_grad` and `detach`") and is enforced by a live source guard. Removing it is a
plan-contract change, not a cleanup. The reviewer's side observation is worth keeping though:
`detach()` hardcodes `requires_grad: false`, so `requires_grad_observed` can never be
anything but `false` — the witness field is currently vacuous, and reading the flag off
`embedded` instead would make it a real observation. **That is a genuine weakness in the
evidence and should be raised as a finding against TRN-05 rather than silently optimised.**

### D-6. Removing `HeadFittedEvidence::ordered_labels` as a field (delegate to `head().labels()`)
Raised by: efficiency (medium). It is a third copy of the same label list.

**Why deferred:** the plan pins the evidence struct's field list, 03-08 and 03-10 assert on
it, and a live guard greps the declaration for each named field.

### D-7. Deleting `pub type NoEvidence = ()`
Raised by: altitude (medium) — a public API symbol whose stated reason to exist is to make a
string-scanning test non-vacuous.

**Why deferred:** it is currently load-bearing. The B-3 guard counts `type Evidence = (`;
if `Prepared` went back to `type Evidence = ();` the count becomes 1 and the guard fails. The
altitude fix (narrow the needle) is real but re-cuts a delicate two-sided guard for no
functional gain.

### D-8. Deleting the file-local `resolve_lambda` count guard as "subsumed"
Raised by: simplification (low-medium), altitude (medium).

**Not subsumed.** The directory-wide guard in `mod.rs` asserts exactly one resolver exists
*anywhere* in the directory; the `head_input.rs` guard asserts it is *here*. Move the resolver
to another file and the directory-wide guard still passes. The narrower guard pins location,
which is a distinct property. Kept deliberately.

## Deferred — test-only, low value relative to churn

### D-9. Memoising the expensive fixture pipelines (`LazyLock`)
Raised by: efficiency (high, measured). ~3.5 s of redundant `head_fitted_run` builds plus
~0.8 s of redundant `trusted()` builds, out of a ~17-minute single-threaded suite.

**Why deferred:** requires the run/fixture types to be `Sync` for a `static`, which is not
established; the measured win is ~0.4% of suite wall-clock.

### D-10. Consolidating the source-scan test helpers into `test_fixtures.rs`
Raised by: reuse (medium), simplification (medium-high), altitude (medium). `shipped_source`
in `head_input.rs` and `shipped_mod_source` in `mod.rs` are the same helper; the
`CARGO_MANIFEST_DIR` path is spelled inline in three places across two files.

**Partially applied:** the duplicate read and duplicate path expression *within*
`head_input.rs` were removed (it now reads the file once). The cross-module consolidation into
`test_fixtures.rs` was left — it edits a file outside the reviewed diff.

### D-11. `dataset_with_foreign_ids` duplicates `fx::synthetic_dataset`
Raised by: reuse (medium), simplification (medium-high). ~35 lines reproducing the fixture to
vary one row-id prefix, with `["alpha","beta","gamma"]` written out twice more while
`test_fixtures.rs:76` already owns `LABEL_NAMES`.

**Why deferred:** the fix parameterises `test_fixtures.rs`, outside the reviewed diff.

### D-12. Test fixtures hand-roll the ledger/dataset/selection/encoder quadruple
Raised by: reuse (high). Three copies bypass `fx::prepared_run` + `into_parts`, which
`baseline.rs:187` already uses, and each copy builds the synthetic corpus twice.

**Why deferred:** rewiring fixtures changes what the negative/control tolerances were measured
against. Behaviour-sensitive; not a mechanical cleanup.
