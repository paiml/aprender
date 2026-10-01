---
phase: 04-apr-artifact-and-production-parity
plan: 16
subsystem: training-lifecycle
tags: [setfit, apr, provenance, selection-lock, credential, trn-07, apr-04, d-11, d-16]

# Dependency graph
requires:
  - phase: 04-17
    provides: "SetFitCredential — the sealed three-value trait the three lock doors are now typed against, and the seal impl site this plan added its second entry to"
  - phase: 04-03
    provides: "load_setfit_apr / VerifiedSetFitModel — the eight-rung production loader incl. probe replay this door runs FIRST"
  - phase: 04-13
    provides: "bundle field 20 (ProvenanceRecord) — the three recorded identifiers the identity gate compares"
  - phase: 04-05
    provides: "AprCodec + APR_FORMAT_ID + artifact_error, and the APR-capable test fixture that is the only model in this crate able to produce real .apr bytes"
provides:
  - "train::setfit::apr_reload::reload_verified_run_from_apr(&[u8], &PreparedDataset<Canonical>, &Selection) -> Result<ReloadedSetFitCredential, SetFitTrainError> — the fresh-process door"
  - "train::setfit::apr_reload::ReloadedSetFitCredential — a SetFitCredential a process that trained nothing can hold, carrying the VerifiedSetFitModel through"
  - "train::setfit::apr_reload::AprReloadError — four typed refusals, three of them naming both compared values"
  - "SetFitTrainError::AprReload(AprReloadError)"
affects: [04-07, 04-08, 04-12, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A nested identity gate is ordered COARSE TO FINE, because a fine identifier that digests the coarse ones makes every later arm unreachable AND misreports the coarse failure"
    - "A shared test fixture is MOVED to the module both consumers can see, never copied — a second APR-capable fixture is free to drift in exactly the dimensions the probes are sensitive to"
    - "A sealed trait's second implementor is a deliberate edit at ONE site: the seal impl must live in the module that owns the private supertrait, so `what satisfies the seal` stays one grep"
    - "Re-blessing a trybuild snapshot is followed by re-falsifying the case in its NEW scope; the diff is reviewed line by line, never taken on trust"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/apr_reload.rs
    - crates/aprender-train/src/train/setfit/apr_reload_tests.rs
  modified:
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/credential.rs
    - crates/aprender-train/src/train/setfit/credential_tests.rs
    - crates/aprender-train/src/train/setfit/apr_codec.rs
    - crates/aprender-train/tests/ui/setfit_external_credential_impl.stderr

key-decisions:
  - "The gate is ordered corpus -> ledger -> rows, NOT the plan's rows -> ledger -> corpus. The three identifiers nest: `Selection::semantic_hash` is SHA-256 of the whole `SelectionPayload`, which embeds `access_ledger` and `ledger_hash`. Semantic-first makes the ledger arm unreachable code AND reports a differently-audited selection as `different rows were selected` — a wrong diagnosis, not a coarse one."
  - "The plan's stated rationale for comparing the ledger hash was FALSE in the same way the draft's it replaced was. The check is right; the reason and the order were not. Measured in `apr_reload_the_three_recorded_identifiers_are_ordered_coarse_to_fine`."
  - "No byte-identity gate. Nothing here re-serializes, so `VerifiedSetFitModel::artifact_sha256()` IS `artifact_sha256_hex` of the input slice; restating that equality would compare a value with itself."
  - "The door takes `&PreparedDataset<Canonical>` and `&Selection`, not owned values: the credential retains neither, and `CanonicalTestAccess::grant` needs the dataset next."
  - "The two selection hashes are read off the CALLER'S objects rather than the artifact's record, because the artifact records hex strings and `selection_ledger_hash` returns `[u8; 32]` — a decode that can fail would be a failure mode arriving after the credential exists."
  - "`ReloadedSetFitCredential` has a hand-written `Debug` printing three digests, because a derive recurses into `VerifiedSetFitModel` and renders every rebuilt tensor."

patterns-established:
  - "When a plan's own justification for a check is measurably false, keep the check and ship the measured reason — declining it would repeat the mistake, and shipping the false reason would license a later reader to delete it"

requirements-completed: []

# Metrics
duration: ~2h05m
completed: 2026-08-15
---

# Phase 04 Plan 16: The Fresh-Process Door — Summary

**A process that trained nothing now reaches `create_selection_lock` -> `mint_test_token`
-> `CanonicalTestAccess::grant` from a `.apr` file, and it gets there without one field of
`HeadFittedEvidence` or `PassedEvidence` being fabricated, defaulted or reconstructed. The
door runs `aprender-core`'s eight-rung production loader FIRST and then gates on all three
recorded provenance identifiers — in an order the plan got backwards, for a reason the plan
stated incorrectly, both of which were measured rather than argued.**

## Task Commit

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `apr_reload.rs` + `apr_reload_tests.rs` (17 tests), the `SetFitTrainError::AprReload` arm, the second seal impl, the fixture move, the re-blessed trybuild snapshot | `53d42b35a` |

One commit: the door does not compile without the seal impl, and the seal impl does not
compile without the type.

## The exact public signature

```rust
pub fn reload_verified_run_from_apr(
    bytes: &[u8],
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
) -> Result<ReloadedSetFitCredential, SetFitTrainError>
```

```rust
pub struct ReloadedSetFitCredential { /* private */ }

impl ReloadedSetFitCredential {
    pub fn artifact_hash(&self) -> &str;
    pub fn selection_semantic_hash(&self) -> &str;
    pub const fn selection_ledger_hash(&self) -> [u8; 32];
    pub const fn model(&self) -> &VerifiedSetFitModel;
    pub fn into_model(self) -> VerifiedSetFitModel;
}
```

`impl sealed::Sealed for ReloadedSetFitCredential {}` and `impl SetFitCredential for
ReloadedSetFitCredential` both live in `credential.rs`, because `mod sealed` is private
there and a sibling cannot name it. That is the seal working, not an inconvenience: the
complete answer to "what satisfies the seal" stays one grep in one file, and
`credential_seal_is_a_private_supertrait` counts it (bumped 1 -> 2, both named).

## What the plan asked for, and what it got instead

The `<replan_note>` kept step 3 (the provenance identity gate) verbatim and dropped steps
4-8. Step 3 shipped — and one thing inside it did not survive contact with the source.

### The plan's rationale for the ledger comparison is false

The plan says, verbatim, that comparing the ledger hash

> distinguishes "the selection this artifact was trained under" from "a different selection
> over the same dataset with the same shots" — which the semantic hash alone does not.

It does not. `Selection::assemble` (`aprender-contrastive-data/src/select.rs:280`) computes
`semantic_hash = SHA-256(payload.to_canonical_bytes())`, and `SelectionPayload`
(`manifest.rs:108-135`) carries **both** `access_ledger` and `ledger_hash` as fields. The
semantic hash therefore digests the ledger hash. Comparing the ledger buys nothing the
semantic hash does not already have — in that direction.

**The relationship runs the other way, and it is what makes the ledger check worth having.**
Three measurements, all in `apr_reload_the_three_recorded_identifiers_are_ordered_coarse_to_fine`:

| Change | ledger hash | semantic hash |
| --- | --- | --- |
| different root seed, same corpus, same shots | **UNCHANGED** | changed |
| same seed, same corpus, one extra ledger append | changed | changed |
| different corpus | changed | changed |

The access ledger records the dataset fingerprint and the access purposes and nothing that
depends on the draw, so it is BLIND to the seed. So the two identifiers are genuinely
different instruments — one coarse, one fine — and the gate order decides which one answers.

### So the gate order was inverted, deliberately

The plan ordered it (a) semantic, (b) ledger, (c) dataset. Shipped order is
**corpus -> ledger -> rows**:

1. `DatasetFingerprintMismatch` — the wrong corpus.
2. `SelectionLedgerHashMismatch` — the right corpus, a different audit trail.
3. `SelectionSemanticHashMismatch` — the right corpus and ledger, a different draw.

Under the plan's order the semantic gate fires for all three cases: the ledger arm is
unreachable code with a test that could never have distinguished it, and a selection whose
rows are byte-identical but whose ledger differs is reported as *"different rows were
selected"*. That is not an opaque diagnosis, it is a **wrong** one — the failure mode T-04-60
names. `apr_reload_refuses_a_selection_with_a_different_ledger_hash` asserts
`polluted.ordered_ids() == honest.ordered_ids()` before it asserts the refusal, so the test
would fail if the wrong arm answered.

The plan's own acceptance criterion — that the ledger comment must cite `select.rs`'s
manifest replay path and must not carry the earlier draft's rationale — is met verbatim, and
both false rationales are pinned absent by
`apr_reload_the_ledger_rationale_is_the_true_one`.

### The byte-identity gate (step 8) is deliberately absent

Step 8 belonged to the rebuild design: re-serialize through the codec, require the bytes to
equal the file. Nothing here re-serializes. `VerifiedSetFitModel::artifact_sha256()` is
`artifact_sha256_hex` of the slice the loader was handed, so `credential.artifact_hash() ==
artifact_sha256_hex(bytes)` holds by construction. It is asserted once, in
`apr_reload_mints_a_credential_whose_hash_is_the_artifacts_own`, as a **witness that the
loader's digest is the one the credential carries** — and the module says plainly that this
is not a gate, because a check comparing a value with itself is worse than no check: it
reads like one.

## The three refusals and their messages

| Variant | Fields | Message names |
| --- | --- | --- |
| `DatasetFingerprintMismatch` | `recorded`, `supplied` | "this artifact was trained on the dataset fingerprinted `{recorded}`, and the dataset supplied fingerprints to `{supplied}`. A canonical-test grant taken over a different corpus would admit rows this model has no claim to" |
| `SelectionLedgerHashMismatch` | `recorded`, `supplied` | "…access-ledger hash is `{recorded}`, and the selection supplied carries `{supplied}`. The ledger hash travels IN the persisted selection manifest and `Selection::replay` refuses to assemble a selection whose records do not produce it, so two processes loading one manifest hold one value: a disagreement here is a different selection, not a different process" |
| `SelectionSemanticHashMismatch` | `recorded`, `supplied` | "…semantic hash is `{recorded}`, and the selection supplied hashes to `{supplied}`. The semantic hash covers the ordered ids, the label map and both content hashes, so a different value means DIFFERENT ROWS were selected — the lock this artifact would take would then record a selection decision nobody made" |
| `ProvenanceUnreadable` | `reason` | field 20 is not derivable from the other nineteen, so a reload cannot proceed without it |

`apr_reload_the_three_refusals_are_distinct_and_each_names_both_values` requires the three to
be distinct values, the three renderings to be distinct strings, and every rendering to
contain both compared values — so a `Display` that dropped one side is caught.

## Zero fabricated evidence — held by the compiler and by a scan

The credential is three values plus the loader's own output. It carries no evidence type, no
`Default`, no placeholder.

- `apr_reload_module_fabricates_no_evidence_and_mints_no_state` scans CODE LINES ONLY (the
  header explains at length why the evidence is absent, and that explanation is worth more
  than the names' absence) for `HeadFittedEvidence`, `PassedEvidence`, `UpdateEvidence`,
  `validate_evidence`, `verify_artifact`, `unimplemented!`, `todo!`, plus any `SetFitRun {`
  or `ArtifactVerifiedEvidence {` construction — with a non-vacuity check that the comment
  filter did not eat the module, and a positive half requiring the prose (including the
  string `E0451`) to still be there.
- `ArtifactReloadedAndVerified` appears **twice in the whole module, both in doc comments**
  (lines 8 and 210). It is not in the return type, not imported, and not constructed.
- `credential_validate_evidence_is_still_the_only_passed_evidence_producer` (04-17's, run
  unchanged) still reports 0 construction sites in every module and exactly 1 in `tune.rs`.

## Verification — status captured directly, never through a pipe

Every command ran as `cmd > log 2>&1; echo "rc=$?"`.

```
$ cargo test -p aprender-train --features setfit --lib setfit::apr_reload::
rc=0     17 passed, 7930 filtered out          (0 before: +17, and the filter is non-empty — F-04)

$ cargo test -p aprender-train --features setfit --lib
rc=101   7908 passed; 24 failed; 15 ignored    (7891 before: +17, EXACTLY the new tests)

$ cargo test -p aprender-train --features setfit --lib credential_
rc=0     14 passed                             (7 before; +7 are apr_reload tests matching the filter)

$ cargo test -p aprender-train --features setfit --lib lock_
rc=0     90 passed                             (88 before; +2 apr_reload tests matching the filter,
                                                all 88 originals still green — the regression witness)

$ cargo test -p aprender-train --features setfit --lib verify_
rc=0     46 passed                             (46 before: UNCHANGED)

$ cargo test -p aprender-train --features setfit --lib setfit::apr_codec::
rc=0     17 passed                             (the fixture move changed no test)

$ cargo test -p aprender-train --test ui --features setfit
rc=0     9/9 compile-fail cases ok

$ cargo check -p apr-cli --features setfit                        rc=0
$ cargo fmt -p aprender-train -- --check                          rc=0
$ cargo clippy -p aprender-train --features setfit --lib --no-deps
rc=0     0 diagnostics naming apr_reload.rs / credential.rs / apr_codec.rs / setfit/mod.rs
$ cargo clippy -p aprender-train --features setfit --lib --all-targets --no-deps
rc=0     29 warnings, 28 in aprender-compute + 1 PRE-EXISTING in verify_tests.rs:628
         (`is_none()` after `find()`), a file this plan does not touch
```

`--no-deps` is mandatory (F-03): without it the run exits on `aprender-compute`'s
pre-existing arm64 debt and "no findings in my crate" becomes indistinguishable from "my
crate was never linted". The run above **did** lint `aprender-train` — it reported the
pre-existing `verify_tests.rs:628` warning, which is how the mechanism is proved engaged
rather than asserted.

### The 24 red tests are the known-red baseline, DIFFED not counted

```
$ diff <observed failure names, sorted> <known-red-baseline.md names, sorted>
Files are identical      (24 names: 21 gpu:: + 3 prune::snapshot_tests)
```

**Zero regressions.** Test-count arithmetic closes exactly: 7891 + 17 = 7908.

### The seal was re-falsified in its NEW scope (CLAUDE.md rule 4)

The trybuild snapshot needed re-blessing: with two implementors rustc's diagnostic changes
from *"the trait … is implemented for `SetFitRun<…>`"* to *"the following other types
implement trait …"* and enumerates both. The re-blessed diff was reviewed line by line and is
**only** that enumeration — `error[E0277]` and the `"sealed trait"` note are byte-identical.

Extending a guard's scope does not inherit the old proof, so the mutation was re-run here:

```
$ perl -pi -e 's/pub trait SetFitCredential: sealed::Sealed \{/pub trait SetFitCredential {/' credential.rs
$ cargo test -p aprender-train --test ui --features setfit
rc=101   test tests/ui/setfit_external_credential_impl.rs ... error
         Expected test case to fail to compile, but it succeeded.
```

Mutation reverted from a pristine copy; `git diff` on `credential.rs` shows only the intended
change, and 9/9 ui cases are green again. A sealed-trait case that had only ever been
observed passing would be equally consistent with trybuild not compiling the file at all.

## Deviations from Plan

### 1. [Rule 1 — bug] The gate order is inverted relative to the plan, and the plan's rationale for one gate is replaced

Documented in full above. The plan's order left the ledger arm unreachable and misreported
the ledger case; its stated reason for the ledger comparison is contradicted by
`select.rs:280` + `manifest.rs:108-135`. The check is KEPT — declining it on a rationale is
the exact anti-pattern T-04-60 names, and the plan was right that it belongs here — but it is
ordered so it can fire and documented with the measured reason.

### 2. [Rule 3 — blocking] Four files beyond the plan's two-file constraint

The plan (written for wave 5's ownership contract with 04-06 and 04-12, both since landed)
allowed only `apr_reload.rs` and `mod.rs`.

- **`credential.rs`** — unavoidable. `mod sealed` is private to that module, so
  `impl sealed::Sealed for ReloadedSetFitCredential {}` cannot be written anywhere else.
  04-17's summary anticipated exactly this ("add `impl sealed::Sealed for YourType {}` in
  `credential.rs`"). The `SetFitCredential` impl went beside it so both answers live together.
- **`credential_tests.rs`** — `credential_seal_is_a_private_supertrait` asserts the seal-impl
  count is 1 and 04-17 wrote it to force the second to be deliberate. Bumped to 2, with BOTH
  types named by their exact impl text, so swapping one for another still fails.
- **`apr_codec.rs`** — three edits, all `#[cfg(test)]` or visibility:
  `fn artifact_error` -> `pub(super)` (reused rather than respelled; a second spelling is a
  second place for the format id to drift), `mod fixture` -> `pub(in crate::train::setfit)`,
  and `apr_capable_run`/`artifact_bytes_of` MOVED from `mod round_trip` into `mod fixture`
  with `round_trip` importing them (call sites unchanged). The move is the point: that
  fixture is the ONLY model in this crate that can produce real `setfit-apr-v1` bytes — the
  phase-3 slice fixture provably cannot compute two of the six probes — so a copy in
  `apr_reload_tests.rs` would have been a second APR-capable fixture free to drift in exactly
  the dimensions the probes are sensitive to. No test was added, removed or changed there;
  `setfit::apr_codec::` is 17 before and after.
- **`tests/ui/setfit_external_credential_impl.stderr`** — re-blessed, diff reviewed, seal
  re-falsified. See above.

### 3. [Rule 2 — missing critical] `ProvenanceUnreadable`, a fourth variant the plan did not list

`doc.provenance` is a `serde_json::Value` (core cannot name train's types). Deserializing it
into `ProvenanceRecord` can fail — for an artifact written by a different provenance shape —
and the alternative to a typed refusal is an `expect`. Fails closed, named, with serde's own
message carried.

### 4. [Rule 2 — missing critical] A hand-written `Debug` on the credential

The plan did not discuss `Debug`. A derive recurses into `VerifiedSetFitModel`, whose own
derive renders the rebuilt encoder — every tensor — into any log line that formatted a
credential. 04-17 measured the same hazard on a 1.74 MiB buffer and answered it the same way.
`apr_reload_credential_debug_does_not_print_the_model` asserts the rendering by EXACT string
equality; a `contains` check would pass for a derive that printed the digests and then the
model.

### 5. The signature borrows its inputs; the plan's took them by value

`(&[u8], &PreparedDataset<Canonical>, &Selection)`. The plan's by-value shape came from the
rebuild design, where the run OWNED the dataset and selection. The credential retains
neither, and `CanonicalTestAccess::grant(token, model, dataset)` needs the dataset next —
consuming it here would take from the caller the object it needs one line later. Pinned by
`apr_reload_signature_returns_a_credential_and_borrows_its_inputs`.

### 6. The `<must_haves>` `key_links` entry to `verify_artifact` is superseded

The plan's frontmatter still lists a required link from `apr_reload.rs` to
`SetFitRun::verify_artifact` with `pattern: "verify_artifact"`. The `<replan_note>` dropped
that design, and this module's own guard now **forbids** `verify_artifact` on a code line of
`apr_reload.rs`. A mechanical check of that key_link will not find the pattern on a code
line; it appears twice in the module header, explaining why the policy is not re-entered.
The surviving key_link (`load_setfit_apr`) is satisfied.

---

**Total deviations:** 1 correctness (gate order + rationale), 1 file-scope, 2 missing-critical,
1 signature, 1 superseded frontmatter link. No new package, no `Cargo.toml`/`Cargo.lock`
change.

## What was NOT touched — the hardest negative constraint

```
$ git diff --name-only HEAD~1 HEAD | grep -E "contracts/|bundle\.rs|artifact\.rs|STATE\.md|ROADMAP\.md"
(no match, rc=1)
```

7 files changed. `contracts/setfit-apr-v1.yaml` is byte-identical; `bundle.rs` (the 20-field
bijection gate), `apr_codec.rs`'s mapping code, `aprender-core/src/setfit/artifact.rs`,
`STATE.md` and `ROADMAP.md` are untouched — the orchestrator owns the last two. The artifact
schema did not move: this plan is a READ path plus three comparisons.

## Threat Register, as shipped

| Threat | Mitigation as shipped |
| --- | --- |
| T-04-46 (a second minting path with weaker evidence) | The door constructs no lifecycle state and no evidence; `ArtifactReloadedAndVerified` appears only in prose; the code-line scan and 04-17's `PassedEvidence` producer count (0 everywhere, 1 in `tune.rs`, with a positive control) hold both ends |
| T-04-47 (reloading against a dataset/selection it was not trained on) | Three typed refusals over all three recorded identifiers, before the credential exists, each naming both values, each with a constructed negative that holds every coarser identity fixed |
| T-04-48 (an artifact that reloads into something else) | `load_setfit_apr` runs the full ladder INCLUDING the six-probe replay before any gate; a flipped byte is refused by an integrity rung, asserted |
| T-04-49 (restating a recorded fact as this host's) | No device is resolved and no config re-probed: this door claims nothing about the machine it runs on |
| T-04-60 (a check declined on a false rationale) | The ledger check is kept AND its false rationales — both of them — are pinned absent by a source assertion; the true relationship is measured in a dedicated test rather than argued in a comment |
| T-3-09 / TRN-07 (a forged credential) | The trait is sealed; E0277 re-blessed and re-falsified in the two-implementor scope |
| T-04-SC (package installs) | Zero: no `Cargo.toml` or `Cargo.lock` change in the diff |

## Known Stubs

**None.** Every value the credential exposes is either computed by the production loader over
the input bytes or read off the caller's own `Selection`, after the gate proved the two agree.

## Threat Flags

None. One new public function and one new public type, both narrowing: the type is
non-constructible outside the module, the trait it satisfies is sealed, and the door refuses
before it builds. No network endpoint, no auth path, no schema at a trust boundary.

## Notes for Later Plans

- **04-07 (`apr eval`).** Your `<interfaces>` block still lists
  `reload_verified_run_from_apr(bytes, dataset, selection) -> Result<SetFitRun<ArtifactReloadedAndVerified>, _>`.
  It returns a **`ReloadedSetFitCredential`** and takes its dataset and selection **by
  reference**. Pass the credential straight to `lock::create_selection_lock`,
  `SelectionLock::mint_test_token` and `CanonicalTestAccess::grant` — unchanged — and use
  `credential.model()` / `.into_model()` for the classification, so the artifact is loaded
  once. `apr_reload_credential_reaches_lock_token_and_grant` is the shape that already
  compiles.
- **Building a candidate set in a fresh process is still an open edge.** A
  `SelectionCandidate` reads its artifact hash out of a `ValidationEvaluation`, and
  `evaluate_validation` takes a train-time `SetFitRun<ArtifactReloadedAndVerified>`. So a
  process that only has `.apr` files cannot yet build the candidate set it locks over — it
  can only consume a lock somebody else wrote. This plan did not need it (the lock chain is
  reachable, which is TRN-07's positive tier) and did not widen the evaluator, because that
  is a second evaluation policy question and belongs to whichever plan owns `apr eval`'s
  candidate story.
- **04-12.** The reload door is the load rung to `into_artifact_bytes`'s save rung.
- **Anyone adding a third `SetFitCredential`.** Add `impl sealed::Sealed for YourType {}` in
  `credential.rs` and bump `credential_seal_is_a_private_supertrait` to 3, naming it — the
  guard exists to make that a deliberate edit. Then re-bless
  `setfit_external_credential_impl.stderr`, because rustc enumerates the implementors, and
  re-falsify the seal in the new scope.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-train/src/train/setfit/apr_reload.rs
FOUND: crates/aprender-train/src/train/setfit/apr_reload_tests.rs
```

Commit claimed, checked in the log: `53d42b35a`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git status --short` after the commit | empty | **empty** |
| `git diff --diff-filter=D --name-only HEAD~1 HEAD` | empty | **empty** |
| files changed | 7, all in `aprender-train` | **7** |
| `contracts/`, `bundle.rs`, core `artifact.rs` in the diff | 0 | **0** |
| `STATE.md` / `ROADMAP.md` modified | no | **not touched** |
| scoped filter `setfit::apr_reload::` | >= 8 tests, non-empty | **17 passed, rc=0** |
| `ArtifactReloadedAndVerified` constructed in `apr_reload.rs` | 0 | **0** — 2 mentions, both doc comments |
| `grep -c selection_ledger_hash apr_reload.rs` | >= 1 | **8** |
| `grep -c "own ledger state" apr_reload.rs` | 0 | **0** |
| known-red failure names | identical to baseline | **`diff` reports identical**, 24 names |
| the seal | compiler-proven, and shown able to fail | **E0277 re-blessed; mutation gives rc=101 "Expected test case to fail to compile, but it succeeded"** |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 16 — COMPLETE (retargeted; supersedes 04-16-BLOCKED.md)*
*Completed: 2026-08-15*
