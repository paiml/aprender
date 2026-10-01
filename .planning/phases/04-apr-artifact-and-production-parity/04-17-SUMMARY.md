---
phase: 04-apr-artifact-and-production-parity
plan: 17
subsystem: training-lifecycle
tags: [setfit, apr, public-api, typestate, sealed-trait, selection-lock, apr-04, ops-01, ops-02, trn-07]

# Dependency graph
requires:
  - phase: 04-05
    provides: "AprCodec behind the sealed codec seam — the codec whose bytes now reach a caller"
  - phase: 04-13
    provides: "bundle field 20 (ProvenanceRecord) — untouched by this plan, and that is a deliverable"
  - phase: 04-14
    provides: "SetFitTrainConfig::to_request — the merge door apr setfit train already used"
  - phase: 04-06
    provides: "commands/setfit_train.rs::verified_artifact_bytes — the ONE call site this plan closed, plus the two ignored witnesses"
provides:
  - "SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes(self) -> Vec<u8> — the ONLY public door to the verified artifact's bytes (G1)"
  - "train::setfit::credential::SetFitCredential — a SEALED three-value credential, and the fresh-process door to the lock chain (G2)"
  - "train::setfit::lock::create_selection_lock<C: SetFitCredential> — the single implementation the inherent method now forwards to"
  - "`apr setfit train` WRITES its artifact: OPS-02's train leg is unblocked at the library tier"
affects: [04-12, 04-16, 04-07, 04-08, 04-11, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A consuming door beats a borrowing accessor when the value is large and the handover is final: the caller cannot hold the payload AND the live model at once"
    - "A retype is only one implementation if the moved body has ONE home and the old spelling forwards to it — asserted by counting the refusal's construction sites, not by reading the diff"
    - "A newtype with a hand-written Debug is how a retained payload stops being a logging hazard, because `LifecycleState::Evidence: Debug` makes the derive reachable from every run"
    - "A source-scan guard is blessed by a TWO-SIDED control against a real prior revision (`git show HEAD~1:file`), not by inspection of the needle"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/credential.rs
    - crates/aprender-train/src/train/setfit/credential_tests.rs
    - crates/aprender-train/tests/ui/setfit_external_credential_impl.rs
    - crates/aprender-train/tests/ui/setfit_external_credential_impl.stderr
  modified:
    - crates/aprender-train/src/train/setfit/verify.rs
    - crates/aprender-train/src/train/setfit/verify_tests.rs
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/lock.rs
    - crates/aprender-train/src/train/setfit/lock_tests.rs
    - crates/aprender-train/tests/ui.rs
    - crates/apr-cli/src/commands/setfit_train.rs

key-decisions:
  - "The retained bytes are wrapped in a `RetainedArtifactBytes` newtype with a hand-written Debug, because `ArtifactVerifiedEvidence` must be `Debug` and a derived `Vec<u8>` would render 1,824,298 bytes as decimal integers into any log line that formatted a run"
  - "`into_artifact_bytes` went into its OWN impl block, not the reproducibility accessor block: that block's guard proves read-only-ness by calling every member through a shared reference, and a consuming method cannot be called that way — so the guard was EXTENDED to count the second block rather than have its claim quietly weakened"
  - "`create_selection_lock`'s body moved to a public free fn over the credential and the inherent method forwards to it; the candidate-membership refusal is now asserted to be constructed exactly once in lock.rs, so the two spellings cannot drift"
  - "`ARTIFACT_BYTES_GAP` and the `verified_artifact_bytes` wrapper were DELETED rather than kept as a forwarder — the wrapper existed only to hold the typed refusal, and the guard is stronger when it names the library door instead of a local alias"
  - "The seal was falsified before it was trusted: deleting `: sealed::Sealed` makes the trybuild case COMPILE and trybuild reports `Expected test case to fail to compile, but it succeeded`"

patterns-established:
  - "Extending a counted-surface guard to a second block requires asserting there is no THIRD block, because `find` returns the first match and would otherwise keep passing"

requirements-completed: []

# Metrics
duration: ~2h50m
completed: 2026-08-15
---

# Phase 04 Plan 17: The Two Public-API Doors — Summary

**Both doors wave 5 proved missing are landed, both are compiler-guarded, and neither
touched the artifact schema. `apr setfit train` now writes the file whose SHA-256 it
reports — and the bytes it writes are proven, by re-hashing, to be the bytes the trusted
policy hashed and round-trip-closed.**

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | G1: `VerifiedOutcome.bytes`, `into_artifact_bytes`, the corrected drop comment, 2 tests + the extended surface guard | `74124e2dc` |
| 2 | G2: `credential.rs`, the three retyped doors, one `create_selection_lock` implementation, 7 tests | `cd017c8b4` |
| 2 | G2 seal proved by trybuild + the blessed `.stderr` + the ui harness doc | `8f71a7b68` |
| 3 | apr-cli: the gap closed at its one call site, the G1 witness un-ignored | `b92db643a` |

Task 2 landed in two commits at the coordinator's instruction after a stream stall — the
working credential and retype were committed as soon as they were green rather than held
while the trybuild case was blessed. Both commits build and test alone.

## G1 — the verified artifact's bytes have a door

`verify::run_verify_policy` did `drop(bytes)` (verify.rs:656) and kept `bytes.len()`.
`VerifyReport::artifact_bytes()` returns a `usize`, which was the trap: it compiles and
reads as if the payload were in hand. The bytes now move into `VerifiedOutcome.bytes` and
travel to `ArtifactVerifiedEvidence`, and leave through exactly one public door:

```rust
impl SetFitRun<ArtifactReloadedAndVerified> {
    pub fn into_artifact_bytes(self) -> Vec<u8>
}
```

`drop(reloaded)` was NOT moved. It is the larger buffer (the bundle carries its tensors as
hex, so ~2x the artifact) and it has no caller.

### The bytes are proven to be the HASHED ones

`verify_into_artifact_bytes_are_the_hashed_bytes` re-hashes what the door returns and
requires the digest to equal `run.artifact_hash()`. A non-emptiness check would have passed
for a re-serialization, which is the exact substitute this door exists to make unnecessary.
The test also reloads them through `SerdeJsonCodec` and re-serializes, observing
`close_round_trip`'s property from OUTSIDE the policy on the value a caller actually
receives.

### Peak RSS — MEASURED, five runs per side

`/usr/bin/time -l` around one full verify of the calibrated fixture
(`train::setfit::verify::verify_tests::verify_full_pipeline_reaches_the_final_state`,
`--exact --test-threads=1`, binary `target/debug/deps/entrenar-86eb079f2a405f41`,
aarch64 Darwin 25.6.0). `maximum resident set size`, bytes:

| | run 1 | run 2 | run 3 | run 4 | run 5 | **min** | **median** | **max** |
|---|---|---|---|---|---|---|---|---|
| before | 59,146,240 | 59,293,696 | 65,748,992 | 59,588,608 | 59,473,920 | **59,146,240** | **59,473,920** | 65,748,992 |
| after | 67,518,464 | 60,882,944 | 61,292,544 | 61,571,072 | 61,128,704 | **60,882,944** | **61,292,544** | 67,518,464 |

- **delta on min: +1,736,704 bytes (+1.66 MiB)**
- **delta on median: +1,818,624 bytes (+1.73 MiB)**
- **retained buffer, measured exactly: 1,824,298 bytes (1.74 MiB)**

The median delta accounts for the retained buffer to within **0.3 %**. That agreement is
the reason these numbers are worth quoting: the five-run spread is ~6 MiB, so a single pair
of runs would have proved nothing, and the median difference landing on the deterministic
figure is what shows the measurement is resolving the retention rather than the noise.

The retained length is re-measurable, not just recorded:

```
cargo test -p aprender-train --lib --features setfit verify_into_artifact_bytes -- --nocapture
04-17 G1: retained artifact buffer = 1824298 bytes
```

**Does the delta "materially exceed the ~180 MB the old comment cited"? No — and the honest
answer needs a caveat.** The old figure was for a FULL PIN, and no test in this repository
can produce one (F-10). It was an estimate then and it remains one. What can be stated
without estimating is the scaling law, because the retained buffer IS the artifact: this
holds exactly one artifact's worth, never a multiple. The corrected comment in `verify.rs`
says all of that, including that the 180 MB figure stays an estimate.

### The drop-justification comment is corrected, not deleted

It used to read *"only its LENGTH is still wanted … on a full pin it is ~180 MB that would
otherwise stay live"*. The first half was measured false — wanting only the length is
precisely what left `apr setfit train` unable to write its output. The replacement states
what is retained, why (there is no other public route to the verified bytes), the five-run
table above, and that `drop(reloaded)` keeps its position and its reason.

### Debug safety (Rule 2)

`ArtifactVerifiedEvidence` must be `Debug` (`LifecycleState::Evidence: fmt::Debug`), so a
bare `Vec<u8>` field under the derive would render 1,824,298 bytes as decimal integers —
several MB of text — into any log line that formatted a run, and two orders of magnitude
worse on a pin. The bytes are wrapped in `RetainedArtifactBytes` with a hand-written
`Debug` printing `RetainedArtifactBytes { len: N }`, asserted by exact string equality on
the newtype (a `contains` check would pass for a derive that printed the length and then
every byte).

## G2 — the sealed fresh-process credential

The three doors read exactly three values off the run. Not one touches the evidence table.
`SetFitCredential` is those three values and nothing else:

```rust
pub trait SetFitCredential: sealed::Sealed {
    fn artifact_hash(&self) -> String;
    fn selection_semantic_hash(&self) -> String;
    fn selection_ledger_hash(&self) -> [u8; 32];
}
```

`SetFitRun<ArtifactReloadedAndVerified>` implements it by delegation, written in
fully-qualified inherent form (`SetFitRun::<ArtifactReloadedAndVerified>::artifact_hash(self)`)
so it cannot silently become infinite recursion the day the inherent method is renamed.

The retype:

| Door | Before | After |
|---|---|---|
| `SelectionLock::mint_test_token` | `&SetFitRun<ArtifactReloadedAndVerified>` | `<C: SetFitCredential>(&self, model: &C)` |
| `CanonicalTestAccess::grant` | `&SetFitRun<ArtifactReloadedAndVerified>` | `<'a, C: SetFitCredential>(token, model: &C, dataset)` |
| `create_selection_lock` | inherent method, body inline | inherent method **forwards** to a `pub fn create_selection_lock<C: SetFitCredential>` |

`mint_test_token` still calls `self.verify_integrity()` FIRST, before the identity
comparison. Every error variant and every field is unchanged; this is a retype.

### The seal is COMPILER-PROVEN, and was shown able to fail

`tests/ui/setfit_external_credential_impl.rs` implements `SetFitCredential` for an
out-of-crate `ForgedCredential` returning three strings of its own choosing. The blessed
`.stderr` pins:

```
error[E0277]: the trait bound `ForgedCredential: credential::sealed::Sealed` is not satisfied
   = note: `SetFitCredential` is a "sealed trait", because to implement it you also need to
     implement `entrenar::train::setfit::credential::sealed::Sealed`, which is not accessible
```

**Falsification (CLAUDE.md rule: a negative that has only been described is not a
negative).** Deleting `: sealed::Sealed` from the trait and re-running:

```
test tests/ui/setfit_external_credential_impl.rs ... MISMATCH
Expected test case to fail to compile, but it succeeded.
rc=101
```

Mutation reverted (`cp /tmp/credential.pristine`, `git diff` empty); 9/9 ui cases green
again. A sealed-trait case that had only ever been observed passing would prove nothing
about the seal — it would be equally consistent with trybuild not running the file at all.

### The evidence constraint — nothing is fabricated

- `credential.rs` names `HeadFittedEvidence`, `PassedEvidence`, `UpdateEvidence` and
  `validate_evidence` **only in prose**. `credential_module_fabricates_no_evidence` scans
  CODE LINES only (comment lines filtered, with a non-vacuity assertion that the filter did
  not eat the module) for those four plus `EvidenceSummary`, `Default`, `default()`,
  `unimplemented!` and `todo!`, and separately requires the explanatory prose to still be
  there — so deleting the explanation is caught as well as adding the code.
- `credential_validate_evidence_is_still_the_only_passed_evidence_producer` counts
  struct-literal construction sites of `PassedEvidence` across `credential.rs`, `lock.rs`,
  `mod.rs`, `verify.rs`, `evaluate.rs` and `bundle.rs` — **0 in each** — with a **positive
  control** requiring `tune.rs` to still hold exactly **1**. Without the control, a counter
  that could only return zero would report success on a tree where the type had been
  deleted. The counter excludes return types (`-> &tune::PassedEvidence {` opens a function
  body, and `mod.rs` has two of those), declarations and `impl` headers; getting that wrong
  in either direction was the failure mode, so both directions are pinned.
- No second minting policy, no second tolerance, no `Default`, no placeholder.

### The generic driver IS the evidence

`credential_tests.rs` holds:

```rust
fn drive_every_door<'a, C: SetFitCredential>(
    model: &C, candidates: Vec<SelectionCandidate>, dataset: &'a PreparedDataset<Canonical>,
) -> Result<CanonicalTestGrant<'a>, LockError>
```

Its body **cannot name** `SetFitRun`, `ArtifactReloadedAndVerified`, `HeadFittedEvidence`
or `PassedEvidence` — the bound does not provide them. The fact that it compiles is the
whole of G2. `credential_drives_the_lock_chain_end_to_end` then runs it with the train-time
run and asserts the grant admits the right rows; when 04-16 lands
`reload_verified_run_from_apr`, its credential goes through this same function unchanged.

Non-vacuity: `credential_that_is_not_the_locked_model_is_still_refused` locks over one run,
mints against a genuinely different one (asserted `assert_ne!` on the two artifact hashes
first) and requires `StaleLock` naming both.

## Task 3 — 04-06's ignored witnesses

**The plan's model of these two tests was slightly off, and the difference matters.** Both
were `#[ignore]`d for *integration weight* (04-10 runs them as their own invocation), not
because the blockers made them fail; both already passed. The mapping the plan intended
still holds cleanly:

| 04-06 test | Status now | Why |
|---|---|---|
| `setfit_train_e2e_records_the_blocker_that_stops_short_of_an_artifact` | **UN-IGNORED**, passes in the default invocation | This is the G1 witness. It stats three fixture files and scans this module's source — it was never the heavy one, it inherited the e2e's reason. |
| `setfit_train_e2e_clears_every_stage_up_to_the_encoder_over_real_phase_two_artifacts` | still `#[ignore]`, reason UPDATED | Genuinely heavy (builds a benchmark directory and a selection), and it still cannot reach a written artifact. Its reason now names **F-10 as the sole remaining cause**, and records that it is a deliberate Phase 5 item rather than a defect. |

**The un-ignored test's claim advanced; it was not weakened.** Its BLOCKER 1 (F-10) half is
byte-for-byte 04-06's. Its BLOCKER 2 half was
`ARTIFACT_BYTES_GAP.contains("into_artifact_bytes")` — *"the gap must keep naming the exact
door that closes it"*. That door landed, so the successor assertions are: the library door
is called **exactly once**, those bytes are what `atomic_write` receives, and **no gap
constant survived** (gone, not merely unused).

**Two-sided control on the new needles** (CLAUDE.md rule 7 — a guard regex is blessed by a
case table, not by re-reading it), against the real prior revision:

| Needle | `HEAD~1` (`git show`) | this tree |
|---|---|---|
| `ARTIFACT_BYTES_GAP:` | **1** | **0** (asserted) |
| `verified.into_artifact_bytes()` | **0** | **1** (asserted) |

Both assertions are therefore measurements, not tautologies.

`ARTIFACT_BYTES_GAP` and the `verified_artifact_bytes` wrapper are deleted. The wrapper
existed only to carry the typed refusal; with the door landed it would have been a one-line
forwarder, and the guard is stronger naming the library door than a local alias. The
completion report's three recorded values are read BEFORE the consuming call — an ordering
the borrow checker enforces, so no source assertion is needed for it.

## Verification — real numbers, status captured directly (never through a pipe)

Every command ran as `cmd > log 2>&1; echo "rc=$?"`.

```
$ cargo test -p aprender-train --lib --features setfit
rc=101   7891 passed; 24 failed; 15 ignored

$ cargo test -p aprender-train --lib --features setfit verify_
rc=0     46 passed, 7877 filtered out            (44 before this plan: +2)

$ cargo test -p aprender-train --lib --features setfit credential_
rc=0     7 passed, 7923 filtered out             (0 before: +7)

$ cargo test -p aprender-train --lib --features setfit lock_
rc=0     88 passed, 7842 filtered out            (88 before: UNCHANGED — the regression witness)

$ cargo test -p aprender-train --test ui --features setfit
rc=0     9/9 compile-fail cases ok               (8 before: +1)

$ cargo test -p apr-cli --features setfit --lib
rc=0     6697 passed, 15 ignored, 0 FAILED       (6696 / 16 before)

$ cargo test -p apr-cli --features setfit --lib setfit_train
rc=0     15 passed, 1 ignored                    (14 / 2 before)

$ cargo test -p apr-cli --features setfit --lib setfit_train -- --ignored
rc=0     1 passed                                (2 before — the other one moved, it did not vanish)

$ cargo check -p apr-cli --features setfit                      rc=0
$ cargo check -p apr-cli                        (feature OFF)   rc=0
$ cargo fmt -p aprender-train -- --check                        rc=0
$ cargo fmt -p apr-cli -- --check                               rc=0
$ cargo clippy -p aprender-train --features setfit --lib --no-deps
rc=0     0 diagnostics naming credential.rs / lock.rs / verify.rs / mod.rs
$ cargo clippy -p apr-cli --features setfit --lib --all-targets --no-deps
rc=0     0 diagnostics naming setfit_train.rs / setfit_io.rs / setfit_commands.rs
```

`--no-deps` on both clippy legs is mandatory here (F-03): without it the run exits on
`aprender-compute`'s pre-existing arm64 debt and "no findings in my crate" becomes
indistinguishable from "my crate was never linted".

### The 24 red tests are the known-red baseline, DIFFED not eyeballed

```
$ diff <sorted observed failures> <sorted known-red-baseline.md names>
Files are identical      (24 names, 21 gpu:: + 3 prune::snapshot_tests)
```

**Zero regressions.** The test-count arithmetic closes exactly: base 7882 → +2 (`verify_`)
→ +7 (`credential_`) = 7891. apr-cli's binary is 6712 tests before and after; the only
change is one test moving from `ignored` to `passed`.

## Deviations from Plan

### 1. [Rule 3 — blocking] `crates/apr-cli/src/commands/setfit_train.rs` is in the diff, and it is not in `files_modified`

The frontmatter lists six `aprender-train` files. Task 3 requires un-ignoring 04-06's
witnesses and making them pass, and 04-06 shipped them in `apr-cli` — the plan's own
`<context>` says so ("it already codes against `into_artifact_bytes` at the single call
site `commands/setfit_train.rs::verified_artifact_bytes`"). The task is not satisfiable
without editing that file. No other apr-cli file was touched.

### 2. [Rule 2 — missing critical] `RetainedArtifactBytes` newtype was added; the plan said only "a `bytes: Vec<u8>` field"

`VerifiedOutcome.bytes` IS a plain `Vec<u8>` as instructed (that type is not `Debug`). The
newtype is on the `ArtifactVerifiedEvidence` field, which the plan did not discuss and which
`LifecycleState::Evidence: fmt::Debug` makes reachable from every `SetFitRun`'s `Debug`.
Retaining the payload behind a derived `Debug` would have turned any `{:?}` on a run into
megabytes of output. Correctness-of-operation, so Rule 2 rather than a question.

### 3. [Rule 3 — blocking] Three existing guards had to be UPDATED, not left alone

Each of these would have gone red or become false, and each was updated to make the same
claim about the new shape rather than relaxed:

- `verify_reproducibility_accessors_are_read_only_and_complete` — extended to count the
  second `impl` block by exact signature, **plus a new assertion that there is no THIRD
  block** (`find` returns the first match, so without it the guard would keep passing while
  a third block grew unobserved).
- `lock_mint_test_token_takes_the_run_object_and_no_hash_bytes` — the needle moved from
  `pub fn mint_test_token(` to `pub fn mint_test_token<`, and the assertion from "names the
  concrete run type" to "`C: SetFitCredential` and `model: &C`". The claim was never about
  the concrete type; it is that the identity is READ OFF AN OBJECT rather than supplied
  beside it, and the seal is what preserves it. A compile-time half was added: the
  train-time run must still satisfy the door.
- `lock_grant_signature_takes_the_token_the_model_and_the_canonical_dataset` — same change,
  same reasoning; the `Split<Test>` and `CompatibilityTest` negatives are untouched.
- `lock_run_side_door_is_a_single_read_only_method` — gained the two halves that make the
  body-move honest: the inherent method must contain the forwarding call, and
  `Err(LockError::ChosenModelNotACandidate {` must be constructed exactly once in `lock.rs`.

### 4. [Rule 3 — blocking] The plan's model of 04-06's `#[ignore]` reasons was inaccurate

Documented in full under Task 3 above. The plan assumed both tests were ignored *because of*
the blockers; they were ignored for integration weight and both already passed. The outcome
the plan specified (un-ignore the G1 one, leave F-10's ignored with its reason naming F-10
as the sole remaining cause) is exactly what was done — the reasoning behind it is just
different from what the plan expected, and saying so is the point of this entry.

### 5. Task 2 landed in TWO commits rather than one

At the coordinator's instruction after a stream stall: the working credential and retype were
committed as soon as they were green rather than held while the trybuild snapshot was
blessed. Both commits compile and test alone.

---

**Total deviations:** 1 file-scope (blocking), 1 missing-critical, 2 blocking guard/model
corrections, 1 process. No scope creep: 11 files, all inside `crates/aprender-train/` and one
`apr-cli` command module. No new package, no `Cargo.toml` or `Cargo.lock` change.

## What was NOT touched — the plan's hardest negative constraint

```
$ git diff --name-only 3ceb04261 HEAD | grep -E "contracts/|bundle.rs|apr_codec.rs|STATE.md|ROADMAP.md"
(no match, rc=1)
```

11 files changed, none of them `contracts/setfit-apr-v1.yaml`, `bundle.rs` (the 20-field
bijection gate), `apr_codec.rs`, `aprender-core/src/setfit/artifact.rs`, `STATE.md` or
`ROADMAP.md`. The artifact schema is byte-identical; this plan was a TYPING change and a
retention change, exactly as scoped.

## Threat Register, as shipped

| Threat | Mitigation as shipped |
| --- | --- |
| T-04-46 (a second minting path with weaker evidence) | `SetFitCredential` carries no evidence and can construct none; the fabrication scan and the `PassedEvidence` producer count (with a positive control) hold both ends |
| T-3-09 / TRN-07 (a forged credential wearing a verified model's authority) | The trait is sealed; proved by trybuild E0277 and falsified by removing the supertrait |
| The written file disagreeing with the reported digest | `into_artifact_bytes` moves the hashed buffer, never rebuilds it; `verify_into_artifact_bytes_are_the_hashed_bytes` re-hashes the returned value against `artifact_hash()` |
| Two paths from a verified run to a file | Exactly one `into_artifact_bytes` call site in `setfit_train.rs`, asserted, with a two-sided control |
| Retained payload as a logging hazard | `RetainedArtifactBytes`'s hand-written `Debug`, asserted by exact string equality |
| T-04-SC (package installs) | Zero: no `Cargo.toml` or `Cargo.lock` change in the diff |

## Known Stubs

**None.** The one stub 04-06 shipped (`verified_artifact_bytes`, which always returned a
typed refusal) is deleted, and `apr setfit train`'s write path is fully wired. The command
still cannot complete end-to-end on this host, but for a reason that is neither a stub nor
this plan's: F-10, a deliberate Phase 5 item, asserted structurally so its author is pointed
here.

## Threat Flags

None. No new network endpoint, no new auth path, no schema at a trust boundary. The one new
public surface (`into_artifact_bytes`) narrows rather than widens: it is the only route to a
value that previously had none, it consumes the run so the payload cannot be held alongside
the live model, and the credential trait it ships beside is sealed.

## Notes for Later Plans

- **04-16.** Your blocker is closed. Build `reload_verified_run_from_apr`'s output as a type
  that implements `SetFitCredential` (add `impl sealed::Sealed for YourType {}` in
  `credential.rs` — `credential_seal_is_a_private_supertrait` asserts exactly one impl today
  and will require your addition to be deliberate). Then hand it to
  `lock::create_selection_lock`, `SelectionLock::mint_test_token` and
  `CanonicalTestAccess::grant` unchanged; `credential_tests::drive_every_door` is the shape
  that already compiles. **Do not mint a `SetFitRun<ArtifactReloadedAndVerified>`** — the
  five unrecoverable evidence fields you measured are still unrecoverable, and this plan
  changed nothing about that.
- **04-12.** `OPS-01-F1` is closed. `SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes(self)`
  is the save rung. It CONSUMES the run, so read `artifact_hash()` before you call it.
- **04-07 / 04-08.** Nothing changed for you except that a real `.apr` can now be produced
  from outside the crate, which makes your fixtures cheaper to build.
- **04-11.** **Nothing was flipped in REQUIREMENTS.md.** OPS-02's train leg is unblocked at
  the LIBRARY tier, but `apr setfit train` still cannot complete on any encoder this repo
  has (F-10), so marking OPS-02 complete would put a claim in the traceability table that no
  run can demonstrate. Same policy Phase 2 used: mark each at the plan that actually closes
  it.
- **Phase 5.** F-10 is now the SOLE blocker between `apr setfit train` and a written
  artifact. Widening `CALIBRATED_REGIMES` is a deliberate contract edit, never an inline
  change.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-train/src/train/setfit/credential.rs               (6.6K)
FOUND: crates/aprender-train/src/train/setfit/credential_tests.rs         (13.8K)
FOUND: crates/aprender-train/tests/ui/setfit_external_credential_impl.rs  (2.3K)
FOUND: crates/aprender-train/tests/ui/setfit_external_credential_impl.stderr (1.5K)
```

Commits claimed, checked in the log: `74124e2dc`, `cd017c8b4`, `8f71a7b68`, `b92db643a`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git status --short --untracked-files=all` | empty | **empty** |
| `git diff --diff-filter=D --name-only` per commit | empty | **empty** on all four |
| files changed since `3ceb04261` | aprender-train + one apr-cli module | **11**, no others |
| `contracts/`, `bundle.rs`, `apr_codec.rs` in the diff | 0 | **0** |
| `STATE.md` / `ROADMAP.md` modified | no | **not touched** — the orchestrator owns them |
| known-red failure names | identical to baseline | **`diff` reports identical**, 24 names |
| peak-RSS numbers | measured, not estimated | **5 runs per side**, `/usr/bin/time -l`, table above |
| the seal | compiler-proven | **E0277 blessed**, and shown able to fail (rc=101) |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 17 — COMPLETE*
*Completed: 2026-08-15*
