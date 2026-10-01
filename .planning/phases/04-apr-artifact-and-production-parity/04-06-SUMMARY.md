---
phase: 04-apr-artifact-and-production-parity
plan: 06
subsystem: apr-cli
tags: [setfit, cli, config-merge, ops-02, ops-06, bounded-read, atomic-write, d-05, d-06, d-07]

# Dependency graph
requires:
  - phase: 04-03
    provides: "read_setfit_apr_bytes_bounded + MAX_ARTIFACT_BYTES — the library half of review B5, which setfit_io is the CLI half of"
  - phase: 04-05
    provides: "AprCodec + verify_artifact(AprCodec) — the codec this adapter drives; also THE FINDING (F-10) that no encoder can both pass the calibration gate and carry an artifact"
  - phase: 04-14
    provides: "SetFitTrainConfig::to_request — the public validated merge door that made this plan's --seed/--device design compilable at all"
  - phase: 02-09
    provides: "read_attested_canonical / read_selection_manifest / Selection::replay — the Phase 2 ingest doors this command reuses rather than re-implements"
provides:
  - "`apr setfit train` — the training-only `setfit` namespace, its dispatch, and the complete filesystem adapter"
  - "apr-cli `setfit` feature (NOT default) — the gate 04-10's tier legs and 04-08's serve leg extend"
  - "setfit_io::read_setfit_apr_file_bounded — the ONE bounded artifact-reading door, ready for 04-07 and 04-08"
  - "data_contrastive::read_attested_canonical / read_selection_manifest at pub(crate)"
  - "A MEASURED public-API gap: aprender-train exposes no door to the verified artifact's bytes"
affects: [04-07, 04-08, 04-10, 04-11, 04-12, 04-16]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "One bounded door per resource class: the CLI reads artifact bytes through exactly one function, and that function's own tests ban the unbounded read it replaces"
    - "Fail on the REQUEST before blaming the data — and assert the ORDER, not just the checks"
    - "A blocked step is surfaced as a typed refusal naming the exact remedy, never as a stub, a re-implementation or a silent success"
    - "Guard falsification by behaviour where possible: the bounded-read mutation is caught by observing the 268435457 bytes actually read, not only by a source scan"

key-files:
  created:
    - crates/apr-cli/src/setfit_commands.rs
    - crates/apr-cli/src/setfit_io.rs
    - crates/apr-cli/src/commands/setfit_train.rs
  modified:
    - crates/apr-cli/Cargo.toml
    - crates/apr-cli/src/lib.rs
    - crates/apr-cli/src/extended_commands.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - crates/apr-cli/src/commands/mod.rs
    - crates/apr-cli/src/commands/data_contrastive.rs
    - Cargo.lock

key-decisions:
  - "The missing artifact-bytes door is SURFACED, not worked around: re-serializing in the CLI would be a second bundle-to-artifact implementation whose output is not what was verified, and widening aprender-train is outside this plan's wave-5 file ownership"
  - "--dry-run stops BEFORE --model-dir is opened, and the report says so in a checks_skipped list — a pre-flight that costs as much as the thing it precedes is not a pre-flight"
  - "The output no-clobber gate runs twice: once as a pre-flight before training and once inside atomic_write. The pre-flight closes the 20-minute-run-then-refuse window; the write-time check is what makes the guarantee true"
  - "data_contrastive's two ingest doors were widened to pub(crate) rather than duplicated, so there stays ONE reader of benchmark-manifest.json and ONE of selection-manifest.json"
  - "Task boundaries were shifted so every commit builds alone (F-08): the ExtendedCommands variant and dispatch arm landed with the adapter in Task 2, not with the enum in Task 1"

requirements-completed: []

# Metrics
duration: 2h35m
completed: 2026-08-15
---

# Phase 04 Plan 06: `apr setfit train` Summary

**The `setfit` namespace, its complete filesystem adapter and the CLI's one bounded
artifact-reading door all ship and are tested — and the plan's headline (`… atomically
writes a verified setfit-apr-v1`) is BLOCKED by a measured public-API gap that this plan
surfaces rather than patches around.**

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `setfit` feature, `SetfitCommands` (Train only), `setfit_io` + 5 tests | `089c6d167` |
| 2 | `commands/setfit_train.rs`, the variant, the dispatch arm, ingest-door widening, 13 tests | `4b6d1c49f` |
| 3 | e2e over real Phase 2 artifacts, both blockers made executable, writer/reader composition, 3 tests | `67843ce5a` |

Every commit builds alone. Task 1 deliberately ships no user-visible surface (the enum
exists, nothing dispatches to it) so that the ExtendedCommands variant and the adapter it
calls land in the same commit — the F-08 discipline 04-05 adopted after 04-13's
intermediate commit did not compile in isolation.

## THE FINDING: the verified artifact's bytes are not reachable out-of-crate

**This is the plan's one unmet `must_haves.truth`, and it is a library gap, not an
implementation shortfall.**

`SetFitRun::<HeadFitted>::verify_artifact` returns a verified RUN. The trusted policy
behind it, `verify::run_verify_policy`, **drops the artifact bytes** — deliberately, with
the reason in a comment (`verify.rs:650-656`): "The artifact has done its whole job by
here … only its LENGTH is still wanted. Dropping it before the rebuild is not tidiness —
on a full pin it is ~180 MB that would otherwise stay live." `VerifiedOutcome` carries the
encoder, the head, the report, the probe and the 32-byte hash. It does not carry the
bytes, and `SetFitRun<ArtifactReloadedAndVerified>` has no accessor that would.

So `apr setfit train` can train, verify, and report the artifact's SHA-256 — and cannot
write the file whose digest that is.

**Both ways around it are worse than the gap, which is why neither was taken:**

| Route | Why it was refused |
| ----- | ------------------ |
| Re-serialize in the CLI | `SetFitBundle::from_run_parts` is `pub(crate)`, so an adapter cannot obtain a bundle for `AprCodec::serialize`. Assembling a `SetFitArtifactView` from the run's rebuilt encoder and head instead is a SECOND implementation of the twenty-field mapping 04-05 proved field by field — and its bytes would not be the bytes that were verified, which is exactly the claim ("the served model is the evaluated model") this phase exists to make true. OPS-03 forbids two implementations of one operation. |
| Widen `aprender-train` here | The door belongs in `crates/aprender-train/src/train/setfit/`, which this plan does not own. The wave-5 ownership contract splits `apr-cli` (this plan) from `aprender-train` (04-12 and 04-16, both running concurrently); the plan's own `<verification>` says `git diff --name-only` must contain only `crates/apr-cli/` paths. |

**04-12's plan states the phase's rule for exactly this case** — *"If a public-API gap
blocks the test … STOP and surface — that gap is itself an OPS-01 finding, not something
to patch around by editing core or train sources here."* That is what was done.

**The remedy, precisely.** One public door in `aprender-train`, e.g.

```rust
impl SetFitRun<ArtifactReloadedAndVerified> {
    pub fn into_artifact_bytes(self) -> Vec<u8>
}
```

which requires `VerifiedOutcome` to retain `bytes` instead of dropping them (or a
`verify_artifact_retaining_bytes` variant, if the ~180 MB peak is judged unacceptable on
the default path). `commands/setfit_train.rs::verified_artifact_bytes` is the ONE call
site — closing the gap is a one-function change there, and a test asserts that call site
stays unique.

It is kept executable rather than written down: `ARTIFACT_BYTES_GAP` must keep naming
`into_artifact_bytes`, and `setfit_train_e2e_records_the_blocker_that_stops_short_of_an_artifact`
fails if it stops doing so.

### The SECOND, independent blocker (F-10, already known to the phase)

Even with that door, no run reaches it today:

- `tune_encoder` judges `encoder.architecture_fingerprint()` against `CALIBRATED_REGIMES`,
  whose **only** entry is `minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4`
  (`thresholds.rs:61-62`).
- The slice that satisfies it **cannot carry an artifact** — two contract-resident probes
  are uncomputable on a 97-row vocab closure with 64 position rows (F-10, measured by
  04-05).
- The production pin computes every probe and returns `UncalibratedRegime`. 04-CONTEXT
  lists this as a deliberate Phase 5 item: *"Phase 4 does not silently widen the regime."*

Both halves of that are asserted structurally by the second ignored test, so a fixture-
estate change or a calibration edit turns it red and points its author here.

## What DID land

Five of the plan's six `must_haves.truths` are delivered and tested.

| Truth | Status |
| ----- | ------ |
| 1. trains from Phase 2 artifacts and atomically writes a verified `setfit-apr-v1` | **BLOCKED** — everything up to the write is implemented and driven; see THE FINDING |
| 2. config-file-first; overrides through the PUBLIC validated merge API; unknown field or invalid merged value fails BEFORE training | **YES** — 6 tests |
| 3. the MERGED resolved config is what gets reported (D-07) | **YES** at the report tier — `ResolvedConfigReport { requested: <merged>, resolved_device }` on both the dry-run and completion paths. The ARTIFACT-tier half rides on truth 1 |
| 4. `--device cuda` on a CPU-only host fails closed, nonzero exit | **YES** — asserted including `exit_code() != SUCCESS`; a silent-fallback mutation turns it red |
| 5. `setfit_io::read_setfit_apr_file_bounded` is the ONE filesystem door; stats first, refuses over-cap before reading, reads through core's bounded reader | **YES** — 5 tests |
| 6. existing output refused without `--force`; a failed write leaves no partial file | **YES** — plus the ordering assertion that the refusal happens before `--data` is opened |

### The exact feature line

```toml
setfit = ["training", "aprender/setfit", "entrenar?/setfit"]
```

`crates/apr-cli/Cargo.toml:109`, with a comment noting that 04-08 appends
`"realizar?/setfit"`. **Not** in `default`. `toml = { workspace = true }` was added to
`[dependencies]`.

**T-04-SC measured, not assumed:** `git diff Cargo.lock` is `1 file changed, 1 insertion(+)` —
a single edge `toml 0.8.23` inside apr-cli's dependency block. **Zero new `[[package]]`
entries**; the package was already in the workspace lock.

### `setfit_io`'s public signature

```rust
// crates/apr-cli/src/setfit_io.rs:67
pub(crate) fn read_setfit_apr_file_bounded(path: &Path) -> Result<Vec<u8>, CliError>
```

`pub(crate)` and not `pub`: every consumer (04-07's predict/inspect/eval, 04-08's serve
startup) is in this crate, and a wider door would be a supported entry point nobody asked
for. The module header states the rule for those plans in one sentence — *no code in
`apr-cli` may call `fs::read` on an artifact path* — and the module's own tests assert it
does not contain that call itself.

Error mapping: `FileNotFound` (3) for an absent path, `NotAFile` (3) for a directory or
FIFO, `InvalidFormat` (4) for an over-cap file naming which of the library's two length
checks fired, `ModelLoadFailed` (6) for a read failure or an unknown `#[non_exhaustive]`
variant, `Io` (7) for a stat failure that is neither.

## Test Counts (scoped filters, status captured directly, never through a pipe)

Every command ran as `cmd > log 2>&1; rc=$?` (CLAUDE.md verification rule 1). Every count
is a stated non-zero number against a stated minimum (F-04).

```
$ CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit_io
rc=0     5 passed, 6707 filtered out           (Task 1 criterion: >= 1)

$ CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit_train
rc=0    14 passed, 2 ignored, 6696 filtered    (Task 2 criterion: >= 9)

$ CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib setfit_train -- --ignored
rc=0     2 passed, 6710 filtered out           (Task 3's e2e leg)

$ CARGO_INCREMENTAL=0 cargo test -p apr-cli --features setfit --lib
rc=0  6696 passed, 16 ignored, 0 FAILED

$ CARGO_INCREMENTAL=0 cargo check -p apr-cli --features setfit
rc=0
$ CARGO_INCREMENTAL=0 cargo check -p apr-cli            # feature OFF
rc=0
$ CARGO_INCREMENTAL=0 cargo clippy -p apr-cli --features setfit --lib --all-targets --no-deps
rc=0    (0 diagnostics naming setfit_train.rs / setfit_io.rs / setfit_commands.rs)
$ cargo fmt -p apr-cli -- --check
rc=0
```

**The whole-suite delta is a difference between two measurements, not a subtraction from a
quoted figure.** Test binary before this plan: **6691**. After: **6712** (6696 passed +
16 ignored) = **+21** — 5 `setfit_io`, 14 `setfit_train`, 2 ignored e2e. Zero failures at
any point, so there is no known-red set to diff for this crate.

`--no-deps` on the clippy leg is F-03's requirement; without it the run exits 101 on
`aprender-compute`'s pre-existing debt and "no findings in my crate" is indistinguishable
from "my crate was never linted".

## Guards Shown Able To Fail (CR-02 / F-04 discipline)

Six mutations, each reverted immediately with `cp /tmp/<file>.pristine <file>` and the
green baseline re-measured after every revert. A test filter matching zero tests exits 0,
so each of these was watched fail **by name**.

| # | Mutation | Result | Killed by |
| - | -------- | ------ | --------- |
| — | baseline (`setfit_io`) | 5 passed, rc=0 | — |
| I1 | `read_setfit_apr_bytes_bounded(file, None)` instead of the stat'd length | **3 passed, 2 FAILED, rc=101** | `setfit_io_reads_through_the_library_door_and_passes_the_statted_length` **and** `setfit_io_refuses_an_over_cap_file_from_its_declared_length` |
| — | baseline (`setfit_train`) | 13 passed, rc=0 | — |
| M1 | `merge_overrides` drops the `--seed` override | **12 passed, 1 FAILED** | `setfit_train_seed_override_is_reflected_in_the_merged_config` |
| M2 | output pre-flight moved AFTER the Phase 2 ingest | **12 passed, 1 FAILED** | `setfit_train_refuses_an_existing_output_before_it_reads_the_data` |
| M3 | `CudaNotAvailable` silently returns `Device::Cpu` | **12 passed, 1 FAILED** | `setfit_train_device_cuda_fails_closed_with_a_nonzero_exit_code` |
| M4 | the dry run leaves a file behind | **12 passed, 1 FAILED** | `setfit_train_dry_run_validates_the_real_inputs_and_writes_nothing` |
| — | all reverted | 5 / 14 passed, rc=0 | — |

**I1 is the one that earned its keep, and it is BEHAVIOURAL, not a source scan.** With
`None` the sparse over-cap file is genuinely read before being refused, and the failure
message says so:

```
artifact is larger than the contracted cap — stream observed 268435457 bytes against a
cap of 268435456
```

`stream`, not `declared_length`. That is review B5's exact defect, reproduced and killed:
the bound still fires, and 256 MiB of a hostile file has already been read by the time it
does. A source assertion alone would have proven only that a call looks right.

M4's first draft (`return Ok(())` deleted) did not compile — borrowck rejects the moved
`resolved` — so it was replaced with a realistic version (the dry run writes a plan file),
which the tempdir listing comparison catches. Recorded because a mutation that fails to
compile is not evidence about a test.

## Deviations from Plan

### Auto-fixed

**1. [Rule 3 — blocking] `data_contrastive.rs` was edited: two `fn` visibilities widened to `pub(crate)`**

- **Found during:** Task 2, wiring the Phase 2 ingest.
- **Plan text:** *"Load Phase 2 inputs through the existing attested ingest doors …
  reuse, never re-parse JSONL by hand."* `data_contrastive.rs` is not in the plan's
  `files_modified`.
- **Issue:** `read_attested_canonical` and `read_selection_manifest` are module-private.
  The plan's own instruction is not satisfiable without a visibility change.
- **Alternatives rejected:** a second ingest sequence in `setfit_train.rs` would make two
  readers of `benchmark-manifest.json` — the exact defect the comment inside
  `read_attested_canonical` exists to prevent — and two places for the selection
  envelope's digest check to be omitted. A training run could then accept a directory
  `apr data select` refuses.
- **Fix:** `pub(crate)` on both, with the reason recorded on each. **No behaviour
  changed**: the diff is two `fn` keywords and doc comments.
- **Commit:** `4b6d1c49f`

**2. [Rule 3 — blocking] `Cargo.lock` is in the diff, and the wave contract says apr-cli paths only**

- **Issue:** adding `toml = { workspace = true }` changes the lock. Leaving it
  uncommitted would leave the manifest and lock disagreeing (any `--locked` build fails)
  and the change would be destroyed when the orchestrator removes the worktree.
- **Why the conflict risk is acceptable:** the delta is **one line** — an edge to an
  already-locked package — inside apr-cli's own dependency block. 04-12 creates one
  integration-test file and 04-16 edits `aprender-train/src`; neither changes a
  dependency, so neither touches that block.
- **Commit:** `089c6d167`

**3. [Rule 3 — blocking] Task boundary moved: the `ExtendedCommands` variant and dispatch arm went to Task 2**

- **Plan text:** Task 1 steps 4-5 place both in Task 1, dispatching to
  `commands::setfit_train::run` — which Task 2 creates.
- **Issue:** Task 1's commit would either not compile (calling an absent module) or would
  ship a user-visible `apr setfit train` that parses and does nothing. F-08 recorded an
  intermediate commit that did not build alone as a defect; 04-05 rearranged its task
  split for the same reason.
- **Fix:** Task 1 ships the feature, the enum and `setfit_io`; Task 2 ships the variant,
  the dispatch arm and the adapter together. Both commits compile gated and ungated and
  pass their scoped tests on exactly the tree that was committed.

**4. [Rule 2 — missing critical] The output no-clobber gate runs BEFORE training as well as at the write**

- **Plan text:** step 6 puts the `--force` gate in the atomic write.
- **Issue:** a run that trains for twenty minutes and then refuses to write is hostile,
  and on the intended (pinned-weights) path that is exactly what would happen.
- **Fix:** `refuse_existing_output` is called at step (2), before `--data` is opened, and
  again inside `atomic_write`. The second is not redundancy: it is what makes the
  guarantee true for a file that appeared while the run was going. The ordering is
  asserted by a test that passes a nonexistent `--data` and requires the error to name
  the OUTPUT.
- **Commit:** `4b6d1c49f`

**5. [Rule 2 — missing critical] `--dry-run` states what it does NOT check**

- **Issue:** the plan says a dry run stops after the device probe and input validation.
  As written that leaves a reader free to assume `--model-dir` was validated. It was not,
  and validating it would cost a multi-hundred-megabyte read — which is not a pre-flight.
- **Fix:** the report carries `checks_performed` (4 items) and `checks_skipped` (3 items),
  both rendered in JSON and human output, and the flag's `--help` says the same.
- **Commit:** `4b6d1c49f`

### Scope deviation (surfaced, not fixed)

**6. Task 3's three acceptance criteria are NOT met, and are not claimed.**

The plan asks the e2e to assert the output file exists, that it loads through
`load_setfit_apr`, that the reported hash equals the file's, and that a second `--force`
run is byte-identical. All four require a written artifact, which THE FINDING (plus F-10)
withholds. What shipped instead:

- the e2e **does** run over real Phase 2 artifacts — a benchmark directory built by
  actually running `apr data tweet-eval-stance` over a synthetic source, and a selection
  manifest built by actually running `apr data select` — and asserts every adapter stage
  up to the encoder door;
- the byte-identity property is asserted at the reachable tier (`atomic_write` →
  `setfit_io` round trip, then a `--force` rewrite);
- both blockers are asserted **structurally, by name**, so closing either turns a test red.

Task 3's `read_first` also names `aprender-train`'s `test_fixtures.rs` as the cheap path
to a real lifecycle. That module is `#[cfg(test)]` and its header says explicitly that it
stays that way ("plan 03-10's acceptance criteria reject a `#[doc(hidden)]` test-support
door on the shipped surface"), so it is unreachable from `apr-cli` by design.

---

**Total deviations:** 5 auto-fixed (3 blocking, 2 missing-critical) + 1 surfaced scope
deviation. No scope creep: nothing outside `crates/apr-cli/` (plus the one-line lock
edge) was touched, and no new registry package was added.

## Threat Register, as shipped

| Threat ID | Mitigation as shipped |
| --------- | --------------------- |
| T-04-17 (config injection / unvalidated merge) | `deny_unknown_fields` on the library's wire form plus `to_request` → `new`, so the merged whole is revalidated. Four tests: unknown key, invalid value, invalid device override, seed override. M1 shows a dropped override is caught |
| T-04-18 (silent device fallback) | `resolve_requested_device` runs at step (3), before any data is read. `--device cuda` on this CPU-only host is refused with a nonzero exit code, asserted. M3 shows the silent fallback is caught |
| T-04-19 (partial / clobbered output) | Temp-in-destination + `sync_all` + one `fs::rename`, `create_new` on the temp, cleanup on every error path that never masks the original error. Source-asserted: exactly 1 rename site, 0 `File::create`. The pre-rename fault seam proves no partial file and no stray temp |
| T-04-50 (unbounded artifact read) | `setfit_io::read_setfit_apr_file_bounded`; over-cap refused from the declared length before a byte is read. I1 shows the alternative reads 256 MiB first |
| T-04-SC (package installs) | Zero new packages — measured on `Cargo.lock` (one edge, no `[[package]]` entry) |

## Notes for Later Plans

- **04-07 / 04-08.** `crate::setfit_io::read_setfit_apr_file_bounded` is the door. Do not
  call `fs::read` on an artifact path; the module's own test bans it there and the same
  ban should be extended to your files. The feature line already carries the comment
  telling 04-08 where to append `"realizar?/setfit"`.
- **04-10 (gates).** Counted filters, all requiring `--features setfit`:
  `setfit_io` (**5**), `setfit_train` (**14**, plus 2 ignored). The ignored pair needs its
  OWN invocation — `… --lib setfit_train -- --ignored` — because libtest takes one
  positional filter. The clippy leg needs `--no-deps` (F-03). Both `cargo check -p apr-cli`
  and `--features setfit` must be legs: the ungated build is what proves the gating.
- **04-11 (requirements audit).** **Nothing was flipped in REQUIREMENTS.md.** OPS-02's
  train leg is not deliverable until THE FINDING is closed, and marking it would put a
  false claim in the traceability table — the Phase 2 policy ("mark each at the plan that
  actually closes it") applied here. OPS-06's CLI half IS delivered and tested; whether
  that closes the requirement depends on plans this one does not own, and REQUIREMENTS.md
  is shared across three concurrent worktree agents.
- **04-12 / 04-16.** You are in `aprender-train`. **THE FINDING is yours to close if the
  phase wants OPS-02's train leg this phase** — 04-12's plan predicts "take the artifact
  bytes" at its step 3 and will hit the same wall. One `into_artifact_bytes` door serves
  both plans.
- **Phase 5.** The calibrated-regime widening remains the documented blocker. Note that
  `apr setfit train` will report `UncalibratedRegime` with a remedy that explicitly says
  widening the set is a deliberate contract edit, never an inline change.

## Known Stubs

**One, and it is deliberate and typed:** `verified_artifact_bytes` always returns
`CliError::Aprender` carrying `ARTIFACT_BYTES_GAP`. It is not a placeholder value flowing
into a report — no artifact path, no fake hash and no success is ever produced. The
command fails, loudly, with the exact library door that must land. Everything else on the
path (config parse, merge, device probe, ingest, replay, the four lifecycle transitions,
the atomic writer, both report renderers) is fully wired and exercised.

## Threat Flags

None. No new network endpoint, auth path or schema at a trust boundary. The two file
surfaces this plan adds are both narrower than what existed: artifact reads are bounded
before allocation, and artifact writes go through a single atomic site.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/apr-cli/src/setfit_commands.rs
FOUND: crates/apr-cli/src/setfit_io.rs
FOUND: crates/apr-cli/src/commands/setfit_train.rs
FOUND: crates/apr-cli/Cargo.toml           (setfit feature at line 109)
FOUND: crates/apr-cli/src/lib.rs           (mod setfit_io + include!)
FOUND: crates/apr-cli/src/extended_commands.rs   (Setfit variant, cfg-gated)
FOUND: crates/apr-cli/src/dispatch_analysis.rs   (arm + dispatch_setfit_command)
FOUND: crates/apr-cli/src/commands/mod.rs
FOUND: crates/apr-cli/src/commands/data_contrastive.rs
```

Commits claimed, checked in the log: `089c6d167`, `4b6d1c49f`, `67843ce5a`.

Source assertions, measured on the committed tree:

| Assertion | Criterion | Observed |
| --------- | --------- | -------- |
| `grep -c "SetfitCommands" extended_commands.rs` | >= 1, cfg-gated | **1**, under `#[cfg(feature = "setfit")]` |
| `SetfitCommands` variant count (D-06) | exactly 1 | **1** (`Train`) |
| `setfit_io.rs` passes the stat'd length to core's bounded reader | present | asserted in-test; I1 falsifies it |
| `setfit_train.rs` contains `to_request` | >= 1 | asserted in-test |
| `setfit_train.rs` names any `*Wire` type | 0 | asserted in-test (two needles) |
| `fs::rename` sites in `setfit_train.rs` | exactly 1 | asserted in-test |
| `File::create` outside the atomic helper | 0 | asserted in-test |
| `git diff --diff-filter=D --name-only HEAD~3 HEAD` | empty | **empty** — no deletions |
| `git diff --name-only HEAD~3 HEAD` | apr-cli paths + Cargo.lock | 9 apr-cli files + `Cargo.lock` (deviation 2) |
| `git status --short --untracked-files=all` | empty | **empty** |
| `STATE.md` / `ROADMAP.md` modified | no | **not touched** — the orchestrator owns them |

---
*Phase: 04-apr-artifact-and-production-parity*
*Completed: 2026-08-15*
