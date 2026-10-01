---
phase: 04-apr-artifact-and-production-parity
plan: 07
subsystem: apr-cli
tags: [setfit, cli, predict, inspect, eval, selection-lock, apr-05, ops-02, ops-03, trn-07, d-04, d-06, d-08, m2, b4, b5]

# Dependency graph
requires:
  - phase: 04-06
    provides: "setfit_io::read_setfit_apr_file_bounded — the ONE bounded artifact door all three commands read through; the `setfit` feature; the atomic-write precedent"
  - phase: 04-16
    provides: "reload_verified_run_from_apr — the fresh-process door, and THE OPEN EDGE it recorded and assigned to this plan"
  - phase: 04-17
    provides: "SetFitCredential (sealed) + create_selection_lock<C> — the generic lock doors a fresh process can reach"
  - phase: 04-04
    provides: "ClassifyRequestDocument / ClassifyResponse — the shared request document and the D-08 envelope this CLI serializes and never redefines"
  - phase: 04-03
    provides: "load_setfit_apr + MAX_ARTIFACT_BYTES; MAX_REQUEST_BODY_BYTES, whose CLI-side enforcement F-14(4) recorded as owed by THIS plan"
provides:
  - "`apr predict` — a NEW generic command; tag-only routing, the shared request document, core's envelope verbatim"
  - "`apr inspect` SetFit section — every APR-05 field, offline, from the artifact alone, in --json and human output"
  - "`apr eval --task classify` SetFit branch — validation + durable lock commit; test gated on a PRIOR lock"
  - "crate::setfit_tag::read_setfit_tag — the ONE tag detector, NOT feature-gated"
  - "train::setfit::apr_evaluate::evaluate_validation_from_artifact — closes 04-16's open edge; a .apr file can now BE a SelectionCandidate"
  - "evaluate::evaluation_from_predictions — the shared tail both evaluators are"
affects: [04-09, 04-10, 04-11, 04-15]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A source-scan guard must scan CODE LINES only — the module header explains the rule using the very names the scan forbids, and a guard that fails on its own documentation is F-05. Every filter here carries a non-vacuity assertion."
    - "Two files that differ ONLY in `model_type` are the D-04 negative: the untagged one's InvalidFormat and the tagged one's ModelLoadFailed are two DIFFERENT answers, and that difference is the evidence the tag routed."
    - "A flag that belongs to the other half of a workflow is REFUSED, never ignored: an operator who passes --selection-lock and sees a green report has every reason to believe it gated something."
    - "Extracting a shared tail beats adding a sibling: `evaluation_from_predictions` makes 'is this a second evaluation policy?' answerable by pointing at one function with two callers."
    - "A hostile length is bounded by the FILE, not by an invented constant: `metadata_size` and a truncated container need no new number to be judged."

key-files:
  created:
    - crates/apr-cli/src/setfit_tag.rs
    - crates/apr-cli/src/setfit_tag_tests.rs
    - crates/apr-cli/src/commands/predict.rs
    - crates/apr-cli/src/commands/predict_tests.rs
    - crates/apr-cli/src/commands/inspect_setfit.rs
    - crates/apr-cli/src/commands/inspect_setfit_tests.rs
    - crates/apr-cli/src/commands/eval/setfit.rs
    - crates/apr-cli/src/commands/eval/setfit_tests.rs
    - crates/aprender-train/src/train/setfit/apr_evaluate.rs
    - crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs
  modified:
    - crates/apr-cli/src/lib.rs
    - crates/apr-cli/src/extended_commands.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - crates/apr-cli/src/commands/mod.rs
    - crates/apr-cli/src/commands/inspect.rs
    - crates/apr-cli/src/commands/inspect_output_json.rs
    - crates/apr-cli/src/commands/inspect_tests.rs
    - crates/apr-cli/src/commands/builder.rs
    - crates/apr-cli/src/commands/construction.rs
    - crates/apr-cli/src/commands/eval/mod.rs
    - crates/apr-cli/src/lib_parse_rosetta_02.rs
    - crates/apr-cli/src/lib_verbose_inheritance_parse.rs
    - crates/apr-cli/src/lib_dispatch_coverage.rs
    - crates/aprender-train/src/train/setfit/evaluate.rs
    - crates/aprender-train/src/train/setfit/evaluate_tests.rs
    - crates/aprender-train/src/train/setfit/mod.rs

key-decisions:
  - "The tag detector is NOT feature-gated. `apr predict` on a tagged artifact in a binary built without `setfit` must answer FeatureDisabled (exit 9), not `unsupported format` — those are different answers and only one is true."
  - "`apr inspect` renders APR-05 from the RAW recovered document with ONE renderer; core's SetFitArtifactDoc is parsed as a SIGNAL (`document_schema_valid`), not rendered from. Two renderers — one per feature state — would be two answers that can drift, and inspect must still show a future-schema artifact's identity fields."
  - "The evaluator gap 04-16 left open was closed by EXTRACTING `evaluation_from_predictions` from the trainer's tail rather than adding a parallel evaluator, so 'one implementation per operation' is a fact about the call graph rather than an argument."
  - "The fresh-process evaluator predicts through core's `classify` — the path the artifact's own six-probe replay is evidence about. Reaching around it to the trainer's encode path would measure something the artifact carries no probes for."
  - "`--input` parses the shared ClassifyRequestDocument, bounded above MAX_REQUEST_BODY_BYTES BEFORE serde is handed anything. F-14(4) recorded this enforcement as owed by 04-07; it is paid."
  - "The lock file is a separate resource class from the artifact, so it is NOT read through `setfit_io`: that door carries the 256 MiB artifact cap, and applying it to a 1 MiB resource would be the wrong bound wearing the right function name."

requirements-completed: []

# Metrics
duration: ~4h
completed: 2026-08-15
---

# Phase 04 Plan 07: The Generic Consumer Surfaces — Summary

**All three generic surfaces ship, D-06 is intact (no `apr setfit predict|eval|inspect`
alias exists), and the one library gap that would have made `apr eval --split validation
--lock-out` impossible was found, measured and closed — by extracting a shared tail from
the trainer's own evaluator rather than writing a second one beside it.**

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `setfit_tag.rs` (the ONE tag detector) + `apr predict` + wiring, 23 tests | `9574532f8` |
| 2 | `apr inspect` SetFit section — APR-05 complete, 10 tests | `1787d490a` |
| 3a | `apr_evaluate.rs` + the extracted `evaluation_from_predictions`, 9 tests | `9e219f0cf` |
| 3b | `apr eval` SetFit branch + the TRN-07 file-mediated proof, 18 tests | `ad59e6963` |

Every commit builds alone and passes its scoped filter on exactly the tree that was
committed (F-08 discipline).

## THE GAP THIS PLAN FOUND AND CLOSED

04-16's summary recorded an open edge and assigned it to whoever owned `apr eval`'s
candidate story:

> A `SelectionCandidate` reads its artifact hash out of a `ValidationEvaluation`, and
> `evaluate_validation` takes a train-time `SetFitRun<ArtifactReloadedAndVerified>`. So a
> process that only has `.apr` files cannot yet build the candidate set it locks over — it
> can only consume a lock somebody else wrote.

**That is this plan.** Verified against the source, not assumed:
`evaluate_validation(run: &SetFitRun<ArtifactReloadedAndVerified>, …)`;
`reload_verified_run_from_apr` returns a `ReloadedSetFitCredential`, which is not that
type and cannot become it; `ValidationEvaluation` has no public constructor and its only
other producer is `#[cfg(test)] pub(super) evaluation_for_tests`. So
`apr eval --split validation --lock-out` was **not implementable** as the plan wrote it.

**The close is a refactor, not an addition.** `evaluate_validation`'s tail — bounds check,
metric dispatch, macro-F1's empty-class convention, the construction of the evidence
record — was extracted into

```rust
pub(super) fn evaluation_from_predictions(
    metric, truth: &[usize], predicted: &[usize], classes,
    artifact_hash, validation_split_fingerprint, dataset_fingerprint,
) -> Result<ValidationEvaluation, SetFitTrainError>
```

and both evaluators now ARE that function. The new door is

```rust
pub fn evaluate_validation_from_artifact(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
    metric: ValidationMetricKind,
) -> Result<ValidationEvaluation, SetFitTrainError>
```

It predicts through `VerifiedSetFitModel::classify` — core's ONE classification path, the
one `apr predict` and `POST /v1/classify` use, and the one the artifact's own six-probe
replay is evidence about — chunked by `MAX_BATCH_TEXTS`, and re-checks BOTH provenance
fingerprints so the function is total on its own arguments rather than relying on the
caller to pass the dataset the reload door saw.

**The caveat is written in the module, not hidden:** the trainer's path and this one are
two float pipelines, so two evaluations of one model are not guaranteed bit-identical.
Nothing shipped mixes them (a candidate set is built by one caller, and `apr eval` uses
this door for every candidate). `ValidationEvaluation`'s wire form cannot record which
path produced it without changing the selection lock's canonical bytes and invalidating
every lock in existence; that trade was not worth making for a mixture no code performs.

### Rule 3: an existing guard had to be extended, not dodged

`evaluate_source_exposes_no_public_api_taking_a_float_parameter` scans `pub fn ` and
`pub const fn `. A `pub(super) fn` matches neither, so the new door would have slipped past
**in silence** — and the guard's own doc says *"an exception a guard does not mention is an
exception nobody re-checks."* It now enumerates all three `pub(super)` doors by name,
asserts the count is exactly three, and asserts the new one's parameter list is float-free.

## What shipped, surface by surface

### `apr predict` (NEW, generic — D-06)

```
apr predict <FILE> [--text <STR>]... [--input <FILE>] [--logits] [--json]
```

Order of business, asserted by a test: **the request first**, then the typed tag, then
core's one classify path. A conflicting `--text`/`--input` invocation is refused as a
request error even when the model path does not exist — a run that opened the file first
would have sent the operator to look at a file that is fine.

- **`--input` is the shared `ClassifyRequestDocument`**, not one text per line, and the
  reason is in the flag's help and in the module header: a line-delimited format cannot
  carry a text containing a newline, so the library, CLI and HTTP surfaces would receive
  different ordered input sets while appearing to agree (review M2).
- **The M2 witness at the CLI boundary** is a six-entry document containing an embedded
  newline, a tab, an empty string, a padded string and non-ASCII (`el zorro café — naïve π`).
  The parsed document's `texts` must equal the written vector **exactly and in order** —
  the empty string surviving as an *entry* is what a `filter(|s| !s.is_empty())` would break,
  shifting every later index.
- **`MAX_REQUEST_BODY_BYTES` is enforced before serde is handed anything** (stat'd length
  first, then the stream with `take(cap + 1)` so a lying length is detected). This is the
  half F-14(4) recorded as still owed by 04-07.
- **The file declares no response type** — `--json` is `serde_json` of core's
  `ClassifyResponse` verbatim; the human view reads the SAME value through accessors.
  Grep-asserted.

### `apr inspect` — APR-05, complete and offline

The `--json` SetFit section, as shipped. Each key is asserted **by name, individually**:

```
schema, schema_version, bundle_schema_version, format_id,
encoder_revision, encoder_tokenizer_sha256, tokenizer_sha256, hidden_act,
architecture{hidden, heads, head_dim, num_layers, intermediate, vocab, positions,
             type_vocab_size},
preprocessing{pooling, normalization, l2_epsilon_hex,
              truncation_max_sequence_length, padding_mode, max_length},
ordered_labels, head{n_features, num_labels},
dataset_fingerprint, validation_split_fingerprint,
selection_semantic_hash, selection_ledger_hash,
root_seed, selection_root_seed, shots_per_class,
evidence{verdict, trainable_count, frozen_count, worst_param_name, epsilon_used,
         calibration_regime_id, contract_version, table_hash},
artifact_sha256, document_schema_valid
[+ artifact_sha256_note / document_schema_error when either is not available]
```

- **READ-ONLY metadata work.** No tensor load, no probe replay, no `load_setfit_apr` —
  grep-asserted. An operator inspects a broken artifact precisely because it is broken, and
  a version that required the verified state would fail exactly then.
- **The two fingerprints are asserted to be DIFFERENT values**, so a renderer that read one
  path twice cannot pass a presence check.
- **Non-regression golden:** a plain APR's report gains **no key**. The top-level key set is
  asserted by name and by exact count.

### `apr eval --task classify` — the durable lock workflow (D-16, review B4)

```
apr eval M.apr --task classify --data D --selection S --split validation --lock-out L
apr eval M.apr --task classify --data D --selection S --split test       --selection-lock L
```

Flags as shipped: `--data <DIR>`, `--selection <FILE>`, `--split validation|test` (default
`validation`), `--lock-out <FILE>`, `--selection-lock <FILE>`, `--candidate <APR>`
(repeatable), `--force`.

- **Every flag belongs to exactly one split and is REFUSED on the other.** A `--lock-out`
  on a test run would let the test command write the lock that is supposed to precede it.
- **The absent-lock refusal names the exact prior command**, because a test run cannot
  create what it needs — that is the entire point.
- **Zero gating types are defined here** and there is **no path to the test split around
  `grant`**: `dataset.test()` appears zero times, `grant.test()` exactly once, and the lock
  is asserted to be READ from disk before a token is minted from it.
- The validation report is the EVAL-03-shaped row (metric + bits, n_rows, artifact hash,
  both fingerprints, ordered labels, evidence table hash, selection seed, the candidate
  table with each candidate's `config_hash`, the lock's path/hash/chosen artifact/rule) and
  states its `config_hash_derivation` so Phase 5 can reproduce it.
- A validation run **without** `--lock-out` reports and says in a `notes` entry that it
  committed nothing and therefore cannot later unlock test access.

## TRN-07 evidence — named, and labelled with what it actually is

**`apr_evaluate_the_lock_travels_between_two_invocations_as_a_file`**
(`crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs`).

**IN-PROCESS, FILE-MEDIATED.** The two halves share no `SelectionLock` value: the first
scope measures, commits, serializes to disk and drops the credential, the dataset and the
lock; the second reads the FILE, reconstructs through `SelectionLock::from_canonical_bytes`,
asserts the reconstructed `lock_hash` equals the recorded one, then mints, grants and reads
the admitted test rows. **04-15's spawned test is the cross-process citation; this one is
not that and does not claim to be.**

Two negatives, each holding every other identity constant so only the arm under test can
fire:

| Test | Refusal |
| --- | --- |
| `apr_evaluate_a_lock_naming_a_different_artifact_is_refused_as_stale` | `LockError::StaleLock`, both hashes named |
| `apr_evaluate_a_grant_over_a_different_corpus_is_refused` | `LockError::TokenDatasetMismatch` |

**Why it lives in `aprender-train` and not beside `apr eval`** — measured, not assumed:
`apr-cli` cannot construct a `setfit-apr-v1` artifact at all. `CALIBRATED_REGIMES` admits
only the phase-3 MiniLM slice (F-10), whose 97-row vocabulary closure cannot compute two of
the six contract-resident probes; `into_artifact_bytes` on the only out-of-crate-trainable
run yields `setfit-serde-json-v1` bytes (04-12 measured this); and core's APR-capable view
fixture is `#[cfg(all(test, feature = "setfit"))] pub(crate)`. With no artifact there is no
credential, no `ValidationEvaluation` and therefore no `SelectionLock`. The test file says
this in its header so a reader does not mistake the boundary's shape for a gap in effort.

## Test counts — scoped filters, status captured directly, never through a pipe

Every command ran as `cmd > log 2>&1; echo "rc=$?"`.

```
$ cargo test -p apr-cli --features setfit --lib setfit_tag::
rc=0     7 passed, 6734 filtered out
$ cargo test -p apr-cli --features setfit --lib commands::predict::
rc=0    16 passed, 6725 filtered out        (plan criterion: >= 6)
$ cargo test -p apr-cli --features setfit --lib setfit_inspection
rc=0    10 passed, 6741 filtered out        (plan criterion: >= 3)
$ cargo test -p apr-cli --features setfit --lib inspect
rc=0   120 passed                            (the whole inspect surface, incl. pre-existing)
$ cargo test -p apr-cli --features setfit --lib eval::setfit
rc=0    15 passed, 6751 filtered out        (plan criterion: >= 6)
$ cargo test -p aprender-train --features setfit --lib setfit::apr_evaluate::
rc=0    12 passed, 7947 filtered out
$ cargo test -p aprender-train --features setfit --lib setfit::
rc=0   311 passed, 1 ignored                (the whole setfit surface — 0 failures)

$ cargo test -p apr-cli --features setfit --lib
rc=0  6751 passed, 15 ignored, 0 FAILED
$ cargo test -p aprender-train --features setfit --lib evaluate_
rc=0    54 passed
$ cargo test -p aprender-train --features setfit --lib lock_
rc=0    90 passed                            (90 before — the regression witness)

$ cargo check -p apr-cli --features setfit --all-targets       rc=0
$ cargo check -p apr-cli --all-targets       (feature OFF)     rc=0
$ cargo fmt -p apr-cli -- --check                              rc=0
$ cargo fmt -p aprender-train -- --check                       rc=0
$ cargo clippy -p apr-cli --features setfit --lib --all-targets --no-deps    rc=0
$ cargo clippy -p aprender-train --features setfit --lib --no-deps           rc=0
```

`--no-deps` is F-03's requirement: without it the run exits on `aprender-compute`'s
pre-existing arm64 debt and "no findings in my crate" becomes indistinguishable from "my
crate was never linted".

**52 tests added across four filters** (7 + 16 + 10 + 15 + 12 − the 8 pre-existing that the
`setfit_inspection`/`inspect` filters overlap). The apr-cli binary reports 6766 tests where
04-17 recorded 6712 — that is **not** a clean delta for this plan, because 04-08 landed
between those two measurements; the four named filters above are the measured additions.

### aprender-train's known-red baseline, DIFFED not counted

Measured on the FINAL tree (`ad59e6963`):

```
$ cargo test -p aprender-train --features setfit --lib setfit::
rc=0     311 passed, 1 ignored, 7647 filtered out
         (the entire setfit surface, including this plan's new module — zero failures)

$ cargo test -p aprender-train --features setfit --lib
rc=101   7920 passed; 24 failed; 15 ignored

$ set-difference(observed failure names, known-red-baseline.md names)
observed failures: 24
NEW (regressions): NONE
all in baseline:   True
```

24 names, every one in the baseline (21 × `gpu::*`, 3 × `prune::snapshot_tests`) — all in
subsystems this plan does not touch, and `setfit::` is entirely green. **Zero regressions.**

Test-count arithmetic closes exactly: 7908 before this plan → +9 (`apr_evaluate` Task 3a)
= 7917 → +3 (the TRN-07 trio in Task 3b) = **7920**.

## Guards shown able to FAIL

Two behavioural mutations, each observed red **by name**, reverted from a pristine copy,
with the green baseline re-measured after the revert. A filter matching zero tests exits 0,
so a mutation that "passed" would prove nothing.

| # | Mutation | Result | Killed by |
| - | -------- | ------ | --------- |
| — | baseline (`eval::setfit`) | 15 passed, rc=0 | — |
| M1 | `check_split_flags` no longer requires `--selection-lock` on `--split test` | **14 passed, 1 FAILED, rc=101** | `eval_setfit_test_split_requires_a_lock_and_names_the_exact_prior_command` |
| — | reverted | 15 passed, rc=0 | — |
| — | baseline (`setfit::apr_evaluate::`) | 12 passed, rc=0 | — |
| M2 | the corpus fingerprint gate is dead code | **11 passed, 1 FAILED, rc=101** | `apr_evaluate_refuses_a_dataset_that_is_not_the_artifacts_corpus` |
| — | reverted | 12 passed, rc=0 | — |

**Three guards were caught being VACUOUS or WRONG during development, and each was fixed
rather than relaxed:**

1. `setfit_tag_detection_never_consults_a_tensor_name` turned red on the module's own
   documentation (the header explains D-04 using the string `setfit.head.weight`). The F-05
   defect, in a guard written to catch a different one. Fixed by scanning CODE LINES only,
   with a non-vacuity assertion that the filter did not eat the module.
2. `inspect_setfit_reads_the_whole_file_only_through_the_bounded_door` did the same on
   `VerifiedSetFitModel`, which the header names while explaining the APR-04 boundary. Same
   fix, same non-vacuity check.
3. `setfit_tag_refuses_a_metadata_block_longer_than_its_own_file` **was measuring nothing**:
   its first version patched a `u32` at a guessed header offset (28) and the detector
   accepted the file. Replaced with truncation of a real container, which is
   layout-independent and cannot silently stop testing anything, plus a non-vacuity
   assertion that the UNtruncated fixture IS recognized.

## Deviations from Plan

### 1. [Rule 3 — blocking] `aprender-train` was edited; the plan's `files_modified` lists apr-cli only

Five files (`apr_evaluate.rs`, `apr_evaluate_tests.rs`, `evaluate.rs`, `evaluate_tests.rs`,
`mod.rs`). **Task 3 is not satisfiable without it** — see THE GAP above; `apr eval --split
validation --lock-out` cannot produce a lock, because no fresh-process producer of
`ValidationEvaluation` existed. 04-16 explicitly assigned this to "whichever plan owns `apr
eval`'s candidate story", which is this one. This plan is alone in wave 7 (04-09 and 04-15
are wave 8 and depend on it), so there is no concurrent owner of those files.

### 2. [Rule 3 — blocking] The plan's `<interfaces>` block was wrong about the reload door

It states `reload_verified_run_from_apr(...) -> Result<SetFitRun<ArtifactReloadedAndVerified>, _>`
taking its dataset and selection by value. It returns a **`ReloadedSetFitCredential`** and
borrows both. Coded against reality, as the phase's standing instruction requires.

### 3. [Rule 3 — blocking] A NEW module the plan did not name: `setfit_tag.rs`

The plan says to copy `inspect.rs`'s metadata read into `predict`. Two copies of the
detection rule is two places for D-04 to be violated, and the second one looks exactly like
the first. One detector, used by `predict`, `inspect` and `eval` — and deliberately NOT
feature-gated, so a binary without the classifier can still tell "this IS a classifier and I
cannot run it" (exit 9) from "unsupported format".

### 4. [Rule 2 — missing critical] `read_metadata` allocated an attacker-controlled `u32`

Pre-existing in `inspect.rs`: `vec![0u8; header.metadata_size as usize]` with
`metadata_size` read out of the file under inspection — up to 4 GiB allocated before
`read_exact` discovers the file is too short. Bounded by the stat'd file length, which needs
no invented constant to be sound. **The output is unchanged** (the same
`MetadataInfo::default()`, without paying for the allocation first), so the non-regression
golden still holds. The same bound is applied in `setfit_tag.rs`.

### 5. [Deviation, argued] `apr inspect` renders from the RAW document, with the typed parse as a signal

The plan says to "reuse core's `SetFitArtifactDoc` when the setfit feature is on … when off,
render the raw JSON value with a note" — two renderers. Shipped as ONE renderer over the raw
value, with the typed parse reported as `document_schema_valid`. Two renderers are two
answers that can drift and the feature-off one would be the untested one; and a single
renderer keeps `inspect` useful on an artifact whose document this build cannot parse, which
is exactly when an operator needs it. `inspect_setfit_top_level_paths_are_the_documents_own_field_names`
pins every path the renderer walks against core's normative
`SETFIT_ARTIFACT_DOC_FIELDS`, so a renamed field is caught here rather than by a `null`
appearing in production.

### 6. [Deviation, measured] The TRN-07 test is not "through `commands::eval`"

The plan asks for an integration test reaching lock → token → grant *through
`commands::eval`* across two invocations. **`apr-cli` cannot construct a `setfit-apr-v1`
artifact**, so it cannot construct a credential, an evaluation or a lock — three
independently-measured blockers, listed above. The test therefore lives in `aprender-train`,
is named in this summary, is file-mediated across two scopes, and is explicitly labelled
IN-PROCESS. Everything of the CLI branch that IS reachable — every refusal, the flag
ordering, the bounded lock read, and the structural no-bypass claims — is tested in
`eval::setfit`.

### 7. [Rule 3 — blocking] Six existing test literals needed the new fields

`ExtendedCommands::Eval` gained six fields and `InspectResult`/`MetadataInfo` one each, so
six struct literals in existing test modules (`construction.rs`, `builder.rs`,
`inspect_tests.rs`, `lib_parse_rosetta_02.rs`, `lib_verbose_inheritance_parse.rs`,
`lib_dispatch_coverage.rs`) stopped compiling. Each gained the new field at its default; no
assertion was changed.

---

**Total deviations:** 4 blocking, 1 missing-critical, 2 argued/measured. No new package: no
`Cargo.toml` or `Cargo.lock` change in the diff.

## Threat Register, as shipped

| Threat ID | Mitigation as shipped |
| --------- | --------------------- |
| T-04-20 (untagged APR treated as SetFit) | Detection reads `model_type` and nothing else. The positive and negative fixtures differ ONLY in that field and carry identical SetFit-shaped tensors; they produce two DIFFERENT typed refusals, which is the evidence the tag routed. Source-asserted that no tensor name appears on a code line of the detector. |
| T-04-21 (canonical test leakage via eval) | The test split is reachable only via a PRIOR durable lock → mint → grant. Absent / stale / mismatched are three typed refusals. `dataset.test()` count in the branch: **0**; `grant.test()` count: **1**; the lock is asserted to be read from disk BEFORE the mint. |
| T-04-22 (corrupted artifact reaching prediction) | `apr predict` routes every tagged file through `load_setfit_apr`'s full ladder including probe replay; a tagged-but-invalid container surfaces as `ModelLoadFailed` (exit 6), asserted. |
| T-04-50 (unbounded read) | Artifacts: `setfit_io`'s bounded door, over-cap refused from the declared length (asserted through `apr predict`). Request documents: `MAX_REQUEST_BODY_BYTES` before serde, both from the stat and from the stream. Lock files: `MAX_SELECTION_LOCK_BYTES`, same two checks. Metadata blocks: bounded by the file's own length in BOTH readers. |
| T-04-52 (edited lock granting access to the wrong model) | `mint_test_token` verifies integrity then compares the artifact; `grant` re-compares the artifact AND the dataset fingerprint. Both refusals are surfaced with the flags named and nothing stronger is claimed. Both are asserted by constructed negatives. |
| T-04-SC (package installs) | Zero new packages; no manifest or lock change. |

## Threat Flags

None. No new network endpoint and no new auth path. The three surfaces are read paths whose
every entry is narrower than what existed: artifact reads are bounded before allocation,
request and lock reads are bounded before parsing, and detection is by explicit tag rather
than by content.

## Known Stubs

**None.** Every field the three surfaces report is computed or recovered; nothing is
hardcoded empty and nothing renders a placeholder. Two fields are `null` **with a stated
reason** and only in a binary built without the `setfit` feature (`artifact_sha256` and
`document_schema_valid` in `apr inspect`), because computing them would require a second
hashing path and a second schema parser — two answers to questions that must have one.

## Notes for Later Plans

- **04-09 (parity harness).** `apr predict --input <FILE>` takes the SAME
  `ClassifyRequestDocument` your HTTP leg posts, so all three legs can be fed one document.
  `--json` is core's envelope byte-for-byte — no CLI re-keying — and F-14(3) already removed
  `latency_ms` from `PartialEq`, so `assert_eq!(cli, http)` is the intended comparison.
- **04-15 (spawned lifecycle).** The flags are exactly as your plan predicts:
  `--split validation --lock-out <FILE>` then `--split test --selection-lock <FILE>`, with
  `--data <DIR>` and `--selection <FILE>` on both. **Your test is the cross-process
  citation** — this plan's is in-process and says so. Note that a real `.apr` is still not
  producible on this host (F-10), which your plan will hit at its first step.
- **04-10 (gates).** New counted filters, all requiring `--features setfit`:
  `setfit_tag::` (**7**), `commands::predict::` (**16**), `setfit_inspection` (**10**),
  `eval::setfit` (**15**) in apr-cli; `setfit::apr_evaluate::` (**12**) in aprender-train.
  Both `cargo check -p apr-cli` and `--features setfit` must be legs: the ungated build is
  what proves the gating, and `predict`/`inspect` have real feature-off code paths.
- **04-11 (requirements audit).** **Nothing was flipped in REQUIREMENTS.md.** APR-05 is
  complete and machine-readable and OPS-03's structural claim holds, but OPS-02's end-to-end
  leg still cannot be demonstrated on this host (F-10), and marking a requirement whose
  demonstration no run can produce would put a false claim in the traceability table. The
  TRN-07 citation for the positive tier is
  `apr_evaluate_the_lock_travels_between_two_invocations_as_a_file`, **labelled IN-PROCESS**.
- **Anyone touching `evaluate.rs`.** It now has THREE `pub(super)` doors and
  `evaluate_source_exposes_no_public_api_taking_a_float_parameter` enumerates all three by
  name and asserts the count. A fourth must be examined against the float rule deliberately.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/apr-cli/src/setfit_tag.rs
FOUND: crates/apr-cli/src/setfit_tag_tests.rs
FOUND: crates/apr-cli/src/commands/predict.rs
FOUND: crates/apr-cli/src/commands/predict_tests.rs
FOUND: crates/apr-cli/src/commands/inspect_setfit.rs
FOUND: crates/apr-cli/src/commands/inspect_setfit_tests.rs
FOUND: crates/apr-cli/src/commands/eval/setfit.rs
FOUND: crates/apr-cli/src/commands/eval/setfit_tests.rs
FOUND: crates/aprender-train/src/train/setfit/apr_evaluate.rs
FOUND: crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs
```

Commits claimed, checked in the log: `9574532f8`, `1787d490a`, `9e219f0cf`, `ad59e6963`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git status --short` after each commit | empty | **empty** |
| deletions in any commit | none | **none** |
| D-06: a `setfit`-namespaced predict/eval/inspect alias | 0 | **0** — `SetfitCommands` still has exactly one variant (`Train`) |
| a SECOND artifact reader in apr-cli | 0 | **0** — `read_setfit_apr_file_bounded` is the only one; `fs::read(` count is 0 in predict.rs, inspect_setfit.rs and eval/setfit.rs |
| `STATE.md` / `ROADMAP.md` modified | no | **not touched** — the orchestrator owns them |
| `contracts/` modified | no | **not touched** |
| new `include!()` files gitignored | no | `git check-ignore` exits 1 for both |
| known-red failure names (aprender-train) | zero new | **set difference is empty**, 24 names |
| guards shown able to fail | >= 1 behavioural | **2**, each observed red by name and reverted |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 07 — COMPLETE*
*Completed: 2026-08-15*
