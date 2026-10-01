---
phase: 02-deterministic-pair-and-data-protocol
plan: 06
subsystem: contrastive-data
tags: [attestation, d05-seam, thin-adapter, byte-parity, d18, d19, d27, schema-version, real-duplicate-golden, contract-growth]
requires:
  - "02-01 (the tracked D-06 baseline this relocation is diffed against, and the tweet-eval contract at v1.1.0)"
  - "02-03 (Split typestate + 5-gate ladder, both content hashes, DatasetFingerprint, ExclusionRecord, PreparedDataset<P>)"
  - "02-05 (PreparedJsonl::as_map, label_names(), the ledger conventions this reuses)"
provides:
  - "attestation.rs — DatasetAttestation + PreparedDataset::<Canonical|Compatibility>::from_attested_bytes, the crate-owned identity boundary"
  - "DATASET_ATTESTATION_SCHEMA_VERSION = 2 and SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS = [2], crate-owned, no version-1 shim"
  - "PreparedDataset::from_validated_splits — the SINGLE assembly point both ingest doors land in"
  - "ContrastiveDataError::UnsupportedNormalizationVersion"
  - "data_tweeteval.rs as a thin adapter on exactly the D-05 seam, with a read-back verification step"
  - "benchmark-manifest.json schema_version 2 with dataset_attestation + exclusions sections"
  - "contracts/tweet-eval-stance-benchmark-v1.yaml v2.0.0 — 9 obligations, 13 falsification tests"
  - "the real-duplicate golden, RUN and GREEN against the live pinned revision"
affects:
  - "02-09 (its data select / data pairs commands must read splits through from_attested_bytes; attestation_bytes_from_manifest is the read path they need)"
  - "02-08 (binding registry — two new #[contract] sites on dataset_attestation; OBLIG-CPP-ERROR-TAXONOMY grew one variant)"
  - "Phase 3 / Phase 5 (typed splits are now reachable without depending on any apr-cli module)"
  - "anyone holding a schema_version-1 benchmark directory — it is refused, deliberately, and must be re-prepared"
tech-stack:
  added: []
  patterns:
    - "verify the attested digest of a buffer BEFORE parsing it, so a substituted file is diagnosed as the wrong file rather than as a malformed row"
    - "stage the ledger and commit it only after every comparison passes, so a rejected dataset leaves no access record"
    - "the writer holds itself to the reader's gate: prepare re-opens what it wrote through the attested boundary and rolls back on rejection"
    - "re-derive a golden wire format by string formatting inside the test rather than by calling the encoder under test"
    - "name --lib in every contract test command, because a bare filter emits 'test result: ok' from suites that matched nothing"
key-files:
  created:
    - crates/aprender-contrastive-data/src/attestation.rs
  modified:
    - crates/apr-cli/src/commands/data_tweeteval.rs
    - contracts/tweet-eval-stance-benchmark-v1.yaml
    - crates/aprender-contrastive-data/src/prepared.rs
    - crates/aprender-contrastive-data/src/split.rs
    - crates/aprender-contrastive-data/src/error.rs
    - contracts/contrastive-pair-protocol-v1.yaml
decisions:
  - "from_attested_bytes routes through Split::from_jsonl_bytes, which required a pub(crate) from_validated_splits in prepared.rs the plan forbade touching"
  - "the setfit merged split's rows now carry source_split compatibility_test — the one deliberate output-byte change, declared by the schema_version bump"
  - "prepare verifies its own output through the attested boundary, which is what stops the read path being test-only dead code"
  - "ContrastiveDataError gains UnsupportedNormalizationVersion; a String tag cannot ride in UnsupportedSchemaVersion's u32"
  - "contract v2.0.0 (major), pv diff's own suggestion and correct on the merits"
metrics:
  duration: ~1h50m
  tasks: 3
  files: 6
  completed: 2026-08-09
requirements: [DATA-01, DATA-02]
---

# Phase 2 Plan 06: Attested Dataset Boundary and the D-05 Relocation Summary

The generic model moved out of `data_tweeteval.rs` into `aprender-contrastive-data` without
moving a single canonical output byte, and canonical splits are now reachable only through a
boundary that re-derives every attested field before any `Split<R>` accessor exists.

## Commits

| # | Task | Commit | Result |
|---|------|--------|--------|
| 1 | attestation boundary | `be145da14` | 18 new crate tests; 126 -> 144 lib tests |
| 2 | D-05 thin-adapter relocation | `e1accbf93` | 10 -> 17 data_tweeteval tests, name set a strict superset |
| 3 | real-duplicate golden + contract | `e939a53ae` | contract v1.1.0 -> 2.0.0; golden RUN and GREEN on live data |

## The `#[allow(dead_code)]` question, answered first

Plan 02-03 left a scoped allow on `Split::from_jsonl_bytes` with an explicit prediction: *"if
02-06 lands and the allow is still needed, the attested-bytes path did NOT route through the
gate ladder and that is a defect."*

**The allow is REMOVED.** `PreparedDataset::from_attested_bytes` is its non-test caller. That
was not free, and the cost is the plan's most consequential deviation — see Deviation 1.

The reason it mattered is not lint hygiene. Task 1's required cross-path fingerprint test
("build via `from_labeled_rows`, attest, feed `PreparedJsonl` bytes back through
`from_attested_bytes`, assert the fingerprints are EQUAL") is only an assertion about anything
if the two doors derive `source_hash` **differently**: `from_rows` hashes the canonical
re-encoding of accepted rows, `from_jsonl_bytes` hashes the supplied buffer. Had
`from_attested_bytes` simply called `from_labeled_rows` internally — the one route that needs
no change to `prepared.rs` — the two ends would be the same code and the test would be a
tautology dressed as a two-derivation agreement.

## What Was Built

### Task 1 — `attestation.rs`, the identity boundary

`DatasetAttestation` carries profile, schema version, label map, normalization version,
per-split JSONL SHA-256 + per-class counts, the exclusion-record digest, and the dataset
fingerprint. `from_prepared` is generic over a small `AttestedProfile` trait implemented only
by `Canonical` and `Compatibility`.

`from_attested_bytes` exists on **both** profile types and runs a fixed ladder:

parse -> schema version -> normalization version -> profile -> role set -> **per-split SHA-256,
before the buffer is parsed** -> the ordinary 02-03 gate ladder (where per-class counts are
checked) -> exclusion digest -> dataset fingerprint.

Nine rejection tests, one per failure, plus the two success paths, the cross-path fingerprint
reproduction, canonical/strict serialization, a supported-set vacuity guard, a
foreign-role rejection, and a "rejected leaves no ledger record" test — 18 in all.

**The ordering claim is not asserted, it is demonstrated.** Two mutations were induced, run,
and reverted:

| Mutation | Predicted | Observed |
|---|---|---|
| disable the per-split digest comparison | corrupt-bytes test reports a parse error, mixed-directory test reports the wrong error | `MalformedRow { split: "test", index: 0, reason: "expected ident at line 1 column 2" }` and `FingerprintMismatch` — 2 of 18 RED |
| make `check_derived` return `Ok` unconditionally | fingerprint, exclusion and ledger tests fail | exactly those three RED, 15 passed |

The first mutation is the important one: without the digest-first ordering the mixed-directory
case is *still rejected*, just by `FingerprintMismatch` instead of `SplitHashMismatch`. A test
that only asserted "it was rejected" would have been green under the mutation.

### Task 2 — `data_tweeteval.rs` on the D-05 seam, and only that seam

**Stayed CLI-side** (verified by grep, all present): `decode_utf8`, the text/label
length-mismatch check, `parse::<usize>()`, `LABEL_NAMES.get(label)`, `{canonical_name}:{index}`
id minting, the pinned revision and URLs, `SOURCE_FILES`, the three count constants,
`FEW_SHOT_SIZES`, `BENCHMARK_SEEDS`, the `ureq` fetch, `--source`, `validate_revision`,
rollback-on-partial-write, `create_new` no-clobber, stale-`validation.jsonl` removal, and JSON
output.

**Moved to the crate** (verified by grep, all gone): `StanceSample`, `fn encode_jsonl`,
`fn class_counts`, `let mut counts` (×2), `sha256(&bytes)` over split content, the empty-text
branch, and the local class-count comparison.

| Measurement | Baseline `7eb1a67` | Now |
|---|---|---|
| total lines | 813 | 1516 |
| non-test lines | 561 | 845 |
| test lines | 252 | 671 |
| `fs::read` outside the test module | 1 | 1 |
| `Sha256::digest` occurrences | 1 | 1 (raw SOURCE files only) |

**The plan predicted "net deletion of hashing and per-row validation logic" and the file grew
instead.** That prediction was about the *model*, and the model did leave. What arrived is new
adapter surface the plan itself asked for: the `dataset_attestation` and `exclusions` manifest
sections, the crate-referencing schema-version gate, `attestation_bytes_from_manifest`,
`verify_prepared_directory`, the two per-profile `prepare_*` functions, and eight new tests.
Stating this as a net deletion would have been a nicer-sounding claim than the diff supports.

### Task 3 — the real duplicate, as a live golden

**The opt-in network test was RUN, against the live pinned revision, and it passed.** Verbatim:

```
  Output: /var/folders/3s/xftgktnj6qs681vbh0tg5hmc0000gn/T/.tmpulMnej/output

  Test: test.jsonl (280 samples)
  Train: train.jsonl (587 samples)
  Validation: validation.jsonl (66 samples)
  Cross-split duplicates: 1 group(s), 1 training row(s) excluded from the selection pool

OK Benchmark prepared; canonical test data remains isolated from validation.
test commands::data_tweeteval::tests::pinned_upstream_records_exactly_one_coalesced_duplicate_group ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 6651 filtered out; finished in 0.52s
```

Status captured directly (`cargo test ... > /tmp/t-net2.log 2>&1; echo rc=$?` -> `rc=0`), never
through a pipe. The mechanism is proven by the assertion content rather than by the label: the
test requires `hex(exact_hash(train:70.input)) == e3af840b5398272c89e5e1e3e1730c26b0f5bc856d3d46531a2a64dd8844a2c3`
and the same for `validation:3`, which no synthetic or empty download could satisfy.

Asserted and observed: prepare **succeeded** (D-27), `excluded_train_ids == ["train:70"]`,
`groups().len() == 1`, members `[(train, train:70), (validation, validation:3)]`, `detected_by`
exact **and** normalized both true, `label_conflict == false`, reduced pools 158 / 319 / 109,
each `>= 64`.

`groups().len() == 1` is the executable form of review finding F8: under the pre-review
independent-grouping design this one duplicate would have produced two groups and decremented
the `none` pool twice.

**The contract grew to v2.0.0** — `pv diff`'s own suggestion via the two-path form:

```
git show 4ea35e1ee40aa19fef9905090ac0f96adb612acf:contracts/tweet-eval-stance-benchmark-v1.yaml \
  > /tmp/tweet-eval-prev.yaml
cargo run --release -p aprender-contracts-cli --bin pv -- diff \
  /tmp/tweet-eval-prev.yaml contracts/tweet-eval-stance-benchmark-v1.yaml
-> "Contract diff: v1.1.0 → v1.1.0 / Suggested bump: major"
```

Major is also right on the merits, which is worth checking separately because a tool's
suggestion is easy to accept for the wrong reason: a 1.x reader is refused by the schema-version
gate, and the setfit profile's row bytes changed. Four obligations added
(`DUP-EXCLUSION`, `CONFLICTING-ROLES`, `SCHEMA-VERSION`, `ATTESTATION`), two reworded
(`HASH-FROM-PARSED-BYTES`, `LABEL-BOUNDS`), five falsification tests added (009-013). Final
`pv status`: **v2.0.0, 9 proof obligations, 13 falsification tests, 1 Kani harness**. No tweet
text anywhere — the duplicate is recorded by SHA-256 and row id only.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 - Bug] `prepared.rs` was modified, against an explicit acceptance criterion**

- **Found during:** Task 1, wiring `from_attested_bytes`.
- **Issue:** the criterion is *"`git diff --stat crates/aprender-contrastive-data/src/prepared.rs`
  is empty (this task adds impls in its own module so wave 4 stays parallel with plan 02-05)"*.
  Rust field privacy is module-scoped and `attestation` is a **sibling** of `prepared`, so an
  inherent impl written in `attestation.rs` can call `PreparedDataset`'s public constructors but
  cannot build the value from typed splits. Honouring the criterion therefore forces
  `from_attested_bytes` to call `from_labeled_rows`, which means it never calls
  `Split::from_jsonl_bytes` — leaving 02-03's allow in place, contradicting 02-03's handoff, and
  making the mandated cross-path fingerprint test vacuous.
- **Fix:** extracted `pub(crate) from_validated_splits` for both profiles as a pure
  extract-function refactor; `from_labeled_rows` now calls it, so there is exactly ONE assembly
  point and the two doors cannot fingerprint differently. Diff: +57/-19 lines in `prepared.rs`,
  no behaviour change (all 10 pre-existing `prepared_tests` pass untouched).
- **Why this is not simply ignoring the plan:** the criterion's own stated rationale is wave-4
  parallelism with plan 02-05. 02-05 completed and committed at `485ad6123` before this plan
  started; there is no concurrent writer to conflict with, so the constraint's reason had
  already expired while its letter had not.
- **Files modified:** `crates/aprender-contrastive-data/src/prepared.rs`
- **Commit:** `be145da14`

**2. [Rule 2 - Missing critical functionality] `ContrastiveDataError::UnsupportedNormalizationVersion`**

- **Found during:** Task 1, writing the preflight ladder.
- **Issue:** the attestation carries the content-normalization tag its exclusion record was
  computed under, and nothing checked it. `UnsupportedSchemaVersion` cannot carry it — its
  `got`/`supported` are `u32` and the tag is a string. An attestation produced under a different
  normalization pipeline would have been accepted with a record whose collapsing rules this
  build does not implement, which is exactly the drift D-17 versions the pipeline to prevent.
- **Fix:** new variant with `got: String, supported: &'static str`, raised only by
  `attestation::preflight`. `error.rs` documents the process for adding a variant and mandates
  two things; both were done — recorded here, and `OBLIG-CPP-ERROR-TAXONOMY` in
  `contracts/contrastive-pair-protocol-v1.yaml` extended with the variant plus the reasoning for
  why it is separate. That contract re-validates clean.
- **Files modified:** `crates/aprender-contrastive-data/src/error.rs`,
  `contracts/contrastive-pair-protocol-v1.yaml`
- **Commit:** `be145da14`

**3. [Rule 2 - Missing critical functionality] prepare verifies its own output**

- **Found during:** Task 2, wiring the schema-version gate.
- **Issue:** the plan requires a manifest READ path (the version gate) but the command only ever
  wrote. A `pub(crate)` reader with no non-test caller is dead code, and silencing that with an
  allow would have reproduced the very pattern this plan exists to retire.
- **Fix:** `write_outputs` now re-opens the directory it just wrote through
  `PreparedDataset::from_attested_bytes` and rolls the write back if the boundary rejects it.
  This is a real guarantee rather than a way to keep a function alive: a directory this command
  emits is one the gate that guards canonical splits has actually opened. Cost is one extra pass
  over the splits.
- **Files modified:** `crates/apr-cli/src/commands/data_tweeteval.rs`
- **Commit:** `e1accbf93`

**4. [Rule 1 - Bug] The contract named a falsification recipe that no longer applies**

- **Found during:** Task 3.
- **Issue:** `qa_gate.falsification`, `OBLIG-TWEET-EVAL-LABEL-BOUNDS` and
  `FALSIFY-TWEET-EVAL-008`'s `if_fails` all instructed a reader to *"move `counts[label] += 1`
  above the LABEL_NAMES bound check in data_tweeteval.rs"*. Per-class counting moved to the
  crate's Gate 5 in Task 2, so that mutation is no longer performable. A contract naming an
  inapplicable falsification is the same defect class 02-01 fixed in CLAUDE.md.
- **Fix:** the recipe is now *"replace `LABEL_NAMES.get(label).copied().ok_or_else(...)` with
  `LABEL_NAMES[label]`"*, which is performable today and discriminates exactly the same
  property. The obligation records where the counter went and why the recipe moved with it.
- **Commit:** `e939a53ae`

**5. [Rule 1 - Bug] The contract's crate-level test commands could pass vacuously**

- **Found during:** Task 3, checking that each command actually runs (CLAUDE.md rule 7).
- **Issue:** `cargo test -p aprender-contrastive-data attestation` runs three suites and prints
  three `test result: ok` lines — `18 passed`, then **`0 passed; 1 filtered out`**, then
  `1 passed; 6 filtered out`. An `expected_output: 'test result: ok'` grep is satisfied by the
  middle line, which ran nothing.
- **Fix:** both crate-level commands now carry `--lib` (one suite, real counts: 18 and 24), with
  the measurement and the reasoning recorded in a COMMAND FORM block in the contract.
- **Commit:** `e939a53ae`

### Pre-existing tests modified (both called out as the plan requires)

No test was deleted, no assertion was weakened, and the name set is a strict superset (10 -> 18).
Two pre-existing tests changed shape, neither in what it asserts:

1. **`class_count_contract_rejects_modified_source`** — the class-count contract now fires one
   step later, at the crate's ingest boundary, so the test calls `load_canonical_dataset(...)`
   (which now succeeds) and then `build_outputs(...)`. **All three assertions are verbatim:**
   `"class-count contract failed"`, `"[159, 319, 109]"`, `"[158, 319, 110]"`. The crate's
   `InvalidClassCounts` Display happens to produce the same three substrings, which is why the
   assertions did not have to move. Still raised before anything is written.
2. **`label_index_is_in_bounds_or_a_typed_error`** — `load_split` lost its `expected_counts`
   parameter, so the call site drops it. Every assertion is unchanged. The doc comment was
   updated to describe the current mechanism (the fallible `LABEL_NAMES.get`) and records where
   `counts[label] += 1` went, so the prose does not silently become false.

No test's `schema_version` assertion was migrated 1 -> 2, because no pre-existing test asserted
`schema_version` at all; the new `manifest_is_schema_version_two_...` test introduces it.

### Behaviour change: the setfit profile's row bytes

This is the plan's `must_haves.truths[3]` ("the JSONL row output is byte-identical to the D-06
baseline") holding for the canonical profile and **not** for the compatibility profile, and it
is unavoidable rather than a choice.

D-19 requires the merged split to be `Split<CompatibilityTest>`, a role distinct from `Test`.
02-03's Gate 2 requires every row's embedded `source_split` to equal the role being built. So a
merged row must carry `compatibility_test`, where the baseline wrote `validation` / `test`.
Nothing is lost — row ids still read `validation:N` / `test:N` and the manifest still records
`source_splits: [validation, test]` — but the bytes did change. It is declared in three places:
the `schema_version` 1 -> 2 bump, `profiles.setfit.row_source_split` in the contract, and a test
(`setfit_merged_split_carries_the_compatibility_role_and_keeps_id_provenance`) that pins the new
form together with the train split's unchanged bytes. Canonical byte parity is proven against a
wire format re-derived by string formatting **inside the test**, not by calling the encoder.

## Verification

All statuses captured directly (`cmd > log 2>&1; rc=$?`), never read through a pipe. `rtk proxy`
used for every git-porcelain and grep result, since the hook prints a literal `ok` on a clean
porcelain path and abridges other output.

| Gate | rc | Result |
|---|---|---|
| `cargo test -p aprender-contrastive-data --lib attestation` | 0 | 18 passed |
| `cargo test -p aprender-contrastive-data` (all targets) | 0 | 158 passed, 1 ignored |
| `cargo test -p apr-cli --lib data_tweeteval` | 0 | 16 passed, 2 ignored, 0 failed |
| `cargo test -p apr-cli --lib data_tweeteval -- --ignored` | 0 | **2 passed** (live network) |
| `cargo test -p apr-cli --lib` | 0 | 6639 passed |
| **cross-crate baseline** (`--lib -- --skip gpu::`) | 0 | **14,186 passed, 0 failed** |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 | clean |
| `cargo clippy -p aprender-contrastive-data --all-targets --no-deps -- -D warnings` | 0 | clean |
| `cargo clippy -p apr-cli --no-deps --lib -- -D warnings` | 0 | clean |
| `cargo fmt --check` (both crates) | 0 | clean |
| `make contrastive-data-boundary` | 0 | deps subset of allowlist; no fs/net/path under `src/` |
| `pv validate contracts/tweet-eval-stance-benchmark-v1.yaml` | 0 | 0 errors, 0 warnings |
| `pv validate contracts/contrastive-pair-protocol-v1.yaml` | 0 | 0 errors, 0 warnings |
| `make contract-validate` | 0 | 44 contracts valid |
| `pv status` (tweet-eval) | 0 | v2.0.0, 9 obligations, 13 falsification tests |
| `make tier2` | **2** | **RED — pre-existing, see below** |
| `.snap.new` files after the full run | — | restored, no deletion in any commit |

**The baseline reconciles exactly.** 14,161 -> 14,186 is +25: apr-cli 6632 -> 6639 (+7 new
non-ignored data_tweeteval tests, plus 1 new ignored network test), aprender-contrastive-data
126 -> 144 (+18 attestation tests), aprender-train 7403 unchanged. No pre-existing test changed
state.

**Contract commands were executed, not assumed.** All four forms named in the contract were run:
`cargo test -p apr-cli --lib data_tweeteval` (rc=0), the same with `-- --ignored` (rc=0),
`cargo test -p aprender-contrastive-data --lib attestation` (18 passed), and
`cargo test -p aprender-contrastive-data --lib split` (24 passed).

### `make tier2` is RED, and it is not this plan's doing

24 clippy errors, attributed by file: **`aprender-compute` 21, `aprender-zram-core` 3, zero
anywhere else** — no error names `apr-cli`, `aprender-contrastive-data`, or any file this plan
touched. This is D-ITEM-02 exactly as 02-03 and 02-05 recorded it: arch-gated SIMD arms that
every CI runner (`[self-hosted, X64, Linux]`) skips. The count differs from 02-05's 25 only in
how the two runs enumerated locations.

Because `make` halts at the first failure, tier2's later steps never ran under `make`; each was
run individually and is green (see the table above).

**`bashrs` is not installed on this host**, so no shell linting was possible. This plan added no
shell logic — the Makefile was not touched — so nothing was owed.

## Threat Model Dispositions

| Threat | Disposition |
|---|---|
| T-02-17 dataset substitution across the refactor | mitigated — canonical byte parity proven against an independently re-derived wire format; count contracts preserved through the seam; hash-from-same-buffer intact (one `fs::read` outside tests) |
| T-02-18 local dir attested as pinned revision | mitigated — `revision_verified = downloaded.is_some()` still derived CLI-side; FALSIFY-TWEET-EVAL-006 passes unchanged |
| T-02-19 tweet text entering repo/contract | mitigated — the duplicate is recorded by SHA-256 and row id; the golden computes the hash from written rows rather than quoting text |
| T-02-20 compat-profile leakage into selection | mitigated — the setfit path builds `PreparedDataset<Compatibility>`, a distinct TYPE with no validation field; `from_attested_bytes` refuses a canonical attestation at the compatibility door and vice versa |
| T-02-42 mixed or forged prepared directory | mitigated — profile, schema version, per-split digests, class counts, exclusion digest and fingerprint all re-derived before exposure; the digest-first ordering demonstrated by induced mutation |
| T-02-43 silent migration of a stale manifest | mitigated — crate-owned `SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS = [2]`, no shim, error names the set and the remediation; a stale normalization tag is now its own typed refusal |

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or trust-boundary schema change
beyond those already in the register. The one new read path (`verify_prepared_directory`) reads
files this same command wrote, inside the directory the user named.

## Known Stubs

None from this plan. `pairs.rs` remains a `//!`-only stub owned by plan 02-07, unchanged here.

## Notes for later plans

- **02-09** — `apr data select` / `apr data pairs` must obtain splits through
  `PreparedDataset::from_attested_bytes`, not by parsing the JSONL directly.
  `data_tweeteval::attestation_bytes_from_manifest` is the read path (currently private; make it
  `pub(crate)` when a second caller appears) and `role_files` is the role -> filename map,
  including the `compatibility_test` -> `test.jsonl` case.
- **02-08** — two new `#[contract]` sites bind `dataset_attestation` (one per profile impl), and
  `OBLIG-CPP-ERROR-TAXONOMY` grew `UnsupportedNormalizationVersion`. The build still emits
  *"binding.yaml not found … skipping"*, so none of these are compile-time enforced yet.
- **Anyone regenerating a benchmark directory** — schema_version 1 directories are refused with
  no migration. Re-prepare with `--force`.
- **The real-duplicate golden is `#[ignore]`d.** It is a release-path and
  dedup/seam/pinned-revision gate, not a per-commit one. "The suite is green" does not include
  it; a report that claims it passed must quote its output, as this one does.

## Self-Check: PASSED

| Item | Status |
|---|---|
| `crates/aprender-contrastive-data/src/attestation.rs` | FOUND, contains `from_attested_bytes` |
| `crates/apr-cli/src/commands/data_tweeteval.rs` | FOUND, contains `aprender_contrastive_data` |
| `contracts/tweet-eval-stance-benchmark-v1.yaml` | FOUND, contains `e3af840b`, version 2.0.0 |
| `crates/aprender-contrastive-data/src/{prepared,split,error}.rs` | FOUND (3/3) |
| `contracts/contrastive-pair-protocol-v1.yaml` | FOUND, contains `UnsupportedNormalizationVersion` |
| commits `be145da14` `e1accbf93` `e939a53ae` | FOUND (3/3) |
| `#[allow(dead_code)]` on `Split::from_jsonl_bytes` | GONE |
| Three `.snap.new` files under `crates/aprender-train/src/prune/snapshots/` | intact, no deletion in any commit |
