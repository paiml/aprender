---
phase: 02-deterministic-pair-and-data-protocol
plan: 03
subsystem: contrastive-data
tags: [tdd, typestate, union-find, content-hashing, jsonl-codec, gate-ladder, access-ledger, d27, data-06]
requires:
  - "02-02 (the 13-module skeleton, ContrastiveDataError's 34 variants, contrastive-pair-protocol-v1.yaml, make contrastive-data-boundary)"
provides:
  - "schema.rs — LabeledExample + JSONL bytes codec, deny_unknown_fields, round-trip exact"
  - "hash.rs — exact SHA-256 + normalized_hash (nfc-trim-ws-v1) + DatasetFingerprint + SplitFingerprint under distinct domain tags"
  - "split.rs — Split<Train|Validation|Test|CompatibilityTest> typestate behind a 5-gate pub(crate) ingest ladder"
  - "ledger.rs — AccessLedger with deterministic canonical bytes + ledger_hash (Phase 5's selection-lock artifact)"
  - "dedup.rs — union-find coalesced cross-split duplicate components + deterministic ExclusionRecord"
  - "prepared.rs — PreparedDataset<Canonical|Compatibility> typestate, ValidationWitness canonical-only"
  - "DatasetProfile::Splits associated type — makes D-19 structural (see Deviation 1)"
affects:
  - "02-04/02-05 (bucketing + selection consume &PreparedDataset<Canonical> and the ledger)"
  - "02-06 (its from_attested_bytes is the non-test caller that retires the scoped allow on Split::from_jsonl_bytes)"
  - "02-08 (trybuild non-constructibility gate — Deviation 1 makes the property provable at the field level)"
  - "Phase 5 (selection-lock reads ledger_hash off disk rather than an in-memory value)"
tech-stack:
  added: []
  patterns:
    - "typestate via PhantomData + an associated type, so an absent capability is an absent FIELD rather than an Option + expect()"
    - "compile_fail doctest ALWAYS paired with a positive control, so it cannot be green for an unrelated reason"
    - "hash the SAME buffer that is parsed — never re-read the source, never re-encode before hashing"
    - "union-find over the UNION of exact- and normalized-hash edges, so an exact dup (necessarily also a normalized dup) decrements the pool exactly once"
    - "ordering obligations discharged by data structure (BTreeMap) rather than by a caller-side sort that can be omitted"
key-files:
  created: []
  modified:
    - crates/aprender-contrastive-data/src/schema.rs
    - crates/aprender-contrastive-data/src/hash.rs
    - crates/aprender-contrastive-data/src/split.rs
    - crates/aprender-contrastive-data/src/ledger.rs
    - crates/aprender-contrastive-data/src/dedup.rs
    - crates/aprender-contrastive-data/src/prepared.rs
    - Makefile
    - .planning/phases/02-deterministic-pair-and-data-protocol/deferred-items.md
decisions:
  - "DatasetProfile gains an associated type Splits so PreparedDataset<Compatibility> has NO validation field at all, rather than an Option + expect() in three accessors"
  - "Split::exact_hash_pairs() returns pairs pre-sorted from the internal BTreeMap, discharging SplitFingerprintInput's ordering obligation structurally"
  - "make tier2 is NOT green on this host, for 25 pre-existing arm64-only clippy errors in 5 untouched crates — logged as D-ITEM-02, not fixed"
metrics:
  duration: "~55m active (23m TDD implementation, ~30m verification), plus ~1h blocked on a full disk"
  completed: 2026-08-09
requirements: [DATA-01, DATA-02, DATA-06]
---

# Phase 2 Plan 03: Bytes→Typed Data Layer Summary

Six TDD cycles built the crate's bytes→typed layer: a JSONL codec, two content hashes under
distinct domain tags, a five-gate ingest ladder behind phantom-typed splits, an append-only
ledger with a persistable `ledger_hash`, union-find duplicate coalescing, and a
`PreparedDataset` whose profile isolation is a **compile-time** property rather than a runtime
check.

## What Was Built

| Module | Tests | What it establishes |
|---|---|---|
| `schema.rs` | 10 | `LabeledExample` + JSONL codec, `deny_unknown_fields`, `encode(parse(b)) == b` for canonical input |
| `hash.rs` | 17 | exact SHA-256, `normalized_hash` (`nfc-trim-ws-v1`), `DatasetFingerprint`, `SplitFingerprint` — distinct domain tags (`apr-split-fp-v1`) so a split digest can never be mistaken for a dataset digest |
| `split.rs` | 12 | `Split<R>` typestate, 5-gate ladder: parse → role agreement → duplicate IDs → unknown/mismatched labels → class-count validity |
| `ledger.rs` | 9 | `AccessLedger` append-only records, deterministic canonical byte form, `ledger_hash` |
| `dedup.rs` | 9 | disjoint-set forest over exact ∪ normalized edges; transitive chains collapse to one component |
| `prepared.rs` | 10 | `PreparedDataset<Canonical\|Compatibility>`, `ValidationWitness` canonical-only |

**67 lib tests + 5 doctests = 72**, all green.

### The three requirements

- **DATA-01** — the generic model is crate-resident and bytes-in/bytes-out: nothing in
  `src/` touches fs, net, or paths (enforced by `make contrastive-data-boundary`, still green
  with no `cfg(test)` exemption).
- **DATA-02's missing half is typed** — duplicate IDs, conflicting source roles, cross-split
  duplicate content, and `label_text` disagreement all have typed variants. Duplicate IDs and
  conflicting source roles are two of the exact three classes D-07 scopes to Phase 2; the
  third (cross-split duplicate content) is `dedup.rs`.
- **DATA-06 exists as all three layers (D-16)** — compile-time typestate, runtime boundary,
  and ledger.

### D-27, evidenced by two named tests

- `prepare_time_duplicate_content_is_excluded_not_fatal` — construction **succeeds**; the
  duplicate is excluded and recorded (`excluded_train_ids() == ["train:0"]`, one group, the
  class-0 pool decremented by exactly one).
- `split_role_span_is_fail_closed` — actual split-role span is a typed error.

The asymmetry is the point: exclusion is bookkeeping, span is corruption.

### Why the dedup is union-find and not a pair list

An exact duplicate is *necessarily* also a normalized duplicate, so the two hash relations
produce overlapping edges. Treating them as independent pair lists would decrement the
candidate pool twice for one logical duplicate. Building a disjoint-set forest over the union
of both edge sets makes each duplicate one component regardless of how many relations witness
it, and makes transitive chains (a≈b via exact, b≈c via normalized) collapse into a single
group — covered by the three-way-chain fixture.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] `DatasetProfile` gained an associated type `Splits`**

- **Found during:** Task 3, implementing `PreparedDataset`.
- **Issue:** The plan's interface declared only `const PROFILE`. With that shape both profiles
  share one struct, so `PreparedDataset<Compatibility>` still needs a validation *field* —
  `Option<Split<Validation>>` — plus an `expect()` in three accessors. That makes D-19's claim
  ("the compatibility profile emits no `Split<Validation>` at all") false at the field level:
  the slot exists and is merely empty at runtime. A `None` that must never be `Some` is a
  runtime invariant wearing a type's clothing.
- **Fix:** `DatasetProfile` carries an associated type `Splits`. `Canonical::Splits` has
  train/validation/test; `Compatibility::Splits` has train/test. The compatibility dataset
  now has no validation field to be empty, and no `expect()` exists on any accessor.
- **Why it matters beyond tidiness:** 02-08's trybuild non-constructibility gate has to *prove*
  the absence. Proving "this field is always `None`" needs whole-program reasoning; proving
  "this field does not exist" is a type error. Deviation 1 is what makes that gate writable.
- **Files modified:** `crates/aprender-contrastive-data/src/prepared.rs`
- **Commit:** `885adffa5`
- **Every DATA-06 property in the plan holds unchanged** — `PreparedDataset<Canonical>` and
  `PreparedDataset<Compatibility>` remain distinct types, `validation_witness()` remains
  canonical-only, and selection still consumes `&PreparedDataset<Canonical>`.

**2. [Rule 2 - Missing critical functionality] `ValidationWitness::validation()` and `Split::exact_hash_pairs()`**

- **Found during:** Task 3.
- **Issue:** (a) `ValidationWitness` held a borrowed split that nothing could reach, making the
  borrow pointless. (b) `SplitFingerprintInput` requires `(id, exact_hash)` pairs in ascending
  id order, but the plan left that ordering to each caller — an obligation a caller can silently
  omit, and a wrong order yields a plausible-looking wrong digest.
- **Fix:** Added `ValidationWitness::validation()`. Added `pub(crate) Split::exact_hash_pairs()`
  yielding pairs straight from the internal `BTreeMap`, already sorted ascending by id.
- **Why it matters:** the ordering obligation is now discharged **by the data structure** rather
  than by a convention. There is no unsorted path to construct the input from.
- **Files modified:** `crates/aprender-contrastive-data/src/split.rs`, `.../prepared.rs`
- **Commit:** `885adffa5`

**3. [Rule 3 - Blocking issue] Corrupt incremental-compilation cache**

- **Found during:** final verification.
- **Issue:** `cargo test -p aprender-core ... setfit::` failed rc=101 with
  `failed to move dependency graph from .../entrenar-*/dep-graph.part.bin ... (os error 2)` —
  fallout from the earlier ENOSPC, not a code error.
- **Fix:** none needed; cargo purged the corrupt entry itself on the next invocation. Retry was
  rc=0. Nothing under `target/debug/incremental/` predated this session (the orchestrator had
  cleared it wholesale), so no work was at risk.

### Makefile comment updated (standing instruction, not a discretionary edit)

The tier2 contrastive-data block carries its own instruction: *"Re-measure and update this
number when the crate's suite grows; a tier2 line whose comment records a stale number is worse
than one with no comment, because it will be trusted."* The suite grew from one determinism
doctest to 67 lib + 5 doc tests, so the re-measurement was owed. Three consecutive warm runs:
**1s / 2s / 1s, rc=0 each**. The number did not move across a 72× growth in test count, which
strengthens rather than contradicts the block's original claim — the wall clock is cargo's
per-invocation freshness check, not test execution (0.02s lib + 0.76s doc of actual test time
inside a 1–2s invocation). Comment updated with the new figures and that reasoning.

## Verification

| Gate | rc | Result |
|---|---|---|
| `cargo test -p aprender-contrastive-data` | 0 | 72 passed (67 lib + 5 doc) |
| `cargo test -p aprender-contrastive-data --doc` | 0 | 5 passed |
| `cargo clippy -p aprender-contrastive-data --all-targets -- -D warnings` | 0 | clean |
| `make contrastive-data-boundary` | 0 | deps subset of allowlist; no fs/net/path in `src/` |
| `pv validate contracts/contrastive-pair-protocol-v1.yaml` | 0 | 0 errors, 0 warnings |
| cross-crate baseline | 0 | **14,102 passed**, 0 failed |
| `make tier2` | **2** | **RED — pre-existing, see below** |

**Cross-crate baseline reconciles exactly.** `cargo test -p apr-cli -p aprender-train
-p aprender-contrastive-data --lib -- --skip gpu::` gave 6632 + 67 + 7403 = **14,102 passed,
0 failed**. The prior floor was 14,035; the delta is 67, which is precisely this crate's new
lib-test count. No pre-existing test changed state.

**Both `compile_fail` doctests are paired with positive controls.** A `compile_fail` block that
fails to compile for the *wrong* reason (a typo, a missing import) is green and proves nothing.
Each is therefore twinned with the same call on the canonical type, which must compile.

### `make tier2` is RED, and it is not this plan's doing

25 clippy errors across 5 crates — `aprender-compute` (19), `aprender-zram-core` (3),
`aprender-core` (1), `aprender-present-terminal` (1), `aprender-serve` (1). **Zero in
`aprender-contrastive-data`.** Measured, not inferred:

- `cargo clippy -p aprender-zram-core -- -D warnings` → rc=101, 4 errors, and
  `cargo tree -p aprender-zram-core` contains **no** reference to `aprender-contrastive-data`.
  The failure reproduces with this phase's crate absent from the graph entirely.
- `git log d50a0d818^..HEAD -- crates/aprender-zram-core crates/aprender-compute` is empty.

Every location is arch-gated SIMD code, and the host is arm64 while every CI `runs-on` is
`[self-hosted, X64, Linux, clean-room]` — so these aarch64-live arms are never clippy-linted by
CI, and `ci / gate` stays green while every arm64 developer's tier2 is red. Logged as
**D-ITEM-02**; out of scope per the executor Scope Boundary (5 untouched crates, needs per-arch
review plus an arm64 CI lane, or the lints silently return).

Because make halts at the first failure, tier2's remaining steps never ran under `make`. All
were run individually: setfit lib gate 162 passed (rc=0), setfit conformance 27 passed 1 ignored
(rc=0), contrastive-data 72 passed (rc=0). The only red is the pre-existing one.

A second finding fell out of reading that log: tier2's headline `cargo test --lib` selects only
the root facade package and executes **zero** tests. Logged as **D-ITEM-03**.

## Notes for Later Plans

- **02-06** owns retiring the scoped `#[allow(dead_code)]` on `Split::from_jsonl_bytes`. Its
  only non-test caller is `from_attested_bytes`. The comment at the allow says it explicitly:
  if 02-06 lands and the allow is still needed, the attested-bytes path did not route through
  the gate ladder, and that is a defect rather than a lint to silence.
- **02-08** benefits from Deviation 1 — the non-constructibility gate can now assert a missing
  field rather than reason about an always-`None` option.
- The `#[contract]` annotations bind `cross_split_exclusion` (dedup) and
  `prepared_dataset_typestate` (prepared). Both equations exist in the contract and `pv validate`
  is green, but note the build emits *"binding.yaml not found … skipping"* — so the bindings are
  **not** compile-time-enforced yet. Wiring `contracts/aprender/binding.yaml` is 02-08's job, as
  planned; flagging only so nobody mistakes the annotations for an active gate before then.

## Self-Check: PASSED

All six modified source files exist and all six commits are present in `git log`.

| Item | Status |
|---|---|
| `crates/aprender-contrastive-data/src/{schema,hash,split,ledger,dedup,prepared}.rs` | FOUND (6/6) |
| `d50a0d818` `ff6dfe55e` `3934c73a4` `5cc94cfe4` `b8def8f34` `885adffa5` | FOUND (6/6) |
| Three `.snap.new` files under `crates/aprender-train/src/prune/snapshots/` | restored, no deletion staged |
