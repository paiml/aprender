---
phase: 02-deterministic-pair-and-data-protocol
plan: 05
subsystem: contrastive-data
tags: [tdd, philox, counter-based-rng, fisher-yates, selection-manifest, strict-replay, goldens, d20, d21, d08, data-03]
requires:
  - "02-02 (crate skeleton, ContrastiveDataError variants, contrastive-pair-protocol-v1.yaml equations)"
  - "02-03 (Split typestate, both content hashes, DatasetFingerprint/SplitFingerprint, AccessLedger, ExclusionRecord, PreparedDataset<Canonical>)"
provides:
  - "rng.rs — DomainKey/derive_key/draw/bounded(NonZeroU64) + the six frozen domain strings, byte encoding pinned by an independently derived golden"
  - "buckets.rs — ClassBuckets: BTreeMap of sorted per-class pools, exclusions subtracted once, every DECLARED label kept"
  - "select.rs — SelectedExample, SelectedId, Selection, FewShotSelector::select, Selection::replay"
  - "manifest.rs — SelectionPayload (digest-free, ledger persisted) + SelectionManifest envelope + from_bytes/to_file_bytes"
  - "tests/goldens/ — frozen 3-class corpus + four selection payload goldens under manifest.sha256"
  - "tests/goldens_regenerate.rs — #[ignore]d re-baseline entry point"
affects:
  - "02-07 (pair sampler consumes SelectedId/label_of/ids_in_class rather than class-size totals)"
  - "02-08 (trybuild non-constructibility gate; the three new #[contract] bindings feed contract-audit-phase2)"
  - "02-09 (CLI writes SelectionManifest::to_file_bytes verbatim and reconstitutes only via Selection::replay)"
  - "Phase 5 (selection-lock reads the persisted access ledger out of the manifest payload)"
tech-stack:
  added: []
  patterns:
    - "counter-based draw: every random decision is a pure function of (key, stream_id, ordinal); no mutable RNG state crosses a boundary"
    - "golden constants derived by a SECOND implementation written from the contract, then cross-checked against shasum — never captured from a first run"
    - "the hashed object contains no digest of itself; the digest and volatile metadata live in an outer envelope"
    - "a validation ladder split into one small predicate per rung, so the ORDER of rejection is a readable sequence rather than an emergent property"
    - "strict replay ends in a full deterministic recomputation, because every earlier rung accepts an internally consistent forgery"
    - "vacuity guards: assert the fixture is adversarial (unsorted ingest order, non-empty exclusion, two whitespace-variant rows) before asserting the property over it"
key-files:
  created:
    - crates/aprender-contrastive-data/tests/goldens/golden_corpus_train.jsonl
    - crates/aprender-contrastive-data/tests/goldens/golden_corpus_validation.jsonl
    - crates/aprender-contrastive-data/tests/goldens/golden_corpus_test.jsonl
    - crates/aprender-contrastive-data/tests/goldens/selection_seed13_shots8.payload.json
    - crates/aprender-contrastive-data/tests/goldens/selection_seed13_shots16.payload.json
    - crates/aprender-contrastive-data/tests/goldens/selection_seed17_shots8.payload.json
    - crates/aprender-contrastive-data/tests/goldens/selection_seed17_shots16.payload.json
    - crates/aprender-contrastive-data/tests/goldens/manifest.sha256
    - crates/aprender-contrastive-data/tests/goldens_regenerate.rs
  modified:
    - crates/aprender-contrastive-data/src/rng.rs
    - crates/aprender-contrastive-data/src/buckets.rs
    - crates/aprender-contrastive-data/src/select.rs
    - crates/aprender-contrastive-data/src/manifest.rs
    - crates/aprender-contrastive-data/src/prepared.rs
    - Makefile
decisions:
  - "Permutation invariance is claimed for the SELECTION, not for the semantic hash — the payload embeds a dataset fingerprint that digests the split's JSONL in ingest order, so a permuted file is legitimately a different dataset; the test asserts BOTH halves"
  - "The selection AccessRecord carries the DATASET fingerprint, not the validation-split digest the plan named, so one ledger field does not mean two different things depending on which code path wrote it"
  - "PreparedDataset retains its declared label map rather than reconstructing one from row label_text, because a class with no rows would silently vanish from a reconstruction and the payload contracts a label MAP"
  - "SelectionManifest::from_selection leaves created_at empty: the crate reads no clock, for the same reason it opens no file"
  - "The golden corpus carries one cross-split duplicate and two whitespace-variant rows, so the goldens pin a non-empty exclusion record and a normalized hash distinct from the exact one"
metrics:
  duration: ~2h10m
  tasks: 3
  files: 15
  completed: 2026-08-09
requirements: [DATA-03]
---

# Phase 2 Plan 05: Deterministic Few-Shot Selection Summary

Three TDD cycles turned D-20 from a design decision into a mechanism: a domain-separated
counter-based RNG whose byte encoding is pinned by constants a second implementation
produced, a partial Fisher-Yates selector whose output order IS its draw order, and a
materialized selection manifest that a forged copy cannot reconstitute a `Selection` from.

**126 lib tests + 7 integration + 7 doctests, all green** (79 → 140 for the crate).
Cross-crate baseline **14,102 → 14,161**, and the delta is exactly the 59 new lib tests.

## Six commits, RED before GREEN in every cycle

| # | Task | Commit | Result |
|---|------|--------|--------|
| 1 RED | rng byte-encoding + purity tests | `5733cdcba` | 0 passed / 7 failed |
| 1 GREEN | `rng.rs` | `cd27706cc` | 7 lib + 7 doctests pass |
| 2 RED | buckets / selection / payload tests | `62e9390d7` | 24 of 25 new tests fail |
| 2 GREEN | `buckets.rs`, `select.rs`, payload half of `manifest.rs` | `c79e9c3ae` | 99 lib tests pass |
| 3 RED | envelope / replay / golden tests | `60df8d959` | 23 fail, 103 pass |
| 3 GREEN | envelope, `Selection::replay`, goldens | `fc0eeefe0` | 126 lib tests pass |
| — | tier2 runtime re-measurement | `dc44dd6c4` | standing instruction discharged |

The RED runs are the point, not ceremony. Task 2's first RED reported **24 of 25** new
tests failing; the one that passed is `buckets_do_real_sorting_work`, a fixture-vacuity
guard that asserts the training rows arrive *unsorted* — it is supposed to be green
without an implementation, and if it ever goes red the sorting assertions beside it stop
proving anything. Task 3's RED left four golden tests green for the same reason: they
exercise capability that landed in task 2 and are pinned, not introduced, by task 3.

## Are the goldens algorithm-derived or capture-and-blessed? Both, and they are labelled

This is the question a reviewer has to be able to answer without reading the generator, so
it is answered in the module doc as well as here.

**Algorithm-derived — the RNG byte encoding and the selection itself.** A Python
implementation was written from the contract text (`rng_key_derivation`, `bounded_draw`,
`few_shot_selection`) and the Philox 4x32-10 definition in Salmon et al. (2011). It never
read this crate's Rust. It produced:

| Constant | Value | Cross-check |
|---|---|---|
| `derive_key(13, "select/0")` | `[0x0228_10b9, 0x71ce_22dc]` | `printf 'apr-contrastive-v1\0\x0d\0\0\0\0\0\0\0select/0' \| shasum -a 256` → `b9102802 dc22ce71 …`, read little-endian in both lanes |
| `derive_key(13, "select/1")` | `[465_649_502, 1_683_967_742]` | same derivation |
| `derive_key(14, "select/0")` | `[1_239_703_332, 3_359_937_302]` | same derivation |
| `draw(key13, 0, 7)` | `[1281016082, 3815106876, 1099144567, 2908329261]` | Python Philox |
| `assemble64(that block)` | `16_385_759_264_445_743_378` | pinned as a VALUE, not restated as `(lanes[1] << 32) \| lanes[0]` — restating it would stay green if both sides were swapped together |
| `bounded(key13, 0, 7, 587)` | `521` | Python multiply-shift |
| `bounded(key13, 0, 7, 24576)` | `21_830` | " |
| `bounded(key13, 3, 12_345_678_901, 587)` | `419` | " — exercises the HIGH ordinal word and a non-zero stream id, the two counter lanes a naive implementation drops |

and the four ordered-selection digests:

| seed | shots | `SHA-256(ordered_ids joined by "\n")` |
|---|---|---|
| 13 | 8 | `1c99eec4d905430e4b5d05471a01af99f27b4ef65707767f2765cb10ef57701c` |
| 13 | 16 | `1ea9826fd29e0a298c911097d973e3ca4eb7d804bd542b481264804cd658ffaa` |
| 17 | 8 | `7bb11c386e9622d151c83d6a3471b56a21da5c32fc86c9f5b62d97e91d64c763` |
| 17 | 16 | `ca7c7c4c291beb63b9e878428e46f6a9217c23d0742043dd9e7489398f9dd301` |

Those digests pass against the Rust implementation, which is a genuine two-implementation
agreement over the whole chain — key derivation, counter layout, lane assembly,
multiply-shift, bucket sorting, exclusion subtraction and the Fisher-Yates walk. The
8-shot prefix of each seed equals the first 8 of its 16-shot list, which is the
prefix-stability property partial Fisher-Yates is chosen for.

**Capture-and-blessed — the four `*.payload.json` files.** They are this crate's own
canonical serialization of those selections, written once by
`tests/goldens_regenerate.rs`. They pin the BYTE FORM against future drift; they do not
independently corroborate it. Their *content* is corroborated by the digests above, which
is the reason both kinds are committed rather than only the cheap one.

Re-baselining is a named, reviewable command rather than a lost incantation:
`cargo test -p aprender-contrastive-data --test goldens_regenerate -- --ignored`.

## Determinism, stated as what it survives and what it must not

The plan asked for the selection *and* the semantic hash to be invariant under permuted
ingest order. Only the first is true, and the second must not be — see Deviation 1. The
test asserts both halves in one place so neither can regress unnoticed:

```
same ordered selection  +  different dataset fingerprint  =>  different semantic hash
```

Everything else is invariant by construction rather than by discipline:

- **Worker count and iteration order** — draw *i* is `Philox4x32::generate_at(key, counter(i))`.
  Nothing precedes it, so nothing can change it. The purity test requests ordinals
  `[5, 1, 3]` and compares against `[1, 3, 5]`; a stateful stream fails this and no
  "we always draw in order" convention can make it pass.
- **Hash-map iteration** — `grep -rn "HashMap\|HashSet" src/` returns **nothing**
  (control: `BTreeMap` appears in six files, so the grep is live). Buckets, class indexes,
  the seen-set, the payload maps and the exclusion record are all `BTreeMap` /
  `BTreeSet` / sorted `Vec`. `SelectedId` derives `Hash` but is never used as a hash key.
- **Platform** — every byte decision is frozen and pinned by the table above rather than
  inherited from the host's endianness.
- **Seed space** — a proptest sweeps `0..u64::MAX` rather than only the ten contracted
  seeds, and the ten-seed loop additionally asserts that the ten produce ten *distinct*
  orderings, so "deterministic" cannot be satisfied by a constant.

## Strict replay: twelve distinct rejections, and the one that matters

`Selection::replay` is the sole path from manifest bytes back to a `Selection`. Its ladder
runs in a fixed order, each rung a named predicate:

versions → profile → dataset fingerprint → validation fingerprint → exclusions →
membership → uniqueness → class balance → class ordering → both per-row hashes →
semantic hash → **recomputation of the ordered list**.

Twelve rejection tests, one per rung, each constructing a `SelectionManifest` *value*
rather than parsing one — `from_bytes` verifies the digest and would reject most of them
before `replay` ever ran, so parsing them would have tested the wrong thing.

`dataset_fingerprint` and `validation_fingerprint` genuinely hold different values (the
whole-dataset digest and the validation split's own), which is what makes tampering with
each a distinct test rather than the same test twice; the validation-fingerprint test
asserts that inequality inline so it cannot silently become a duplicate.

**The twelfth is the whole reason the recomputation exists.** It substitutes one selected
row for another row from the same class's pool, with that row's real hashes, keeps the
counts and the ordering intact, and then *reseals the envelope digest*. Every earlier rung
passes. The manifest is internally consistent in every way a static audit can check. It is
simply not a selection any seed could have produced, and only recomputing the list catches
it. A membership-plus-hash check — the obvious implementation — would have accepted it.

The ledger invariant that could have been unsatisfiable is now stated as two tests rather
than as a comment. `from_selection` **refuses** a ledger that has grown since `select`
returned (the payload's embedded records would no longer describe it), while `replay`
**succeeds** against a ledger that has already moved on — because replay appends its own
record, so the live ledger has diverged by construction and a digest rule that rebuilt the
payload from it could never pass for any honest manifest.

## Tamper detection, observed rather than assumed

One byte of `selection_seed13_shots8.payload.json` was changed (`"root_seed":13` → `17`)
and the suite re-run. It failed `rc=101` in **three independent ways**:

```
digest drift in selection_seed13_shots8.payload.json
  left:  "f831c5a57b7266577a906ea25a8dd1f2aadb00e4ab43b127501ef8a57373a908"
  right: "e4d22826f4128b11f1e2c758a8ffd387477950d098eb272ee548386f51c94679"
```

plus the byte-for-byte payload comparison and the round-trip equality test. The digest
check names the file and both hashes; the byte comparison would still catch a
*coordinated* edit that also updated `manifest.sha256`. Reverted; 6 golden tests green
again.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] The plan's permutation property was unsatisfiable as written**

- **Found during:** Task 2, writing the permuted-ingest-order test.
- **Issue:** the plan asks for "identical ordered examples **and semantic_hash** … across
  permuted input row order". The payload embeds `dataset_fingerprint`, which absorbs each
  split's `source_hash` — and `source_hash` is `SHA-256` of the split's canonical JSONL
  *in ingest order*. A permuted file is different bytes, so its fingerprint differs and
  its semantic hash must differ too. Asserting equality would have required either
  deleting provenance from the payload or making the fingerprint order-blind; both are
  worse than the property they would buy.
- **Fix:** the test asserts the property that is actually true and the one that is
  actually required — same ordered selection, *different* fingerprint, therefore different
  semantic hash — with the reasoning in the module doc so a later reader does not "fix"
  the second assertion into the first.
- **Files modified:** `crates/aprender-contrastive-data/src/select.rs`
- **Commit:** `62e9390d7` / `c79e9c3ae`

**2. [Rule 1 — Bug] The selection AccessRecord records the dataset fingerprint**

- **Found during:** Task 2, implementing the ledger record.
- **Issue:** the plan asks that the record's `fingerprint_hex` equal
  `dataset.validation_witness().fingerprint_hex()`. After 02-03's checker-warning-1 change
  that accessor returns the **validation split's own** digest, while every ingest record
  in the same ledger carries the **dataset** digest. Following the plan literally would
  make one field mean two different things depending on which code path wrote it, and a
  Phase 5 selection-lock reading that column would have no way to tell which.
- **Fix:** the record carries the dataset fingerprint, consistent with ingest. The test
  still discharges the D-19 evidence the plan was after, by asserting the value equals
  `dataset.validation_witness().dataset_fingerprint_hex()` — an accessor reachable only
  from a dataset that HAS a validation witness — and by asserting `profile == "canonical"`,
  which is the field D-19 actually names. The validation-split digest is recorded where it
  belongs: `payload.validation_fingerprint`.
- **Files modified:** `crates/aprender-contrastive-data/src/select.rs`
- **Commit:** `c79e9c3ae`

**3. [Rule 2 — Missing critical functionality] `PreparedDataset` retains its label map**

- **Found during:** Task 2, building `SelectionPayload`.
- **Issue:** the payload contracts a label map (`selection_canonical_payload`), but
  `PreparedDataset` kept none — it absorbed `decls.label_names` into the fingerprint and
  dropped it. Reconstructing the map from row `label_text` values would silently omit any
  class with no rows in the split, turning a label MAP into "the labels that happened to
  appear" and quietly shrinking `check_class_balance`'s expectation vector during replay.
- **Fix:** `PreparedDataset<P>` retains `label_names` and exposes `label_names()` on both
  profiles. No fingerprint changes — the map was already absorbed, so this only grants
  access to an identity that was already committed to.
- **Files modified:** `crates/aprender-contrastive-data/src/prepared.rs`
- **Commit:** `c79e9c3ae`

**4. [Rule 2 — Missing critical functionality] The golden corpus was not adversarial enough**

- **Found during:** Task 3, reading the first generated goldens.
- **Issue:** with plain text every row's `exact_hash` equalled its `normalized_hash`, so
  the goldens said nothing about `nfc-trim-ws-v1` — a normalization regression could have
  swapped the two derivations and left every golden byte intact.
- **Fix:** two training rows carry a whitespace variant (leading, internal and trailing
  space), and the corpus-shape test asserts that **exactly two** rows have differing
  hashes, so the property cannot decay into vacuity. The corpus also carries one
  cross-split duplicate, so the goldens pin a **non-empty** exclusion record rather than
  only the easy path.
- **Files modified:** `crates/aprender-contrastive-data/tests/goldens/*`
- **Commit:** `60df8d959`

**5. [Rule 2 — Missing critical functionality] A committed re-baseline entry point**

- **Found during:** Task 3, generating the goldens.
- **Issue:** the plan pins the goldens with `include_bytes!` from `src/`, which is right —
  but it leaves no in-tree way to regenerate them. A golden whose procedure lives only in
  a summary gets hand-edited by the next person, and the diff stops meaning anything.
- **Fix:** `tests/goldens_regenerate.rs`, `#[ignore]`d so an ordinary `cargo test` can
  never overwrite the artifacts it is checking. It sits under `tests/`, outside the D-04
  `src/` scan, and builds its dataset from the *same committed corpus files* the verifier
  embeds — so the only duplicated thing is the three-line declaration block, which the
  verifier asserts against the corpus.
- **Commit:** `60df8d959`

### Interface details chosen where the plan was silent

- **`SelectionManifest`'s three fields are public**, and `from_selection` leaves
  `created_at` **empty** for the caller to fill. The crate reads no clock for the same
  reason it opens no file: a timestamp minted inside the library is ambient input the
  caller cannot control. `tool_version` comes from `CARGO_PKG_VERSION`. Both live outside
  the hashed region, proven by the two-manifests-differing-only-in-volatile test.
- **`Selection` gained `root_seed()`, `shots_per_class()`, `payload()` and `is_empty()`**
  beyond the interfaces block — the first three because replay and the manifest need them,
  the last because `len()` without `is_empty()` is a clippy error.
- **`compute_ordered` is a free function**, split out of `select` so `replay` can recompute
  the identical list without appending a second ledger record.

### Standing instruction discharged, not a discretionary edit

The tier2 contrastive-data block instructs: *"Re-measure and update this number when the
crate's suite grows; a tier2 line whose comment records a stale number is worse than one
with no comment, because it will be trusted."* The suite grew 72 → 140. Three warm runs:
**1.75 s / 1.78 s / 1.74 s, rc=0 each** (1.17 s lib + 0.23 s doc of actual test time). The
number moved for the first time in this phase, and the comment now records why: the
proptest sweep over the seed space. Commit `dc44dd6c4`.

## Verification

| Gate | rc | Result |
|---|---|---|
| `cargo test -p aprender-contrastive-data` (all targets) | 0 | **140 passed** (126 lib + 7 integration + 7 doc), 1 ignored |
| `cargo test -p aprender-contrastive-data rng` | 0 | 7 passed; `next_f32` count in non-comment lines = **0** |
| `cargo test -p aprender-contrastive-data select` | 0 | 16 passed |
| `cargo test -p aprender-contrastive-data buckets` | 0 | 4 passed |
| `cargo test -p aprender-contrastive-data payload` | 0 | 7 passed |
| `cargo test -p aprender-contrastive-data manifest` | 0 | 35 passed |
| `cargo clippy -p aprender-contrastive-data --all-targets --no-deps -- -D warnings` | 0 | clean |
| `cargo fmt -p aprender-contrastive-data --check` | 0 | clean |
| `make contrastive-data-boundary` | 0 | deps subset of allowlist; **no fs/net/path symbols under `src/`** |
| `pv validate contracts/contrastive-pair-protocol-v1.yaml` | 0 | 0 errors, 0 warnings |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 | clean |
| cross-crate baseline (`--lib -- --skip gpu::`) | 0 | **14,161 passed**, 27 ignored, 0 failed |
| one-byte golden tamper | 101 | 3 tests fail, digest check names the file and both hashes; reverted |
| `make tier2` | **2** | **RED — pre-existing, see below** |

**The baseline reconciles exactly.** 14,102 → 14,161 is +59, which is precisely this
crate's new lib-test count (67 → 126). No pre-existing test changed state.

**The `.snap.new` files are intact.** All three tracked
`crates/aprender-train/src/prune/snapshots/*.snap.new` files exist after the full
`aprender-train` run, and no deletion is staged in any of the seven commits.

### `make tier2` is RED, and it is not this plan's doing

Identical to what 02-03 recorded as **D-ITEM-02**. `make` halts at
`cargo clippy -- -D warnings` with arch-gated SIMD errors in crates this plan never
touched; `aprender-compute` accounts for 21 of the reported locations and
`aprender-zram-core` for 3 (raw-pointer constness casts). **Zero in
`aprender-contrastive-data`** — its only appearance in the whole tier2 log is the
`Checking aprender-contrastive-data v0.63.0` line, i.e. it compiled clean.

Measured rather than inferred, again:

- `cargo clippy -p aprender-zram-core --no-deps -- -D warnings` → rc=101 on its own.
- `cargo tree -p aprender-zram-core` contains **no** reference to
  `aprender-contrastive-data`, so the failure reproduces with this phase's crate absent
  from the graph entirely.

Because make halts, tier2's later steps never ran under `make`. All were run individually
and are green: the contrastive-data suite (140 passed, three consecutive warm runs), the
Phase 1 setfit gates unchanged, and the full cross-crate baseline above.

### Measurement notes (CLAUDE.md Verification Discipline)

- **Statuses captured directly, never through a pipe.** Every `rc` above came from
  `cmd > log 2>&1; rc=$?`. Several outer shell invocations reported non-zero purely
  because a trailing `grep` matched nothing.
- **`rtk proxy` for every porcelain and grep result**, since the hook prints a literal
  `ok` on a clean porcelain path and abridges other output.
- **The `HashMap` absence claim carries a control** — the same recursive grep for
  `BTreeMap` returns hits in six files, so its silence on `HashMap` is meaningful rather
  than a broken command.
- **The tamper was induced and observed**, not assumed, and reverted with the suite
  re-run green afterwards.
- **`bashrs` is not installed on this host** (`command not found`), so the Makefile edit
  could not be linted. It is a comment-only change touching no shell logic.

## Notes for later plans

- **02-07** — derive pair targets from `Selection::label_of(SelectedId)` and
  `ids_in_class(label)`, never from `class_sizes()`. `SelectedId`'s constructor is private
  to `select.rs`, so a value of that type is proof of membership; that is the mechanism
  behind non-leaky endpoints, and it only holds if the sampler takes `SelectedId` rather
  than `&str`.
- **02-08** — three new `#[contract]` bindings land here (`rng_key_derivation`,
  `bounded_draw`, `rng_domain_strings`) plus `few_shot_selection`,
  `selection_canonical_payload` and `selection_replay`. `rng_domain_strings` sits on
  `domains::select` because it is the only *function* in that module — the other five
  strings are `const` items an attribute macro cannot annotate — and the only one that
  formats, hence the only one that could drift across platforms. Note the build still
  emits *"binding.yaml not found … skipping"*, so none of these are compile-time enforced
  until 02-08 wires the registry.
- **02-09** — write `SelectionManifest::to_file_bytes()` verbatim and compose no JSON.
  Read back only through `SelectionManifest::from_bytes` (which verifies the digest before
  returning) and `Selection::replay`. Fill `manifest.volatile.created_at` at the CLI
  boundary; it never affects a digest.
- **Phase 5** — the access ledger is inside `payload.access_ledger` with its hash beside
  it, and the persisted-ledger test proves `AccessLedger::from_bytes` over those records
  reproduces `payload.ledger_hash`. The selection lock has a real artifact to read.
- **Out-of-scope observation:** `crates/apr-cli/src/commands/nf4_classifier.rs:280` and a
  neighbouring line emit `unused_mut` warnings during the baseline run. Untouched by this
  plan and not fixed, per the executor scope boundary.

## Known Stubs

None. Every module this plan touched is implemented; `pairs.rs` and `attestation.rs`
remain `//!`-only stubs owned by plans 02-07 and 02-06 respectively, unchanged here.

## Self-Check: PASSED

| Item | Status |
|---|---|
| `crates/aprender-contrastive-data/src/{rng,buckets,select,manifest,prepared}.rs` | FOUND (5/5) |
| `crates/aprender-contrastive-data/tests/goldens/` — 3 corpus + 4 payload + manifest.sha256 | FOUND (8/8) |
| `crates/aprender-contrastive-data/tests/goldens_regenerate.rs` | FOUND |
| `5733cdcba` `cd27706cc` `62e9390d7` `c79e9c3ae` `60df8d959` `fc0eeefe0` `dc44dd6c4` | FOUND (7/7) |
| Three `.snap.new` files under `crates/aprender-train/src/prune/snapshots/` | intact, no deletion staged |
