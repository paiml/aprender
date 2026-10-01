---
phase: 02-deterministic-pair-and-data-protocol
verified: 2026-08-09T08:26:45Z
verified_at_commit: 6cfc0f901
verified_on_branch: gsd/phase-2-contract-gate
status: human_needed
score: 31/32 must-haves verified (5/5 ROADMAP success criteria)
overrides_applied: 0
requirements:
  DATA-01: satisfied
  DATA-02: satisfied
  DATA-03: satisfied
  DATA-04: satisfied
  DATA-05: satisfied
  DATA-06: satisfied
  orphaned: []
human_verification:
  - test: "Decide whether the setfit-profile row-byte change is ACCEPTED as a deviation from plan 02-06 must_haves.truths[2] (\"The JSONL row output is byte-identical to the D-06 baseline\")."
    expected: "Either paste the override YAML in `## Suggested override` below into this file's frontmatter and re-run verification, or reject it and open a gap-closure plan. Note that reverting the change would break D-19 and 02-03's Gate 2, and therefore ROADMAP SC5 — gap closure is almost certainly the WRONG route here."
    why_human: "Literal-vs-intent judgement on a plan-level must-have. The deviation is forced by a locked decision (D-19), disclosed in the contract, declared by a schema_version 1->2 bump with a no-migration rejection policy, and pinned by a passing test — but it IS an output-byte change and a machine cannot decide whether that acceptance is granted."
  - test: "Approve (or defer) the crates.io publish cascade: publish `aprender-contrastive-data` BEFORE `apr-cli`."
    expected: "Until this human-approved release action happens, EVERY `cargo package -p apr-cli` — verifying and `--no-verify` alike — stays red, because manifest resolution rewrites the path dep into a registry dep that does not exist. This is the only thing that clears pre-release Gate 5."
    why_human: "CLAUDE.md explicitly forbids self-serving the publish cascade. Recorded as a caveat in plans 02-02 and 02-08 and in STATE.md Blockers/Concerns; verified here as expected state, NOT a phase regression."
  - test: "Decide the fix route for FINDING-W2: contracts/contrastive-pair-protocol-v1.yaml FALSIFY-CPP-007 predicts N=512 singleton classes with negative_capacity = C(512,2) = 130816, but no test in the repo uses 512 and the literal 130816 appears nowhere. The largest K exercised is 128."
    expected: "Either add a K=512 case to `state_report_grows_linearly_in_k_under_a_fixed_budget` (cheap — the layout is `vec![1u64; 512]`), or amend the prediction text to the K in {8, 32, 128} sweep that is actually run."
    why_human: "Choice between widening a test and narrowing a contract claim. The obligation OBLIG-CPP-KN-ADVERSARIAL IS discharged in substance at K=32 and K=128; only the recorded prediction's literal numbers are unexercised."
  - test: "Decide the fix route for FINDING-W1: crates/apr-cli/src/commands/data_contrastive.rs lines 25-31 still carry the module-doc heading \"# `run_pairs` is still Task 1's placeholder\", but `run_pairs` is fully implemented and shipping."
    expected: "Delete or rewrite the stale doc block, or ticket it. A reader of this module is currently told that a working, user-facing command is a placeholder."
    why_human: "Trivial to fix but outside a read-only verifier's remit; needs a fix-now-vs-ticket call."
findings:
  - id: FINDING-D1
    severity: deviation
    title: "setfit profile row bytes changed (source_split: validation/test -> compatibility_test)"
    status: accepted
    where: "crates/apr-cli/src/commands/data_tweeteval.rs:543-545"
  - id: FINDING-W1
    severity: warning
    title: "Stale module doc claims the shipped `run_pairs` command is a placeholder"
    where: "crates/apr-cli/src/commands/data_contrastive.rs:25-31"
  - id: FINDING-W2
    severity: warning
    title: "FALSIFY-CPP-007 predicts N=512 / C(512,2)=130816; no test exercises 512 (max K = 128)"
    where: "contracts/contrastive-pair-protocol-v1.yaml:1013-1020"
  - id: FINDING-W3
    severity: warning
    title: "OBLIG-CPP-CAPACITY-INVARIANT.formal references state_report().retained_bytes, a field that does not exist and that the negative test explicitly bans as a self-reported-size needle"
    where: "contracts/contrastive-pair-protocol-v1.yaml:758"
  - id: FINDING-I1
    severity: info
    title: "proptest-regressions/*.txt hold 8 real counterexamples found during this phase but are gitignored repo-wide (.gitignore:22), so they do not travel to CI or a fresh clone"
    where: "crates/aprender-contrastive-data/proptest-regressions/"
  - id: FINDING-I2
    severity: info
    title: "'thin adapter' is a stretch: data_tweeteval.rs production code is 858 lines / 24 functions, up from the 813-line D-06 baseline. All model/hash/dedup/JSONL logic IS delegated to the crate; the growth is attestation composition plus verification."
    where: "crates/apr-cli/src/commands/data_tweeteval.rs"
  - id: FINDING-E1
    severity: info
    title: "VERIFICATION HAZARD (new, not previously recorded): `diff` under the rtk hook reports \"[ok] Files are identical\" with rc=0 for files whose SHA-256 digests differ. Any emptiness/equality assertion made through `diff` in this repo is unsound. Use `cmp`, `shasum`, or Python."
    where: "environment"
overrides:
  - must_have: "The JSONL row output is byte-identical to the D-06 baseline"
    reason: >
      Scoped to the canonical profile, where it holds and is proven by independent
      re-derivation (FALSIFY-TWEET-EVAL-011). The setfit profile's merged rows deliberately
      carry source_split "compatibility_test" instead of "validation"/"test": D-19 requires
      Split<CompatibilityTest> and 02-03 Gate 2 requires source_split == role, so the two
      cannot both hold. Declared by the schema_version 1->2 bump with an enforced
      no-migration rejection, disclosed in contracts/tweet-eval-stance-benchmark-v1.yaml
      under profiles.setfit.row_source_split, and pinned by a passing test. Row ids and
      manifest source_splits preserve full provenance.
    accepted_by: "Guy Ernest"
    accepted_at: "2026-08-09T11:05:00Z"
---

# Phase 2: Deterministic Pair and Data Protocol — Verification Report

**Phase Goal:** Users can prepare and replay exact few-shot training inputs and bounded contrastive
pairs without split leakage, provenance ambiguity, silent row loss, or Cartesian-product growth.

**Verified:** 2026-08-09T08:26:45Z at `6cfc0f901` on `gsd/phase-2-contract-gate`
**Status:** human_needed
**Re-verification:** No — initial verification

## How this was verified

Goal-backward, at the "a user can…" tier the ROADMAP demands. The five Success Criteria were
exercised **against the real `apr` CLI and the live pinned TweetEval revision
`4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`**, not against SUMMARY claims and not only against the
test suite. Where the phase's own tests assert a property, I additionally constructed an
**independent adversary** (a self-consistent forged selection manifest with a recomputed digest, a
tampered split, a swapped split, a downgraded schema version, a compatibility-profile directory) and
required the shipped binary to reject it.

**Binary pinning (CLAUDE.md rule 3).** `. scripts/apr_bin.sh` refuses `target/debug/apr`: it reports
`0.63.0 (05d484aaf)` against `HEAD = 6cfc0f901`. I did not override the guard blindly — I measured
the gap: `git diff --stat 05d484aaf..HEAD` is **four files, all documentation**
(`.planning/ROADMAP.md`, `.planning/STATE.md`, `02-09-SUMMARY.md`, `docs/examples/tweet-eval-stance.md`).
Zero Rust, zero `Cargo.*`, zero `Makefile`. The binary is therefore byte-equivalent in behaviour to
HEAD for every code path under test, and every CLI result below is attributable to this checkout.

**Measurement hygiene.** Every status was captured directly (`cmd > log 2>&1; rc=$?`), never through
a pipe (CLAUDE.md rule 1). `git status --porcelain` was read through `rtk proxy`. I also discovered
a *new* environment hazard mid-verification and stopped relying on the tool — see FINDING-E1.

## Goal Achievement

### ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | A user can acquire the pinned TweetEval abortion-stance source and produce canonical 587/66/280 JSONL plus exact labels, hashes and provenance without committing tweet text; malformed / duplicate-ID / conflicting / unknown data fails with typed errors, while cross-split duplicate *content* is excluded from the training pool and recorded (D-27). | VERIFIED | Live run of `apr data tweet-eval-stance` against the pinned revision produced **train 587 / validation 66 / test 280** with exact per-class counts (159/319/109, 18/36/12, 45/189/46), per-split SHA-256, six upstream `files_sha256`, `revision_verified: true`, and `license_notice` confirming no vendored text. Exclusion record carries **exactly one coalesced group** (`train:70` ≡ `validation:3`, `detected_by {exact, normalized}`, `label_conflict: false`) with reduced pools 158/319/109 — so D-27's exclude-and-record half is exercised by real data, not a fixture. Typed errors reproduced live: `TweetEval label 9 in train sample 1 is outside 0..3`; `train text/label length mismatch: 2 texts vs 1 labels`. The crate's `ContrastiveDataError` carries 34 variants covering every DATA-02 class (`MalformedRow`, `DuplicateId`, `UnknownLabel`, `LabelTextMismatch`, `InvalidClassCounts`, `ConflictingSourceRole`, `SplitRoleMismatch`, `CrossSplitDuplicateUnderflow`, …). |
| SC2 | A user can select exactly 8/16/32/64 unique canonical-training examples per class for every contracted seed and replay the same ordered selected-ID manifest and semantic hashes. | VERIFIED | **All 40 cells run live** (4 shot counts × 10 contracted seeds): every cell rc=0, `selected == 3·shots`, **unique-ID count == selected** in all 40, perfect per-class balance in all 40, **zero** non-train IDs, **zero** occurrences of the excluded `train:70`, and **40 distinct semantic hashes**. Replay re-run independently on 6 cells (8/17, 16/29, 32/41, 64/53, 16/13, 64/23): identical `semantic_hash`, identical ordered ID list, identical `ledger_hash` every time. |
| SC3 | A user can replay positive and negative pair manifests whose endpoints are distinct selected training IDs, whose targets agree with class identity, whose unordered identities cannot conflict, and whose singleton-class behavior is explicit and versioned. | VERIFIED | Live `apr data pairs --dump` over the seed-13/8-shot selection emitted 384 pairs (192 pos / 192 neg). Independent Python audit of the dump against the selection: **0 endpoints outside the selection, 0 self-pairs, 0 non-canonical orderings** (`lo` ordinal < `hi` ordinal in all 384), **0 target/class mismatches**, **0 conflicting unordered identities**, **0 reversed duplicates**. Singleton behaviour is explicit and versioned in the JSON output: `singleton_policy: negatives_only`, `singleton_policy_version: 1`, `degenerate_policy_version: 1`, `affected_singleton_classes: 0`, plus three `deviation` clauses naming the self-pair divergence from pinned setfit 1.1.3. |
| SC4 | A user can impose a per-epoch pair budget, and increasing the example count under a fixed budget retains `O(examples + pair budget)` state and storage instead of materializing a Cartesian product. | VERIFIED | **Storage, live:** `--budget 64` at shots 8/16/32/64 (24 → 192 examples, 8×) produced dumps of exactly 64 lines and 3105/3120/3114/3110 bytes — flat. **Runtime state, live:** peak RSS 24 526 848 B at 8 shots vs 24 674 304 B at 64 shots — 0.6 % drift under an 8× example growth. **Structural:** `PairLayout` holds exactly three `Vec<u64>` (`offsets`, `pos_prefix`, `neg_prefix`) and no pair storage field; `PairSampler` adds only borrowed `Vec<&[SelectedId]>` slices; `PairIter` is a cursor calling the pure `pair_at`. `state_report()` derives every count from real container lengths — `materialized_pairs: 0` is honest because there is no field to count. Budget binding proven live: `--budget 2000000` and `--budget 100 --hard-cap 50` both exit 5 naming both numbers, never a silent clamp. |
| SC5 | Validation/test endpoints, cross-split duplicate content, and the merged SetFit compatibility test used for model selection are rejected fail-closed, while the manifest proves canonical train/validation/test isolation. | VERIFIED | Five independent fail-closed attacks, all rejected live with rc=5 and a typed, offender-naming message: (1) selecting on a `--profile setfit` directory (merged 346-row test) → `dataset profile mismatch: expected "canonical", got "compatibility"`; (2) pairs against a compatibility directory → same; (3) one tampered train row → `train split hash mismatch` with both digests; (4) `validation.jsonl` replaced by `test.jsonl` → `validation split hash mismatch`; (5) `schema_version` downgraded to 1 → explicit no-migration refusal naming `[2]` and the `--force` remedy. **The strongest single result:** I built a *self-consistent* forged selection manifest — swapped a selected ID for `validation:3` **and recomputed the payload digest** (my recomputation method was proven correct first: `sha256(compact(payload)) == recorded semantic_hash` on the honest file) — and strict replay still rejected it: `pair endpoint "validation:3" is not in the selection (found in validation)`. Isolation is additionally structural: five trybuild compile-fail cases prove a compatibility dataset cannot even be *passed* to selection or replay. |

**ROADMAP score: 5/5 VERIFIED.**

### Plan-level must-haves independently checked

These are the must-haves I judged most fakeable, plus every item the orchestrator flagged for
scrutiny. Each was checked against the codebase, not against its SUMMARY.

| # | Plan | Truth | Status | Evidence |
|---|------|-------|--------|----------|
| P1 | 02-01 | D-06 baseline lands as ONE standalone tracked commit | VERIFIED | `7eb1a67fa feat(data): land TweetEval abortion-stance baseline as-is (D-06, pre-Phase-2)` — 813-line loader + contract + docs + eval F_avg, message records the hash-attestation discipline. Branch `gsd/phase-2-d06-baseline` exists on `origin`. |
| P2 | 02-01 | `pv validate` accepts the tweet-eval contract; tier3 actually reaches it | VERIFIED | Ran `pv validate` on both phase contracts: **0 errors, 0 warnings, "Contract is valid."** Both appear in the explicit `$(CONTRACTS)` list (Makefile:1059-1060), which `contract-validate` iterates. |
| P3 | 02-01 | Every `pv diff` example uses the two-filesystem-path form | VERIFIED | CLAUDE.md:432-442 now shows `git show HEAD~3:… > /tmp/…` then `pv diff /tmp/old.yaml contracts/new.yaml`, and cites `cli.rs:73-78` as the reason. |
| P4 | 02-02 | D-04 boundary is a POSITIVE allowlist + a `cfg(test)`-blind `src/` symbol ban | VERIFIED (non-vacuity probed) | `make contrastive-data-boundary` → rc=0. I did **not** take that at face value: I re-ran the target's own grep pipeline over `tests/` (which legitimately uses `std::fs`/`Path`) and it produced hits, then over `src/` and it produced **0** — so the scan works on this BSD-grep host and the pass is real, not the `grep -oP` vacuity of D-ITEM-01. The recipe also carries explicit anti-vacuity guards on empty `cargo tree` output, a missing allowlist, and an empty allowlist. |
| P5 | 02-02 | Every DATA-02 failure class has a typed variant | VERIFIED | 34 variants in `error.rs`, `#[non_exhaustive]`, including the later-needed `BudgetExceedsHardCap`, `ZeroBudget`, `ZeroHardCap`, `ArithmeticOverflow`, `OrdinalOutOfRange`, `PairTargetMismatch`, `UnsupportedSchemaVersion`. |
| P6 | 02-03 | Profile isolation is TYPE-LEVEL, not runtime | VERIFIED | `cargo test --test ui` → 5/5 compile-fail cases pass. Snapshots name the crate's real types, not rustc phrasing: `&PreparedDataset<Compatibility>`, `expected &PreparedDataset<Canonical>`, `expected &Selection, found &Vec<String>`, `associated function from_jsonl_bytes is private`. |
| P7 | 02-03 | Exact + normalized duplicate edges are COALESCED into one component | VERIFIED | Live: the real pinned dataset yields exactly **one** group with `detected_by {exact: true, normalized: true}` and decrements pool 0 exactly once (159 → 158). Two groups would have double-decremented. |
| P8 | 02-03 | D-27: prepare-time duplicate content is excluded-and-recorded, never fatal | VERIFIED | Live prepare exits 0 with the exclusion recorded; the reduced pools still supply 64 shots (158/319/109 ≥ 64), confirmed by 64-shot selections succeeding on all ten seeds. |
| P9 | 02-03 / 02-05 | The AccessLedger is PERSISTED, not an in-memory value that dies | VERIFIED | `selection-manifest.json` payload carries `access_ledger` **and** `ledger_hash`; `ledger_hash` reproduces identically across independent runs of the same cell. |
| P10 | 02-04 | Fixtures record MEASURED setfit behaviour and CONTRACTED Aprender behaviour, neither derived from the Rust implementation | VERIFIED | `generate_fixtures.py:1054-1102` derives both families from first principles (Aprender contract equations; a reading of `setfit/sampler.py`) and states explicitly that deriving them from Rust would be circular. Each fixture carries the reproducibility triple (`setfit_version 1.1.3`, `uv_lock_sha256 c2b36114…`, `uv_version 0.9.5`) and a three-way agreement rule (measurement = closed form = literal, or abort). The `[4,1]` divergence (measured 22 vs contracted 12) and the K=N `singletons_32` row are both present in both families. |
| P11 | 02-04 | Manifest integrity is real and manifest-relative | VERIFIED | `shasum -a 256 -c manifest.sha256` in `tests/setfit_reference/` → **12/12 OK**; in `tests/goldens/` → **9/9 OK**. The Rust verifier `reference_fixtures.rs` passes (7 tests), and `golden_manifest_coverage_is_total` asserts the embedded `include_bytes!` set and the manifest line set are the same 9 names — so a manifest line nothing checks is itself a failure. |
| P12 | 02-05 | Same seed ⇒ identical ordered list and semantic hash | VERIFIED | See SC2. Also corroborated by an **independent** derivation: `golden_ordered_ids_match_the_independently_derived_digests` pins four ordered-ID digests produced by a Python implementation of the contract equations that never read the Rust source. |
| P13 | 02-05 | The hashed payload EXCLUDES its own digest — the definition is not circular | VERIFIED (independently) | The payload object has 13 keys and **no digest field**; the digest lives in the outer envelope beside `volatile`. I recomputed `sha256(json.dumps(payload, separators=(',',':')))` myself and it equals the recorded `semantic_hash`. Two runs of the same cell produce byte-identical `payload` and `semantic_hash`; the *only* difference in the whole file is `volatile.created_at`. |
| P14 | 02-05 | Replay is strict — a forged manifest cannot reconstitute a Selection | VERIFIED (independently) | Two forgeries built by me: digest-not-fixed → `selection semantic_hash mismatch` naming both digests; **digest-recomputed** → still rejected, naming the offender and the split it came from. |
| P15 | 02-06 | Canonical JSONL rows are byte-identical to the D-06 baseline | VERIFIED | `canonical_jsonl_rows_are_byte_identical_to_the_d06_baseline` re-derives the D-06 wire format by string formatting **in the test**, not by calling the encoder under test, and includes a non-empty-expectation vacuity guard. Passes. |
| P16 | 02-06 | setfit JSONL rows are byte-identical to the D-06 baseline | **UNCERTAIN — deviation** | **False as written.** Every merged row now carries `source_split: "compatibility_test"` (verified live: 346/346 rows) where the D-06 baseline wrote `validation`/`test`. See FINDING-D1 and the suggested override below. |
| P17 | 02-06 | `from_attested_bytes` is the crate-owned identity boundary; a mixed/forged/stale directory fails before any row is exposed | VERIFIED | Three live forgeries rejected (see SC5). The two ingest doors derive `source_hash` *differently* — `from_rows` hashes the canonical re-encoding, `from_jsonl_bytes` hashes the supplied buffer (`split.rs:115` vs `:144`) — so `attestation_round_trip_reproduces_the_dataset_fingerprint` is a genuine two-derivation agreement and **not** a tautology, exactly as its doc comment claims. |
| P18 | 02-06 | The real pinned dataset excludes exactly one group and still supplies 64 shots | VERIFIED (live, twice) | Live CLI run, plus both opt-in network tests run explicitly: `cargo test -p apr-cli --lib data_tweeteval -- --ignored` → **2 passed** (`pinned_upstream_satisfies_the_dataset_contract`, `pinned_upstream_records_exactly_one_coalesced_duplicate_group`). |
| P19 | 02-07 | Negative sampling is O(K), not O(K²) | VERIFIED | Structural: three K-long arrays, no class-pair array anywhere in `PairLayout`. Measured: `total_retained_entries()` is exactly `3K` at K ∈ {8, 32, 128} = {24, 96, 384} under a **fixed** budget — 4× in K gives 4× in state, where a class-pair design would give 16×. At K = N = 32 the shipped design retains 32 negative-weight entries against the rejected design's C(32,2) = 496, and 496 is **read from the contracted fixture**, not typed into the test. |
| P20 | 02-07 | The hard cap BINDS an explicit budget; it never silently clamps | VERIFIED (live) | Both the default-cap and explicit-cap paths exit 5 naming both numbers. |
| P21 | 02-07 | The pair-manifest hash commits the replay tuple | VERIFIED (live) | Same selection ⇒ same `pair_manifest_hash` across independent runs (3 cells); different cells ⇒ different hashes. |
| P22 | 02-08 | The leaky negative is in-band, with a control AND a mirror, through ONE public call | VERIFIED | `negative_leaky.rs` (250 lines) poisons an `UntrustedPairRecord` — the *only* type that can express the attack, since `SelectedId`'s constructor is private — runs the control (`remove the poisoned record ⇒ Ok`) **before** the rejection assertion, asserts the specific variant `EndpointNotInSelection`, requires the message to name the ID, and mirrors that the same call reproduces the sampler's stream **pair-for-pair**. A self-scan requires ≥5 call sites of the single entry point. 4 tests pass. |
| P23 | 02-08 | The materializing negative is RED at K=N through the SAME public `state_report()` call | VERIFIED | `negative_materializing.rs` (500 lines) pins the population (18 336 pairs) **before** bounding it, asserts the failure message names `materialized_pairs=18336` and `exceeds 209`, mirrors that the honest sampler passes the identical call at the identical layout **at a 24 576-pair budget** (so it is not bounded merely by being asked for little), and repeats the whole thing at K = N = 32. A self-scan bans `memory_used` / `retained_bytes` / `bytes_allocated` so the gate cannot drift to self-report. 9 tests pass. |
| P24 | 02-08 | Binding audit is blocking in tier3, and covers every equation | VERIFIED (ran) | `make contract-audit-phase2` → rc=0, **24/24 bound** for contrastive-pair-protocol-v1 and **1/1** for tweet-eval-stance-benchmark-v1, "No binding gaps found." The recipe captures `status=$$?` per contract and `exit 1`s on any unbound — unlike the repo-wide `contract-audit` (D-ITEM-04), which I confirmed by reading has no status capture at all. Wired at Makefile:294 inside tier3. |
| P25 | 02-08 | `official_f_avg` is bound | VERIFIED (contradicts the orchestrator's note) | `binding.yaml:1062-1073` binds it to `entrenar::eval::classification::metrics::f1_average_for_classes`, `status: implemented`, and the function exists at `crates/aprender-train/src/eval/classification/metrics.rs:10` with the caller-facing `MultiClassMetrics::f1_avg_for_classes` at `:91`. The note honestly records that no `#[contract]` attribute was added and that the plan's `ClassificationMetrics` type name was wrong. |
| P26 | 02-09 | select/pairs CLI: attested ingest, required contracted seed, atomic + no-clobber writes | VERIFIED (live) | `--seed 42` → rejected naming all ten contracted seeds; `--any-seed` accepts it and records `seed_mode: "uncontracted"` so an experimental run can never be mistaken for a benchmark cell; `--shots 12` → rejected naming `{8,16,32,64}`; re-running into an existing directory → `Refusing to replace existing file … (pass --force)`. `atomic_write` is the single writer (temp-in-destination → `write_all` → `sync_all` → `rename`) with a `#[cfg(test)]`-only pre-rename fault-injection seam and a `#[cfg(not(test))] const fn` returning `false` in production. |
| P27 | 02-09 | The CLI is a pure filesystem adapter | VERIFIED (with FINDING-W1) | `data_contrastive.rs` imports the crate for attestation, ledger, manifest, pairs, prepared, select and split; no parsing, hashing, selection, sampling or budget resolution lives there. `data_tweeteval.rs`'s one remaining `Sha256` call (`:832`) hashes only the six **upstream source files** for provenance — a CLI concern under D-05 — and says so; all row/split content digests come from `DatasetAttestation`. The module doc, however, is stale (FINDING-W1). |

**Plan-level score: 26/27 VERIFIED, 1 UNCERTAIN.**
**Combined: 31/32.**

### Deviations examined and accepted (not findings)

The orchestrator flagged six judgement calls beyond FINDING-D1. My adjudications:

- **02-06 violated its own acceptance criterion "`git diff --stat prepared.rs` is empty"** by
  extracting `pub(crate) fn from_validated_splits`. **Accepted.** The criterion was a *wave-4
  coordination* constraint against 02-05, which had already completed, so it protected nothing by
  then. The extraction does not widen the public surface (it is `pub(crate)`, and
  `tests/ui/split_constructed_directly.rs` pins that the only doors in remain `from_labeled_rows`
  and `from_attested_bytes`), it is live at four call sites (`prepared.rs:218,407`;
  `attestation.rs:361,405`), and honouring the letter would have left `#[allow(dead_code)]` in place
  **and** made the mandated cross-path fingerprint test a tautology — see P17, where I confirmed the
  two derivations are genuinely different.
- **02-07 changed the API 02-08 depended on** (`PairLayout::from_class_sizes`). **Accepted, and it
  was necessary.** `FewShotSelector::select` admits only `shots_per_class ∈ {8,16,32,64}`, so the
  K = N all-singleton layout DATA-05 requires is genuinely unconstructible through
  `PairSampler::new`. Without the extraction the adversarial case could not be written at all. I
  confirmed DATA-05 **is** exercised at K = N: `the_adversarial_k_equals_n_layout_retains_o_k_state_not_o_k_squared`,
  `the_materializing_sampler_is_red_at_the_adversarial_layout_too`, and
  `honest_state_is_linear_in_the_class_count_across_three_all_singleton_layouts` at K ∈ {8,32,128}.
- **02-05's goldens are mixed provenance.** **Adequately disclosed — in the code, not only the
  SUMMARY.** `manifest.rs:692-707` states plainly which values are algorithm-derived (the four
  `ordered_ids_sha256`, from an independent Python implementation) and which are capture-and-blessed
  (the four `*.payload.json` and both pair goldens), and says outright that the latter "pin the BYTE
  FORM against future drift; they do not independently corroborate it." A reviewer reading the
  source cannot miss it.
- **02-08 changed the capacity bound to `c*(examples + classes)`.** **Correct.** That IS the
  contract's text (`contrastive-pair-protocol-v1.yaml:750-751`), and the plan's
  `c*(examples + budget)` form contradicts the same obligation's budget-independence clause. The
  test's own bound (`1·examples + 3·classes + 8`) is *tighter* and implies the contract at c = 3,
  with every coefficient named rather than rounded.
- **02-09 Task 3 had no executable RED phase** and substituted three induced mutations.
  **Not independently reproducible now** (the mutations were reverted), so I did not score it — but
  the shipped behaviour it was meant to evidence is verified directly by live CLI runs above, which
  is stronger evidence than a RED phase would have been.
- **"Thin adapter."** Measured: FINDING-I2. Defensible in substance, loose as a word.

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/aprender-contrastive-data/` | New publishable crate, D-04 bytes boundary | VERIFIED | 14 `src/` modules, 10 585 lines; 244 tests pass (1 ignored = the deliberate golden re-baseline writer); clippy `--all-targets -D warnings` rc=0; `cargo fmt --check` rc=0; **zero `.unwrap()` anywhere in `src/`**. |
| `…/src/pairs.rs` | O(K) sampler, fallible capacity, binding cap, degenerate policy, `SamplerStateReport`, untrusted DTO | VERIFIED | 2 378 lines. Three O(K) `Vec`s and no pair storage; `triangular_unrank` uses integer binary search, never `sqrt`, explicitly to avoid host-fragile floats. |
| `…/src/select.rs`, `manifest.rs`, `rng.rs`, `buckets.rs` | Philox selection, non-circular manifest, persisted ledger, strict replay | VERIFIED | Non-circularity independently recomputed (P13); strict replay independently attacked (P14). |
| `…/src/prepared.rs`, `split.rs`, `dedup.rs`, `schema.rs`, `hash.rs`, `ledger.rs`, `attestation.rs` | Bytes→typed ladder, typestate splits, coalesced dedup, attested boundary | VERIFIED | Typestate proven by compiler (P6); dedup proven on real data (P7). |
| `…/tests/setfit_reference/` (12 files + manifest) | Measured + contracted fixture families incl. K=N | VERIFIED | 12/12 digests OK; both families present for all six layouts; reproducibility triple on every file. |
| `…/tests/goldens/` (9 files + manifest) | Frozen corpus, selection payloads, pair prefixes | VERIFIED | 9/9 digests OK; coverage proven total. |
| `…/tests/negative_leaky.rs`, `negative_materializing.rs`, `ui.rs` + 5 `ui/*.rs`+`.stderr` | In-band negatives and compile-fail proofs | VERIFIED | 4 + 9 + 5 cases, all pass; snapshots name real types. |
| `contracts/contrastive-pair-protocol-v1.yaml` | 24 equations, 15 obligations, falsification tests | VERIFIED (2 text defects) | `pv validate` clean; 24/24 bound. FINDING-W2, FINDING-W3. |
| `contracts/tweet-eval-stance-benchmark-v1.yaml` | Dataset half of DATA-01/02, schema-version policy, setfit disclosure | VERIFIED | `pv validate` clean; 1/1 bound; `profiles.setfit.row_source_split` + a 10-line note disclose FINDING-D1. |
| `contracts/aprender/binding.yaml` | Binding for every equation of both contracts | VERIFIED | 25/25 bound; the three insertion traps documented in-file. |
| `Makefile` | `$(CONTRACTS)`, `contrastive-data-boundary`, `contract-audit-phase2`, tier3 wiring | VERIFIED | Both new targets rc=0 when run standalone; both wired into tier3 (lines 294, 303). |
| `crates/apr-cli/src/commands/data_contrastive.rs` | select + pairs over the crate API | VERIFIED | 1 816 lines, 36 tests. FINDING-W1 (stale doc only). |
| `crates/apr-cli/src/commands/data_tweeteval.rs` | Thin adapter on the D-05 seam | VERIFIED | 1 552 lines (858 production), 16 + 2 tests. FINDING-I2. |
| `docs/examples/tweet-eval-stance.md` | prepare → select → pairs workflow | VERIFIED | 378 lines covering seed policy, `--any-seed`, budget/hard-cap semantics, atomic writes, fail-closed isolation, schema-version policy. |

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `apr data tweet-eval-stance` | `PreparedDataset::<Canonical>::from_labeled_rows` | D-05 seam: CLI decodes paired files, crate owns the model | WIRED (`data_tweeteval.rs:497`) |
| `apr data select` | `PreparedDataset::from_attested_bytes` | attested ingest before any row is selected | WIRED (proven by live rejection of 3 forgeries) |
| `apr data select` | `FewShotSelector::select` → `SelectionManifest::to_file_bytes` | atomic write of the manifest | WIRED (live) |
| `apr data pairs` | `SelectionManifest::from_bytes` → `Selection::replay` → `PairSampler` | strict replay before any pair | WIRED (proven by live rejection of a digest-consistent forgery) |
| `pairs.rs` | `select.rs` | endpoints are `SelectedId` ordinals; targets from `Selection::label_of` | WIRED (`derive_target`, `pair_at`) |
| `Makefile` tier3 | both phase contracts | `$(CONTRACTS)` → `contract-validate` | WIRED (ran) |
| `Makefile` tier3 | `contracts/aprender/binding.yaml` | `contract-audit-phase2`, blocking | WIRED (ran, rc=0) |
| `Makefile` tier3 | `crates/aprender-contrastive-data` | `contrastive-data-boundary` | WIRED (ran, rc=0, non-vacuity probed) |

## Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data? | Status |
|----------|------|--------|-----------|--------|
| `benchmark-manifest.json` | splits, attestation, exclusions | live upstream fetch → `PreparedDataset` → `DatasetAttestation::from_prepared` | Yes — 587/66/280 real rows, digests reproduce | FLOWING |
| `selection-manifest.json` | ordered examples, ledger, fingerprints | `FewShotSelector::select(&PreparedDataset<Canonical>)` | Yes — 24…192 real train IDs with both per-row hashes | FLOWING |
| pair dump (`--dump`) | 384 `{lo,hi,target}` | `PairSampler::iter_from(0)` over the replayed Selection | Yes — every endpoint resolves into the selection; targets re-derived from real labels | FLOWING |
| `SamplerStateReport` | five counts | real container lengths (`pos_prefix.len()` etc.) | Yes — no self-reported number; `materialized_pairs: 0` is honest because no such field exists | FLOWING |

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Prepare canonical from the live pinned revision | `apr data tweet-eval-stance --output <dir> --json` | rc=0, 587/66/280, 1 exclusion group | PASS |
| Prepare setfit compatibility profile | `… --profile setfit` | rc=0, train 587 + merged test 346 | PASS |
| All 40 benchmark cells | `apr data select` × 4 shots × 10 seeds | 40× rc=0, 40 distinct hashes, 0 invariant violations | PASS |
| Selection replay determinism | re-run 6 cells into fresh dirs | identical semantic hash, ordered IDs, ledger hash | PASS |
| Manifest non-circularity | recompute `sha256(compact(payload))` | equals recorded `semantic_hash`; payload has no digest field | PASS |
| Pair invariants | audit 384-pair dump against the selection | 0 violations across 6 invariant classes | PASS |
| Pair determinism | re-run pairs on 3 cells | identical `pair_manifest_hash` | PASS |
| Storage bound | `--budget 64` at 8/16/32/64 shots | 64 lines, ~3.1 kB, flat | PASS |
| Memory bound | `/usr/bin/time -l`, 8 vs 64 shots, fixed budget | 24.53 MB vs 24.67 MB (0.6 %) | PASS |
| Hard-cap binding | `--budget 2000000`; `--budget 100 --hard-cap 50` | rc=5 both, naming both numbers | PASS |
| Seed policy | `--seed 42`; `--seed 42 --any-seed` | rc=5 naming ten seeds; rc=0 with `seed_mode: uncontracted` | PASS |
| Shots policy | `--shots 12` | rc=5 naming `{8,16,32,64}` | PASS |
| No-clobber | re-run select into an existing dir | rc=5, `--force` remedy named | PASS |
| Compatibility fail-closed | select / pairs on a setfit dir | rc=5 both, `ProfileMismatch` | PASS |
| Tampered / swapped / stale dir | 3 forged directories | rc=5 each, distinct typed error | PASS |
| Digest-consistent forged manifest | swap a selected ID for `validation:3`, recompute digest | rc=5, names the offender **and** the split | PASS |
| Malformed source data | out-of-range label; text/label length mismatch | rc=5 each, typed | PASS |
| Contract validation | `pv validate` × 2 | 0 errors, 0 warnings | PASS |
| Binding audit | `make contract-audit-phase2` | rc=0, 24/24 + 1/1 | PASS |
| D-04 boundary | `make contrastive-data-boundary` | rc=0, non-vacuity probed | PASS |
| Fixture integrity | `shasum -c` × 2 manifests | 12/12 and 9/9 OK | PASS |
| Crate suite | `cargo test -p aprender-contrastive-data` | rc=0, 244 passed, 1 ignored | PASS |
| Lint / format | clippy `--all-targets -D warnings`; `cargo fmt --check` | rc=0 both | PASS |

## Probe Execution — contract-declared test harnesses

The two phase contracts declare 21 executable harnesses. CLAUDE.md warns that
`cargo test -p <crate> <filter>` prints `test result: ok` from a suite matching **zero** tests, so
"the harness is green" is not evidence unless the population is pinned. **I ran every one and
counted the passing tests.** None is vacuous.

| Harness | rc | passed |
|---------|----|--------|
| `cargo test -p apr-cli --lib data_tweeteval` | 0 | 16 |
| `cargo test -p apr-cli --lib data_tweeteval -- --ignored` | 0 | 2 (live network, against the pinned revision) |
| `cargo test -p apr-cli --lib label_index` | 0 | 1 |
| `cargo test -p aprender-contrastive-data --lib attestation` | 0 | 18 |
| `cargo test -p aprender-contrastive-data --lib split` | 0 | 24 |
| `cargo test -p aprender-contrastive-data --test negative_leaky` | 0 | 4 |
| `cargo test -p aprender-contrastive-data --test negative_materializing` | 0 | 9 |
| `cargo test -p aprender-contrastive-data --test reference_fixtures` | 0 | 7 |
| `cargo test -p aprender-contrastive-data --test ui` | 0 | 1 (5 compile-fail cases) |
| `cargo test -p aprender-contrastive-data attestation` | 0 | 19 |
| `cargo test -p aprender-contrastive-data canonical_pair_ordering` | 0 | 1 |
| `cargo test -p aprender-contrastive-data capacity_no_overflow` | 0 | 2 |
| `cargo test -p aprender-contrastive-data dedup` | 0 | 10 |
| `cargo test -p aprender-contrastive-data error` | 0 | 4 |
| `cargo test -p aprender-contrastive-data ledger` | 0 | 14 |
| `cargo test -p aprender-contrastive-data pairs` | 0 | 54 |
| `cargo test -p aprender-contrastive-data prepared` | 0 | 10 |
| `cargo test -p aprender-contrastive-data rng` | 0 | 7 |
| `cargo test -p aprender-contrastive-data select` | 0 | 40 |
| `cargo test -p aprender-contrastive-data split` | 0 | 24 |
| `make contrastive-data-boundary` | 0 | n/a (gate) |

## Requirements Coverage

Every ID mapped to Phase 2 in REQUIREMENTS.md appears in at least one plan's `requirements`
frontmatter. **No orphaned requirements.**

| Req | Claimed by | Status | Evidence |
|-----|-----------|--------|----------|
| DATA-01 | 02-01, 02-03, 02-06 | SATISFIED | SC1 — live 587/66/280 with labels, counts, per-split and per-source hashes, revision provenance, and an explicit non-vendoring notice. |
| DATA-02 | 02-01, 02-02, 02-03, 02-06 | SATISFIED | SC1 + SC5 — 34 typed variants; malformed/length-mismatch reproduced live; split-span and pool-underflow are typed; D-27 exclude-and-record proven on real upstream data and recorded in `exclusions` with `normalization_version`. |
| DATA-03 | 02-02, 02-05, 02-09 | SATISFIED | SC2 — all 40 cells, unique, balanced, replayable, with a stable ordered-ID manifest. |
| DATA-04 | 02-02, 02-04, 02-07, 02-09 | SATISFIED | SC3 — 384-pair audit: labels match, endpoints differ, unordered identities canonical and non-conflicting, singleton policy explicit and versioned. |
| DATA-05 | 02-02, 02-04, 02-07, 02-08, 02-09 | SATISFIED | SC4 — flat storage and flat RSS under an 8× example growth at fixed budget; structural O(K) proven at K = N and by 4× scaling; the materializing counterfactual is RED under the identical gate. |
| DATA-06 | 02-02, 02-03, 02-08 | SATISFIED | SC5 — compatibility profile rejected at both doors, five compile-fail proofs of non-constructibility, and a digest-consistent forged manifest still rejected by strict replay. |

REQUIREMENTS.md already carries all six as `[x]` / `Complete`. **I re-derived each independently
above rather than inheriting the checkmarks**; all six are genuinely delivered at the "a user can…"
tier.

## Anti-Patterns Found

Scanned all 91 files in `9808cd1b8..HEAD` excluding `.planning/`.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | `TBD` / `FIXME` / `XXX` | — | **NONE.** The only match repo-wide is CLAUDE.md:409, which is the text of a `pmat analyze satd` command, not a debt marker. The debt-marker gate is clean. |
| `crates/apr-cli/src/commands/data_contrastive.rs` | 25-31 | `placeholder` in module doc | WARNING | FINDING-W1 — false statement about shipped behaviour. |
| `crates/aprender-contrastive-data/src/pairs.rs` | 2252 | word "placeholder" | INFO | Prose *denying* placeholder status ("…are the evidence, not a placeholder for it"). Not a defect. |
| `crates/aprender-contrastive-data/src/select.rs` | 1253, 1258 | word "placeholder" | INFO | Prose explaining why a value is **not** a placeholder. Not a defect. |
| all `src/` of the new crate | — | `.unwrap()` | — | **ZERO.** Repo policy satisfied. |
| all phase files | — | empty impls / hardcoded empty returns | — | None flowing to output. `materialized_pairs: 0` is a structural fact, not a stub (no pair-storage field exists). |

## Findings

### FINDING-D1 — setfit profile row bytes changed (deviation, human decision required)

**Status:** UNCERTAIN — WARNING, human decision requested.
**Where:** `crates/apr-cli/src/commands/data_tweeteval.rs:543-545`.
**Contradicts:** plan 02-06 `must_haves.truths[2]`, "The JSONL row output is byte-identical to the
D-06 baseline".

Measured live: all **346/346** merged rows of the setfit profile carry
`"source_split":"compatibility_test"`; the D-06 baseline wrote `validation` / `test`.

Why I did **not** classify this a BLOCKER:

1. It is **forced**, not incidental. D-19 requires the merged split to ingest as
   `Split<CompatibilityTest>`, and 02-03's Gate 2 requires each row's `source_split` to equal its
   role. Reverting it would break ROADMAP SC5 — the criterion this deviation exists to serve.
2. It is **disclosed in the contract**, not just a SUMMARY: `profiles.setfit.row_source_split` plus a
   ten-line `row_source_split_note` naming the change, the reason, and what is preserved.
3. It is **declared by a version bump with no migration path**, and that policy is enforced — I
   downgraded a manifest to `schema_version: 1` and the CLI refused it, naming `[2]` and the remedy.
4. It is **pinned by a passing test**
   (`setfit_merged_split_carries_the_compatibility_role_and_keeps_id_provenance`), which also
   asserts row IDs still read `validation:N` / `test:N` and the manifest still records
   `source_splits: [validation, test]` — so no provenance is lost.
5. The **canonical** profile — which every ROADMAP SC is about — is byte-identical, proven against a
   wire format re-derived independently inside the test.

#### Suggested override

If accepted, paste into this file's frontmatter and re-run verification:

```yaml
overrides:
  - must_have: "The JSONL row output is byte-identical to the D-06 baseline"
    reason: >
      Scoped to the canonical profile, where it holds and is proven by independent
      re-derivation (FALSIFY-TWEET-EVAL-011). The setfit profile's merged rows deliberately
      carry source_split "compatibility_test" instead of "validation"/"test": D-19 requires
      Split<CompatibilityTest> and 02-03 Gate 2 requires source_split == role, so the two
      cannot both hold. Declared by the schema_version 1->2 bump with an enforced
      no-migration rejection, disclosed in contracts/tweet-eval-stance-benchmark-v1.yaml
      under profiles.setfit.row_source_split, and pinned by a passing test. Row ids and
      manifest source_splits preserve full provenance.
    accepted_by: "REPLACE_WITH_YOUR_NAME"
    accepted_at: "REPLACE_WITH_ISO_TIMESTAMP"
```

### FINDING-W1 — stale module doc claims a shipped command is a placeholder

`crates/apr-cli/src/commands/data_contrastive.rs:25-31` still heads a section
`# \`run_pairs\` is still Task 1's placeholder`. `run_pairs` is fully implemented at `:784-800`
(`pairs_outcome` → JSON or human render) and I ran it successfully many times. Cosmetic, but it is a
false statement about shipped behaviour in a file this phase authored, and it is exactly what a
placeholder scan is supposed to catch.

### FINDING-W2 — a contract prediction no test produces

`contracts/contrastive-pair-protocol-v1.yaml:1013-1020` (FALSIFY-CPP-007) predicts
"N = 512 singleton classes … negative_capacity is C(512, 2) = 130816". The largest K exercised
anywhere is **128** (`state_report_grows_linearly_in_k_under_a_fixed_budget`, K ∈ {8,32,128}), and
the literal `130816` appears **nowhere** in the repository. The obligation
OBLIG-CPP-KN-ADVERSARIAL is discharged in substance — K = N is genuinely tested at 32 and 128, with
the rejected design's C(32,2)=496 read from a fixture — but the recorded prediction is not the
measurement performed. `pv validate` checks shape, not whether a prediction is produced, so nothing
catches this. This is the same class of defect (a contract claim nothing checks) that
`contract-audit-phase2` was added to close, one level down.

### FINDING-W3 — contract `formal` clause names a field that does not exist

`contracts/contrastive-pair-protocol-v1.yaml:758` states
`formal: 'state_report().retained_bytes <= c * (examples + classes) && …'`. `SamplerStateReport` has
no `retained_bytes` field, and `tests/negative_materializing.rs:489-499` **explicitly bans the
string `retained_bytes`** as a self-reported-size needle. The obligation's *prose* ("allocation
counts and container lengths") matches the implementation exactly; only the formal line is stale.
Trivial to fix, worth fixing because the formal line is the machine-facing half.

### FINDING-I1 — proptest counterexamples are gitignored

`crates/aprender-contrastive-data/proptest-regressions/{hash,pairs,rng,select}.txt` exist locally and
record **8 real counterexamples** found while building the sampler (e.g. `which = 1, ordinal = 192`).
`.gitignore:22` (`proptest-regressions/`, pre-existing and repo-wide) excludes them, so those seeds
will not be re-run in CI or in a fresh clone — contradicting the header proptest itself writes into
the file ("It is recommended to check this file in to source control"). Not phase-caused; regression
protection is modestly weaker than it appears.

### FINDING-I2 — "thin adapter", measured

`data_tweeteval.rs` production code is **858 lines across 24 functions** (vs the 813-line D-06
baseline); the file total is 1 552 with tests. The substantive claim holds — typed rows, split
roles, per-row hashes, the dataset fingerprint, cross-split dedup and JSONL codec **are** delegated
to the crate, and the remaining functions are fetch / UTF-8 decode / label parse / ID mint /
manifest serialize / atomic write, all CLI-side under D-05. But "thin" understates a module that
still owns two dozen production functions.

### FINDING-E1 — verification hazard discovered during this run (new)

`diff` under the rtk hook printed **`[ok] Files are identical` with rc=0** for two
`selection-manifest.json` files whose SHA-256 digests differ. I caught it only because I had run
`shasum` first and the two results contradicted each other. This inverts equality assertions the
same way the already-recorded `git status --porcelain` → `ok` bug inverts emptiness assertions.
**Any `diff`-based assertion in this repo is unsound until this is fixed.** Recommend adding it to
the CLAUDE.md environment notes alongside the `git status --porcelain` entry, and using `cmp`,
`shasum` or Python for byte-equality.

## Known-red, pre-existing — NOT scored against this phase

Confirmed as expected state, consistent with the orchestrator's independent measurements and the
plans' own `caveats`:

- `cargo package -p apr-cli` red in **every** form (including `--no-verify`) until the human-approved
  publish cascade. STATE.md:179 records the correction that `--no-verify` does not help, with a
  control (removing the dep line ⇒ rc=0, 581 files). Cleared only by human verification item 2.
- `make tier2` clippy: 24-25 arch-gated SIMD errors, **zero** in any Phase 2 crate (D-ITEM-02).
- `aprender-profile` / `aprender-serve` cannot build on Darwin/arm64, so no green
  `cargo check --workspace --all-targets` is possible here.
- 21 GPU tests in `aprender-train/src/gpu/` fail without a GPU; the phase diff touches none of it.
- `make contract-audit` (repo-wide) prints 132 BIND-001 errors and exits 0 — I confirmed by reading
  Makefile:1091-1098 that the loop body never captures a status. Neither phase contract is among the
  132. The phase's own gate is the correctly-statused scoped `contract-audit-phase2` (D-ITEM-04).
- Both CB-510 guards vacuous on macOS via `grep -oP` (D-ITEM-01); tier2's `cargo test --lib` runs
  zero tests (D-ITEM-03). Both pre-existing, both logged.

## Gaps Summary

**No gaps.** Nothing is missing, stubbed, orphaned, or unwired; no must-have failed in a way that
would block the phase goal, and no debt marker is present anywhere in the phase diff.

The phase goal is achieved and demonstrated end-to-end against the live pinned dataset: a user can
prepare 587/66/280 with full provenance and a recorded cross-split exclusion, select any of the 40
contracted cells reproducibly, replay a strict manifest into a bounded deterministic pair stream, and
be refused — fail-closed, with a typed and offender-naming error — on every leakage, forgery,
profile, schema and budget violation I could construct.

Four items need a human decision before the phase is formally closed: acceptance of the setfit
row-byte deviation (FINDING-D1), the publish-cascade approval that is the only thing clearing
pre-release Gate 5, and the fix routes for FINDING-W2 and FINDING-W1. None of them blocks Phase 3,
which depends on the selection/pair protocol verified above, not on any of the four.

---

_Verified: 2026-08-09T08:26:45Z_
_Verifier: Claude (gsd-verifier) — goal-backward, FORCE stance_
