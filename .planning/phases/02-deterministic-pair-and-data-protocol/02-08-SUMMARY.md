---
phase: 02-deterministic-pair-and-data-protocol
plan: 08
subsystem: contrastive-data
tags: [d25, d26, negative-tests, in-band-negatives, trybuild, compile-fail, binding-registry, contract-audit, cargo-mutants, capacity-invariant, k-equals-n, data-05, data-06]
requires:
  - "02-04 (tests/common loaders + the singletons_32 contracted fixture the K = N case reads 496 from)"
  - "02-06 (the tweet-eval obligations and the dataset_attestation #[contract] sites Task 3 binds)"
  - "02-07 (PairLayout::from_class_sizes, RetainedState/SamplerStateReport, UntrustedPairRecord, validate_pair_records)"
provides:
  - "tests/negative_leaky.rs — the in-band leaky dump negative, its two controls, and the mirror (OBLIG-CPP-LEAKY-RED)"
  - "tests/negative_materializing.rs — the in-band Cartesian sampler, one shared &dyn RetainedState gate, the K = N adversarial case and the scaling contrast (OBLIG-CPP-CAPACITY-INVARIANT, OBLIG-CPP-KN-ADVERSARIAL)"
  - "tests/ui.rs + five reviewed compile-fail .stderr snapshots (OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE)"
  - "tests/common — synthetic_dataset / synthetic_selection / synthetic_id / contracted_negative_capacity"
  - "contracts/aprender/binding.yaml — all 25 Phase 2 equations bound (24 + official_f_avg)"
  - "make contract-audit-phase2 — BLOCKING binding-coverage gate wired into tier3"
  - "six mutation-killing unit tests in pairs.rs / manifest.rs / select.rs"
affects:
  - "02-09 (tier3 now fails if any Phase 2 equation loses its binding; the CLI's own equations must be added to binding.yaml when it lands one)"
  - "anyone re-baselining tests/ui/*.stderr after a toolchain bump — the review rule is in the ui.rs doc"
tech-stack:
  added: []
  patterns:
    - "the negative and the honest subject implement the SAME public trait so the gate is literally one call, not two claims"
    - "every 'must be REJECTED' assertion carries a control proving the defect under test is the ONLY defect"
    - "read the adversarial constant out of a committed fixture rather than typing it, so two artifacts cross-check"
    - "prove a gate can FAIL before declaring it blocking — induce, observe, revert"
    - "scope a blocking gate to what the phase owns when the repo-wide form is already broken, and log the broad form rather than absorbing it"
    - "surviving mutants are killed or individually justified with the reason the mutation is unobservable"
key-files:
  created:
    - crates/aprender-contrastive-data/tests/negative_leaky.rs
    - crates/aprender-contrastive-data/tests/negative_materializing.rs
    - crates/aprender-contrastive-data/tests/ui.rs
    - crates/aprender-contrastive-data/tests/ui/select_on_compat_dataset.rs
    - crates/aprender-contrastive-data/tests/ui/compat_dataset_has_no_validation_witness.rs
    - crates/aprender-contrastive-data/tests/ui/replay_on_compat_dataset.rs
    - crates/aprender-contrastive-data/tests/ui/pairs_from_raw_ids.rs
    - crates/aprender-contrastive-data/tests/ui/split_constructed_directly.rs
  modified:
    - crates/aprender-contrastive-data/tests/common/mod.rs
    - crates/aprender-contrastive-data/src/pairs.rs
    - crates/aprender-contrastive-data/src/manifest.rs
    - crates/aprender-contrastive-data/src/select.rs
    - contracts/aprender/binding.yaml
    - Makefile
    - .planning/phases/02-deterministic-pair-and-data-protocol/deferred-items.md
decisions:
  - "The capacity bound is the CONTRACT's c*(examples + classes), not the plan's c*(examples + budget) — the plan's form contradicts the same obligation's own budget-independence clause"
  - "The blocking tier3 gate is the scoped contract-audit-phase2; the repo-wide contract-audit is NOT wired because it prints 132 BIND-001 errors and exits 0"
  - "cargo-mutants ran at --timeout 20 rather than the plan's 60, from a measured sample: 9% of mutants hang, and 48 hangs x 60 s alone exceeds the whole 2700 s budget"
  - "official_f_avg binds to MultiClassMetrics::f1_avg_for_classes, not the plan's ClassificationMetrics — that type does not exist"
  - "10 of the 22 surviving mutants are individually justified as unobservable rather than killed; each justification names the reason the mutation cannot change output"
metrics:
  duration: ~1h45m
  tasks: 3
  files: 15
  completed: 2026-08-09
requirements: [DATA-05, DATA-06]
---

# Phase 2 Plan 08: The Honesty Gates Summary

Both of the phase's fakeable claims now have an in-band wrong implementation that the real
gate demonstrably catches, and **every gate in this plan was watched failing before it was
trusted** — the leaky record un-poisoned, the materializer made to report honest counts, a
`ui/` case made to compile, and a binding entry deleted.

**Five commits on `gsd/phase-2-contract-gate`. No branch, no PR, no push** (the policy 02-01 set).

| # | Task | Commit | Result |
|---|------|--------|--------|
| 1 | in-band leaky + materializing negatives | `2c8d8e872` | 224 -> 237 crate tests |
| 2 | trybuild compile-fail proofs | `2e1093bbb` | 237 -> 238; five reviewed `.stderr` |
| 3a | binding registry + blocking tier3 gate | `1ec11db66` | 25 BIND-001 -> 0 |
| 3b | mutation-killing tests | `eef887216` | 238 -> 244; 22 missed -> 10 |
| 3c | measured-comment reconciliation | `6376570c0` | tier2 line 3.0 s -> 6.4 s |

---

## Task 1 — the two negatives

### `negative_leaky.rs` (OBLIG-CPP-LEAKY-RED), 4 tests

The poison is an `UntrustedPairRecord`, not a `LabeledPair`, exactly as review finding F13
required: a trusted pair's endpoints are `SelectedId`s whose constructor is private to
`select.rs`, so **the attack is not representable in the trusted type** — which is the
structural half of `split_span_fail_closed` working, not a gap.

| Test | What it establishes |
|---|---|
| `mirror_the_same_call_accepts_the_honest_dump_and_reproduces_the_sampler` | `validate_pair_records` over the untouched 32-record dump returns `Ok` **and equals the sampler's own `Vec<LabeledPair>` pair for pair**. Not `is_ok()` — a validator that accepted everything and returned garbage would satisfy that |
| `a_dump_endpoint_naming_a_validation_row_is_rejected_and_the_id_is_named` | record 7's `lo` replaced by `validation:1-0`; `Err(EndpointNotInSelection)` and the rendered message contains the id |
| `a_dump_record_declaring_the_wrong_target_is_rejected_naming_both_targets` | a cross-class pair relabeled `1.0`; `Err(PairTargetMismatch)` naming both endpoints, `declares target 1` and `derive 0` |
| `every_leakage_assertion_here_goes_through_the_same_public_entry_point` | a runtime-assembled scan proving all five call sites are the same public function, so the negative and the mirror cannot drift into two validators |

**Each negative carries its own control, run BEFORE the rejection assertion.** For the
endpoint case the same list with record 7 REMOVED validates `Ok`; for the target case the
same list with the target restored validates `Ok`. Without those, "the poisoned list is
rejected" is equally satisfied by a gate that rejects everything, or by a list that had
become malformed for an unrelated reason. The test first asserts the poison names a row the
selection genuinely does not contain — otherwise it would be testing nothing about membership.

### `negative_materializing.rs` (OBLIG-CPP-CAPACITY-INVARIANT, -KN-ADVERSARIAL), 9 tests

`MaterializingSampler` keeps the same three O(K) arrays as the honest layout — so the only
difference the gate sees is the materialization — plus every unordered pair of distinct
examples, and it OWNS its example ordinals rather than borrowing them. It implements the
crate's public `RetainedState`, so the gate is **one function** for both subjects:

```rust
fn check_capacity_invariant(subject: &dyn RetainedState, examples: u64, classes: usize)
    -> Result<SamplerStateReport, String>
```

with `bound = C_EXAMPLES*examples + C_CLASSES*classes + C_CONST` = `1*E + 3*K + 8`, each
coefficient derived from the honest sampler's structure and documented at its definition.

| Layout | Honest | Materializing | Bound |
|---|---|---|---|
| 3 classes x 64 (S = 192) | 201 | **18,537** | 209 |
| K = N = 32 singletons | 96 | **624** | 136 |

The materializer is RED at both. Its failure message names `materialized_pairs=18336` and
`exceeds 209`, because a bound that fails without saying which term blew it is not
diagnosable. The mirror asserts the honest sampler passes the identical call **at a budget of
24,576** — otherwise "bounded" would be compatible with "was asked for very little".

**The K = N case reads 496 from the fixture, not from a literal.**
`common::contracted_negative_capacity("singletons_32")` supplies it, and the test also asserts
`common::all_singleton_layout(32) == common::contracted_layout("singletons_32")`, so the
builder and plan 02-04's committed artifact cross-check. Reported at that layout:
`negative_weight_entries == 32`, `class_offset_entries == 32`, `positive_weight_entries == 32`,
`materialized_pairs == 0` — against the 496 the rejected class-pair design would allocate.
Sixty-two times the budget (16 -> 992) changes not one entry.

**Scaling is measured three ways, none of them a single-size assertion.** Real `PairSampler`
over 24 -> 192 examples under one fixed budget: 33 -> 201 (ratio 6.1, allowance 9x), while the
materializer over the same span goes 309 -> 18,537 (ratio 60, i.e. > 4x the example ratio) —
that second half is the control, without which "grew linearly" would be untestable rather than
satisfied. `PairLayout` at `[8,8,8]` vs `[80,80,80]` — the **exact 10x** the plan asked for,
which no `Selection` can express — reports byte-identical state. K = 8/32/128 all-singleton
under one budget: 24/96/384, each 4x step in K a 4x step in state.

### Falsification of both gates — performed, observed, reverted

| Mutation | Predicted | Observed |
|---|---|---|
| un-poison both leaky records | the two rejection tests go RED | `rc=101`, **2 passed / 2 failed**; both `expect_err`s fire with the messages they were written with. The mirror and the entry-point scan stayed green |
| materializer reports `materialized_pairs: 0` (honest structural counts, list untouched) | the capacity negatives go GREEN | `rc=101`, **6 passed / 3 failed**. `the_materializing_sampler_is_red_at_the_adversarial_layout_too` printed the accepted report verbatim — `SamplerStateReport { bucket_entries: 32, …, materialized_pairs: 0 }` — proving the gate's verdict flipped because the REPORT changed while the type and its 18,336-element `Vec` did not. The scaling control also fired: *"the materializer grew only 33 -> 201; it is not behaving quadratically"* |

Both reverted; `git diff` clean before the commit.

---

## Task 2 — five compile-fail proofs

`cargo test -p aprender-contrastive-data --test ui` passes with all five, and five `.stderr`
snapshots are committed. **Every snapshot's first diagnostic line, quoted:**

| Case | First line |
|---|---|
| `select_on_compat_dataset` | ``error[E0308]: mismatched types`` — body: `expected `&PreparedDataset<Canonical>`, found `&PreparedDataset<Compatibility>`` |
| `compat_dataset_has_no_validation_witness` | ``error[E0599]: no method named `validation_witness` found for reference `&PreparedDataset<Compatibility>` in the current scope`` |
| `replay_on_compat_dataset` | ``error[E0308]: mismatched types`` — same pair of types, on `Selection::replay` |
| `pairs_from_raw_ids` | ``error[E0308]: mismatched types`` — body: `expected `&Selection`, found `&Vec<String>`` |
| `split_constructed_directly` | ``error[E0624]: associated function `from_jsonl_bytes` is private`` |

Not one is a syntax error or an unresolved import. Each case file names
`OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE` and DATA-06 in its header comment.

**Falsification of the harness:** case 2 was edited to use `Canonical` instead of
`Compatibility` — a program that DOES compile, because the canonical type really does have
that method. trybuild reported

```
test tests/ui/compat_dataset_has_no_validation_witness.rs ... error
Expected test case to fail to compile, but it succeeded.
...
1 of 5 tests failed
```

`rc=101`. Reverted. That probe doubles as the mirror for case 2: the same call compiles on one
profile and does not on the other, so the failure is about the profile rather than about the
method being absent everywhere.

**Snapshots pin rustc's wording, and the doc says so.** `ui.rs` records the re-baseline command
(`TRYBUILD=overwrite …`) and the rule for reviewing its diff: the five names above must survive
any legitimate reword, because the assertions are about types and visibility.

---

## Task 3a — every equation bound, and a gate that can fail

**Before (plan 02-02's recorded baseline, re-measured at HEAD before the change):**

```
contrastive-pair-protocol-v1.yaml : Total 24, Bound 0 — 24 x [ERROR] BIND-001, rc=1
tweet-eval-stance-benchmark-v1.yaml: Total  1, Bound 0 —  1 x [ERROR] BIND-001, rc=1
```

**After:**

```
contrastive-pair-protocol-v1.yaml : Total 24, Bound 24, Implemented 24, Obligations covered 360
                                    "No binding gaps found."           rc=0
tweet-eval-stance-benchmark-v1.yaml: Total  1, Bound  1, Implemented  1, Obligations covered   9
                                    "No binding gaps found."           rc=0
```

All 25 module paths and signatures were read out of the `#[contract]` sites, not invented.
Three spot-checks against source:

| Equation | Entry | Source |
|---|---|---|
| `pair_stream` | `aprender_contrastive_data::pairs` / `PairLayout::raw_pair_at` | `pairs.rs:733` annotation, `:734` `pub fn raw_pair_at(&self, ordinal: u64) -> Result<RawPair, …>` |
| `budget_resolution` | `…::pairs` / `resolve_budget` | `pairs.rs:379` annotation, `:383` `pub fn resolve_budget(cfg: &PairConfig, class_sizes: &[u64]) -> Result<(u64, bool), …>` |
| `split_ingest_boundary` | `…::split` / `validate_ingest_ladder` | `split.rs:240` annotation, `:244` module-private `fn validate_ingest_ladder(rows, role, decl)` |

Plan 02-02's three traps were honoured: bare filename in `contract:`, `status: implemented`
(never `planned`), insertion before `critical_path:`. All are restated in a comment block at
the head of the Phase 2 entries so the next author does not rediscover them.

### The gate

`make contract-audit-phase2` iterates `$(PHASE2_CONTRACTS)`, captures each audit's status
directly into `status=$$?` (never through a pipe), and exits 1 naming the offending contracts.
Standalone before wiring: **rc=0, 9 s cold, then 1 s / 0 s / 1 s over three warm runs.** Wired
into tier3 beside `$(MAKE) contract-validate`; `make -n tier3` reaches it.

**Its failure mode was induced, observed and reverted.** Deleting the `pair_manifest_hash`
entry from `binding.yaml`:

```
[ERROR] BIND-001: Equation 'pair_manifest_hash' in contrastive-pair-protocol-v1.yaml has no binding entry
FAIL: unbound equations remain in: contracts/contrastive-pair-protocol-v1.yaml
rc=2
```

Restored; `git diff --stat contracts/aprender/binding.yaml` is **+220 / −0**, so the revert is
exact. That check mattered more than usual: 02-02 found that a `contract:` field with a `../`
prefix parses cleanly and binds NOTHING, so this gate could otherwise have been green while
inspecting nothing.

### `official_f_avg`, and a plan error corrected

Bound to `entrenar::eval::classification::metrics::f1_average_for_classes`,
`fn f1_average_for_classes(f1: &[f64], classes: &[usize]) -> Option<f64>` — pre-existing
baseline code in `crates/aprender-train`. **No `#[contract]` attribute was added to
`aprender-train`**: that crate is outside this plan's `files_modified` and a source edit there
is not sanctioned here. The plan named the caller-facing method holder `ClassificationMetrics`;
**that type does not exist** — it is `MultiClassMetrics::f1_avg_for_classes(&self, classes:
&[usize]) -> Option<f64>` (`metrics.rs:91`), which delegates to the free function. The plan
itself said "READ the file to take the exact signature; do not invent one", so the source wins
and the correction is recorded in the entry's own `notes`.

---

## Task 3b — the bounded mutation run (D-26)

**Exact invocation, and it COMPLETED — this is not a partial result:**

```bash
timeout 2700 cargo mutants -p aprender-contrastive-data --no-times --timeout 20 --in-place \
  > /tmp/mutants.log 2>&1; rc=$?      # rc=0
529 mutants tested: 22 missed, 395 caught, 101 unviable, 11 timeouts
```

**`--timeout 20`, not the plan's 60, and the reason is a measurement rather than a preference.**
A `--shard 1/50` sample ran 11 mutants in 89 s — of which **60 s was a single hanging mutant**,
leaving ~2.5 s for the other ten. One hang in eleven extrapolates to ~48 hangs over 529; at the
plan's 60 s that is 2,880 s of hangs alone, which exceeds the entire 2,700 s wall budget before
any real work. At 20 s — still 10x the whole crate suite's 2 s runtime, so the plan's stated
rationale ("generous for a crate whose full suite runs in seconds") is preserved — the run
finished inside the bound. The 11 observed timeouts are all infinite loops in
`triangular_unrank`'s binary search and one in `PairIter::next`; a hang is a behaviour change
the suite detects, just not by asserting.

**Caught rate over viable mutants after the fixes: 407 caught + 11 timeouts of 428 viable
(529 − 101 unviable) = 97.4%; strictly caught-only, 95.1%.** Both above the repo's 85% floor.

### The 22 survivors: 12 killed, 10 justified

Six new tests killed twelve. A **targeted re-run of all 14 mutants at those sites reported
`14 mutants tested: 14 caught`.**

| Survivor | Why it survived | Kill |
|---|---|---|
| `pairs.rs:415` `closed_form > hard_cap` -> `>=` | at `==` the `min` picks the same budget either way; only the reported flag changes, and that flag is copied verbatim into the replay record — a manifest claiming the run differed from the request | `a_default_budget_exactly_equal_to_the_hard_cap_is_not_reported_as_clamped`: boundary asserts `(384, false)`, one-below asserts `(383, true)` |
| `pairs.rs:667` `class_count -> 1` | the closed-form FUNCTIONS were pinned everywhere; the accessors that feed `PairReplayRecord::from_sampler` were not | `layout_capacity_accessors_agree_with_the_closed_forms_on_two_layouts`, over `[8,8,8]` and `[8,4,8]` — two layouts, because one is satisfied by an accessor returning the right constant |
| `pairs.rs:677` `positive_capacity -> 0 \| 1` | as above | same test: `(84, 192)` and `(62, 128)` |
| `pairs.rs:682` `negative_capacity -> 0 \| 1` | as above | same test |
| `manifest.rs:457` `pair_canonical_bytes -> [0;12] \| [1;12]` | **every existing hash test varies the HEADER; none varied the stream.** Under the mutant the digest depends on the record and the pair COUNT and nothing about the pairs — the streamed half of the attestation would attest nothing | `the_pair_wire_encoding_is_lo_then_hi_then_target_bits_little_endian` (field-by-field, plus two different pairs must not encode alike) and `two_streams_under_one_header_produce_different_manifest_hashes` |
| `select.rs:116` `SelectedId::ordinal -> 0 \| 1` | the ordinal is what the pair wire format encodes and what `SelfPair` names; nothing asserted the accessor | `selected_id_ordinals_are_the_positions_in_the_ordered_list` — round-trip over all 24 rows, all ordinals distinct |
| `select.rs:349` `validation_fingerprint_hex -> "" \| "xyzzy"` | the field a replay compares to prove D-19 isolation; two empty strings compare equal | `the_validation_fingerprint_is_the_witness_digest_and_differs_from_the_dataset_one` — equals the witness digest, 64 lowercase hex, and differs from the dataset fingerprint |

**The remaining ten are individually justified, and were RE-RUN to confirm they still survive**
(`20 mutants tested: 10 missed, 10 caught` — exactly the ten below). "All in test code" is not
a justification and none of these is that; each is a mutation whose output is provably
unobservable:

| Survivor | Justification |
|---|---|
| `rng.rs:116` `\|` -> `^` in `assemble64` | `(u64::from(lanes[1]) << 32) \| u64::from(lanes[0])`. The two operands occupy **disjoint** bit ranges — the shift clears bits 0..32, and `u64::from(u32)` clears bits 32..64 — so `\|` and `^` are identical functions here. Provably equivalent |
| `dedup.rs:146` `size[a] < size[b]` -> `<=` | union-by-size is a BALANCING heuristic. Swapping which root wins changes tree shape, never the partition |
| `dedup.rs:146` -> `==` | as above |
| `dedup.rs:146` -> `>` | as above |
| `dedup.rs:150` `size[a] += size[b]` -> `-=` | affects only future balancing decisions. No underflow: the swap above guarantees `size[a] >= size[b]` |
| `dedup.rs:150` -> `*=` | as above. Confirmed unobservable at the OUTPUT: `coalesced_exclusions` keys components by root in a `BTreeMap`, but every group's `member_pairs` is sorted and `groups` is then sorted by those members, so the root choice is normalized away |
| `pairs.rs:687` `PairLayout::strategy -> Default::default()` | `PairStrategy` has exactly one variant (`Oversampling`) and it IS `#[default]`. The mutant returns the same value for every constructible layout. Becomes killable the day v2 adds a second variant |
| `pairs.rs:692` `PairLayout::singleton_policy -> Default::default()` | identical argument for `SingletonPolicy::NegativesOnly` |
| `pairs.rs:965` `PairSampler::affected_singleton_classes -> 0` | **a `Selection` cannot hold a singleton class** — `FewShotSelector::select` enforces `shots_per_class ∈ {8,16,32,64}` — so this accessor returns 0 for every constructible `PairSampler`. That is the same fact that forced `PairLayout` to exist (02-07 Deviation 1); the LAYOUT's version of this accessor is caught, at `[4, 1]` |
| `select.rs:323` `Selection::is_empty -> false` | no constructible `Selection` is empty: `assemble` is `pub(crate)` and both callers draw `shots_per_class >= 8` from at least one class. A mutant that returns the always-correct answer is equivalent |

No gate was weakened to improve the score. Disk was watched throughout (`--in-place`, no copies);
`mutants.out` is 2.5 MB and already gitignored, and the source tree was verified additions-only
after every run (`git diff` over `src/` contains no `-` line).

---

## Task 3c — the phase gate sweep

Every status captured directly (`cmd > log 2>&1; rc=$?`), never through a pipe (CLAUDE.md rule 1).
`rtk proxy` used for every porcelain, grep and `make` count, since the hook prints a literal `ok`
on a clean porcelain path and abridges `make` output.

| Gate | rc | Result |
|---|---|---|
| `cargo test -p aprender-contrastive-data` (all targets) | 0 | **244 passed**, 1 ignored, 8 suites (212 lib + 24 integration + 7 doc + 1 trybuild host) |
| `cargo test -p aprender-contrastive-data --test negative_leaky` | 0 | 4 passed |
| `cargo test -p aprender-contrastive-data --test negative_materializing` | 0 | 9 passed |
| `cargo test -p aprender-contrastive-data --test ui` | 0 | 5 compile-fail cases |
| **cross-crate baseline** (`--lib -- --skip gpu::`) | 0 | **14,254 passed**, 28 ignored, 0 failed |
| `cargo check -p aprender-contrastive-data` / `-p apr-cli` | 0 / 0 | clean |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 | clean |
| `cargo clippy -p aprender-contrastive-data --all-targets --no-deps -- -D warnings` | 0 | clean |
| `cargo fmt -p aprender-contrastive-data --check` | 0 | clean |
| `make contrastive-data-boundary` | 0 | deps subset of allowlist; no fs/net/path under `src/` |
| `make contract-audit-phase2` | 0 | 24/24 and 1/1 bound |
| `make contract-validate` (the tier3 contract path over all 44) | 0 | **44/44** "Contract is valid." (counted through `rtk proxy`; the unproxied run shows only 16 — 02-02's abridging finding, reproduced) |
| `make setfit-feature-matrix` | 0 | the carried-forward Phase 1 D-06 matrix, unaffected by the new crate |
| Phase 1 setfit lib gate | 0 | 162 passed — unchanged |
| Phase 1 setfit conformance suite | 0 | 27 passed, 1 ignored — unchanged |
| `cargo package --no-verify -p aprender-contrastive-data` (clean tree) | 0 | **59 files**, 599.8 KiB |
| `cargo package --no-verify -p apr-cli` | **101** | **KNOWN-RED, expected — see below** |
| `cargo check --workspace` | **101** | **pre-existing, host-specific — see below** |
| `make tier2` | **2** | **RED — pre-existing D-ITEM-02, see below** |

**The baseline reconciles exactly.** 14,248 -> 14,254 is +6, precisely this plan's six new lib
tests (206 -> 212). The 13 integration tests are not in that figure because it is a `--lib` run.
No pre-existing test changed state.

**The `.snap.new` files are intact.** All three
`crates/aprender-train/src/prune/snapshots/*.snap.new` were deleted by the `aprender-train` run
and restored with `git checkout --` before any commit;
`git diff --diff-filter=D --name-only HEAD~5 HEAD` reports **none**.

### `cargo check --workspace` is RED for a Linux-only crate

```
error: renacer requires Linux (ptrace syscall tracing)
error[E0601]: `main` function not found in crate `aprender_profile`
error: could not compile `aprender-profile` (bin "aprender-profile")
```

`crates/aprender-profile/src/main.rs:6` is a `compile_error!` under
`#[cfg(not(target_os = "linux"))]`, and its `fn main` is under `#[cfg(target_os = "linux")]`.
The E0601 is a consequence of the first error, not a second defect. **Control:**
`cargo check --workspace --exclude aprender-profile` exits **0**. Host-specific and
pre-existing; nothing in this plan touches that crate. The plan's acceptance criterion
"`cargo check --workspace` exits 0" is unachievable on Darwin and should be read as the
`--exclude aprender-profile` form.

### `make tier2` is RED, and it is not this plan's doing

D-ITEM-02, identical to what 02-03, 02-05, 02-06 and 02-07 recorded. `make` halts at
`cargo clippy -- -D warnings` with **24 errors**, attributed by file: **`aprender-compute` 38
locations, `aprender-zram-core` 3, `aprender-present-terminal` 1, `aprender-core` 1,
`aprender-serve` 1 — zero in `aprender-contrastive-data`.** Its only appearance in the whole
tier2 log is the `Checking aprender-contrastive-data v0.63.0` line: it compiled clean. CI is all
`[self-hosted, X64, Linux]` and never lints the aarch64 arms.

Because make halts, tier2's later steps never ran under `make`. All were run individually and
are green: the Phase 1 setfit lib gate (162), the Phase 1 conformance suite (27), and the
contrastive-data suite (three warm runs, 6.48/6.34/6.46 s, rc=0 each).

---

## KNOWN-RED, EXPECTED, NOT A REGRESSION — the publish cascade

`cargo package -p apr-cli` is red from wave 2 through phase exit, **verifying or not**:

```
error: failed to prepare local package for uploading
Caused by:
  no matching package named `aprender-contrastive-data` found
  location searched: crates.io index
```

**The plan's acceptance criterion "Both `cargo package --no-verify` runs exit 0" is FALSIFIED**,
as plan 02-02 measured and as the orchestrator's brief restates: `--no-verify` skips the
packaged-crate BUILD, not the MANIFEST RESOLUTION that rewrites the path dependency into a
registry dependency. 02-02's control (removing the dependency line makes the identical command
exit 0 with 581 files) attributes the cause unambiguously; it was not re-run here because
nothing about that finding changed.

**What IS gateable and is green:** `cargo package --no-verify -p aprender-contrastive-data`,
rc=0, 59 files (19 at 02-02 — the growth is this phase's `src/`, `tests/`, goldens and
fixtures). The crate ships its own evidence: the packaged archive contains
`tests/negative_leaky.rs`, `tests/negative_materializing.rs`, `tests/ui.rs` and all five
`ui/*.rs` + `ui/*.stderr`.

**Required publish order: `aprender-contrastive-data` BEFORE `apr-cli`.** That is a
human-approved release action; CLAUDE.md forbids self-serving it and **nothing was published
here**. `/gsd:verify-work` must read a red `pre-release` Gate 5 as this expected state.

---

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] The plan's capacity bound contradicts the obligation it discharges**

- **Found during:** Task 1, defining the shared gate.
- **Issue:** the plan specifies `total_retained_entries() <= C_LINEAR * (examples + budget) + C_CONST`.
  `OBLIG-CPP-CAPACITY-INVARIANT` states the bound as `c * (examples + classes)` **and** says
  retained state is "INDEPENDENT of the budget". A bound with `budget` in it cannot express
  budget-independence — worse, it grows the allowance exactly when the pair space grows, which
  is when a materializing implementation most needs catching.
- **Fix:** the contract's form, tightened to two coefficients read off the honest structure:
  `1*examples + 3*classes + 8`. This IMPLIES the contract's `c*(examples+classes)` at `c = 3`,
  so satisfying it satisfies the obligation. Budget-independence then becomes a separate,
  stronger assertion (`draining_the_whole_stream_changes_no_retained_entry`, and the 16 -> 992
  budget comparison at K = N) rather than something the bound quietly permits.
- **Files modified:** `crates/aprender-contrastive-data/tests/negative_materializing.rs`

**2. [Rule 1 — Bug] The plan names a type that does not exist**

- **Found during:** Task 3a, authoring the `official_f_avg` entry.
- **Issue:** the plan directs the binding to "the method form `ClassificationMetrics::f1_avg_for_classes`".
  No `ClassificationMetrics` type exists in `crates/aprender-train`; the method is on
  `MultiClassMetrics` (`src/eval/classification/metrics.rs:91`).
- **Fix:** the entry binds the free function `f1_average_for_classes` (the annotated-shaped
  surface) and its `notes` record `MultiClassMetrics::f1_avg_for_classes` as the caller-facing
  method plus the plan's naming error, so a reader who checks the plan against the registry is
  not left guessing which is wrong.
- **Files modified:** `contracts/aprender/binding.yaml`

**3. [Rule 3 — Blocking] The plan's `--timeout 60` cannot finish inside the plan's own wall bound**

- **Found during:** Task 3b, sampling before committing to a 45-minute run.
- **Issue:** measured on a `--shard 1/50` sample — 11 mutants, 89 s, of which one hanging mutant
  consumed 60 s. At ~9% hang rate, 529 mutants implies ~48 hangs; 48 x 60 s = 2,880 s > the
  2,700 s `timeout` the plan sets. The run would have been reported as a partial result for a
  reason entirely within our control.
- **Fix:** `--timeout 20`, which is still 10x the crate suite's ~2 s and therefore keeps the
  plan's own stated rationale. The full 529-mutant run then COMPLETED (rc=0). Eleven hangs were
  observed, consistent with the sample.
- **Files modified:** none (invocation only; recorded verbatim above)

**4. [Rule 3 — Blocking] Both self-scanning tests tripped on their own needles**

- **Found during:** Task 1, first run of `negative_materializing.rs` (`8 passed, 1 failed`).
- **Issue:** the "no self-reported size" scan searched its own source for the literal
  `"retained_bytes"`, which appears in the assertion itself. A guard that is permanently red is
  not a guard.
- **Fix:** every needle in both files is assembled at runtime
  (`format!("{}{}", "retained_", "bytes")`), the pattern `detach_negative.rs` already uses for
  exactly this reason, with the reason commented at the site.
- **Files modified:** `crates/aprender-contrastive-data/tests/negative_materializing.rs`

**5. [Rule 2 — Missing critical functionality] The `mutants.out` artifact and the `.gitignore`**

- **Found during:** Task 3b.
- **Issue/outcome:** `cargo mutants` writes `mutants.out/` (2.5 MB) into the repository root.
  Checked rather than assumed: `git check-ignore -v mutants.out` reports
  `.gitignore:26:**/mutants.out*/`. Already covered; no change needed and none made. Recorded
  because "it did not appear in `git status`" is not by itself evidence that it is ignored
  rather than that the hook abridged the output.

### Scope corrections against the orchestrator brief

- **`make contract-audit` is not "already RED" — it is worse.** The brief states it is red at
  HEAD from Phase 1's 10 BIND-001 errors. Measured: it reports **132** BIND-001 errors across
  **38 of the 44** contracts and **exits 0 anyway**, because its loop body never reads the
  audit's status. Phase 1's setfit contract is the largest single contributor at 10; the other
  122 are spread over the kernel contracts. The brief's operational conclusion is unchanged and
  was followed — wire the scoped `contract-audit-phase2`, not the broad target — but the reason
  is stronger: wiring it would turn tier3 red on 132 pre-existing gaps, and leaving it is a
  target that checks nothing. Logged as **D-ITEM-04**, not fixed.
- **`$(CONTRACTS)` holds 44 entries, not 46.** Counted; both the Makefile comment and D-ITEM-04
  were corrected before commit rather than shipping the wrong number.

---

## Threat Model Dispositions

| Threat | Disposition |
|---|---|
| T-02-25 vacuous leakage gate | **mitigated** — in-band `UntrustedPairRecord` negative RED + mirror GREEN under the SAME `validate_pair_records` call, two controls proving the poison is the only defect, and a runtime-assembled scan proving there is one call site rather than two validators. The gate was observed failing (un-poisoned: 2 passed / 2 failed) |
| T-02-26 unbounded-memory sampler passing tests | **mitigated** — structural invariant read through the public `RetainedState::state_report`, materializer RED at both 3x64 and K = N, honest mirror GREEN at a 24,576-pair budget, three independent scaling measurements. The gate was observed flipping to GREEN when only the REPORT changed |
| T-02-27 leakage expressible through the public API | **mitigated** — five trybuild compile-fail proofs with reviewed committed `.stderr`, each naming a real type/method/visibility; the harness was observed rejecting an unexpected success |
| T-02-28 tests that pass by construction | **mitigated** — 529-mutant scoped run completed under a bounded wall clock; 12 survivors killed with a 14/14 targeted re-run, 10 individually justified and re-confirmed |
| T-02-46 contract obligations bound to nothing | **mitigated** — 25/25 equations bound, `contract-audit-phase2` BLOCKING in tier3, failure mode induced (`BIND-001: pair_manifest_hash`, rc=2) and reverted exactly (+220/−0) |
| T-02-47 publish cascade on an unpublished dependency | **mitigated as far as it can be** — the crate-only `--no-verify` package is green (59 files) and gated; the `apr-cli` form is honestly declared un-gateable until the human-approved publish, with the required order recorded |

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or trust-boundary schema change.
The new test files read only committed fixtures under `tests/`, outside the D-04 library
boundary (`make contrastive-data-boundary` green).

## Known Stubs

None. Every artifact this plan declared exists and runs in a plain
`cargo test -p aprender-contrastive-data` — no feature gate, no `#[ignore]`.

## Notes for later plans

- **02-09** — tier3 now FAILS if a Phase 2 equation loses its binding. If the CLI plan adds a
  `#[contract]` site or a new equation to either phase contract, it must add the matching
  `binding.yaml` entry in the same commit. The three traps are in the comment block above the
  Phase 2 entries; the shortest check is `make contract-audit-phase2`.
- **02-09** — `tests/common/mod.rs` now ships `synthetic_dataset(classes, train_per_class,
  ledger)`, `synthetic_selection(classes, train_per_class, seed, shots)` and
  `synthetic_id(role, label, index)`, all built through the real ingest ladder. A CLI test that
  needs a canonical dataset without touching the golden corpus should use them.
- **Anyone bumping the toolchain** — `tests/ui/*.stderr` pin rustc's wording. Re-baseline with
  `TRYBUILD=overwrite cargo test -p aprender-contrastive-data --test ui` and review the diff
  against the five names listed in `ui.rs`'s doc comment before committing it.
- **Whoever picks up D-ITEM-04** — the one-line fix (read the audit's status) turns
  `contract-check` red on 132 equations across 38 contracts. The gradation the registry offers is
  `status: pending`, which yields a BIND-004 *warning* instead of a BIND-001 *error*; that is
  probably the honest first step.
- **Out-of-scope observation, unchanged since 02-05:** `crates/apr-cli/src/commands/nf4_classifier.rs`
  still emits two `unused_mut` warnings during the baseline run. Untouched, per the executor
  scope boundary.

## Measurement notes (CLAUDE.md Verification Discipline)

- **Rule 1 — statuses captured directly.** Every `rc` above came from `cmd > log 2>&1; rc=$?`.
  `rtk` rewrites `cargo test` output into a summary line, so per-suite counts were read from the
  raw logs; `make contract-validate`'s 44/44 was counted through `rtk proxy` because the
  unproxied run abridges to 16.
- **Rule 2 — mechanisms proven, not labelled.** "The gate reads the report, not the type" is
  backed by the mutated run's printed `SamplerStateReport`, not by the design intent. "The
  mutation run completed" is backed by `rc=0` plus a total line summing to 529.
- **Rule 6 — one failing input is an anecdote.** The scaling claim uses three spans, the accessor
  test uses two disagreeing layouts, and the K-linearity test uses three values of K.
- **`bashrs` is NOT installed on this host** (CLAUDE.md mandates it over shellcheck). This plan
  added shell logic to the Makefile — the `contract-audit-phase2` loop — and it could not be
  linted. **shellcheck was not substituted.** The recipe was reviewed by hand against bashrs's
  rules (status captured into `status=$$?` immediately after the command and never through a
  pipe, every `$$var` quoted, no `ls` iteration, explicit `exit 1` on failure) and `make -n`
  used to prove it parses; `make -n tier2` and `make -n tier3` both exit 0.
- **`cargo test` takes ONE positional.** Every command quoted here carries at most one, and each
  was executed rather than transcribed.
- **`cargo-kani` remains uninstalled** and no `#[kani::proof]` harness exists; this plan added
  none and claims none.
- **Disk pressure was real and is reported.** Free space fell from 30 GB to 3.6 GB across the
  three mutation runs and the gate sweep. `--in-place` was used precisely to avoid per-mutant
  tree copies. Nothing was deleted, no `cargo clean` was run, and no ENOSPC occurred.

## Self-Check: PASSED

| Item | Status |
|---|---|
| `crates/aprender-contrastive-data/tests/negative_leaky.rs` | FOUND, contains `EndpointNotInSelection` and `validate_pair_records` |
| `crates/aprender-contrastive-data/tests/negative_materializing.rs` | FOUND, contains `RetainedState` and `state_report` |
| `crates/aprender-contrastive-data/tests/ui.rs` | FOUND, contains `compile_fail` |
| `crates/aprender-contrastive-data/tests/ui/` | FOUND — 5 `.rs` + 5 `.stderr` |
| `contracts/aprender/binding.yaml` | FOUND, contains `contrastive-pair-protocol-v1.yaml`; audit 24/24 and 1/1 |
| `Makefile` | FOUND, contains `contract-audit-phase2` at 5 lines: `.PHONY` (22), tier3 invocation (294), `PHASE2_CONTRACTS` comment (1063), its own evidence comment (1121), target definition (1136) |
| `2c8d8e872` `2e1093bbb` `1ec11db66` `eef887216` `6376570c0` | FOUND (5/5) |
| Three `.snap.new` files under `crates/aprender-train/src/prune/snapshots/` | intact; `git diff --diff-filter=D HEAD~5 HEAD` empty |
| `cargo test -p aprender-contrastive-data` | 244 passed, 1 ignored, rc=0 |
| cross-crate baseline | 14,254 passed, 0 failed, rc=0 |
