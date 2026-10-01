---
phase: 02-deterministic-pair-and-data-protocol
plan: 07
subsystem: contrastive-data
tags: [tdd, pair-sampler, o-k-negatives, capacity-invariant, budget-fail-closed, degenerate-policy, untrusted-ingest, replay-hash, goldens, d09, d11, d12, d14, d15, data-04, data-05]
requires:
  - "02-02 (crate skeleton, the ContrastiveDataError pair variants, contrastive-pair-protocol-v1 equations)"
  - "02-04 (tests/common loaders and the six contracted fixture rows, incl. the [1]*32 K = N layout)"
  - "02-05 (rng::derive_key/bounded + the five frozen pair domain strings, Selection/SelectedId/label_of/ids_in_class, the golden corpus and its include_bytes! verifier)"
provides:
  - "pairs.rs — CanonicalPair, fallible capacity math, DEFAULT_HARD_CAP, resolve_budget, classify_degenerate, PairLayout (the O(K) core), PairSampler/PairIter, SamplerStateReport + RetainedState, UntrustedPairRecord + parse_pair_dump + validate_pair_records"
  - "manifest.rs — PairReplayRecord + PAIR_DEVIATION_CLAUSES + pair_manifest_hash (header-committed, streamed) + dump_pairs<W: Write>"
  - "tests/pair_counts.rs — the fixture-driven count host (mod common;)"
  - "tests/common — all_singleton_layout, contracted_layout, ADVERSARIAL_BUDGET for 02-08's capacity gate"
  - "tests/goldens — two committed 32-pair dumps under manifest.sha256 (now 9 files), verifier coverage asserted total"
affects:
  - "02-08 (validate_pair_records, RetainedState::state_report, UntrustedPairRecord, PairLayout::from_class_sizes — see the handoff section, it is a contract)"
  - "02-09 (PairConfig / PairSampler / PairReplayRecord / dump_pairs are the CLI's whole pair surface)"
  - "Phase 3 (the epoch-independent, offset-resumable stream the trainer consumes)"
tech-stack:
  added: []
  patterns:
    - "the sampler's retained state is a separate PUBLIC type built from bare class sizes, so the adversarial layout a Selection cannot express stays reachable from outside the crate"
    - "an equation's evaluation order may differ from its formula when the alternative overflows earlier, provided a test pins the two derivations against each other"
    - "a diagnostics object measured structurally and implemented by both the honest and the deliberately-wrong sampler, so the gate is one call rather than two claims"
    - "non-quadratic behaviour proven by SCALING under a FIXED budget, never by a single-size assertion"
    - "vacuity guards on every relation over a collection: an empty stream and two empty vectors satisfy almost anything"
    - "the dump writer and the untrusted reader share ONE serde type, so the two schemas cannot drift into almost-agreement"
key-files:
  created:
    - crates/aprender-contrastive-data/tests/pair_counts.rs
    - crates/aprender-contrastive-data/tests/goldens/pairs_seed13_shots8_first32.jsonl
    - crates/aprender-contrastive-data/tests/goldens/pairs_seed17_shots8_first32.jsonl
  modified:
    - crates/aprender-contrastive-data/src/pairs.rs
    - crates/aprender-contrastive-data/src/manifest.rs
    - crates/aprender-contrastive-data/src/select.rs
    - crates/aprender-contrastive-data/tests/common/mod.rs
    - crates/aprender-contrastive-data/tests/reference_fixtures.rs
    - crates/aprender-contrastive-data/tests/goldens_regenerate.rs
    - crates/aprender-contrastive-data/tests/goldens/manifest.sha256
    - Makefile
decisions:
  - "PairLayout is a separate PUBLIC type holding the sampler's entire retained state, because a Selection always carries shots_per_class in EVERY class and therefore cannot express the K = N all-singleton layout DATA-05 must survive"
  - "negative_capacity accumulates the running prefix rather than (S^2 - sum n^2)/2: same O(K), same quantity, but it overflows exactly when the RESULT does instead of long before"
  - "NoPairCapacity is checked before the default budget resolves, following the degenerate equation's INVARIANT prose rather than its formula line, which would have reported ZeroBudget for a layout with no pair space"
  - "PairReplayRecord::from_sampler takes ONLY the sampler; the plan's extra selection and config arguments could only ever disagree with the stream they attest"
  - "A swapped untrusted record is canonicalized, not refused: orientation carries no information, so rejecting it would be rejecting a spelling"
  - "Both untrusted_pair_ingest and split_span_fail_closed annotate the PUBLIC validate_pair_records; the macro accepts stacking (verified by compiling both forms) and a private-function binding is harder to audit"
metrics:
  duration: ~2h45m
  tasks: 3
  files: 11
  completed: 2026-08-09
requirements: [DATA-04, DATA-05]
---

# Phase 2 Plan 07: Bounded, Deterministic, Replayable Pair Protocol Summary

The pair sampler retains three K-long arrays and nothing else. At K = N = 32 singleton
classes it reports 32 negative-weight entries where the rejected class-pair design would
report 496 — and that contrast is not an argument in a comment, it is a mutation that was
applied, observed, and reverted.

**224 tests green in the crate** (206 lib + 11 integration + 7 doc, 1 ignored), up from 158.
Cross-crate baseline **14,186 -> 14,248**, and the delta is exactly the 62 new lib tests.

## Seven commits, RED before GREEN in every cycle

| # | Task | Commit | Result |
|---|------|--------|--------|
| 1 RED | pair identity, capacity, budget, degenerate policy | `eceaecd9f` | 24 lib: **2 passed / 22 failed**; 0/4 integration |
| 1 GREEN | `CanonicalPair`, checked capacity, binding cap, total policy | `5ccd32908` | 24 lib + 4 integration pass |
| 2 RED | sampler, diagnostics, untrusted ingest | `675b825cf` | **0 of 26 new tests pass** (24 = the Task 1 set) |
| 2 GREEN | `PairLayout`, `PairSampler`, `RetainedState`, validation | `159abaf45` | 50 lib pairs tests pass |
| 3 RED | replay record, hash, dump, goldens | `addd0db8e` | 36 passed / **11 failed** |
| 3 GREEN | header-committed streamed hash, dump, two pair goldens | `49ab9e9f9` | 224 crate tests pass |
| — | tier2 runtime re-measurement | `4a2b267c8` | standing instruction discharged |

**Two RED runs had to be corrected before they were evidence.** Task 2's first RED showed
three green tests: `positives_share_a_class_and_negatives_do_not` iterated an EMPTY stream,
`validate_pair_records_accepts_the_samplers_own_pairs` compared two EMPTY vectors, and
`pair_at_and_iter_from_beyond_the_budget_are_typed_errors` passed because with budget 0 every
ordinal is out of range. Each now pins its population first and the corrected RED is 0 of 26.
Task 3's round-trip test compared two all-zero digests, so it gained
`assert_ne!(first, [0u8; 32])`. This is 02-04's vacuity lesson recurring three times in one
plan; the guards are commented at each site so they are not later removed as noise.

Task 1's two legitimate RED-greens are `degenerate_policy_both_kinds_alternate` (the
placeholder returns `Both` by construction) and the version-tag constants. Task 3's two are
`pair_goldens_agree_with_an_independent_rederivation` — a Task 2 capability that Task 3 PINS
rather than introduces — and the record-fields test, whose `from_sampler` needed no stub.

## The DATA-05 blocker: how the O(K) claim is defended

### The scheme

`w_c = n_c · (S − n_c)` — one array of K entries whose sum is `2 · negative_capacity`. Draw
the first class `j` against those weights, the first endpoint uniformly inside `j`, then
`u ∈ [0, S − n_j)` mapped past class `j`'s contiguous block (`u < offset_j ? u : u + n_j`),
resolving the second class by binary search over the O(K) offsets. **No array indexed by
class pairs exists anywhere.**

The equivalence is algebra, and it is in the doc comment: every ORDERED cross-class endpoint
pair has probability `1 / Σw`, there are `Σ_c n_c(S − n_c) = 2·Σ_{j<k} n_j n_k` of them, so
each unordered class pair `{j,k}` receives weight proportional to `n_j · n_k` — exactly what
D-14 specifies. The rewrite is a representation change, not a semantics change.

### Measured, not restated

- **Marginals.** 20,000 negative draws on `[3, 5, 7]`. Expected 15/71, 21/71, 35/71; observed
  within 0.02 (≈ 9σ of headroom), and the three frequencies are asserted in strict order so a
  uniform-over-class-pairs bug cannot hide inside the tolerance. On `[2, 2]` all four
  cross-class endpoint pairs are reached.
- **K = N.** `state_report()` at `[1; 32]` under a FIXED budget of 16 reports
  `negative_weight_entries == 32`, `class_offset_entries == 32`, `materialized_pairs == 0`.
  The test also names 496 — the rejected design's array length, read from `negative_capacity`
  — so the contrast is explicit rather than implied.
- **Scaling, not inspection.** K = 8, 32, 128 all-singleton layouts under the SAME budget:
  `total_retained_entries()` is 24, 96, 384. The test asserts the 4× step in K is a 4× step
  in state, which quadratic growth (16×) fails. A single-size assertion cannot distinguish
  O(K) from O(K²) and none is relied on.
- **Budget independence.** `state_report()` before and after draining the whole 384-pair
  stream is equal, and `materialized_pairs` is 0 at every layout including the 24,576-pair
  one.

### The falsification was performed

The contract's `qa_gate.falsification` names three mutations. All three were applied, run,
and reverted:

| Mutation | Predicted | Observed |
|---|---|---|
| replace the per-class weight array with an enumeration of unordered class PAIRS | 3-class tests stay green, K ≈ N turns RED | **46 passed, 4 failed.** `state_report_weight_arrays_are_k_long_at_k_equals_n` reported `left: 496, right: 32` — literally the rejected design's length. The K-scaling test, the cross-layout proptest and the `[3,5,7]` marginal test also went red; every three-class-only test stayed green, which is the point |
| delete the equality branch of `CanonicalPair::new` | `canonical_pair_ordering` RED | RED at `minimal failing input: a = 2, b = 2` |
| silently clamp an explicit over-cap budget | `BudgetExceedsHardCap` tests RED | 48 passed, 2 failed: `got Ok((10000, false))` where the error was required |

## Handoff contract for plan 02-08 — read this before writing the gates

The three names 02-08's `key_links` declare all exist, with these exact shapes:

```rust
// crates/aprender-contrastive-data/src/pairs.rs
pub fn validate_pair_records(recs: &[UntrustedPairRecord], sel: &Selection)
    -> Result<Vec<LabeledPair>, ContrastiveDataError>;
pub struct UntrustedPairRecord { pub lo: String, pub hi: String, pub target: f32 }
pub fn parse_pair_dump(bytes: &[u8]) -> Result<Vec<UntrustedPairRecord>, ContrastiveDataError>;
pub trait RetainedState { fn state_report(&self) -> SamplerStateReport; }
pub struct SamplerStateReport {
    pub bucket_entries: usize,
    pub positive_weight_entries: usize,
    pub negative_weight_entries: usize,
    pub class_offset_entries: usize,
    pub materialized_pairs: usize,
}
impl SamplerStateReport { pub fn total_retained_entries(&self) -> usize; }
impl RetainedState for PairLayout {}
impl RetainedState for PairSampler<'_> {}
```

`validate_pair_records` returns `Err(EndpointNotInSelection { id, found_in })` naming the
offending id — `found_in` is the string `"unknown"`, deliberately, because the validator's
universe is the Selection so it can prove absence but cannot locate the id. The
target-poisoning case is `PairTargetMismatch { lo, hi, declared_target, derived_target }`.

**ONE THING 02-08 MUST KNOW THAT THE PLAN DID NOT ANTICIPATE.** The K = N capacity case
**cannot** be built from a `Selection`: `FewShotSelector::select` validates
`shots_per_class ∈ {8, 16, 32, 64}`, so every class of every Selection has at least 8
members and an all-singleton layout is not expressible as one. That is why the sampler's
retained state lives in a separate public type:

```rust
pub struct PairLayout { /* three O(K) arrays, no pair storage */ }
impl PairLayout {
    pub fn from_class_sizes(class_sizes: &[u64], cfg: &PairConfig) -> Result<Self, ContrastiveDataError>;
    pub fn raw_pair_at(&self, ordinal: u64) -> Result<RawPair, ContrastiveDataError>;
    pub fn budget(&self) -> u64;
    pub fn class_count(&self) -> usize;
    pub fn total_examples(&self) -> u64;
    pub fn emitted_kinds(&self) -> EmittedKinds;
    /* + positive_capacity, negative_capacity, class_size, affected_singleton_classes,
         default_was_clamped, strategy, singleton_policy, root_seed */
}
pub struct PairSampler<'a> { /* PairLayout + &'a Selection + K borrowed bucket slices */ }
impl PairSampler<'_> { pub fn layout(&self) -> &PairLayout; pub fn selection(&self) -> &Selection; }
```

So 02-08's K = N case is
`PairLayout::from_class_sizes(&common::all_singleton_layout(32), &PairConfig { budget: Some(common::ADVERSARIAL_BUDGET), ..PairConfig::new(seed) })`,
and its 3×64 mirror is an ordinary `PairSampler` over a 64-shot `Selection`. Both implement
`RetainedState`, so the gate is one call for the honest sampler, the bare layout, and 02-08's
`MaterializingSampler` alike. `tests/common/mod.rs` already ships `all_singleton_layout(k)`,
`contracted_layout(fixture_id)` and `ADVERSARIAL_BUDGET = 16`, and `pair_counts.rs` asserts
the builder agrees with plan 02-04's `singletons_32` fixture so the two artifacts cross-check.

**`bucket_entries` means different things for the two implementors, honestly.** It counts
per-EXAMPLE entries the sampler RETAINS. `PairSampler` holds one borrowed slice per class
covering every selected row, so it reports `S`. A bare `PairLayout` addresses an abstract
index space and retains none, so it reports 0. A materializing sampler reports its own copy.
The bound 02-08 writes should therefore be over `total_retained_entries()` with
`examples`/`classes` taken from the layout under test, not from the report.

**Both contract annotations are on `validate_pair_records`, not on a helper.** The plan
allowed extracting `assert_endpoints_in_selection` if the macro rejected stacked attributes.
It does NOT reject them — verified by compiling both forms — so `untrusted_pair_ingest` and
`split_span_fail_closed` are stacked on the public entry point, which is the path
`binding.yaml` should name. The private helper still exists for readability and carries no
annotation. `split_span_fail_closed`'s other, STRUCTURAL arm needs no binding: it is
`SelectedId`'s private constructor, a type rather than a function.

**Two more contract equations are now bound in code:** `pair_stream` on
`PairLayout::raw_pair_at`, `singleton_policy` on `PairSampler::new`, plus `canonical_pair`,
`positive_capacity`, `negative_capacity`, `default_epoch_budget` (on
`effective_default_budget`, since the contracted equation INCLUDES the clamp),
`budget_resolution`, `pair_stream_degenerate_policy`, `pair_manifest_replay` (on
`PairReplayRecord::to_config`) and `pair_manifest_hash`. Twelve new `#[contract]` sites in
all. The build still emits *"binding.yaml not found … skipping"*, so none is compile-time
enforced until 02-08 wires the registry.

## The hash commits the tuple, and two tests prove it

`pair_manifest_hash = SHA-256(record.to_canonical_bytes() ‖ 0x1E ‖ pair_0 ‖ … )`, streamed
via `pair_at` and never collected. Two configurations were needed, not one, because the
sharper case only appeared while writing the first:

1. **Same budget by two routes.** Configuration A takes the DEFAULT budget under
   `hard_cap = 200` (clamped); configuration B asks for `budget = 200` under the default cap
   (not clamped). Same selection, same seed 31, so the two streams are asserted
   **identical pair for pair** (200 of them) before the hashes are compared. They differ only
   in `default_was_clamped`, and their manifest hashes differ.
2. **Same pair BYTES, different selection.** Pairs are encoded by ordinal, so two DIFFERENT
   selections that share a class layout and a seed produce a literally identical ordinal
   stream — the test asserts that as a `Vec<(u32, u32, f32)>` equality. Only `selection_hash`
   separates them, and it lives inside the digest. A pairs-only hash would collide here with
   no tell at all.

`pair_manifest_hash` additionally REFUSES a record whose `selection_hash` or `budget` does not
describe the sampler it is handed (`SelectionReplayMismatch` naming the field). Without that
the function would happily attest a stream under somebody else's header — the exact failure
the header-inside-the-digest design exists to prevent.

## Every degenerate case has a named test

| Case | Layout | Outcome |
|---|---|---|
| no capacity of either kind | `[1]`, `[]` | `NoPairCapacity { 0, 0 }` at layout construction |
| positives impossible | `[1; 32]` | constructs, `emitted_kinds == NegativesOnly`, budget 992 |
| negatives impossible | `[6]` | constructs, `emitted_kinds == PositivesOnly`, budget 30 |
| both | `[8,4,8]` | `Both`, alternating |
| odd budget | `[8,4,8]` @ 255 | 128 positives / 127 negatives, `|#pos − #neg| ≤ 1` at every prefix, `== 0` at every even prefix |
| zero hard cap | any | `ZeroHardCap` — and it outranks a zero budget, because it is fixed in a different file |
| zero explicit budget | any | `ZeroBudget` |
| explicit over cap | 20,000 vs 10,000 | `BudgetExceedsHardCap { 20000, 10000 }`, never a clamp |
| singleton class | `[4, 1]` | never in a positive pair; all 100 negatives touch it |

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] The plan's `PairSampler`-only design made the phase's headline gate unreachable**

- **Found during:** Task 2, writing the K = N evidence tests.
- **Issue:** the interfaces block gives `PairSampler::new(sel: &Selection, cfg)` as the only
  constructor, and the plan then requires `state_report()` assertions at `[1; 32]` — a layout
  no `Selection` can hold, because `FewShotSelector::select` enforces
  `shots_per_class ∈ {8, 16, 32, 64}`. The same gate is 02-08's headline
  (`OBLIG-CPP-KN-ADVERSARIAL`), and it would have been unwritable from outside the crate.
- **Fix:** the retained state was extracted into a public `PairLayout` built from bare class
  sizes; `PairSampler<'a>` is now that layout plus the borrowed Selection and its bucket
  handles. No `SelectedId` is minted anywhere — `PairLayout` works in ordinal space and only
  `PairSampler` maps to selected ids — so the non-leakage guarantee is untouched.
- **Files modified:** `crates/aprender-contrastive-data/src/pairs.rs`
- **Commit:** `159abaf45`

**2. [Rule 1 — Bug] `(S² − Σn²)/2` reports overflow for layouts whose answer fits**

- **Found during:** Task 1, writing the checked-arithmetic tests.
- **Issue:** the contract's `negative_capacity` formula line computes `S²` first, which
  overflows `u64` long before the capacity does. `[2^32, 65537]` has a negative capacity of
  ~2.8·10^14 — comfortably representable — while `S² ≈ 1.8·10^19` does not survive. The
  function would have returned `ArithmeticOverflow` for a perfectly ordinary answer, and the
  `default_epoch_budget` overflow case could not have been reached at all.
- **Fix:** the running-prefix accumulation `Σ_k n_k · (Σ_{j<k} n_j)`, which is the same O(K)
  cost, is NOT class-pair iteration (the property the contract's invariant protects), matches
  the equation's own postcondition verbatim, and overflows exactly when the result does.
  `negative_capacity_agrees_with_the_sum_of_squares_derivation` pins the two forms against
  each other on seven layouts, so the evaluation order cannot drift into a different quantity.
- **Files modified:** `crates/aprender-contrastive-data/src/pairs.rs`
- **Commit:** `5ccd32908`

**3. [Rule 1 — Bug] The degenerate equation's formula and its invariants disagree on ordering**

- **Found during:** Task 1, implementing `resolve_budget`.
- **Issue:** the formula line orders `effective budget == 0 -> ZeroBudget` BEFORE
  `pos == 0 and neg == 0 -> NoPairCapacity`. Under a default budget the closed form for `[1]`
  is 0, so the literal reading reports `ZeroBudget` — naming the request when the layout is
  what is wrong. The same equation's invariant prose says the opposite, and the plan's Task 1
  behavior list requires `NoPairCapacity` for `[1]` and `[]`.
- **Fix:** the invariant reading is implemented — zero cap (configuration defect), then an
  explicit zero or over-cap budget (request defects, wrong whatever the layout is), then
  absent capacity, then resolution. Each rung names the thing the reader must change, and the
  ordering decision plus its reasoning is a doc comment on `resolve_budget` so it is not
  "corrected" back.
- **Files modified:** `crates/aprender-contrastive-data/src/pairs.rs`
- **Commit:** `5ccd32908`

**4. [Rule 2 — Missing critical functionality] `pair_manifest_hash` accepted a foreign record**

- **Found during:** Task 3.
- **Issue:** the plan's signature takes the record as a parameter, which is what makes the
  header commitment possible — but nothing checked that the record DESCRIBES the sampler. A
  caller could hash stream X under record Y and publish the result as an attestation, which
  is the same class of defect the whole equation exists to close.
- **Fix:** `assert_record_describes` compares `selection_hash` and `budget` first and returns
  `SelectionReplayMismatch` naming the field. Two tests cover it.
- **Files modified:** `crates/aprender-contrastive-data/src/manifest.rs`
- **Commit:** `49ab9e9f9`

**5. [Rule 3 — Blocking] `goldens_regenerate.rs` would have deleted the new pair goldens**

- **Found during:** Task 3, generating the goldens.
- **Issue:** plan 02-05's regenerator rewrites `manifest.sha256` from the list of files it
  itself wrote. Adding pair goldens without teaching it about them means the next re-baseline
  drops two lines from the manifest and turns the `include_bytes!` verifier red for a reason
  that has nothing to do with drift.
- **Fix:** `PAIR_CASES` and `PAIR_PREFIX` added; the regenerator now emits both pair goldens
  and includes them in the manifest. It was RUN to produce the committed files, and it is the
  named re-baseline command.
- **Files modified:** `crates/aprender-contrastive-data/tests/goldens_regenerate.rs`
- **Commit:** `49ab9e9f9`

**6. [Rule 3 — Blocking] `SelectedId` had no way to expose its ordinal**

- **Found during:** Task 1.
- **Issue:** `SelfPair { id: u64 }` must name the offending ordinal and the pair hash encodes
  endpoints by ordinal; `SelectedId`'s field is private to `select.rs` and `pairs` is a
  sibling module.
- **Fix:** `SelectedId::ordinal(self) -> u32`, documented as an accessor and not a
  constructor: reading the number cannot mint a `SelectedId`, so the type remains proof of
  membership in the selection that produced it.
- **Files modified:** `crates/aprender-contrastive-data/src/select.rs`
- **Commit:** `eceaecd9f`

**7. [Rule 3 — Blocking] `tests/common` items are dead code in whichever consumer does not call them**

- **Found during:** Task 2, after adding the shared layout builders.
- **Issue:** `tests/common/mod.rs` is compiled into every integration-test crate that names
  it, so `all_singleton_layout` is dead in `reference_fixtures.rs` while being live in
  `pair_counts.rs`. `cargo clippy --all-targets -D warnings` failed.
- **Fix:** `#[allow(dead_code)]` on the `mod common;` declaration in both consumers, with the
  reason at each site. The alternative — duplicating the definitions per consumer — is exactly
  what 02-04 created the shared module to avoid.
- **Files modified:** `tests/pair_counts.rs`, `tests/reference_fixtures.rs`
- **Commit:** `159abaf45`

### Interface details chosen where the plan was silent or where it was diverged from

- **`PairReplayRecord::from_sampler(sampler)`, not `(sampler, selection, cfg)`.** A sampler
  already borrows its selection and retains its resolved configuration; the extra arguments
  could only ever disagree with the stream the record attests. 02-09 is the only consumer.
- **`PairLayout`, `RawPair`, `RawEndpoint`, `PairKind` are new public names** not in the
  interfaces block. See the handoff section for why.
- **`resolve_budget` and `classify_degenerate` are public**, because `pair_counts.rs` (an
  integration test) has to reproduce every contracted fixture row through the shipped
  resolver rather than through a re-implementation.
- **`PairConfig::new(seed)` and `PairConfig::resolved_hard_cap()`** added so a call site is
  `PairConfig { budget: Some(n), ..PairConfig::new(seed) }` rather than five fields every time.
- **A swapped untrusted record validates to the same `LabeledPair`** rather than being
  refused. The contract lists "canonical ordering" among the validation steps but names no
  error for it; orientation carries no information by D-12, so normalizing is the behaviour
  canonicalization exists for. A test asserts the swapped and honest record sets validate
  equal.
- **`PairIter::next` uses `expect` with the invariant stated**, matching
  `Selection::example_of`'s house style, rather than `.ok()?` — silent truncation of an
  impossible error would end a stream early with nothing red.

## Are the pair goldens algorithm-derived or capture-and-blessed?

**Capture-and-blessed, and labelled as such in the module doc.** Unlike plan 02-05's
`ordered_ids_sha256` digests — which a Python implementation written from the contract
produced — `pairs_seed{13,17}_shots8_first32.jsonl` are this crate's own dump of its own
stream. They pin the BYTE FORM against drift; they do not independently corroborate it.

What corroborates their CONTENT is
`pair_goldens_agree_with_an_independent_rederivation`: it rebuilds the first eight pairs from
`rng::bounded`/`derive_key` plus a **naive enumeration of the class triangle** and a **linear
scan of the weight prefixes**, instead of the sampler's binary-search unranking and
`partition_point`. That is a second implementation of the part that could plausibly be wrong,
standing on RNG primitives plan 02-05 already pinned against independently derived constants.
It is weaker than a second language and it is described that way.

**Tamper detection was observed, not assumed.** One byte of
`pairs_seed13_shots8_first32.jsonl` was changed (`"target":1.0` -> `0.0`) and the suite
re-run: `rc=101` with two independent failures — the digest check naming the file and both
hashes, and the byte-for-byte comparison that would still catch a *coordinated* edit updating
`manifest.sha256` too. Reverted; green again.

## Verification

All statuses captured directly (`cmd > log 2>&1; rc=$?`), never through a pipe. `rtk proxy`
used for every porcelain and grep result, since the hook prints a literal `ok` on a clean
porcelain path and abridges other output.

| Gate | rc | Result |
|---|---|---|
| `cargo test -p aprender-contrastive-data` (all targets) | 0 | **224 passed** (206 lib + 11 integration + 7 doc), 1 ignored |
| `cargo test -p aprender-contrastive-data --lib pairs` | 0 | 52 passed |
| `cargo test -p aprender-contrastive-data --lib manifest` | 0 | 47 passed |
| `cargo test -p aprender-contrastive-data --test pair_counts` | 0 | 4 passed |
| **cross-crate baseline** (`--lib -- --skip gpu::`) | 0 | **14,248 passed**, 28 ignored, 0 failed |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 | clean |
| `cargo clippy -p aprender-contrastive-data --all-targets --no-deps -- -D warnings` | 0 | clean |
| `cargo fmt -p aprender-contrastive-data --check` | 0 | clean |
| `make contrastive-data-boundary` | 0 | deps subset of allowlist; no fs/net/path under `src/` |
| `pv validate contracts/contrastive-pair-protocol-v1.yaml` | 0 | 0 errors, 0 warnings |
| Phase 1 setfit lib gate (`aprender-core`, `setfit::`) | 0 | 162 passed — unchanged |
| Phase 1 setfit conformance suite | 0 | 27 passed, 1 ignored — unchanged |
| `cargo test -p aprender-contrastive-data --test goldens_regenerate -- --ignored` | 0 | 9 files written, manifest regenerated |
| one-byte pair-golden tamper | 101 | 2 tests fail, digest check names the file and both hashes; reverted |
| O(K²) mutation (class-pair weights) | 101 | 4 tests fail incl. `left: 496, right: 32`; 46 pass; reverted |
| `CanonicalPair` equality-branch deletion | 101 | `canonical_pair_ordering` RED at `a = 2, b = 2`; reverted |
| silent over-cap clamp mutation | 101 | 2 budget tests RED (`got Ok((10000, false))`); reverted |
| `make tier2` | **2** | **RED — pre-existing, see below** |

**The baseline reconciles exactly.** 14,186 -> 14,248 is +62, which is precisely this crate's
new lib-test count (144 -> 206). No pre-existing test changed state.

**The `.snap.new` files are intact.** All three tracked
`crates/aprender-train/src/prune/snapshots/*.snap.new` were deleted by the `aprender-train`
run and restored with `git checkout --` before any commit; `git diff --diff-filter=D` over all
seven commits reports **none**.

### `make tier2` is RED, and it is not this plan's doing

D-ITEM-02, identical to what 02-03, 02-05 and 02-06 recorded. `make` halts at
`cargo clippy -- -D warnings` with 24 arch-gated SIMD errors. Attributed by file:
**`aprender-compute` 18 locations, `aprender-zram-core` 2, zero anywhere else.** No error
names `aprender-contrastive-data`; its only appearance in the whole tier2 log is the
`Checking aprender-contrastive-data v0.63.0` line — it compiled clean. CI is all
`[self-hosted, X64, Linux]` and never lints the aarch64 arms.

Because make halts, tier2's later steps never ran under `make`. All were run individually and
are green: the contrastive-data suite (three warm runs, 3.08/3.00/3.00 s, rc=0 each), the
Phase 1 setfit lib gate (162 passed) and the Phase 1 conformance suite (27 passed).

### Measurement notes (CLAUDE.md Verification Discipline)

- **Statuses captured directly, never through a pipe.** Every `rc` above came from
  `cmd > log 2>&1; rc=$?`. `rtk` also rewrites `cargo test` output into a summary line, so the
  per-suite counts were read from the raw logs rather than from the rewritten stream.
- **The `HashMap` absence claim carries a control.** `grep -rn "HashMap\|HashSet"
  crates/aprender-contrastive-data/src/` returns **0**; the same recursive grep for `BTreeMap`
  returns hits in **8** files, so its silence is meaningful rather than a broken command. No
  hash-ordered iteration was introduced in the pair path.
- **`unwrap()` count is 0** in both `pairs.rs` and `manifest.rs` over non-comment lines.
- **Every mutation was applied, RUN, and reverted**, with the suite re-run green afterwards.
- **`bashrs` is not installed on this host** (CLAUDE.md mandates it over shellcheck), so the
  Makefile edit could not be linted. It is a comment-only change touching no shell logic;
  shellcheck was NOT substituted.
- **`cargo test` takes one positional.** Every command quoted here carries at most one, and
  each was executed rather than transcribed.

## Threat Model Dispositions

| Threat | Disposition |
|---|---|
| T-02-21 pair endpoints outside the Selection | mitigated — structural for in-process construction (`SelectedId`'s private constructor; `PairLayout` works in ordinal space and mints none), typed for untrusted bytes via `validate_pair_records` -> `EndpointNotInSelection` naming the id |
| T-02-22 pair explosion via budget | mitigated — O(K) checked closed forms; the cap BINDS explicit budgets (`BudgetExceedsHardCap`, mutation-verified); streamed hashing; no enumeration anywhere |
| T-02-44 adversarial class layout (K ≈ N) | mitigated — per-class weights instead of class-pair prefix sums, evidenced by `negative_weight_entries == 32` at K = N = 32, a 8/32/128 scaling test under a fixed budget, and the O(K²) mutation turning exactly those tests red |
| T-02-45 mislabeled pair target in untrusted input | mitigated — the target is re-derived from `Selection::label_of` and disagreement is `PairTargetMismatch` naming both values |
| T-02-23 Aprender semantics attributed to SetFit | mitigated — `PAIR_DEVIATION_CLAUSES` copied verbatim from the contract into every replay record, with clause 3 explicitly stating that the pinned implementation does the opposite |
| T-02-24 replay drift / hash that under-attests | mitigated — the header is inside the digest, proven by TWO identical-stream tests, and a record that does not describe the sampler is refused |

## Threat Flags

None. No new network endpoint, auth path or file-access pattern. `dump_pairs` takes a
`std::io::Write` sink the caller supplies; the crate still opens nothing
(`make contrastive-data-boundary` green).

## Known Stubs

None. Every function this plan declared is implemented and tested. The `unique` strategy does
not ship (D-11, research recommendation adopted by the plan), and its capacity check ships
as a public, documented, tested helper explicitly labelled as pre-provisioned — so the
`BudgetExceedsCapacity` variant is reachable rather than dead.

## Notes for later plans

- **02-08** — read the handoff-contract section above in full; the K = N case needs
  `PairLayout::from_class_sizes`, not `PairSampler::new`. The three helpers it will want are
  already in `tests/common/mod.rs`. When authoring `binding.yaml`, the twelve new
  `#[contract]` sites are: `pairs::CanonicalPair::new` (canonical_pair),
  `pairs::positive_capacity`, `pairs::negative_capacity`, `pairs::effective_default_budget`
  (default_epoch_budget), `pairs::resolve_budget` (budget_resolution),
  `pairs::classify_degenerate` (pair_stream_degenerate_policy), `pairs::PairLayout::raw_pair_at`
  (pair_stream), `pairs::PairSampler::new` (singleton_policy), `pairs::validate_pair_records`
  (untrusted_pair_ingest AND split_span_fail_closed — two entries, one function),
  `manifest::PairReplayRecord::to_config` (pair_manifest_replay), `manifest::pair_manifest_hash`.
- **02-09** — the CLI's pair surface is `PairConfig` -> `PairSampler::new` ->
  `PairReplayRecord::from_sampler` -> `pair_manifest_hash`, with `dump_pairs` behind the
  `--dump` flag (D-09: the dump is a flag, not the default). `to_config` is the only sanctioned
  way back from a persisted record. The crate reads no clock and opens no file; the CLI owns
  both.
- **Phase 3** — the stream is EPOCH-INDEPENDENT and offset-resumable: advance by
  `iter_from(offset)`. If epoch-varying order is wanted, adding `epoch` to the RNG domain
  string is a versioned, non-breaking policy extension, documented on `raw_pair_at`
  (Assumption A4).
- **Out-of-scope observation, unchanged from 02-05:** `crates/apr-cli/src/commands/nf4_classifier.rs`
  still emits two `unused_mut` warnings during the baseline run. Untouched and not fixed, per
  the executor scope boundary.

## Self-Check: PASSED

| Item | Status |
|---|---|
| `crates/aprender-contrastive-data/src/pairs.rs` | FOUND, contains `SamplerStateReport`, `RetainedState`, `PairLayout`, `validate_pair_records` |
| `crates/aprender-contrastive-data/src/manifest.rs` | FOUND, contains `PairReplayRecord`, `pair_manifest_hash`, `dump_pairs` |
| `crates/aprender-contrastive-data/tests/pair_counts.rs` | FOUND, begins with `mod common;`, contains `load_contracted` |
| `crates/aprender-contrastive-data/tests/goldens/pairs_seed13_shots8_first32.jsonl` | FOUND (32 lines) |
| `crates/aprender-contrastive-data/tests/goldens/pairs_seed17_shots8_first32.jsonl` | FOUND (32 lines) |
| `tests/goldens/manifest.sha256` | FOUND, 9 entries, verifier coverage asserted equal |
| `eceaecd9f` `5ccd32908` `675b825cf` `159abaf45` `addd0db8e` `49ab9e9f9` `4a2b267c8` | FOUND (7/7) |
| `grep -c "unwrap()"` over non-comment lines of `pairs.rs` / `manifest.rs` | 0 / 0 |
| `grep -rn "HashMap\|HashSet" src/` (control: `BTreeMap` in 8 files) | 0 |
| Three `.snap.new` files under `crates/aprender-train/src/prune/snapshots/` | intact, no deletion in any of the 7 commits |
