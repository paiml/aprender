---
phase: 04-apr-artifact-and-production-parity
plan: 13
subsystem: training
tags: [setfit, bundle, provenance, schema-version, nullable-allowlist, apr, contract-gate]
requires:
  - phase: 04-01
    provides: "contracts/setfit-apr-v1.yaml — the `provenance` field names, the doc<->bundle bijection over 20 bundle fields, and the four-path nullable allowlist over five walked sub-documents whose completeness this plan proves"
  - phase: 03-faithful-two-stage-trainer-and-head
    provides: "SetFitBundle, from_run_parts, the verify policy's `close`, and the byte-closure tests field 20 had to ride through unchanged"
provides:
  - "ProvenanceRecord — six fields read off the run's Selection, named exactly as the contract's `provenance` object"
  - "SetFitBundle field 20 (`provenance`) + `provenance()` accessor"
  - "BUNDLE_SCHEMA_VERSION 1 -> 2, with both refusal paths (version check, missing-field parse) tested"
  - "from_run_parts takes `selection: &Selection`; no String provenance parameter exists"
  - "the nullable-path allowlist COMPLETENESS GATE — 04-01 item 3 obligation (b), discharged"
  - "a compile-time no-float guarantee on ProvenanceRecord via `derive(Eq)`"
affects:
  - "04-02 (writer: walks all five sub-documents; its NULLABLE_PATH_ALLOWLIST is now pinned by a gate in aprender-train)"
  - "04-05 (codec: 20 bundle fields to recover, not 19; BUNDLE_SCHEMA_VERSION is 2)"
  - "04-03 (loader), 04-04 (envelope), 04-09 (parity), 04-10 (gates — see the skip_serializing_if note below)"
tech-stack:
  added: []
  patterns:
    - "provenance is READ OFF the object, never accepted as strings (the SelectionLock::from_candidates rule, applied to the data side)"
    - "a zero-contribution sub-document is still WALKED and asserted BY NAME to contribute nothing"
    - "derive(Eq) as a compile-time proof that a record carries no float"
    - "every version comparison reads the constant; a `replacen` that stops matching is caught by assert_ne!"
key-files:
  created: []
  modified:
    - "crates/aprender-train/src/train/setfit/bundle.rs"
    - "crates/aprender-train/src/train/setfit/bundle_tests.rs"
    - "crates/aprender-train/src/train/setfit/verify.rs"
    - "crates/aprender-train/src/train/setfit/verify_tests.rs"
key-decisions:
  - "from_run_parts takes the Selection OBJECT (position 5, mirroring `close`'s own parameter order), never six provenance strings"
  - "ProvenanceRecord derives Eq — which compiles only if no field is a float, so the no-float claim is checked by the type system and not only by a grep"
  - "field 20 is APPENDED, keeping the nineteen existing wire positions stable"
  - "the phase-3 contract is referenced, never edited (Ph1 D-23); the 19-vs-20 reconciliation is a module doc note in bundle.rs"
  - "the completeness gate asserts the two zero-contribution subtrees BY NAME before the set comparison, so a new Option on either reports which type grew one"
patterns-established:
  - "Allowlist completeness gate: serialize an all-`None` instance of every walked type, collect ALL null paths (not first-only), compare as a SET, and name both set differences in the failure message"
  - "subdocument_null_paths asserts the value is a non-empty JSON object before walking — a gate whose pass condition is reachable by not looking is vacuous"
requirements-completed: [APR-01, APR-05]
duration: 50min
completed: 2026-08-15
---

# Phase 4 Plan 13: Bundle Provenance and the Allowlist Completeness Gate Summary

**`SetFitBundle` now carries a `ProvenanceRecord` read off the `Selection` the run actually consumed — six fields, no `Option`, no float — and the contract's four-path nullable allowlist is pinned to all five shipped sub-document types by a gate that was made to fail twice, by name, before it was trusted.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-08-15T07:23Z (approx.)
- **Completed:** 2026-08-15T08:14Z
- **Tasks:** 2
- **Files modified:** 4

## Task Commits

1. **Task 1: ProvenanceRecord + field 20 + schema bump + completeness gate** — `ab7ace94a` (feat)
2. **Task 2: thread the run's selection into the assembly path** — `59a4eca93` (feat)

## The ProvenanceRecord As Shipped

`crates/aprender-train/src/train/setfit/bundle.rs:380-393`, in wire order. Field names match
`contracts/setfit-apr-v1.yaml`'s `provenance` object (line 279) exactly, because the codec maps
them 1:1.

| # | Field | Type | Read off |
| - | ----- | ---- | -------- |
| 1 | `dataset_fingerprint` | `String` | `selection.dataset_fingerprint_hex()` |
| 2 | `validation_split_fingerprint` | `String` | `selection.validation_fingerprint_hex()` |
| 3 | `selection_semantic_hash` | `String` | `hex::encode(selection.semantic_hash())` |
| 4 | `selection_ledger_hash` | `String` | `hex::encode(selection.ledger_hash())` |
| 5 | `selection_root_seed` | `u64` | `selection.root_seed()` |
| 6 | `shots_per_class` | `u32` | `selection.shots_per_class()` |

Derives `Debug + Clone + PartialEq + Eq + Serialize + Deserialize`, `#[serde(deny_unknown_fields)]`.

**`Eq` is load-bearing and was chosen deliberately.** `derive(Eq)` compiles only if every field is
`Eq`, and neither `f32`, `f64` nor `Option<f64>` is. So "this record carries no float" — the property
that makes its zero-contribution-to-the-allowlist claim robust against the `evidence.epsilon_used`
residual class — is checked by the type system on every build, not only by the grep in
`bundle_provenance_record_declares_no_option_field`. `ResolvedConfigRecord`, the other
zero-contribution type, does not derive `Eq`; that is pre-existing and out of this plan's scope.

`ProvenanceRecord::of(&Selection)` is **private**. There is no public constructor and no `&str`
provenance parameter anywhere on the path (T-04-39): `from_run_parts`'s only `&str` is `format_id`,
which is the codec's identity, not the run's.

## BUNDLE_SCHEMA_VERSION 1 → 2, and the two refusals it produces

`bundle.rs:106`. All six sites read the constant; the last bare literal
(`bundle_tests.rs`'s `text.replacen("\"schema_version\":1", ...)`) now builds its needle from it.

That literal was a live trap, not a tidy-up. Had it been left, the bump to 2 would have made the
`replacen` match nothing, the payload would have stayed at the CURRENT version, and
`bundle_version_bump_is_a_typed_version_error` would have asserted that a current-version bundle is
refused — failing for a right-looking reason. The pre-existing `assert_ne!(bumped, text, "the
replacement must have applied")` is what would have caught it; the fix removes the need to rely on it.

The plan's behaviour list asks for "bytes declaring the OLD schema version are refused with typed
`UnsupportedSchemaVersion` naming both versions" AND "bytes whose provenance field is absent are
refused". **These are two different code paths and the ordering decides which one a given payload
gets**, which is recorded in the constant's doc comment rather than left for a reader to discover:

| Payload | Where it fails | Error |
| ------- | -------------- | ----- |
| declares `"schema_version":1`, otherwise v2-shaped | after the parse, at the version check | `UnsupportedSchemaVersion { got: 1, supported: 2 }` |
| a genuine v1 artifact (no `provenance` key at all) | at the parse, one step earlier | `Serialization { context: "parse", detail: "missing field `provenance`..." }` |

`from_canonical_bytes` parses BEFORE it checks the version (the order is load-bearing for the
allocation bounds), and `provenance` is not an `Option`, so serde never reaches the version
comparison for a real v1 payload. Both are refusals rather than partial interpretations, which is
the property that matters; both are tested
(`bundle_the_previous_schema_version_is_a_typed_version_error`,
`bundle_a_payload_without_provenance_is_refused_naming_the_field`).

## Verification Output

Statuses captured directly into a variable (`cmd > log 2>&1; rc=$?`), never read through a pipe
(CLAUDE.md Verification rule 1).

| Command | Baseline (b30bff96a) | After | rc |
| ------- | -------------------- | ----- | -- |
| `cargo test -p aprender-train --features setfit --lib setfit::bundle` | 26 passed | **33 passed** (+7) | 0 |
| `cargo test -p aprender-train --features setfit --lib setfit::verify` | 16 passed | **16 passed** (unchanged) | 0 |
| `cargo test -p aprender-train --features setfit --lib` (full) | — | 7850 passed, **24 failed**, 15 ignored | 101 |
| `cargo clippy -p aprender-train --features setfit --lib --all-targets --no-deps -- -D warnings` | — | clean | **0** |
| `cargo fmt -p aprender-train -- --check` | — | clean | **0** |

Task 1 requires ≥ 9 and Task 2 ≥ 10; both clear it with a stated, non-vacuous count.

**The 24 full-suite failures are byte-identical to the phase-3 known-red baseline.** Not asserted —
diffed:

```
$ diff /tmp/failures-baseline.txt /tmp/failures-now.txt ; DIFF_RC=$?
DIFF_RC=0
```

21 `gpu::` + 3 `prune::snapshot_tests`, the same names in the same order as
`.planning/phases/03-faithful-two-stage-trainer-and-head/known-red-baseline.md`. Zero new failures.
Not "fixed" — out of scope, as the plan directs.

### The clippy run was proven to reach the file, not assumed to

`cargo clippy -p aprender-train ... -- -D warnings` **without** `--no-deps` exits **101** on this
host, entirely from pre-existing findings in `aprender-compute` (unreachable expressions, unused
variables, dead code). Reporting "0 hits in aprender-train" from that run would have been the
label-by-intent failure (CLAUDE.md Verification rule 2): with a dependency erroring under
`-D warnings`, "no findings in my crate" is indistinguishable from "my crate was never linted".

So the scoped run was falsified before being believed. A deliberate `let _clippy_probe =
format!("{}", "...")` was inserted into `ProvenanceRecord::of` and the scoped command re-run:

```
error: useless use of `format!`
   --> crates/aprender-train/src/train/setfit/bundle.rs:402:29
   = note: `-D clippy::useless-format` implied by `-D warnings`
rc=101
```

Probe reverted, rc back to 0. The lint pass demonstrably reaches this file.

## The Completeness Gate

`bundle_nullable_path_allowlist_is_complete_over_the_five_subdocuments`
(`bundle_tests.rs`). It discharges 04-01 item 3 **obligation (b)**.

### The observed null-path set

```
architecture      -> ["architecture.vocab_remap"]
requested_config  -> ["requested_config.pair_config.budget",
                      "requested_config.pair_config.hard_cap"]
resolved_config   -> []          <- asserted BY NAME
evidence          -> ["evidence.epsilon_used"]
provenance        -> []          <- asserted BY NAME
------------------------------------------------------------
observed (sorted, 4) == contracts/setfit-apr-v1.yaml allowlist (4)
```

Five sub-documents walked, four paths collected. Exactly the contract's numbers.

### How each all-`None` instance was obtained, and why two of them are not the fixture's

The gate is only meaningful if each instance really is the all-`None` one, so each is asserted, not
assumed:

- **`architecture`** — the fixture is a **slice** encoder, so its `vocab_remap` is `Some`. The
  **production** shape is the full pin's `None` (`import.rs:501` vs `:620`). The gate asserts
  `is_some()` first — so if the fixture ever stops carrying a remap, the override stops silently
  being a no-op — then sets `None`. This is the one path a fixture-shaped suite structurally cannot
  produce, which is exactly why the contract flags it.
- **`requested_config`** — built through the **public constructor path**,
  `SetFitTrainConfig::reference_defaults(fx::FIXTURE_SEED)` (which goes through `PairConfig::new`,
  both knobs `None`), never the private `SetFitTrainConfigWire`. The fixture run's own config sets
  `budget: Some(variant.budget)` (`test_fixtures.rs:293`), so reusing it would have hidden **one of
  the two** pair paths and the gate would have passed while under-counting. Asserted:
  `budget.is_none() && hard_cap.is_none()`.
- **`resolved_config`**, **`provenance`** — nothing to set; that is the claim under test.
- **`evidence`** — `epsilon_used` set to `None` explicitly rather than relying on it being the
  shipped value (`evidence.rs:655`) by accident.

### Anti-vacuity: `subdocument_null_paths` refuses to walk a non-object

A sub-document that serialized to a scalar would contribute **nothing**, and "contributed nothing"
is precisely what this gate reads as "has no nullable fields" — a pass reachable by not looking. So
the helper asserts the value is a JSON object and that the object is non-empty before walking it.

The walk itself is `evidence.rs`'s `first_null_path` (lines 199-224) **widened from first-only to
all**. First-only is right for a refusal (report the offender, stop) and wrong for a set comparison:
it would report `architecture.vocab_remap` and never mention a field added beside it.

### Falsification transcripts (performed, recorded, reverted)

**A gate only ever observed passing is not a gate.** Two mutations, in opposite directions.

**(1) The plan's required falsification — drop one allowlist entry.** `ALLOWLIST` reduced from 4
entries to 3 by deleting `evidence.epsilon_used`:

```
rc=101
thread '...::bundle_nullable_path_allowlist_is_complete_over_the_five_subdocuments' panicked at
crates/aprender-train/src/train/setfit/bundle_tests.rs:1488:5:
the nullable-path allowlist in contracts/setfit-apr-v1.yaml is no longer complete against the shipped types.
  NOT ALLOWLISTED (a new `Option` field): ["evidence.epsilon_used"]
  ALLOWLISTED BUT NOT OBSERVED (...): []
  observed: ["architecture.vocab_remap", "evidence.epsilon_used", "requested_config.pair_config.budget", "requested_config.pair_config.hard_cap"]
  allowlist: ["architecture.vocab_remap", "requested_config.pair_config.budget", "requested_config.pair_config.hard_cap"]
```

Fails, naming the dropped path. Reverted.

**(2) The direction that actually matters — a new `Option` on `ProvenanceRecord`.** Dropping an
allowlist entry is the easy direction; the defect the gate exists to catch is a field added to a
type. A temporary `pub(crate) mutation_probe: Option<u64>` was added to `ProvenanceRecord`:

```
rc=101
assertion `left == right` failed: `provenance` (ProvenanceRecord) must contribute NO nullable path.
Field 20 is `four String + u64 + u32` by construction and the type doc forbids an `Option`; if this
fires, one was added and the contract's allowlist is now incomplete.
  left: ["provenance.mutation_probe"]
 right: []
```

The **named** zero-contribution assertion fires and reports the new path by name — which is the
whole reason the two empty subtrees are asserted separately instead of being folded into the set
comparison, where this would have surfaced as an opaque set mismatch.

The same mutation was used to falsify the **source-level** guard, which catches the case the
behavioural gate cannot (an `Option` that happens to be `Some` on every fixture):

```
rc=101
assertion `left == right` failed: ProvenanceRecord must have NO `Option` field: ...
```

Both mutations reverted; `setfit::bundle` back to 33 passed, rc=0.

## STOP-GUARD: `from_run_parts` call-site enumeration

Task 2's guard, run on the final tree:

```
$ grep -rn "from_run_parts" crates/
crates/aprender-train/src/train/setfit/verify_tests.rs:213:    let mut bundle = SetFitBundle::from_run_parts(   <- CALL (test)
crates/aprender-train/src/train/setfit/verify_tests.rs:337:    let bundle = SetFitBundle::from_run_parts(       <- CALL (test)
crates/aprender-train/src/train/setfit/bundle.rs:352:/// [`SetFitBundle::from_run_parts`] takes ...              (doc, NEW)
crates/aprender-train/src/train/setfit/bundle.rs:398:    /// Private on purpose: [`SetFitBundle::from_run_parts`] ... (doc, NEW)
crates/aprender-train/src/train/setfit/bundle.rs:539:    pub(crate) fn from_run_parts(                        <- DEFINITION
crates/aprender-train/src/train/setfit/verify.rs:211:/// `close` stamps the format id via [...]                  (doc, pre-existing)
crates/aprender-train/src/train/setfit/verify.rs:290:    let bundle = SetFitBundle::from_run_parts(            <- CALL (PRODUCTION, in `close`)
crates/aprender-train/src/train/setfit/bundle_tests.rs:44:    SetFitBundle::from_run_parts(                     <- CALL (test)
```

**Exactly four call sites** — one production (`close`) and three across the two test modules — plus
one definition and three doc-comment references (two of them added by this plan's new doc comments).
No call site exists outside `close`, `verify_tests.rs` and `bundle_tests.rs`, so there is still
exactly ONE place a bundle can be assembled. `run_verify_policy` reaches `from_run_parts` **through**
`close` (it holds `selection` at `verify.rs:583` and hands it over at `:590`), which is the expected
shape and not a second assembly site. Nothing to surface.

## Diff Shape (Task 2's acceptance criteria)

```
$ git diff crates/.../verify.rs        -> 1 file changed, 1 insertion(+)
  @@ fn close  +        selection,

$ git diff crates/.../verify_tests.rs  -> 1 file changed, 2 insertions(+)
  @@ fn echo_cache                                          +        run.selection(),
  @@ fn verify_a_codec_that_omits_the_format_check_...      +        run.selection(),

$ grep -c "run.selection()" crates/.../verify_tests.rs      -> 2
```

Exactly one changed call expression in `verify.rs` (inside `close`) and exactly two in
`verify_tests.rs`, both adding `run.selection()`. No policy step added, removed or reordered; no
tolerance moved; the closure check untouched; no assertion, doc comment or test name changed.
`echo_cache`'s `source_revision` perturbation and the foreign-codec refusal still assert exactly what
they asserted before. 04-05's later sealed-module visibility edit was not anticipated.

## Source Assertions

| Assertion | Result |
| --------- | ------ |
| `grep -c "pub struct ProvenanceRecord" bundle.rs` | **1** |
| `grep -n "BUNDLE_SCHEMA_VERSION: u32 = 2" bundle.rs` | matches (line 106) |
| `from_run_parts` takes `selection: &Selection` | yes, position 5 (line 544) |
| `from_run_parts` has a String parameter for a provenance value | **no** — its only `&str` is `format_id` |
| `Option<` inside the `ProvenanceRecord` block | **0** (also enforced by a test, and by `derive(Eq)`) |
| any live `"schema_version":<literal>` in the tests | **0** — all eight references read the constant |
| `git diff --name-only` (before commit) | exactly the four planned files |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Task 1's verify command cannot pass without Task 2's edits**

- **Found during:** Task 1
- **Issue:** `cargo test -p aprender-train --features setfit --lib setfit::bundle` compiles the
  whole lib, including `verify.rs`. Changing `from_run_parts`'s arity therefore makes Task 1's own
  verification impossible until Task 2's call sites are updated. A signature change and its call
  sites cannot be split across commits and still compile — this is inherent to the plan's task split,
  which the plan anticipates: Task 1's acceptance criteria already list **all four** files under
  `git diff --name-only`.
- **Fix:** All four files were edited before either verification ran; both scoped filters were then
  run green over that state and committed as two commits in the plan's task order.
- **Measured consequence, recorded rather than hidden:** commit `ab7ace94a` **does not compile in
  isolation**. Verified against the committed blob rather than assumed —
  `git show ab7ace94a:crates/.../verify.rs` shows `close` passing 6 arguments to the now-7-parameter
  function (E0061). `59a4eca93` restores it. Both land together; no state between them is ever
  pushed alone.
- **Files modified:** the four planned files. **Commits:** `ab7ace94a`, `59a4eca93`

**2. [Rule 1 — Bug] `bundle_tests.rs` held a bare `"schema_version":1` literal the bump would have neutered**

- **Found during:** Task 1, step 4 ("verify each of the six sites still reads the constant rather
  than a literal")
- **Issue:** `bundle_version_bump_is_a_typed_version_error` built its `replacen` **needle** from a
  literal while building its replacement from the constant. After the bump the needle matches
  nothing, leaving a current-version payload — and the test would then have asserted that a
  CURRENT-version bundle is refused, i.e. the opposite of its name, failing for a right-looking
  reason.
- **Fix:** both sides now read `BUNDLE_SCHEMA_VERSION`, with the trap recorded in a comment beside
  it. **Commit:** `ab7ace94a`

### Process Deviation

**3. TDD RED was captured as targeted falsification, not as a pre-implementation red run**

The plan marks Task 1 `tdd="true"`. In Rust, writing the tests first against a type that does not
exist yet yields a **compile error**, which carries no information about whether any individual
assertion can fail — and the same compile error masks every other test in the module. So the RED
evidence here is the three falsification transcripts above (two independent mutations of the
completeness gate plus one of the source guard), each of which turns a *specific named assertion*
red and was reverted. That is stronger evidence than a compile failure and it is what the plan's own
acceptance criteria demand ("a gate that cannot fail is not a gate — CR-02"). Recorded as a
deviation rather than reported as a TDD cycle that was not run in that shape.

## A Vacuous Test Filter, Caught In Passing

The first falsification run used the filter `setfit::bundle::bundle_nullable`:

```
rc=0
cargo test: 0 passed, 7889 filtered out (1 suite, 0.00s)
```

**It matched zero tests and exited 0.** The module path is `bundle::bundle_tests::`, so the filter
was wrong — and a mutation that should have turned the suite red reported success. Exactly the CR-02
class the plan names, met while trying to falsify a gate designed to prevent it. Every count in this
summary is a stated number compared against a stated minimum, never an exit status; the re-run with
`bundle_nullable_path_allowlist` returned `rc=101` with `0 passed; 1 failed`.

## Note for 04-10 (gates) and 04-02: the `skip_serializing_if` token count moved 0 → 4

Task 1's acceptance criteria include *"`grep -c "skip_serializing_if" crates/aprender-train/src/train/setfit/`
shows no NEW occurrence introduced by this plan"*. Measured on both sides:

| | baseline `b30bff96a` | HEAD |
| - | -------------------- | ---- |
| textual occurrences of `skip_serializing_if` in the directory | **0** | **4** |
| occurrences of the ATTRIBUTE form `serde(skip_serializing_if` | **0** | **0** |

The criterion is satisfied **in intent** — no type gained the attribute, which is what the contract
invariant and T-04-58 are about. It is **not** satisfied as literally written, and the reason is that
the same task's `<action>` step 6(d) *instructs* the gate to explain in prose that
`skip_serializing_if` is forbidden and why. The four new occurrences are two doc comments, one
assertion message and one guard-string in
`bundle_provenance_record_declares_no_option_field` (which scans only the `ProvenanceRecord` struct
block and therefore does not trip on its own file's prose — confirmed by falsification (2), where
the `Option` assertion fired and the `skip_serializing_if` one did not).

**Consequence for whoever writes the gate:** a repo gate matching the bare token in this directory
would now turn red on its own documentation — the failure mode `test_fixtures.rs`'s own module doc
records ("a gate that turns red on its own prose is a gate nobody will keep"). Such a gate must match
the **attribute** form `serde(skip_serializing_if`, and its must-match/must-not-match case table
should include one of these doc comments as a must-NOT-match row (CLAUDE.md Verification rule 7).

The house precedent for the other direction is already in this file:
`bundle_resolved_config_is_provenance_with_no_reconstruction_path` assembles its forbidden token at
runtime (`format!("TryFrom<{}>", "ResolvedConfigRecord")`) for exactly this reason.

## Notes for Later Plans

- **04-02** — the allowlist is now pinned from this side. If `NULLABLE_PATH_ALLOWLIST` in
  `aprender-core/src/setfit/artifact.rs` disagrees with the four paths above, one of the two is
  wrong and this gate is the one with a falsification transcript. The writer must walk **five**
  sub-documents; `provenance` is the fifth and contributes zero paths **today** — walking it is the
  point, not redundancy.
- **04-05** — the bundle has **20** fields and `BUNDLE_SCHEMA_VERSION` is **2**. The codec's reverse
  bijection must recover `provenance` directly (`= from_value(doc.provenance)`), not recompute it:
  it is not a function of the other fields.
- **Every plan that lands a module must flip its `contracts/aprender/binding.yaml` entry from
  `pending` to `implemented`** (04-01's note). This plan lands no new contract-bound module — field
  20 and the gate are obligations of `nullable_path_allowlist` and `doc_bundle_bijection`, both of
  which bind to 04-02/04-05 code that does not exist yet — so no binding was flipped. Flagged
  explicitly so its absence reads as a decision rather than an omission.
- The three sub-documents with `Option` fields are `EncoderArchitecture`, `PairConfigWire` (reached
  through `SetFitTrainConfig`) and `EvidenceSummary`. `ResolvedConfigRecord` and `ProvenanceRecord`
  have none. Only the latter is protected by `derive(Eq)`; adding `Eq` to `ResolvedConfigRecord`
  would extend the same compile-time guarantee and is a one-line change no plan currently owns.

## Threat Flags

None. This plan adds no network endpoint, no auth path, no file-access pattern and no schema at a
trust boundary that is not already in the plan's `<threat_model>`. The three registered threats
(T-04-39 spoofed provenance, T-04-40 silent schema drift, T-04-58 a drifting null allowlist) are all
`mitigate` and all mitigated above, each with a test.

## Known Stubs

None. Every field is populated from the run; nothing is hardcoded, defaulted or placeheld.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-train/src/train/setfit/bundle.rs
FOUND: crates/aprender-train/src/train/setfit/bundle_tests.rs
FOUND: crates/aprender-train/src/train/setfit/verify.rs
FOUND: crates/aprender-train/src/train/setfit/verify_tests.rs
```

Commits claimed, checked in the log:

```
FOUND: 59a4eca93 feat(04-13): thread the run's selection into bundle assembly
FOUND: ab7ace94a feat(04-13): carry data provenance in the bundle as field 20 (APR-01, APR-05)
```

`git diff --diff-filter=D --name-only HEAD~1 HEAD` empty for both commits — no file deletions.
Working tree clean before this summary. No file outside this plan's four `files_modified` was
touched; `STATE.md` and `ROADMAP.md` were not modified (the orchestrator owns them).
