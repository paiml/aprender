---
phase: 04-apr-artifact-and-production-parity
plan: 14
subsystem: training
tags: [setfit, config-validation, selection-lock, serde, canonical-bytes, trn-07, ops-02]

# Dependency graph
requires:
  - phase: 04-01
    provides: "contracts/setfit-apr-v1.yaml item 13, selection_lock_lifecycle — the normative trust model this door's doc comment must match and not overclaim beyond"
  - phase: 03
    provides: "SetFitTrainConfig::new (the single validating constructor), SelectionLock::from_candidates / to_canonical_bytes / mint_test_token, ValidationEvaluationWire"
provides:
  - "SetFitTrainConfig::to_request — the public, validated override/merge door an out-of-crate caller can use for --seed/--device"
  - "SelectionLock::from_canonical_bytes — durable lock reconstruction, bounded, canonical-form-checked, rebuilt through from_candidates"
  - "MAX_SELECTION_LOCK_BYTES (1 MiB), enforced on the raw slice before serde is handed anything"
  - "ValidationEvaluation::from_wire — pub(super) bits-based evaluation reconstruction"
  - "Five new LockError variants covering the payload's failure modes"
affects: [04-06, 04-07, 04-15, 04-16, apr-cli-setfit-train, apr-cli-eval]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Read-back-as-request merge: a validated type exposes to_request, never a setter; the merge is revalidated as a whole by the single constructor"
    - "Canonical-form closure check on deserialization: compare the reconstruction's re-serialization against the input bytes instead of enumerating per-field disagreements"
    - "Reconstruction rebuilds THROUGH the creating constructor, so invariants and the selection rule have one implementation"
    - "Same-file source scans assemble their needles at runtime so the scan cannot match its own text"

key-files:
  created: []
  modified:
    - crates/aprender-train/src/train/setfit/config.rs
    - crates/aprender-train/src/train/setfit/lock.rs
    - crates/aprender-train/src/train/setfit/lock_tests.rs
    - crates/aprender-train/src/train/setfit/evaluate.rs

key-decisions:
  - "to_request returns a REQUEST and no in-place mutator was added: the only path to a config stays SetFitTrainConfig::new, so a CLI override is validated as a whole rather than per field"
  - "from_canonical_bytes rebuilds through from_candidates rather than re-implementing the invariants, so a file cannot carry a candidate set the constructor would have refused nor name a winner the rule did not pick"
  - "A single canonical-form comparison (re-serialize and compare to the input) replaces a per-field disagreement checklist — the checklist would drift from the wire form, the comparison cannot"
  - "MAX_SELECTION_LOCK_BYTES = 1 MiB, and the derivation is MEASURED from two real locks in a test rather than quoted in a comment"
  - "The doc comment states the contract's trust model verbatim: the lock file is an integrity-checked RECORD, not an unforgeable credential"

patterns-established:
  - "Override/merge door: to_request + the existing validating constructor is the whole surface; a with_seed/with_device convenience is refused because two doors to one merge diverge"
  - "Deserialization closure: a canonical form is canonical, so from_canonical_bytes(to_canonical_bytes(x)) is exact and anything else is refused typed"
  - "Guard falsification is recorded, not asserted: every new guard in this plan was mutated once and the killed tests are named"

requirements-completed: [OPS-02, TRN-07]

# Metrics
duration: 35min
completed: 2026-08-15
---

# Phase 04 Plan 14: Config Override and Durable Lock Doors Summary

**Two public library doors the CLI plans needed and did not have: `SetFitTrainConfig::to_request` (validated `--seed`/`--device` merge through the single constructor) and `SelectionLock::from_canonical_bytes` (bounded, canonical-form-checked reconstruction that makes the selection lock a real cross-process gate).**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-08-15T07:45Z (approx.)
- **Completed:** 2026-08-15T08:20Z
- **Tasks:** 2 (both TDD)
- **Files modified:** 4

## Accomplishments

- **`SetFitTrainConfig::to_request`** closes the 04-06 HIGH finding. `apr-cli` cannot name the private `SetFitTrainConfigWire`, so plan 04-06's original override design was uncompilable from another crate. The merge now goes request → override → `SetFitTrainConfig::new`, so the merged whole is validated by the same single implementation that validated the file. TRN-02's pinned `max_length` and the device grammar both survive the override path, proven by test.
- **`SelectionLock::from_canonical_bytes`** closes the persistence half of review finding B4. `to_canonical_bytes` was public and nothing could read those bytes back, so a separate `apr eval --split test` process had no lock to consume — the only shape that compiled was minting the lock inside the test command, which satisfies "a lock existed before test access" with an object created one line earlier. A lock now survives a filesystem round trip and still refuses a re-tuned model with `StaleLock`.
- **The reconstruction cannot launder a file.** It rebuilds through `from_candidates`, so every consistency invariant and the selection RULE come from one implementation; a file naming a winner the rule did not pick is refused. `from_candidates` is still `pub(super)` — the creation door did not widen.
- **The bound is real and measured.** `MAX_SELECTION_LOCK_BYTES` is enforced on the raw slice before serde sees anything, and the ordering is proven behaviourally (over-cap garbage returns `LockPayloadTooLarge`, not `MalformedLockPayload`) as well as by a source assertion.
- **The trust model is recorded honestly.** The doc comment says what `contracts/setfit-apr-v1.yaml` item 13 says and no more: the file is an integrity-checked RECORD, not an unforgeable credential. A test demonstrates it — a well-formed edit passes its own `verify_integrity` and still fails to mint.

## Task Commits

Each task was committed atomically, RED then GREEN:

1. **Task 1: `SetFitTrainConfig::to_request`** — `169fd14a8` (test, RED) → `ea7d8bd8f` (feat, GREEN)
2. **Task 2: `SelectionLock::from_canonical_bytes`** — `d155ffd86` (test, RED) → `b3d17de85` (feat, GREEN)

No REFACTOR commits: neither implementation needed cleanup after going green.

## New Public Surface (exact signatures)

```rust
// crates/aprender-train/src/train/setfit/config.rs:648
pub fn to_request(&self) -> SetFitTrainRequest

// crates/aprender-train/src/train/setfit/lock.rs:69
pub const MAX_SELECTION_LOCK_BYTES: u64 = 1_048_576;

// crates/aprender-train/src/train/setfit/lock.rs:455
pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, LockError>

// crates/aprender-train/src/train/setfit/evaluate.rs:226 (pub(super), not public API)
pub(super) fn from_wire(wire: ValidationEvaluationWire) -> Self
```

`MAX_SELECTION_LOCK_BYTES` is reachable from `apr-cli` as
`entrenar::train::setfit::lock::MAX_SELECTION_LOCK_BYTES` — `mod.rs` already declares `pub mod
lock;`, so no re-export edit was needed (this plan does not own `mod.rs`).

### `MAX_SELECTION_LOCK_BYTES` derivation

A candidate is a fixed-shape record: a config label, an artifact hash, and a nested evaluation
carrying that artifact hash again plus two more fingerprints, a metric tag, a `u64` of value bits
and a row count. At the 64-character hex hashes the format actually uses that is on the order of
600 bytes with JSON keys and punctuation, so 1 MiB admits roughly 1,600 candidates — about two
orders of magnitude above the tens of candidates a sweep in this phase commits, while still
refusing a payload that claims millions of candidates to make the allocation itself the attack.

The figure is **measured, not quoted**: `lock_max_selection_lock_bytes_clears_a_realistic_sweep`
takes the marginal per-candidate cost as the byte difference between two real locks built with
64-character hashes, then asserts the bound admits ≥ 1,000 candidates. It also pins the constant's
value so a change shows in a diff.

### New `LockError` variants

| Variant | Payload | When |
|---------|---------|------|
| `LockPayloadTooLarge` | `limit: u64, observed: u64` | raw slice longer than the bound, refused before serde |
| `MalformedLockPayload` | `detail: String` | parse failure, including `deny_unknown_fields` rejections |
| `UnsupportedLockSchemaVersion` | `got: u32, supported: u32` | a lock schema version this reader does not implement |
| `ChosenIndexOutOfRange` | `chosen_index: u64, candidates: usize` | the recorded winner names no candidate |
| `NonCanonicalLockPayload` | `observed_len, canonical_len, first_difference_at` | the bytes are not the canonical form of the record they parse to |

`LockError` is `#[non_exhaustive]`, so adding these is not a breaking change for downstream
matches. Every variant carries named fields and a `Display` arm naming the observed values and
the contract equation, in the module's existing style.

## Files Created/Modified

- `crates/aprender-train/src/train/setfit/config.rs` — `to_request` plus five tests and a
  same-file source guard on the merge surface.
- `crates/aprender-train/src/train/setfit/lock.rs` — `MAX_SELECTION_LOCK_BYTES`,
  `from_canonical_bytes`, five `LockError` variants with `Display` arms, and a private
  `first_difference` helper for the non-canonical diagnostic.
- `crates/aprender-train/src/train/setfit/lock_tests.rs` — ten tests plus three local helpers
  (`hex64`, `substitute_first`, `method_body`).
- `crates/aprender-train/src/train/setfit/evaluate.rs` — `ValidationEvaluation::from_wire`.

Untouched by design: `bundle.rs`, `verify.rs`, `mod.rs`, and everything under
`aprender-core/src/setfit/` — the wave-2 ownership contract with the concurrent 04-02 and 04-13
agents. `git diff --name-only` against the wave base lists exactly the four files above.

## Test Counts (per the plan's three scoped filters)

| Filter | Before | After | New |
|--------|--------|-------|-----|
| `cargo test -p aprender-train --features setfit --lib setfit::config` | 35 | **40** | 5 |
| `cargo test -p aprender-train --features setfit --lib setfit::lock` | 26 | **36** | 10 |
| `cargo test -p aprender-train --features setfit --lib setfit::evaluate` | 14 | **14** | 0 (guard unaffected, as required) |
| `cargo test -p aprender-train --features setfit --lib setfit::` (whole module) | — | **249 passed, 1 ignored** | — |

`evaluate_source_exposes_no_public_api_taking_a_float_parameter` still passes: `from_wire` is
`pub(super)` (so it matches neither `pub fn ` nor `pub const fn `) and takes the wire struct, not
a float. The guard's `checked >= 9` floor is unchanged.

## Guard Falsification (CR-02 vacuity discipline)

Every new guard was mutated once and observed RED. A test filter matching zero tests exits 0, so
each of these was watched fail by name:

| Mutation | Killed |
|----------|--------|
| `to_request` emits `root_seed: 0` | `falsify_config_to_request_round_trips_all_twelve_knobs_identically` (knob 10) |
| add `pub fn set_epochs(&mut self, …)` | `falsify_config_to_request_is_the_whole_merge_surface` (mutable-receiver arm) |
| move the length bound below the parse | `lock_from_canonical_bytes_refuses_an_oversized_payload_before_parsing` **and** `lock_from_canonical_bytes_bounds_the_raw_input_before_serde` |
| disable the canonical-form check | `lock_from_canonical_bytes_refuses_a_recorded_winner_the_rule_did_not_pick` |
| `MAX_SELECTION_LOCK_BYTES = 1_000` | 6 tests, incl. `lock_max_selection_lock_bytes_clears_a_realistic_sweep` |

RED for both tasks was also recorded before implementation: Task 1 failed with 4 `no method named
to_request` compile errors, Task 2 with 23 errors naming the absent door, the absent constant and
all five absent variants.

## Decisions Made

1. **No `with_seed` / `with_device` convenience.** Two doors to the same merge is how the two
   diverge. `to_request` plus `new` is the whole surface, and a source guard keeps it that way
   (no mutable receiver, no `with_seed`, no `with_device`, no `set_`).
2. **`from_canonical_bytes` rebuilds through `from_candidates`.** The alternative — reading the
   fields straight into a `SelectionLock` — would have re-implemented the finiteness,
   comparability, duplicate and non-empty invariants as a second answer that could drift, and
   would have trusted the file's `chosen_index` instead of deriving it. Deriving it is what stops
   a file from re-opening the caller-supplied-winner hole `from_candidates` exists to close.
3. **One canonical-form check instead of a per-field checklist.** Comparing the reconstruction's
   re-serialization against the input catches every disagreement class at once — a winner the rule
   did not pick, top-level fingerprints disagreeing with candidate zero's, a candidate whose two
   artifact-hash copies differ, a drifted nested evaluation schema version, a duplicated JSON key,
   a pretty-printed rendering. A checklist would drift from the wire form; this cannot.
4. **`verify_integrity()` at the end is documented as structural, not as a tamper check.** On this
   path the hash was derived from the record three lines earlier, so it cannot fail today. The
   comment says exactly that rather than implying a check it does not perform; the tamper property
   is that an edited file's hash *changes* and every downstream door re-checks identity.
5. **The range check runs after the rebuild.** An empty candidate list is then reported as
   `NoCandidates` — the truer statement — rather than as an out-of-range index into a list that
   does not exist.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 — Missing Critical] Added the canonical-form closure check and `NonCanonicalLockPayload`**
- **Found during:** Task 2 (`from_canonical_bytes` design)
- **Issue:** The plan's step list ((a) bound, (b) parse, (c) schema version, (d) chosen-index
  range, (e) rebuild, (f) `verify_integrity`) leaves four silent-normalization holes. `from_candidates`
  derives the top-level `dataset_fingerprint` / `validation_split_fingerprint` from candidate zero,
  so a file whose top-level values disagreed would be silently corrected. `SelectionCandidateWire`
  carries `artifact_hash` **and** `evaluation.artifact_hash`; a file where they disagree would be
  normalized to one of them, contradicting `SelectionCandidate`'s doc claim that no such
  disagreeable field exists. `ValidationEvaluationWire.schema_version` is dropped by `from_wire`, so
  a drifted nested version would be admitted. And `chosen_index` was read from the file rather than
  derived, so a file could name the loser as winner. In every case `verify_integrity` would still
  pass, because the hash is recomputed from the *repaired* record.
- **Fix:** After rebuilding through `from_candidates`, re-serialize and compare against the input
  bytes; refuse a difference as `NonCanonicalLockPayload { observed_len, canonical_len,
  first_difference_at }`. One check, total over all four holes.
- **Files modified:** `lock.rs`, `lock_tests.rs`
- **Verification:** `lock_from_canonical_bytes_refuses_a_recorded_winner_the_rule_did_not_pick`;
  disabling the check kills it (recorded above).
- **Committed in:** `b3d17de85`

**2. [Rule 2 — Missing Critical] Rebuilt through `from_candidates` rather than assembling the record directly**
- **Found during:** Task 2
- **Issue:** The plan's step (e) says "rebuild the lock" without routing through the constructor. A
  direct assembly would admit a file carrying a NaN metric, duplicate artifact hashes, mismatched
  metric kinds or an empty candidate list — every invariant `from_candidates` enforces — and would
  need the selection rule re-applied by hand or not at all.
- **Fix:** `from_canonical_bytes` calls `Self::from_candidates(...)` and propagates its errors.
- **Files modified:** `lock.rs`
- **Verification:** `cargo test … setfit::lock` (36 passed); `from_candidates` is still
  `pub(super)`, grep-asserted in `lock_from_canonical_bytes_bounds_the_raw_input_before_serde`.
- **Committed in:** `b3d17de85`

**3. [Rule 1 — Bug] The merge-surface guard's own doc comment spelled the needle it searched for**
- **Found during:** Task 1 (GREEN)
- **Issue:** `config.rs`'s tests live in the same file they scan. The first draft's `needle` doc
  comment contained the literal `pub fn to_request`, so the count came back 2 against a source with
  one definition — the exact self-match hazard the helper exists to prevent, committed inside the
  helper that warns about it.
- **Fix:** Rewrote the doc comment so it describes the needle without spelling it, and recorded the
  incident there as the reason the discipline extends to prose.
- **Files modified:** `config.rs`
- **Verification:** `falsify_config_to_request_is_the_whole_merge_surface` passes; `grep -c "pub fn
  to_request" config.rs` == 1.
- **Committed in:** `ea7d8bd8f`

**4. [Rule 1 — Bug] Test asserted `Result<SelectionLock, LockError>` where the source must say `Result<Self, …>`**
- **Found during:** Task 2 (GREEN)
- **Issue:** The RED test asserted the plan's prose signature verbatim. `clippy::use_self` is on
  (pedantic), so the spelled-out type would not survive the lint gate inside `impl SelectionLock`.
- **Fix:** The assertion now requires `Result<Self, LockError>` and states why in a comment, so the
  guard describes the source the lint actually permits.
- **Files modified:** `lock_tests.rs`
- **Verification:** `lock_from_canonical_bytes_bounds_the_raw_input_before_serde` passes;
  `cargo clippy -p aprender-train --features setfit` reports nothing in this crate.
- **Committed in:** `b3d17de85`

---

**Total deviations:** 4 auto-fixed (2 missing-critical, 2 bugs)
**Impact on plan:** Deviations 1 and 2 strengthen the door the plan asked for — they close
silent-normalization holes that would have made the reconstruction accept files
`from_candidates` refuses. Deviations 3 and 4 are self-inflicted test defects caught by the
tests themselves. No scope creep: nothing outside the four owned files was touched, and no new
package was added.

## Issues Encountered

**`cargo clippy -p aprender-train --features setfit -- -D warnings` exits non-zero on
pre-existing `aprender-compute` warnings.** The plan's verification line fails, but not on this
plan's code: the log contains **zero** matches for `aprender-train` or `setfit`. All warnings are
dead-code and unused-import warnings in `crates/aprender-compute/src/{blis,backends}` (SIMD
kernels), present in the baseline before any edit here. Per the scope boundary these were **not**
fixed. Verified by capturing the exit status into a variable rather than reading `$?` through a
pipe.

**Deferred item for the phase (not fixed here, out of scope):**
`crates/aprender-compute` carries ~12 dead-code / unused-import warnings that make any
`-D warnings` clippy invocation fail workspace-wide. Recorded here rather than in a shared
`deferred-items.md` because two sibling agents are committing on parallel worktree branches and a
shared file would be a merge conflict.

## Known Stubs

None. Both doors are fully wired: `to_request` reads the twelve live accessors, and
`from_canonical_bytes` reconstructs from real bytes and is exercised end-to-end against a real
verified run and a real re-tuned run.

## Threat Flags

None. The plan's threat register (T-04-41 DoS, T-04-42 tampering, T-04-43 elevation) is fully
mitigated as specified, and no new network endpoint, auth path, file-access pattern or schema at a
trust boundary was introduced. Note that T-04-42's mitigation is deliberately bounded: an edited
lock fails at the next door, and neither the code nor its documentation claims more.

## Next Phase Readiness

- **04-06** can now apply `--seed` / `--device` overrides through public, validated API. The merge
  is `let mut r = config.to_request(); r.root_seed = seed; SetFitTrainConfig::new(r)?`. Note that
  `new` normalizes `pair_config.root_seed` to the top-level seed, so a `--seed` override reseeds
  the pair stream too — asserted in `falsify_config_to_request_seed_override_moves_the_seed_and_nothing_else`.
- **04-07** can persist a lock with `to_canonical_bytes` and reconstruct it in a separate process
  with `from_canonical_bytes`, which is what makes TRN-07's two-process CLI workflow expressible.
  The CLI still owns the three distinct operator refusals the contract requires: **absent** lock
  (a CLI-level check — the library has no "no path given" error, by design), **stale** lock
  (`LockError::StaleLock` from `mint_test_token`), and **dataset mismatch**
  (`LockError::TokenDatasetMismatch` from `CanonicalTestAccess::grant`).
- **04-15 / 04-16** inherit both doors unchanged.
- No blockers. `mod.rs`, `bundle.rs` and `verify.rs` were not touched, so the wave-2 merge with
  04-02 and 04-13 has no file-level overlap.

## Self-Check: PASSED

- All four modified source files exist on disk.
- All four task commits exist in this worktree's history: `169fd14a8`, `ea7d8bd8f`, `d155ffd86`,
  `b3d17de85`.
- `git diff --diff-filter=D --name-only b30bff96a..HEAD` is empty — no file deletions.
- `git diff --name-only` against the wave base lists exactly the four owned files; `STATE.md`,
  `ROADMAP.md`, `mod.rs`, `bundle.rs` and `verify.rs` are untouched.
- `cargo fmt -p aprender-train -- --check` exits 0.

---
*Phase: 04-apr-artifact-and-production-parity*
*Completed: 2026-08-15*
