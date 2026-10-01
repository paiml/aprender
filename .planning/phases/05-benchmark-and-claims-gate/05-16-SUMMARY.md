---
phase: 05-benchmark-and-claims-gate
plan: 16
subsystem: testing
tags: [eval-02, claims-gate, selection-manifest, pairing-key, bench_gate, setfit, rust, provable-contracts]

requires:
  - phase: 05-benchmark-and-claims-gate (05-11)
    provides: "the D-19 ACTIVE 40-cell narrowing, the re-mutation rule, verify_cell, and the variant-tag arm-count guard"
  - phase: 05-benchmark-and-claims-gate (05-15)
    provides: "EvidenceKind, resolve_committed_evidence_path, read_evidence's kind parameter, the door probe with its positive control, and the measured-floor discipline"
provides:
  - "selection_manifest_path(bench_dir, cell): the pairing key's manifest resolved at a GATE-DERIVED path, whose signature takes no BenchRow"
  - "verify_selection_binding: the row's selection_manifest_hash recomputed through SelectionManifest::from_bytes, plus the manifest's own shots/seed compared against the cell key"
  - "BenchGateError::SelectionManifestMismatch and ::SelectionManifestCellMismatch, plus EvidenceKind::SelectionManifest"
  - "an eleven-row swept case table at the ACTIVE 40-cell scope, every row observed RED as Ok(40 rows verified) and GREEN as its own variant"
  - "spot-checks G and F replayed through the shipped apr door, both measured rc=0 by the verifier and now rc=5"
  - "contract 3.0.0: ACTIVE equation selection_binding_rule, OBLIG-CLAIMS-SELECTION-BOUND, FALSIFY-CLAIMS-011"
affects: [05-17, D-ITEM-05-15, benchmark claims gate, EVAL-02]

actuals:
  tokens: 18698
  tasks: 3
  commits: 3
plan_head_before: edc91e49b447f4b330c85ce10b1211481c3e3bfb

tech-stack:
  added: []
  patterns:
    - "A DERIVED path needs no validation door — and the signature, not the body, is what proves it"
    - "Two checks that an attacker reaches in sequence each need their own negative, plus an INVERTING case proving they are distinguishable"
    - "Do not recompute a digest another crate's constructor already verified before returning (OPS-03)"
    - "Where a swept case table is the right shape, the ROW COUNT must be asserted inside the test — a test-function floor cannot see rows collapsing"
    - "A new step goes AFTER an existing one when going before it would pre-empt an existing negative and silently change which rule that negative observes"

key-files:
  created: []
  modified:
    - "crates/aprender-train/src/train/setfit/bench_gate.rs"
    - "crates/aprender-train/src/train/setfit/bench_gate_tests.rs"
    - "contracts/setfit-benchmark-claims-v1.yaml"
    - "scripts/setfit_bench_gate_door_probe.sh"
    - "scripts/setfit_bench_gate_doctor.py"
    - "Makefile"

key-decisions:
  - "The digest is NOT recomputed in bench_gate. SelectionManifest::from_bytes verifies sha256(payload.to_canonical_bytes()) == semantic_hash before returning, so a second recomputation would be a second definition of a manifest's identity (OPS-03). An unsealed manifest is therefore refused by that door's own error, mapped into SelectionManifestMismatch."
  - "Wired as step 6b, AFTER verify_provenance and after step 5's verify_pairing. Placing it earlier would pre-empt the deferred-scope unpaired_selection negative; a test asserting that it still fires was added rather than reasoning about it."
  - "The contract bump is 3.0.0, the bump `pv diff` suggested. Every element is additive, which alone reads minor — but the ACCEPTANCE SET NARROWS and a tree that verified at 2.0.0 can be refused now."
  - "The Make floor is 52, MEASURED off the suite's `test result:` line. The plan's fails_when arithmetic (`05-15's floor plus 9`) contradicts its own acceptance criterion demanding ONE swept table; the criterion won and the row count is pinned inside the test instead."
  - "ExclusionRecord is built through its public Deserialize impl. Its four fields are private and it has no constructor and no Default; widening another crate's API so a fixture could build one would change production surface for a test's convenience."

patterns-established:
  - "An acceptance ROW is not optional: every refusal table carries a control that must be ACCEPTED, or the table can pass by refusing everything"
  - "A transplant negative that does not also doctor the row's claim proves the WRONG check — the first check fires and the one under test is never reached"

requirements-completed: []

coverage:
  - id: D1
    description: "The gate opens the committed selection manifest at a path derived from the cell key and refuses on disagreement with the row's selection_manifest_hash"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_selection_binding_case_table_at_the_active_scope (11 rows, ACTIVE 40-cell scope, through public verify_run)"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_the_selection_manifest_path_cannot_be_influenced_by_a_row"
        status: pass
      - kind: e2e
        ref: "bash scripts/setfit_bench_gate_door_probe.sh -> PASS, spot-check F rc=5"
        status: pass
    human_judgment: false
  - id: D2
    description: "A manifest transplanted from another cell is refused even when the row's hash is doctored to match it"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_selection_binding_case_table_at_the_active_scope (3 transplant rows + the inverting row)"
        status: pass
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_refuses_a_transplanted_manifest_at_both_ends_of_the_shot_axis"
        status: pass
    human_judgment: false
  - id: D3
    description: "An absent selections/ tree, an absent per-cell directory and a zero-byte manifest each produce a typed refusal naming the cell and the path"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_selection_binding_case_table_at_the_active_scope (rows 2-4)"
        status: pass
      - kind: e2e
        ref: "bash scripts/setfit_bench_gate_door_probe.sh -> spot-check G rc=5, no digest repair needed"
        status: pass
    human_judgment: false
  - id: D4
    description: "The hash comparison is exact byte equality of the 64-character lowercase hex — no truncation, no case folding, no prefix match"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_a_hash_agreeing_in_its_first_characters_is_still_a_refusal"
        status: pass
    human_judgment: false
  - id: D5
    description: "The refusal reported is the first offending cell in contract order, stable across repeated runs"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_reports_the_first_offending_selection_binding_in_contract_order_across_runs (3 invocations)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The scope fence holds: verify_pairing is byte-unchanged and the deferred-scope negative still reports unpaired_selection"
    verification:
      - kind: unit
        ref: "bench_gate_tests.rs#bench_gate_the_deferred_scope_pairing_negative_still_reports_unpaired_selection"
        status: pass
      - kind: other
        ref: "git diff over verify_pairing: zero hunks inside the function"
        status: pass
    human_judgment: false
  - id: D7
    description: "The committed 40-cell evidence still verifies: bench report and verify-cell both rc=0, and two consecutive reports are byte-identical"
    requirement: "EVAL-02"
    verification:
      - kind: e2e
        ref: "apr setfit bench report --bench-dir benchmarks/tweeteval-stance -> rc=0 (apr 0.63.0 (c7271ce2e))"
        status: pass
      - kind: e2e
        ref: "apr setfit bench verify-cell --method setfit --shots 8 --seed 13 -> rc=0"
        status: pass
      - kind: e2e
        ref: "two consecutive reports: cmp -> byte-identical, benchmarks/ untouched"
        status: pass
    human_judgment: false
  - id: D8
    description: "The new rule is falsifiable from the contract alone, validated by pv rather than a bash re-implementation"
    verification:
      - kind: other
        ref: "pv validate contracts/setfit-benchmark-claims-v1.yaml -> 0 errors, 0 warnings"
        status: pass
      - kind: other
        ref: "pv diff /tmp/claims-old.yaml contracts/... -> Suggested bump: major; applied as 3.0.0"
        status: pass
    human_judgment: true
    rationale: "pv proves the contract validates and scores the bump, but whether selection_binding_rule's invariants accurately describe what the code does — and whether the obligation's `what it does NOT cover` prose is complete — is a reading of the source only a human can confirm."
  - id: D9
    description: "The Make floor reads the suite's measured count and the banner prose names the selection-binding negatives"
    verification:
      - kind: integration
        ref: "make setfit-bench-tests -> rc=0, bench_gate leg reports 52 passed against a floor of 52"
        status: pass
      - kind: other
        ref: "grep -n '^\\.SHELLFLAGS' Makefile -> exactly 29 and 57, byte-unchanged"
        status: pass
    human_judgment: false

duration: 61min
completed: 2026-09-11
status: complete
---

# Phase 5 Plan 16: Bind the Pairing Key to the Committed Selection Manifest Summary

**The gate now OPENS `selections/s{shots}-seed{seed}/selection-manifest.json` at a path built from the CELL KEY, recomputes its `semantic_hash` through the manifest's own sealing door, and refuses both a row whose pairing key disagrees and a manifest transplanted from another cell — turning the forty committed manifests from inert files into the thing that makes `apr setfit bench report` refuse the two trees the verifier measured returning 0.**

## Performance

- **Duration:** 61 min (includes one watchdog-killed stream resumed from committed state)
- **Started:** 2026-09-11T22:39:08Z
- **Completed:** 2026-09-11T23:39:56Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- **Verifier gap 2 (EVAL-02, graded PARTIAL) is closed.** The recording half was already complete — 40/40 rows carry a distinct `selection_manifest_hash` equal to the `semantic_hash` of the committed manifest for their own cell — but the gate opened no manifest, so the property held in the DATA by construction and not by enforcement. `verify_selection_binding` now enforces it in both doors.
- **The path is DERIVED, and the signature is the proof.** `selection_manifest_path(bench_dir: &Path, cell: CellKey) -> PathBuf` takes no `BenchRow`, so — unlike `lock_record_path` and `candidate_ledger_path`, which 05-15 had to drag through a two-stage containment door — there is nothing for a producer to choose. A structural guard asserts the declaration verbatim.
- **Two checks, each with its own negative, plus an inversion proving they are distinguishable.** A transplanted manifest seals correctly, so a producer who also doctors the row's key satisfies the hash check completely; the cell-key comparison is the only thing left. The three transplant rows doctor the row hash; a fourth row performs the same transplant WITHOUT it and must report the hash mismatch instead.
- **All eleven mutations were observed RED as `Ok(40 rows verified)`** at the ACTIVE 40-cell scope through the public `verify_run` door — the attacks SUCCEEDED, which is the same false green the verifier saw through the shipped door — and GREEN afterwards, each with its own variant.
- **Spot-checks F and G refuse through the shipped door.** Both measured rc=0 by verification; both now rc=5, with the F replay asserting the refusal is the SELECTION one and not the row-digest one.

## Task Commits

1. **Task 2 (written first, as the plan's own `<action>` permits): the case table, RED** — `3cd44b3d6` (test)
2. **Task 1: the gate-derived path, the recomputation and the two refusals, GREEN** — `107a6de19` (feat)
3. **Task 3: the door replays, the contract co-evolution and the Make floor** — `c7271ce2e` (test)

**Commits:** 3, MEASURED as `git rev-list --count edc91e49b..HEAD`, not narrated.

No REFACTOR commit: the GREEN implementation is 1 helper + 1 check function + 2 variants, and there was no cleanup that did not change behaviour. `cargo fmt -p aprender-train` ran BEFORE the GREEN commit rather than after it, so 05-15's separate `style` commit has no counterpart here.

## TDD Gate Compliance

| Gate | Commit | Status |
|---|---|---|
| RED | `3cd44b3d6` `test(05-16): ...` | PASS — target test failed on an assertion for the planned behaviour |
| GREEN | `107a6de19` `feat(05-16): ...` | PASS |
| REFACTOR | — | Not taken; no behaviour-preserving cleanup was needed |

The RED is INTENTIONAL, not a nonzero exit: `cargo test ... --no-run` was rc=0 first (the tests compile against the unfixed gate, using only symbols that existed pre-fix — they compare `error.variant_tag()` against string literals that simply never match), and the failure was the named target test asserting on the planned behaviour. Not a syntax error, not zero-test discovery, not a fixture crash, not an unrelated assertion.

## THE RED / GREEN EVIDENCE (plan `<output>` requirement)

### The eleven-row case table — ONE LINE PER ROW, both observations

Taken through the PUBLIC `verify_run` door over a freshly built valid ACTIVE 40-cell run per row. PRE-FIX is commit `3cd44b3d6` with `bench_gate.rs` untouched; POST-FIX is `107a6de19`. rc=101 then rc=0.

| row | mutation | PRE-FIX (`3cd44b3d6`) | POST-FIX (`107a6de19`) | cell named |
|---|---|---|---|---|
| 1 | none (CONTROL) | `Ok(40 rows verified)` | `Ok(40 rows verified)` | — |
| 2 | the whole `selections/` tree deleted (spot-check G) | **`Ok(40 rows verified)`** | `evidence_file_missing` | `setfit/s8/seed13` |
| 3 | one cell's `selections/s16-seed29/` deleted | **`Ok(40 rows verified)`** | `evidence_file_missing` | `setfit/s16/seed29` |
| 4 | one manifest truncated to zero bytes | **`Ok(40 rows verified)`** | `evidence_read_failed` | `setfit/s16/seed29` |
| 5 | row key doctored to 64 zeros (spot-check F) | **`Ok(40 rows verified)`** | `selection_manifest_mismatch` | `setfit/s16/seed29` |
| 6 | row key doctored to ANOTHER cell's real digest | **`Ok(40 rows verified)`** | `selection_manifest_mismatch` | `setfit/s8/seed13` |
| 7 | TRANSPLANT (8,17)→(8,13) + row doctored | **`Ok(40 rows verified)`** | `selection_manifest_cell_mismatch` | `setfit/s8/seed13` |
| 8 | TRANSPLANT (8,13)→(16,13) + row doctored | **`Ok(40 rows verified)`** | `selection_manifest_cell_mismatch` | `setfit/s16/seed13` |
| 9 | TRANSPLANT (64,13)→(32,13) + row doctored | **`Ok(40 rows verified)`** | `selection_manifest_cell_mismatch` | `setfit/s32/seed13` |
| 10 | THE INVERSION: same transplant, row NOT doctored | **`Ok(40 rows verified)`** | `selection_manifest_mismatch` | `setfit/s8/seed13` |
| 11 | manifest payload edited without resealing | **`Ok(40 rows verified)`** | `selection_manifest_mismatch` | `setfit/s16/seed29` |

**`Ok` on all eleven is the RIGHT red**, and it is a stronger red than 05-15's. Every mutation left a run that was otherwise perfect: doctored rows are RESEALED and the manifest re-records their new digest, transplanted manifests are validly sealed, and the deleted-directory case edits no row byte at all. The pre-fix gate therefore had no other reason to refuse and returned `Ok`, which is the same false green the verifier observed through the shipped door. A red that had been `provenance_mismatch` or `row_digest_mismatch` would have proved the mutation was DETECTED rather than that the binding was MISSING.

Verbatim RED output (`[bench_gate] SELECTION_BINDING` lines, `--nocapture`, `why` strings elided for width; full log at `scratchpad/red-run.log`):

```
case=control                                          observed=Ok(40 rows verified)
case=selections_dir_deleted                           observed=Ok(40 rows verified)
case=one_cell_dir_deleted                             observed=Ok(40 rows verified)
case=manifest_truncated_to_zero_bytes                 observed=Ok(40 rows verified)
case=row_hash_doctored_to_64_zeros                    observed=Ok(40 rows verified)
case=row_hash_doctored_to_another_cells_real_hash     observed=Ok(40 rows verified)
case=transplant_adjacent_seed_with_row_doctored       observed=Ok(40 rows verified)
case=transplant_one_shot_step_down_with_row_doctored  observed=Ok(40 rows verified)
case=transplant_one_shot_step_up_with_row_doctored    observed=Ok(40 rows verified)
case=transplant_WITHOUT_row_doctored                  observed=Ok(40 rows verified)
case=manifest_payload_edited_without_resealing        observed=Ok(40 rows verified)
```

and the assertion that failed, carrying the same observations independently of the printout:

```
every selection-binding shape must be refused with its own variant through verify_run at the
ACTIVE 40-cell scope, and the control must be ACCEPTED; these were not:
[ ("selections_dir_deleted", "Ok(40 rows verified)|expected=evidence_file_missing"), ... ]
```

Three further tests were red in the same run and green after:

| negative | PRE-FIX | POST-FIX |
|---|---|---|
| `..._refuses_a_transplanted_manifest_at_both_ends_of_the_shot_axis` (s16 under s8; s32 under s64) | `the doctored run 's16_manifest_under_an_s8_cell' was ACCEPTED (40 rows verified)` | `selection_manifest_cell_mismatch` on both, each naming the manifest's own declared shots |
| `..._a_hash_agreeing_in_its_first_characters_is_still_a_refusal` (last byte flipped / truncated to 16 / upper-cased) | `the doctored run 'last_character_flipped' was ACCEPTED (40 rows verified)` | `selection_manifest_mismatch` on all three |
| `..._reports_the_first_offending_selection_binding_in_contract_order_across_runs` | `the doctored run 'two broken selection bindings are a refusal' was ACCEPTED (40 rows verified)` | names `setfit/s8/seed17` on all three invocations, with the cells doctored in REVERSE contract order |

Two additions were green both before and after, and are stated as such rather than dressed up as negatives: `..._the_forty_synthetic_selection_manifests_are_forty_distinct_digests` (a fixture property) and `..._the_deferred_scope_pairing_negative_still_reports_unpaired_selection` (a regression assertion whose whole value is that it did NOT change).

### THE CANONICALIZATION, MEASURED BEFORE ANYTHING DEPENDED ON IT

05-15 carried forward that `bench_row.rs:37-43` is wrong about its own digest scheme. That was re-measured here against the artifact this plan actually binds, the forty committed `selection-manifest.json` files, before any implementation choice was made:

| canonicalization of `payload` | reproduces the committed `semantic_hash` |
|---|---|
| compact JSON, **file / declaration order** | **40 / 40** |
| compact JSON, **key-sorted** | **0 / 40** |

So the 05-15 finding reproduces on this artifact too, and the comment's stated reason ("no workspace crate enables `preserve_order`", therefore key-sorted, therefore immune to field reordering) is false here as well. It is **not** a correctness defect for this plan, and for a reason worth stating: `SelectionPayload::to_canonical_bytes` is `serde_json::to_vec(self)` over a STRUCT with a fixed field order, so declaration order is what serde emits regardless of `preserve_order` — the ambiguity `bench_row`'s comment creates does not exist for this type. The measurement is recorded because the plan required the question to be settled by measurement rather than inherited, not because the answer changed anything.

It also removed the temptation to recompute the digest inside `bench_gate`: the implementation delegates entirely to `SelectionManifest::from_bytes`, so there is exactly one definition of these bytes and this measurement is a cross-check of it rather than a second implementation.

Independently measured on the same tree, and the reason the new check passes on committed evidence: **40/40 rows' `selection_manifest_hash` equals the `semantic_hash` of the manifest whose `payload.shots_per_class`/`payload.root_seed` match its cell, 0 mismatches**, and all 40 digests are distinct. No committed manifest or row was edited.

### The `pv diff` output and the version bump it suggested

```
$ git show HEAD:contracts/setfit-benchmark-claims-v1.yaml > /tmp/claims-old.yaml
$ target/release/pv diff /tmp/claims-old.yaml contracts/setfit-benchmark-claims-v1.yaml
Contract diff: v2.0.0 → v2.0.0
Suggested bump: major

  equations:
    + selection_binding_rule
    ~ pairing_rule: invariants changed
  proof_obligations:
    + equivalence:OBLIG-CLAIMS-SELECTION-BOUND: …
  falsification_tests:
    + FALSIFY-CLAIMS-011
```

`pv 0.63.0`. The suggestion was taken: `metadata.version` 2.0.0 → **3.0.0**, with the output above recorded verbatim in the file's metadata block in 05-11's house style, together with the reasoning — every element is ADDITIVE, which alone reads like a minor, but the ACCEPTANCE SET NARROWS and a benchmark directory that `apr setfit bench report` accepted at 2.0.0 can be REFUSED at 3.0.0. A strengthened guarantee is a breaking change to every producer relying on the old acceptance.

`pv validate contracts/setfit-benchmark-claims-v1.yaml` → **0 error(s), 0 warning(s). Contract is valid.** `pairing_rule` still carries `status: deferred` and `deferred_ticket: D-ITEM-05-15`; only its third invariant changed.

### The measured test-count floor, and where the number came from

38 → 45 (05-15) → **52**. The 52 was MEASURED, not computed from the plan — the suite was run and its own line read:

```
test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 8044 filtered out; finished in 2.55s
```

`make setfit-bench-tests` then reports `52 passed` against a floor of `52` — exact, and 52 > 45. The other three legs are unmoved at their existing floors: row **27**, metrics **14**, cli **65** (27 + 14 = 41, which is the plan's own `bench_metrics|bench_row` bar).

### The door probe's full PASS output, including its positive control and the two new replays

Run at HEAD `c7271ce2e` with `apr 0.63.0 (c7271ce2e)` — the binary pin equals `git rev-parse --short HEAD` (CLAUDE.md rule 3).

```
CONTROL: undoctored slim copy of <repo>/benchmarks/tweeteval-stance verifies (rc=0)
DOCTORED: setfit-s8-seed13 now points at <tmp>/outside/anywhere.json, and the committed lock record is gone
ATTACK: rc=5, refused as a path escape naming <tmp>/outside/anywhere.json
SPOT-CHECK G: rc=5 with selections/ deleted and every row byte untouched, refused as an absent selection manifest
SPOT-CHECK F: rc=5 with setfit-s8-seed13 claiming selection_manifest_hash=0000000000000000000000000000000000000000000000000000000000000000, refused by the recomputation and NOT at the row digest
PASS: <repo>/target/release/apr refuses a row-supplied evidence path that leaves the benchmark directory, a deleted selection manifest, and a doctored pairing key, having first verified the undoctored tree
```

Each case runs on its OWN slim copy, so a later case cannot pass because an earlier one already broke the tree. The probe was run three times (before the lint fixes, after them, and at final HEAD) and differs only in the `mktemp` directory name, with zero scratch directories left behind and `benchmarks/` untouched.

### The idempotency backstop (`verification: backstop` truth)

Two consecutive `apr setfit bench report --bench-dir benchmarks/tweeteval-stance` runs on the untouched committed tree: both rc=0, and `cmp` reports the two outputs **byte-identical** (`sha256 d7e3fd0b…`). `git status --short benchmarks/` is empty afterwards. The new selection-manifest reads introduce no state.

## Files Created/Modified

- `crates/aprender-train/src/train/setfit/bench_gate.rs` — `SELECTIONS_DIR`, `SELECTION_MANIFEST_FILE`, `selection_manifest_path`, `verify_selection_binding`, `EvidenceKind::SelectionManifest`, two new `BenchGateError` variants with their `variant_tag`/`cell()`/`Display` arms; step 6b wired into both `verify_run_scoped` and `verify_cell`; the module header's numbered step list and `verify_cell`'s APPLIES/DOES NOT APPLY enumeration both updated to name the binding.
- `crates/aprender-train/src/train/setfit/bench_gate_tests.rs` — the synthetic builder writes forty real sealed manifests; `synthetic_selection_hash` returns the real digest; the eleven-row `SELECTION_BINDING_CASES` table and its sweep; the shot-axis boundary transplants; the exact-byte-equality negative; the deterministic-order negative; the distinctness assertion; the deferred-scope re-assertion; the derived-path structural guard; the arm-count guard raised 15 → 17 with a per-variant argument; the header inventory extended from fifteen to twenty entries.
- `contracts/setfit-benchmark-claims-v1.yaml` — 3.0.0; `equations.selection_binding_rule` (ACTIVE, six invariants); `pairing_rule`'s third invariant amended; `OBLIG-CLAIMS-SELECTION-BOUND`; `FALSIFY-CLAIMS-011`.
- `scripts/setfit_bench_gate_door_probe.sh` — `slim_copy` and `run_report` helpers, one slim copy per case, and the G and F replays.
- `scripts/setfit_bench_gate_doctor.py` — a `<mode>` argument: `escape` (unchanged) and `selection-hash-zeros`.
- `Makefile` — floor 45 → 52, banner prose fifteen → twenty shapes with the selection-binding account, closing banner updated.

## Decisions Made

- **The digest is not recomputed in `bench_gate` (OPS-03).** `SelectionManifest::from_bytes` verifies before returning, so a caller cannot hold an unsealed manifest. The unsealed-payload case is therefore refused by *that* door's error, mapped into `SelectionManifestMismatch`; a structural guard goes red on the three obvious ways to write a second recomputation.
- **Step 6b, after `verify_provenance` and after step 5.** Moving the binding above `verify_pairing` would pre-empt the deferred-scope `unpaired_selection` negative and silently change which rule that negative observes — it would stay green while it had stopped testing what it names. `verify_pairing` is byte-unchanged, and a dedicated test asserts the negative still reports `unpaired_selection`.
- **`u64::from(cell.seed)`, never `try_from` and never `as`.** `root_seed` is `u64` and `CellKey::seed` is `u32`; widening the key rather than narrowing the manifest keeps the question "are these the same seed" instead of "are these the same seed, given it fits".
- **Only two variants minted, and the missing-manifest case mints nothing.** It reuses `evidence_file_missing` through a new `EvidenceKind`, which is the default answer. The arm-count guard records the argument for each of the two that *were* minted.
- **`requirements-completed` is deliberately empty**, consistent with all fifteen prior phase-5 plans: flipping requirement state is the verifier's act, and `05-VERIFICATION.md` calls that "correct process". `requirements.mark-complete` was not run for EVAL-02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `ExclusionRecord` is not publicly constructible; built through its `Deserialize` impl**
- **Found during:** Task 2 (the fixture)
- **Issue:** The plan's `read_first` instructed "Confirm by inspection that every field and every nested record type is publicly constructible from `aprender-train`'s test module — that is what makes a synthetic manifest buildable." It is not. `ExclusionRecord` (`aprender-contrastive-data/src/dedup.rs:67`) has four PRIVATE fields, no constructor and no `Default`; only its accessors are public.
- **Fix:** Built through the type's own public `Deserialize` impl from a minimal empty record. `SelectionPayload`, `SelectedExampleRecord`, `VolatileMetadata`, `AccessRecord` and `DuplicateGroup` all *are* directly constructible; `ExclusionRecord` is the one that is not.
- **Rejected alternative:** adding a `pub fn new` or `Default` to `aprender-contrastive-data` — that changes production surface in a crate outside this plan's `files_modified`, for a fixture's convenience.
- **Verification:** the fixture builds forty manifests, all forty digests distinct, each sealing through `SelectionPayload::to_canonical_bytes`.
- **Committed in:** `3cd44b3d6`

**2. [Rule 1 — Bug] The plan's `+9 tests` floor arithmetic contradicts its own single-table acceptance criterion**
- **Found during:** Task 3 (the Make floor)
- **Issue:** Task 2's `<fails_when>` requires "fewer than 05-15's floor plus 9 tests" (≥ 54). Task 2's `<acceptance_criteria>` requires "The case table is a single named `const` swept by one loop; there is no hand-written test function duplicating a table row." Both cannot hold: eleven rows swept by one loop add ONE test function, and `assert_tests_ran` counts functions. Satisfying the arithmetic would have meant writing the duplicate per-row test functions the criterion forbids.
- **Fix:** The acceptance criterion wins — it is the one describing the artifact, and padding a suite to hit a number is instrument-tuning. The floor is the MEASURED 52 (45 + 7 new test functions), which is the discipline 05-15 established. **The gap the arithmetic was reaching for is closed a better way:** the sweep now asserts `SELECTION_BINDING_CASES.len() == 11` and `observations.len() == the table length` INSIDE the test, so eleven rows collapsing to three goes red there rather than silently — something a test-function floor could never see. The Makefile banner states this explicitly so the next reader does not re-derive it.
- **Verification:** `make setfit-bench-tests` rc=0 with `52 passed` against a floor of `52`; deleting a table row fails the in-test assertion.
- **Committed in:** `c7271ce2e`

**3. [Rule 3 — Blocking] The probe cannot grep for `row_digest_mismatch`; it greps the refusal's own sentence**
- **Found during:** Task 3
- **Issue:** The plan's `<fails_when>` says the probe fails on "output containing `row_digest_mismatch`". `apr` renders `BenchGateError`'s `Display`, never `variant_tag()` (`apr-cli/src/commands/setfit_bench.rs:2478`), so that literal can never appear in the probe's output — the guard could not fire, which is the theater CLAUDE.md rule 5 names.
- **Fix:** The F replay greps the row-digest refusal's own distinctive sentence, `not the bytes that were attested`, and fails if it is present. The reasoning is recorded in the probe's header so the substitution is visible rather than silent.
- **Verification:** the probe's F case reports `refused by the recomputation and NOT at the row digest`; removing the digest repair from the doctor makes it fail with that message.
- **Committed in:** `c7271ce2e`

**4. [Rule 3 — Blocking] The doctor script gained a `<mode>` argument**
- **Found during:** Task 3
- **Issue:** The F replay needs different doctoring from the E replay, and the script took a fixed 3-argument signature.
- **Fix:** `setfit_bench_gate_doctor.py <mode> <bench> <outside> <cell>` with `escape` and `selection-hash-zeros`. There is exactly one call site (the probe), updated in the same commit. The new mode leaves the COMMITTED MANIFEST alone — only the row's claim moves, so the single disagreement is the one the recomputation exists to find; doctoring the manifest instead would prove it was read but not that the row's claim was compared against it. It also refuses vacuity: if the committed row already claims 64 zeros, it aborts.
- **Verification:** probe rc=0, PASS, all four cases.
- **Committed in:** `c7271ce2e`

**5. [Rule 1 — Bug] Two real `bashrs` findings in the new probe code, fixed; one new false positive, recorded**
- **Found during:** Task 3
- **Issue:** `bashrs lint` went 0 errors / 8 warnings / 22 infos (baseline) → 0 / 14 / 30. Two findings were real: `dest=$1` unquoted in `slim_copy` (SC2320), and four `*_rc=$report_rc` assignments (SC2086). One was a parse artefact: IDEM002 "non-idempotent rm" fired on the substring `rm` inside the word "arm" in a COMMENT.
- **Fix:** quoted all five; reworded the comment to "second METHOD", which is the contract's own vocabulary anyway. Result: **0 errors / 9 warnings / 29 infos**. The one net-new warning over baseline is a SEC014 path-traversal false positive on `rm -rf "${G_DIR:?}/selections"`, whose operand is mktemp-derived — the same class as the pre-existing `cp` one that already carries a `# bashrs:allow SEC014` directive bashrs does not honour.
- **Verification:** `bashrs lint scripts/setfit_bench_gate_door_probe.sh` → 0 errors. `bashrs make lint Makefile` → 1 error / 43 warnings, **byte-identical to the same lint against `git show HEAD:Makefile`** before this plan's edit; the one error is the pre-existing `local`-outside-a-function at `dev-setup`.
- **Committed in:** `c7271ce2e`

---

**Total deviations:** 5 auto-fixed (3 blocking, 2 bugs)
**Impact on plan:** All five were required to satisfy the plan's own acceptance criteria, and two of them (2 and 3) are cases where the plan's `<fails_when>` prose could not be satisfied as literally written. No scope creep: the diff is exactly the five files the plan named plus `setfit_bench_gate_doctor.py`, the load-bearing sibling 05-15 extracted and which the probe cannot run without.

## Issues Encountered

### 1. FINDING for 05-17: `verify-cell`'s PRINTED scope prose now under-describes what the door does

`apr setfit bench verify-cell` prints, at `apr-cli/src/commands/setfit_bench.rs:2931`:

> `scope: steps 1+4+6 of verify_run over ONE cell - manifest digest, this entry's own completeness, the row file/schema/envelope digest/manifest digest/slot, and provenance recomputed from committed bytes.`

That enumeration no longer names the selection binding, which the door now also performs. This does **not** violate this plan's prohibition — the prohibition forbids describing the manifest as binding something the run did not check, and this is the opposite: the door checks MORE than it says. It is nonetheless a door whose own printed enumeration of its coverage is incomplete, which is the class of artifact the phase exists to prevent.

Not fixed here, deliberately: the plan's own threat model assigns it, at T-05-16-05 — *"the gate enforces it before the report can describe it; 05-17 updates the report's own disclosure wording"* — and `apr-cli` is outside this plan's `files_modified`. The same applies to `bench report`'s `verified:` header, which says "provenance was recomputed from the committed lock bytes rather than read off the rows" and is silent about the pairing key.

### 2. The plan's clippy verification line still CANNOT PASS on this tree — unchanged from 05-15, restated rather than smoothed over

`cargo clippy -p aprender-train --lib --features setfit -- -D warnings` exits **101**. Zero of the findings are in `crates/aprender-train`; all are in `aprender-compute` and `aprender-present-terminal`, surfaced because `-D warnings` propagates to dependency crates compiled in the same session. The in-scope signal, taken with `--no-deps` so only the selected package is linted:

```
cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings  →  rc=0, clean
cargo fmt -p aprender-train -- --check                                           →  rc=0
```

Stated so a reader does not mistake `--no-deps` for the plan's literal command. Out of the scope boundary (pre-existing lint debt in unrelated files); CLAUDE.md assigns it to the `toolchain-ceiling.yml` gate.

### 3. The 600s stream-idle watchdog fired once, mid-Task-1

Not slow compilation and not a hanging test — both were measured: `cargo test -p aprender-train --lib --features setfit --no-run` is 43s and the RED suite executes in under a second. The RED test work was uncommitted at the time and survived; the first action on resume was to commit it. The operational lesson carried forward from 05-15 holds: **commit the RED before implementing**, so a killed stream costs a restart rather than the work.

### 4. `cargo test`'s `test result:` line is invisible through the Bash tool

The `rtk` hook rewrites `cargo test` and filters its output BEFORE any redirect, so `cargo test … > log; grep '^test result:' log` finds nothing. Every count in this document was read by invoking the test binary directly (`target/debug/deps/entrenar-*`) or from `make`'s own log, which `make` writes itself. This is an existing, recorded property of the environment, not something this plan changed — noted because a reader reproducing the counts from the plan's literal commands will get an empty grep and conclude the suite did not run.

## Known Stubs

None. No file created or modified by this plan contains a hardcoded empty value flowing to output, a placeholder string, or an unwired data source. The forty committed selection manifests moved in the opposite direction: from orphaned files to load-bearing inputs.

## Threat Flags

None. Every file touched is covered by the plan's own `<threat_model>`, and the five mitigations it assigns are implemented and proven above:

| Threat | Disposition | Evidence |
|---|---|---|
| T-05-16-01 `selection_manifest_hash` never recomputed | mitigate | table rows 5, 6, 10, 11; door probe spot-check F rc=5 |
| T-05-16-02 a valid manifest transplanted from another cell | mitigate | table rows 7–9 + the inverting row 10; both shot-axis boundaries |
| T-05-16-03 the `selections/` tree deleted | mitigate | table rows 2–3; door probe spot-check G rc=5, no digest repair needed |
| T-05-16-04 a manifest edited without resealing | mitigate | table row 11, refused by `from_bytes`'s own digest check (not reimplemented) |
| T-05-16-05 the report implying a binding the gate does not enforce | mitigate | the gate enforces it; the report's own wording is Issue 1 above, assigned to 05-17 by this row |

`T-05-16-06` (path escape) is `accept` and is structurally absent: `selection_manifest_path`'s signature takes no `BenchRow`, which a structural guard now asserts verbatim. No new network endpoint, auth path, file-access pattern or schema change at a trust boundary — the change strictly NARROWS what the gate will accept.

## Next Phase Readiness

- **Ready for 05-17 (EVAL-01).** Two concrete inheritances: (a) `EvidenceKind` gained a fourth variant and is still deliberately not `#[non_exhaustive]`, so a fifth forces every match arm to be revisited; (b) the report's own disclosure wording is now measurably behind what both doors do — Issue 1 gives the two exact strings and their file/line.
- **Both remaining gates must extend the header inventory** in `bench_gate_tests.rs` (now twenty entries) and raise the `setfit-bench-gate` floor to their own MEASURED count. If a future plan also ships a swept table, it should pin the ROW count inside the test as this one does; the Make floor cannot see rows.
- **`D-ITEM-05-15` is materially better off.** The cross-method `pairing_rule` stays deferred with an empty domain, but the key it will pair on is now ENFORCED rather than merely recorded, and `selection_manifest_path` ignores the method — so both halves of a restored pair resolve the same manifest file and the second arm inherits this binding without a second implementation. `verify_pairing` is byte-unchanged and its negative is asserted to still fire on its own variant.
- **Carried forward for a human:** Issue 1 (the report's under-claiming prose, 05-17's by assignment); Issue 2 (the workspace clippy/fmt debt, `toolchain-ceiling.yml`'s by CLAUDE.md); and `bench_row.rs:37-43`'s stale canonicalization rationale, which this plan re-measured and confirmed wrong for a second artifact but did not edit — `bench_row.rs` is outside this plan's `files_modified`, and 05-17 touches that module.
- **Open, not closed by this round:** the binding proves which sampled-ID set a cell CONSUMED, not that the model was TRAINED on it. That distinction is written into `OBLIG-CLAIMS-SELECTION-BOUND`'s own `property` prose rather than left for a reader to infer, along with the fact that it reduces no part of the `ece_top_label_validation` / `brier_multiclass_validation` surface.

---
*Phase: 05-benchmark-and-claims-gate*
*Completed: 2026-09-11*
