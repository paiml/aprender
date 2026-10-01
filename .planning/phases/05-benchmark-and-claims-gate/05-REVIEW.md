---
phase: 05-benchmark-and-claims-gate
reviewed: 2026-09-12T00:00:00Z
depth: standard
supersedes: "the 4e80e48cc review of this phase (previous 05-REVIEW.md, 52 files)"
scope_note: "Tier-2 SUMMARY-derived scope for the 05-15 / 05-16 / 05-17 gap-closure work only. The
  aprender-image / aprender-mcp-chronos / spectral-indices files that a raw diff against
  4e80e48cc also lists belong to unrelated commit fdf6b1802 and were NOT reviewed."
files_reviewed: 12
files_reviewed_list:
  - crates/aprender-train/src/train/setfit/bench_gate.rs
  - crates/aprender-train/src/train/setfit/bench_gate_tests.rs
  - crates/aprender-train/src/train/setfit/bench_metrics.rs
  - crates/aprender-train/src/train/setfit/bench_metrics_tests.rs
  - crates/apr-cli/src/commands/setfit_bench.rs
  - crates/apr-cli/src/commands/setfit_bench_tests.rs
  - scripts/setfit_bench_gate_door_probe.sh
  - scripts/setfit_bench_gate_doctor.py
  - Makefile
  - contracts/setfit-benchmark-claims-v1.yaml
  - benchmarks/tweeteval-stance/report.md
  - .planning/phases/05-benchmark-and-claims-gate/05-15-gate-input-surface.md
findings:
  critical: 1
  warning: 8
  info: 5
  total: 14
status: issues_found
---

# Phase 5: Code Review Report (05-15 / 05-16 / 05-17 gap closure)

**Reviewed:** 2026-09-12
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found
**Supersedes:** the `4e80e48cc` review of this phase.

## Summary

The three gap closures do what they say at the level that matters most. I traced
`resolve_committed_evidence_path` against every bypass on the hunt list and **found none**:
stage 1 is total and filesystem-independent, `Component::ParentDir` / `RootDir` / `Prefix` are
all refused, containment is genuinely `Path::starts_with` over **both** sides canonicalized
(`bench_gate.rs:918`), the TOCTOU residual is disclosed in the doc comment rather than hidden,
`selection_manifest_path` takes no `BenchRow` so there is nothing to steer, `verify_selection_binding`
compares the 64-hex key by full `String` equality with no truncation or case folding
(`bench_gate.rs:1712`), `verify_quality_closed_form` compares every `f64` through `to_bits()`
and never `==`, the confusion-matrix total is accumulated with `saturating_add` and
short-circuits before any allocation (`bench_metrics.rs:462-474`), `matthews_corrcoef`
accumulates in `i64` so the expansion really is order-independent, and the CLI fails closed —
any `BenchGateError` becomes `CliError::ValidationFailed` and the rendering functions are never
reached (`setfit_bench.rs:2517-2522`). There is no `unwrap()` in any non-test code in scope.
The door probe captures status with `cmd > log 2>&1; rc=$?` throughout and never through a pipe,
and every one of its `grep` assertions matches a literal that really is in a `Display` arm — I
checked all six against `bench_gate.rs:630-762`.

What I did find falls into three groups.

**One blocker, and it is in the phase's own tooling rather than the gate.** The Python fixture
doctor's "refuse to doctor the real tree" guard is both cwd-relative and a *string*-prefix test —
the exact containment mistake 05-15 spent a whole plan removing from the Rust — so it fails open
outside the repo root and the script will then move a committed lock record out of the tree and
overwrite committed rows and the run manifest.

**A stale completeness artifact, which is the class invariant this phase exists to enforce.**
`05-15-gate-input-surface.md` still records `payload.quality.ordered_labels` and
`payload.quality.confusion_matrix` as having "no gate consumer" and files `ordered_labels` under
"the identifying facts of the run, which nothing under `bench_dir` attests". Since 05-17 both
**steer** the recomputation — `ordered_labels.len()` sets the class count, the matrix dimension
check and which indices `f_avg` averages — and `ordered_labels` is compared against nothing
anywhere on the gate path. The artifact's whole reason for existing is that a field which moved
class must not be describable by a stale table.

**Several disclosure and negative-control defects.** The published report now prints that the
row's own confusion matrix was "RECOMPUTED ... rather than read off the rows"; the contract
asserts the three residual statements agree "WORD FOR WORD" when they do not; the "ragged
matrix" case in both suites is not ragged, so the branch that catches a ragged matrix is never
the branch that decides any test; one of the path table's three "acceptance rows" is a verbatim
duplicate of another; and the only end-to-end proof that the *shipped* door refuses is a Make
target that nothing depends on and that is not even in `.PHONY`.

Known and already filed (`D-ITEM-05-17-A`, `-B`, `-C`, the workspace clippy debt,
`bench_row.rs:37-43`) are not re-reported. I found no *new* in-scope code whose correctness
depends on which `serde_json` feature set is linked.

## Critical Issues

### CR-01: The fixture doctor's "refuse to doctor the real tree" guard fails open outside the repo root, then destroys committed evidence

**File:** `scripts/setfit_bench_gate_doctor.py:110-111`

```python
if os.path.realpath(bench_dir).startswith(os.path.realpath("benchmarks")):
    raise SystemExit("refusing to doctor the committed benchmark directory")
```

**Issue:** Two independent defects in one line.

1. **`os.path.realpath("benchmarks")` is resolved against the process's cwd**, and `realpath`
   does not require the path to exist — it happily returns `<cwd>/benchmarks` for a directory
   that is not there. The one caller (`setfit_bench_gate_door_probe.sh:85`) does
   `cd "$REPO_ROOT"` first, so the guard happens to be correct there and nowhere else. The
   module docstring ends with a `Usage:` line (`:58`) and the file has a `__main__` block
   (`:172`), so hand invocation is invited. Run from any other directory, e.g.
   `cd crates && python3 ../scripts/setfit_bench_gate_doctor.py escape ../benchmarks/tweeteval-stance /tmp/x setfit-s8-seed13`,
   the guard compares `/repo/benchmarks/tweeteval-stance` against `/repo/crates/benchmarks`,
   does not match, and proceeds.
2. **`str.startswith` is a string-prefix test on a path.** This is precisely the bug
   `resolve_committed_evidence_path` exists to avoid — `bench_gate.rs:833-834` documents the
   `<tmp>/bench` vs `<tmp>/bench-evil` case and the path case table carries a dedicated
   `prefix_sibling_symlink` row that goes red on it. The same mistake is re-introduced in the
   same phase's Python. Here it also over-refuses: a scratch tree at `<repo>/benchmarks-scratch`
   would be wrongly rejected.

**What then happens** is not a no-op. In `escape` mode the script calls
`shutil.move(lock_path, escape_path)` (`:123`), physically moving
`benchmarks/tweeteval-stance/locks/setfit-s8-seed13.lock.json` out of the repository; then
`write_pretty(row_path, row)` (`:153`) overwrites the committed row with a doctored
`lock_record_path`, and `write_pretty(manifest_path, manifest)` (`:166`) overwrites
`run-manifest.json` with a re-sealed digest. The result is a *self-consistent* doctored tree —
every digest is repaired by design — so `apr setfit bench report` would refuse only at the path
escape, and `git status` would show three modified/deleted files that look like a legitimate
re-seal. These 40 rows plus the manifest are the only evidence the entire phase rests on.

**Fix:** make the guard absolute, repo-derived and component-wise, and check it before anything
else touches the filesystem:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMITTED = (REPO_ROOT / "benchmarks").resolve()

target = Path(bench_dir).resolve()
if target == COMMITTED or COMMITTED in target.parents:
    raise SystemExit(
        "refusing to doctor the committed benchmark directory {0}; this script only "
        "doctors a scratch copy".format(target)
    )
```

`Path.parents` is component-wise, so `benchmarks-scratch` is correctly allowed and
`benchmarks/tweeteval-stance` correctly refused, and `__file__` makes the answer independent of
cwd. Add a test row (or a `# must-refuse` case in the probe) that invokes the script with a
`bench_dir` under `benchmarks/` and asserts a non-zero exit with no file touched — the guard
currently has no negative control at all.

## Warnings

### WR-01: `ordered_labels` now steers the recomputation but is compared against nothing, and the input-surface enumeration still says it has "no gate consumer"

**File:** `crates/aprender-train/src/train/setfit/bench_metrics.rs:439-456` ·
`crates/aprender-train/src/train/setfit/bench_gate.rs:1776-1783` ·
`.planning/phases/05-benchmark-and-claims-gate/05-15-gate-input-surface.md:71,73,236-238,258-260`

**Issue:** `verify_quality_closed_form` passes `quality.ordered_labels` straight into
`quality_from_confusion_matrix`, where `n_classes = ordered_labels.len()` decides (a) the
required matrix dimension, (b) the length of all three per-class vectors, (c) the class count
handed to `MultiClassMetrics::from_predictions_with_min_classes`, and (d) whether
`f1_average_for_classes(..., &OFFICIAL_F_AVG_CLASSES)` (`[1, 2]`) is in range. Nothing on the
gate path compares it against anything: `verify_contracted_row_constants`
(`bench_gate.rs:1254-1299`) closes `contract_id`, `calibration_split`, `warmup_count` and
`cold_measured_in_child_process` and does not touch it, and `bench_row.rs:341` declares it as a
bare `Vec<String>`.

Concretely, a producer can ship `ordered_labels = ["none","against","favor","pad"]` with a
consistent 4×4 matrix. Every cross-check passes, the per-class vectors are length 4,
`macro_f1` is averaged over four classes, and `f_avg` is `(F1[1]+F1[2])/2` over a label map that
is not the contract's — while `benchmarks/tweeteval-stance/report.md` prints unconditionally
"F_avg = (F1_against + F1_favor) / 2, the official TweetEval stance metric". This grants an
adversary no *numeric* freedom he did not already have via residual (1) (the matrix is
producer-written), which is why this is a Warning and not a Critical — but it does mean the
published number need not be the metric the report names it as.

The artifact defect is the sharper half. `05-15-gate-input-surface.md:71` still says
`confusion_matrix` has "no gate consumer" and `:73` says the same of `ordered_labels`, filing it
under section **F, "The identifying facts of the run, which nothing under `bench_dir`
attests"** (`:236-238`). Both are now false: they are the two inputs the whole 05-17 cross-check
is a function of. That file is the phase's own claim that the input surface was enumerated
completely, and it is the artifact a future round will read to decide what is still open.

**Fix:** two parts.
1. In the document: move `ordered_labels` and `confusion_matrix` out of section F into a new
   class — they are neither "recomputed" nor "compared" nor "carried and read by nothing"; they
   are *trusted values that steer a comparison*, which is a strictly worse class than (iii) and
   the one this phase's invariant is about. Restate `ordered_labels`'s open item in those terms.
2. In the code, close it the way 05-15 closed `contract_id` — a one-line class-(ii) comparison,
   using the label map the crate already pins:

```rust
// in verify_contracted_row_constants, beside the other contract-pinned constants
if row.payload.quality.ordered_labels != CONTRACTED_ORDERED_LABELS {
    return Err(refuse(format!(
        "the row declares ordered_labels {:?}, but `tweet-eval-stance-benchmark-v1` at its \
         pinned revision declares {CONTRACTED_ORDERED_LABELS:?}. `OFFICIAL_F_AVG_CLASSES` \
         selects indices [1, 2] of THAT map, so a different map publishes a different metric \
         under the official one's name",
        row.payload.quality.ordered_labels
    )));
}
```

`bench_metrics_tests.rs:355` (`..._label_order_is_evidence_from_the_pinned_dataset_revision`)
already establishes the map from the dataset contract; this is the same constant reused on the
read side instead of only the emit side. If pulling a second contract into `bench_gate`'s
compile-time surface is not wanted this round, at minimum bound the arity
(`ordered_labels.len() == 3`), which is a one-line refusal with no new dependency.

### WR-02: the equivalence the whole cross-check rests on is asserted nowhere, and the gate fixture is circular

**File:** `crates/aprender-train/src/train/setfit/bench_gate_tests.rs:319-346` ·
`crates/aprender-train/src/train/setfit/bench_metrics_tests.rs` (absence)

**Issue:** The load-bearing claim is "`quality_from_confusion_matrix` produces bit-for-bit what
`assemble_quality_block` published for the same counts" — stated in `bench_metrics.rs:33-38`, in
`bench_gate.rs:1745-1752`, and as a contract invariant
(`setfit-benchmark-claims-v1.yaml:572-579`, "EXACTNESS IS STRUCTURAL, NOT LUCKY"). Nothing
asserts it directly:

- `synthetic_quality` (`bench_gate_tests.rs:325`) builds the fixture **by calling
  `quality_from_confusion_matrix` itself**. The control row of
  `QUALITY_CROSS_CHECK_CASES` therefore passes by construction and cannot detect an error in
  the recomputation; it can only detect the gate failing to call it. The comment at `:322-324`
  acknowledges this ("the fixture and the gate hold exactly one definition") but the consequence
  — the table's acceptance row is tautological w.r.t. the recomputation — is not stated.
- `bench_metrics_tests.rs` calls `assemble_quality_block` in nine tests and
  `quality_from_confusion_matrix` in six, and **never in the same test**. The only cross-anchor
  is `..._the_forty_committed_rows_agree_with_their_own_confusion_matrices` (`:681`), which
  compares the recomputation against 40 rows emitted by a *previous* build.

So if `assemble_quality_block` drifted tomorrow (a different MCC surface, a different
`Average`), the 40-row test stays green on historical evidence, the gate table stays green on a
circular fixture, and the failure surfaces only as `apr setfit bench report` refusing rows a
fresh run just emitted.

**Fix:** one test, no new machinery, closing the loop on the toy predictions that already exist:

```rust
#[test]
fn bench_metrics_the_recomputation_reproduces_what_the_emitter_published() {
    // THE EQUIVALENCE THE CROSS-CHECK RESTS ON, asserted on freshly emitted numbers rather
    // than only on forty rows a previous build wrote.
    let block = assemble_quality_block(&toy_test_rows(), &toy_validation_rows(), &labels())
        .expect("the toy split assembles");
    let back = quality_from_confusion_matrix(&block.confusion_matrix, &block.ordered_labels)
        .expect("its own matrix is well formed");
    assert_eq!(block.f_avg.to_bits(), back.f_avg.to_bits());
    assert_eq!(block.macro_f1.to_bits(), back.macro_f1.to_bits());
    assert_eq!(block.mcc.to_bits(), back.mcc.to_bits());
    assert_eq!(block.per_class_precision, back.per_class_precision);
    assert_eq!(block.per_class_recall, back.per_class_recall);
    assert_eq!(block.per_class_f1, back.per_class_f1);
    assert_eq!(block.n_test_rows, back.n_rows);
}
```

Also worth adding a line to `synthetic_quality`'s comment naming the circularity and pointing at
this test as the thing that discharges it.

### WR-03: the published report says the confusion matrix was recomputed "rather than read off the rows" — it is read off the row

**File:** `crates/apr-cli/src/commands/setfit_bench.rs:2412-2415, 2584-2588` ·
`benchmarks/tweeteval-stance/report.md:4-9`

**Issue:** The header now renders:

```
verified: ... and the evidence below was RECOMPUTED from
          the committed lock bytes, the committed selection manifest, and
          the row's own confusion matrix
          rather than read off the rows.
```

Two of the three named sources are files opened at paths the row cannot choose. The third —
"the row's own confusion matrix" — is by definition read off the row; `verify_quality_closed_form`
opens no file at all (`bench_gate.rs:1770-1783`, and the contract says so in its own `domain`,
`setfit-benchmark-claims-v1.yaml:558`). "Recomputed from X rather than read off the rows" where
X *is* a row field is self-contradictory as written, and it reads as a stronger claim than the
gate makes. The `residual:` paragraph below does say the matrix is producer-written, but the
`verified:` line is the sentence a reader takes away, and this phase exists because an
attestation line said something that was not true of the run that printed it.

This is shipped: `benchmarks/tweeteval-stance/report.md` is the committed, published artifact.

**Fix:** split the clause so the "rather than read off the rows" qualifier attaches only to the
two file-derived sources:

```rust
pub(crate) const PROVENANCE_SOURCES: &str =
    "the committed lock bytes and the committed selection manifest, at paths the\n\
     \x20         row cannot choose rather than read off the rows; and every published\n\
     \x20         accuracy figure recomputed in closed form from the row's own\n\
     \x20         confusion_matrix, which the residual below scopes";
```

and drop the trailing `rather than read off the rows.\n` line from `render_header`'s format
string. Regenerate `report.md` in the same commit — leaving it stale is the defect 05-17's own
Decision ("The committed `report.md` was refreshed") already ruled on. Add a must-not-match row
to `setfit_bench_report_residual_concedes_exactly_what_the_gate_still_cannot_refuse` for the
literal `confusion matrix\n          rather than read off the rows`, so the shape cannot return.

### WR-04: the contract claims the three residual statements agree "WORD FOR WORD"; they do not, and nothing checks it

**File:** `contracts/setfit-benchmark-claims-v1.yaml` (`selection_safety_evidence.residual_risk.amended_4_0_0`)

**Issue:** The amendment says the remaining residuals are "what `apr setfit bench report`'s
`residual:` line and `bench_gate`'s module header now say WORD FOR WORD", then lists **four**
items. Measured against the other two artifacts:

- The report's `residual:` (`report.md:10-17`) names three, and omits the closure condition
  ("closing that needs a committed per-row prediction artifact no run writes today") that both
  the contract and `bench_gate.rs:85-86` carry.
- The contract's item (4) (the deferred LoRA arm) appears in neither the report nor the gate
  header.
- The three wordings are paraphrases of each other, not identical text.

The 05-17 SUMMARY concedes the gap ("nothing yet catches the gate header or the contract
drifting from it"), but the contract nonetheless asserts the agreement as a fact. A provable
contract asserting an unverified cross-artifact identity is the same defect class this phase
was opened for, and `pv validate` cannot see it.

**Fix:** two options, in order of preference.
1. Make it true and gate it. Hoist the three residual sentences to one `pub const` in
   `bench_gate` (e.g. `RESIDUAL_STATEMENTS: [&str; 3]`), have `apr-cli`'s `RESIDUAL_DISCLOSURE`
   be built from it, and add a test in `bench_gate_tests.rs` asserting each element appears
   verbatim in `CLAIMS_CONTRACT_YAML` (already `include_str!`-ed at `bench_gate.rs:195`). That
   is the cross-artifact test the 05-17 SUMMARY identifies as a candidate, and the constant is
   already compiled in.
2. If the wordings must differ, soften the claim to what is true — "state the SAME THREE
   residuals; the wording differs per artifact and the agreement is not yet gated
   (D-ITEM-…)" — and scope item (4) explicitly to the deferred arm. Do not leave "WORD FOR WORD"
   standing unverified.

### WR-05: no test exercises a ragged confusion matrix; the branch that catches one never decides any case

**File:** `crates/aprender-train/src/train/setfit/bench_metrics.rs:449-456` ·
`crates/aprender-train/src/train/setfit/bench_metrics_tests.rs:853` ·
`crates/aprender-train/src/train/setfit/bench_gate_tests.rs:2235-2237` · `Makefile` (gate banner)

**Issue:** The shape check is a disjunction:

```rust
if confusion_matrix.len() != n_classes || row_widths.iter().any(|width| *width != n_classes) {
```

Every case fed to it is caught by the **first** disjunct, which short-circuits the second:

| case | shape | which disjunct |
|---|---|---|
| `bench_metrics_tests.rs:853` labelled `"ragged"` | `[[1,2,3],[4,5,6]]` — 2 uniform rows of width 3 | `len() 2 != 3` |
| `bench_metrics_tests.rs:855` `four_by_four…` | `[[1;4];4]` | `len() 4 != 3` |
| `bench_gate_tests.rs:2236` `NonSquareMatrix` | `[[1,2,3],[4,5,6]]` | `len() 2 != 3` |
| `bench_gate_tests.rs:2239` `DimensionMismatch` | 4×4 | `len() 4 != 3` |

Neither test file ever constructs a matrix with `len() == ordered_labels.len()` but a row of a
different width — i.e. an actually *ragged* matrix, e.g. `[[1,2,3],[4,5],[6,7,8]]`. So
`row_widths.iter().any(...)` is dead as far as the suite can tell: deleting it leaves every test
green. (I checked what a regression would do — `ConfusionMatrix::from_predictions_with_min_classes`
grows its class count from the observed max index rather than indexing blindly
(`eval/classification/confusion.rs:41-50`), so the consequence is a wrong-arity result caught
downstream by `cross_check_vector`, not a panic. That is luck, not design.)

Three artifacts nonetheless claim the coverage: the case label `"ragged"`, the Makefile gate
banner ("three degenerate matrices (ragged, 4x4 against three labels, all-zero)") and the
contract invariant ("A matrix that is ragged, whose dimension disagrees with
`ordered_labels.len()`, …").

**Fix:** rename the existing row to `wrong_row_count` and add the missing shape, which also
makes the `mutation` half of the phase's own re-mutation rule real:

```rust
("ragged_uniform_row_count", vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]], "ConfusionMatrixShape"),
("row_wider_than_the_label_map", vec![vec![1, 0, 0, 0], vec![0, 1, 0, 0], vec![0, 0, 1, 0]], "ConfusionMatrixShape"),
```

and a matching `QualityMutation::RaggedMatrix` row in `QUALITY_CROSS_CHECK_CASES` so the shape is
also proven at the ACTIVE 40-cell scope (bump `QUALITY_CROSS_CHECK_CASES.len()` pin and the
Make floor to the newly measured count).

### WR-06: one of `EVIDENCE_PATH_CASES`'s three "acceptance rows" is a verbatim duplicate of another

**File:** `crates/aprender-train/src/train/setfit/bench_gate_tests.rs:1307-1328, 1488-1491`

**Issue:** Row `committed_spelling` (`:1307`) and row `committed_spelling_other_kind_dir`
(`:1321`) carry **identical** `declared: Declared::CommittedForKind`, identical
`expect: Expect::Accepted` and identical `detail_contains: ""`. `declared_string` is a pure
function of `(declared, root, bench, kind)` (`:1398-1421`), so for a given `kind` the two rows
build the same string against the same fixture shape and make the same assertion twice.

The second row's `why` — "swept over BOTH kinds by the loop, so the LEDGER_DIR spelling is
proven to be accepted under `EvidenceKind::Ledger`" — describes what the **outer loop**
(`:1437`) already does for every row including the first. The consequence is that
`assert!(accepted >= 3, ...)` (`:1491`) is satisfied by a duplicate: there are two distinct
acceptance shapes, not three. The Makefile banner ("three ACCEPTANCE rows") and the 05-15
SUMMARY (D7: "including three acceptance rows") both repeat the inflated count. An acceptance
row's whole job is to stop a table passing by refusing everything, and one that is a copy adds
no such protection.

**Fix:** replace the duplicate with an acceptance shape that is actually distinct and that the
current table does not cover — a nested path under the kind directory, which is the shape a
restored two-method scope will plausibly produce:

```rust
PathCase {
    label: "committed_spelling_nested_subdirectory",
    declared: Declared::NestedUnderKindDir,   // e.g. "{dir}/2026-09/{file}"
    expect: Expect::Accepted,
    detail_contains: "",
    why: "a contained path with more than two components must still resolve; the `..` and \
          root refusals are per-COMPONENT, and an over-refusal on depth would be invisible \
          to a table whose accepted rows are all two components deep",
},
```

(`path_case_fixture` needs one extra `create_dir_all` + `fs::write`.) Alternatively drop the
row and lower the assertion to `>= 2`, and correct the Makefile banner and the SUMMARY — but
adding the shape is the better trade.

### WR-07: the only end-to-end proof that the shipped door refuses is a Make target nothing depends on, and it is not in `.PHONY`

**File:** `Makefile:2779-2790` (target `setfit-bench-door-probe`), `Makefile:60` (`.PHONY` list)

**Issue:** `setfit-bench-door-probe` is deliberately not wired into `setfit-bench-tests` or any
tier (the comment at `:2779-2785` argues it, and 05-15's D9 records it as a decision). The
argument — that it needs a release `apr` built with a non-default feature and a silently-skipping
leg is worse than no leg — is sound as far as it goes, but the outcome is that **no gate runs
it**: `grep` over `Makefile` and `.github/workflows/*.yml` finds no other reference. The four
trees verification measured `apr setfit bench report` accepting are proven refused exactly once,
by hand, and a regression in the *adapter* (as opposed to the library) — a changed exit path, a
`--bench-dir` default, a renderer that prints before verifying — would be caught by nothing.
That is the mirror of CLAUDE.md Verification Discipline rule 5: the guard scans the decision
surface, but nothing invokes the guard.

Separately, the target is missing from `.PHONY` (`Makefile:60`), where every sibling
(`setfit-bench-tests`, `setfit-repro-inproc`, …) is listed. A file or directory named
`setfit-bench-door-probe` in the repo root would make `make setfit-bench-door-probe` report
"up to date" and run nothing — a silent pass.

**Fix:**
1. Add `setfit-bench-door-probe` to the `.PHONY` list at `Makefile:60`.
2. Give it a real caller that cannot skip silently. Add a `tier3`- or `pre-push`-level target
   that **builds its own prerequisite** and then runs the probe, so there is no "binary is
   stale" degradation to protect against:

```make
setfit-bench-door-probe-ci: ## EVAL-01/02/04 end-to-end: build the feature-enabled apr, then probe
	cargo build --release --bin apr --features setfit
	@$(MAKE) setfit-bench-door-probe
```

and make that a prerequisite of the phase-5 leg of `tier3`. If the build cost is the objection,
say so in the comment as a cost decision rather than leaving the current text, which reads as
though the probe is covered.

### WR-08: `verify_cell`'s doc and inline comment both claim an ACTIVE-scope refusal it does not perform

**File:** `crates/aprender-train/src/train/setfit/bench_gate.rs:1424-1426, 1439-1448` ·
`crates/aprender-train/src/train/setfit/bench_gate_tests.rs:3467-3477`

**Issue:** The `# Errors` section says `ExpectationSetMismatch` is returned "if `cell` is not in
the ACTIVE expectation set", and the inline comment says "It only refuses a REQUEST for a cell
outside the active scope — including a second method's cell." The code performs neither check:

```rust
let Some(entry) = manifest.payload.cells.iter().find(|e| e.cell() == cell) else {
    return Err(BenchGateError::ExpectationSetMismatch { declared: 0, expected: EXPECTED_CELLS });
};
```

That is **manifest membership**, not active-scope membership. A manifest declaring a LoRA cell
makes `verify_cell(…, TARGET_LORA)` proceed all the way through steps 4, 6, 6b and 6c and
return `Ok(())` — and step 7, the LoRA no-selection attestation, is excluded from this door, so
the post-test-selection conjuncts would never run.

The property is currently *reachable-safe* for a different reason: `RunManifest::from_bytes`
(`bench_row.rs:918-924`) enforces `declared == Self::expectation()` over the ACTIVE set, so the
CLI cannot construct such a manifest from a file. But `verify_cell` is a `pub fn` taking a
`RunManifest` by reference, and this module states its own rule three lines earlier at
`:1059-1061` — "a manifest can also be built in memory … and this gate must not depend on which
door its argument came through" — which is exactly the dependency this check has.

The test named for the property cannot distinguish the two reasons either:
`bench_gate_the_single_cell_door_refuses_a_cell_outside_the_active_scope` (`:3468`) builds an
ACTIVE-scope manifest and asks for `TARGET_LORA`, so it goes red on manifest membership and
would stay green if the active-scope check were never added.

**Fix:** add the check the comment describes, one line, before the `find`:

```rust
if !RunManifest::expectation_for(ExpectationScope::Active).contains(&cell) {
    return Err(BenchGateError::ExpectationSetMismatch { declared: 0, expected: EXPECTED_CELLS });
}
```

and make the test distinguishing by building a manifest that *does* declare the LoRA cell
(`declare_for(ExpectationScope::DeferredTwoMethod)`, already `#[cfg(test)]`-available) and
asserting the door still refuses it — which is the assertion that currently cannot fail.

## Info

### IN-01: `symlink_for_test` makes three table rows fail with a misleading diagnosis on non-unix rather than skip

**File:** `crates/aprender-train/src/train/setfit/bench_gate_tests.rs:1423-1432`

**Issue:** The comment argues that gating the materializer rather than the table row avoids a
silently shrinking table. It does — but on a non-unix host the three symlink rows
(`last_component_symlink`, `first_component_symlink_dir`, `prefix_sibling_symlink`) still run
with no symlink created, so the declared path names an absent file and the test fails with
`must be refused as 'evidence_path_escape' … got: evidence_file_missing`. A reader on that host
reads it as a containment regression. Given the repo tracks `WINDOWS.md` items, this will
eventually be somebody's bad hour.

**Fix:** keep the rows in the table (the count assertion at `:1483` depends on them) but have
`symlink_for_test` return `bool` and, on a host where symlinks are unavailable, assert loudly at
the top of the test with a message naming the host limitation — never re-interpret the row's
expectation.

### IN-02: the disclosure test's "overclaim" scan uses near-miss literals that cannot plausibly fire

**File:** `crates/apr-cli/src/commands/setfit_bench_tests.rs` (`setfit_bench_report_residual_concedes_exactly_what_the_gate_still_cannot_refuse`)

**Issue:** The must-not-match set is `["proves the matrix", "verified against the model",
"cannot be forged"]`. None is a phrase the disclosure has ever carried or is likely to drift
into — unlike the two *retired* literals above them, which were copied verbatim out of the
program's own former format string and are genuine negative controls. The overclaim half is
therefore closer to decoration than to a guard.

**Fix:** either scan for the *class* rather than three guesses — e.g. assert the disclosure
still contains the hedge `cannot distinguish` and the words `producer-written`, which any
overclaiming rewrite would have to remove — or document that these three are aspirational and
carry no red-taking evidence, the way the retired-sentence rows do.

### IN-03: `row_widths` is materialized before the cheaper length check, and the class count is unbounded

**File:** `crates/aprender-train/src/train/setfit/bench_metrics.rs:449-456`

**Issue:** `row_widths` is built unconditionally, then the length test runs. For a doctored row
declaring ~2×10⁶ empty matrix rows (which fits inside the 16 MB `MAX_EVIDENCE_FILE_BYTES` cap),
that is a 16 MB `Vec<usize>` that is then embedded in `ConfusionMatrixShape` and rendered by
`{row_widths:?}` into a multi-MB console message. Relatedly, 05-17's `MAX_CROSS_CHECK_ROWS`
bounds the observation *total* but not `ordered_labels.len()`: a ~16 MB row can declare a
2800-label map, and `ConfusionMatrix::new(2800)` then allocates ~62 MB. Both are a small
multiple of an input that is already capped, so neither is a denial of service — but the plan's
own pattern is "Bound any expansion driven by producer-supplied counts", and the class-count
dimension was not bounded.

**Fix:** check `confusion_matrix.len() != n_classes` *first* and only then compute `row_widths`;
truncate `row_widths` in the error to a bounded prefix; and consider a
`MAX_CROSS_CHECK_CLASSES` beside `MAX_CROSS_CHECK_ROWS`, derived the same way.

### IN-04: the door probe's positive control only checks `rc=0`

**File:** `scripts/setfit_bench_gate_door_probe.sh:170-180`

**Issue:** The control asserts `control_rc -eq 0` and nothing about `CONTROL_LOG`'s contents. A
future `apr setfit bench report` that exited 0 without verifying anything — a changed default,
an early return — would satisfy the control, and the four attacks would then be the only signal.
The control is the thing that makes the attacks mean something, so it is worth the extra line.

**Fix:** add one assertion that the control actually produced a verified report, e.g.
`grep -qF 'every cell of the contracted matrix' "$CONTROL_LOG" || fail "the control produced no verified: header"`.

### IN-05: the probe's `grep` assertions are BRE where literal matching is meant

**File:** `scripts/setfit_bench_gate_door_probe.sh:216, 312, 316, 283`

**Issue:** `grep -q -- "$ESCAPE_PATH"`, `grep -q "quality.f_avg"` and friends treat their
arguments as basic regular expressions. `.` matches any character, so `quality.f_avg` would also
match `qualityXf_avg`, and `$ESCAPE_PATH` (an mktemp path containing `.` and `-`) is matched
more loosely than intended. No assertion is currently wrong, but every one of them is weaker
than its author's intent and a path containing a regex metacharacter could in principle change
a verdict.

**Fix:** use `grep -qF --` for all six literal assertions in this script. `-F` also removes the
need for the `--` guard against a leading `-`.

---

_Reviewed: 2026-09-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
