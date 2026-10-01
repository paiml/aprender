---
phase: 05-benchmark-and-claims-gate
verified: 2026-09-12T02:03:33Z
status: human_needed
score: 5/5 must-haves verified
closing_sha: c07edb4edf25c46850fc93157ebbfee5656860fb
branch: gsd/phase-2-contract-gate
covered_files:
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-01-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-01-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-01-calibration-measurements.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-02-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-02-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-03-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-03-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-03-epsilon-basis-decision.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-04-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-04-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-05-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-05-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-06-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-06-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-07-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-07-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-08-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-08-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-09-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-09-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-10-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-10-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-11-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-11-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-11-narrowing-inventory.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-12-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-12-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-12-compute-projection.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-13-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-13-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-14-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-14-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-14-controls.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-15-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-15-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-15-gate-input-surface.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-16-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-16-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-17-PLAN.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-17-SUMMARY.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-CONTEXT.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-DISCUSSION-LOG.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-PATTERNS.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-RESEARCH.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-REVIEW.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-REVIEWS.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-VALIDATION.md"
  - ".planning/phases/05-benchmark-and-claims-gate/05-VERIFICATION.md"
  - ".planning/phases/05-benchmark-and-claims-gate/deferred-items.md"
  - "Makefile"
  - "benchmarks/tweeteval-stance/report.json"
  - "benchmarks/tweeteval-stance/report.md"
  - "benchmarks/tweeteval-stance/run-manifest.json"
  - "contracts/setfit-benchmark-claims-v1.yaml"
  - "contracts/setfit-train-lifecycle-v1.yaml"
  - "crates/apr-cli/src/commands/finetune.rs"
  - "crates/apr-cli/src/commands/setfit_bench.rs"
  - "crates/apr-cli/src/commands/setfit_bench_tests.rs"
  - "crates/aprender-core/src/calibration.rs"
  - "crates/aprender-core/src/stats/hypothesis.rs"
  - "crates/aprender-train/src/train/setfit/bench_gate.rs"
  - "crates/aprender-train/src/train/setfit/bench_gate_tests.rs"
  - "crates/aprender-train/src/train/setfit/bench_metrics.rs"
  - "crates/aprender-train/src/train/setfit/bench_metrics_tests.rs"
  - "crates/aprender-train/src/train/setfit/bench_row.rs"
  - "scripts/run_bench_cells.sh"
  - "scripts/setfit_bench_gate_doctor.py"
  - "scripts/setfit_bench_gate_door_probe.sh"
covered_digest: "v1:sha256:4f383e798af77e0c1bdcea301657ba56ba5f679b0e049fbf4e06fb37afef9d7a"
behavior_unverified: 0
overrides_applied: 0
requirements_disposition:
  EVAL-01: met_in_full        # strengthened by 05-17; label-VOCABULARY residual enumerated, see WARN-2
  EVAL-02: met_in_full        # gap 2 CLOSED — binding recomputed at a gate-derived path
  EVAL-03: met_in_full
  EVAL-04: met_in_full        # gap 1 CLOSED — containment door; recomputation re-proven byte-identical
  EVAL-05: met_narrowly       # D-19 amendment; "comparable" reduces to across-shot-level
re_verification:
  previous_status: gaps_found
  previous_score: 3/5
  previous_verified: 2026-09-08T22:55:00Z
  previous_closing_sha: 4e80e48cca5676e4966db0295e6615a48a7a58ef
  gaps_closed:
    - "EVAL-04 gap 1 — row-supplied evidence path escaping bench_dir. CLOSED: `resolve_committed_evidence_path` (bench_gate.rs:857) is a two-stage door (total syntactic refusal of empty/absolute/rooted/prefix/`..` BEFORE any filesystem call, then canonical containment against the canonicalized bench_dir). ONE helper serves BOTH the SetFit `lock_record_path` arm and the deferred LoRA `candidate_ledger_path` arm. Verified by the VERIFIER'S OWN fixture, not by replaying the executor's probe: `..` traversal -> rc=5, `evidence_path_escape`, naming cell, kind and reason."
    - "EVAL-02 gap 2 — selection-manifest binding unenforced. CLOSED: `verify_selection_binding` (bench_gate.rs:1683) opens the manifest at `selection_manifest_path(bench_dir, cell)` — a function whose SIGNATURE takes `CellKey` + `Path` and no `BenchRow`, so the row provably cannot steer it — recomputes `semantic_hash` via `SelectionManifest::from_bytes`, and independently compares the manifest's own `payload.shots_per_class`/`root_seed` against the cell key. Wired into BOTH `verify_run_scoped` (step 6b, :1119) and `verify_cell` (:1465). Verified by the verifier's own TRANSPLANT attack with every digest repaired -> rc=5, `selection_manifest_cell_mismatch`."
    - "Advisory 2 — closed-form quality cross-check. CLOSED: `verify_quality_closed_form` (bench_gate.rs:1770) recomputes `f_avg`, `macro_f1`, `mcc`, the three per-class vectors and `n_test_rows` through `quality_from_confusion_matrix` (bench_metrics.rs:439), compared by IEEE-754 `to_bits()` with NO epsilon, plus all five `*_bits` siblings. Verified by three independent negatives (per_class_f1[1], n_test_rows via a doctored count, and a bits-type refusal) -> rc=5 each."
  gaps_remaining: []
  regressions: []
  regression_checks:
    - "EVAL-03 (40 rows, every field): report regenerates from all 40 rows and is byte-identical to the committed artifact — so no field was dropped."
    - "EVAL-05 (resource figures with boundaries): the regenerated RESOURCE block is byte-identical; the reload path (`setfit_bench.rs:1237`) is untouched by this round's diff."
    - "Exact recomputation (EVAL-04 first half): re-proven at HEAD, md5 `6fa674df21b1e7e96f7837ca6f7122ba` == committed `report.md`."
    - "Debt-marker gate: zero TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER across all nine phase-modified code artifacts (the sole `XXX` hit is `mktemp`'s `XXXXXX` template)."
advisory:
  - finding: "D-ITEM-05-17-A — the bench-row seal is BUILD-GRAPH DEPENDENT, and `bench_row.rs:39-44` states the opposite as fact"
    category: architectural
    reason: >-
      INDEPENDENTLY CONFIRMED, both halves, without trusting any SUMMARY. (a) Build graph:
      `cargo tree -p aprender --features inference -e features -i serde_json` shows
      `serde_json feature "preserve_order"` enabled via `pmcp v2.19.3`; the same query on
      `-p aprender-train --features setfit` returns ZERO matches. (b) Digests, recomputed by
      the verifier from the committed JSON alone in Python with no Rust build, on
      `rows/setfit-s16-seed13.json`: key-sorted canonicalization -> `fafda6485f47531aaf277a0
      95ea782cd86ba148e694956bb902e3e0d9c5ad04c`; the file's OWN key order -> `1c54f3b4e38540
      040a2a2224a424969bae473e7b28da6b33ee2fb4be9f1eae69`, which is exactly what the envelope
      claims. The committed seal is therefore a DECLARATION-ORDER digest, and the module header
      at `bench_row.rs:39-44` — "`Map` is `BTreeMap`-backed (no workspace crate enables
      `preserve_order`) ... the digest is therefore independent of Rust field-declaration
      order" — is FALSE in the binary that sealed the evidence. A `bench_row.rs` field reorder,
      or pmcp leaving apr's graph, silently flips all 40 rows between "verifies" and "refused"
      with no source edit to the gate. NOT graded a gap for one reason, established by
      measurement rather than argument: the AUDITABILITY the phase goal names SURVIVES — the
      verifier reproduced the committed digest from the committed bytes alone, outside the
      codebase entirely, by preserving the file's own key order. The property holds; the DOC
      that explains why is wrong, and the durability is one cargo resolution away.
    evidence_status: "reproduced independently (cargo tree + Python digest recomputation, both digests matched the 05-17 measurement)"
  - finding: "CR-01 (code review Critical) — `setfit_bench_gate_doctor.py`'s 'refuse to doctor the real tree' guard fails open from any non-root cwd, and its comment claims it cannot"
    category: security
    reason: >-
      REPRODUCED: from `crates/`, `os.path.realpath("benchmarks")` resolves to
      `<repo>/crates/benchmarks` (nonexistent), so
      `realpath("<repo>/benchmarks/tweeteval-stance").startswith(...)` is False and the guard
      at `:109-110` does not fire. The script would then `shutil.move` the committed lock file
      OUT of the real evidence tree and rewrite the committed row and run-manifest. The comment
      says "Refuse to doctor the real tree, whatever a caller passes" — which is false. This is
      the SAME cwd/containment defect class 05-15 just closed in Rust, repeated in Python in
      the very script written to prove that Rust fix. Graded WARNING and not a blocker on
      three checks, each verified rather than assumed: (i) it is not on the shipped gate path;
      (ii) it is not reachable through the probe — `setfit_bench_gate_door_probe.sh:84-85`
      `cd`s to `git rev-parse --show-toplevel` and passes a `mktemp -d` bench dir, and after
      two full probe runs `git status --porcelain -- benchmarks/ crates/ scripts/` showed the
      tracked trees clean; (iii) the damage is git-recoverable and does not remove a user's
      ability to audit. It remains a live footgun against the phase's own deliverable.
    evidence_status: "reproduced (guard predicate evaluated from a non-root cwd; probe path proven safe)"
  - finding: "WR-04 — the claims contract asserts a cross-artifact identity ('WORD FOR WORD') that does not hold, and `pv validate` cannot see it"
    category: other
    reason: >-
      CONFIRMED by reading both artifacts. `setfit-benchmark-claims-v1.yaml:628`
      (`selection_safety_evidence.residual_risk.amended_4_0_0`) says the remaining residuals are
      "what `apr setfit bench report`'s `residual:` line and `bench_gate`'s module header now
      say WORD FOR WORD", then enumerates FOUR items. The report's printed `residual:`
      (`report.md:10-17`) names THREE, omits item (1)'s closure condition ("closing that needs
      a committed per-row prediction artifact no run writes today") that both the contract and
      `bench_gate.rs:85-86` carry, and omits item (4) entirely; the three wordings are
      paraphrases, not identical text. A provable contract asserting an unverified identity is
      the defect class this phase was opened for. Verifier's own addition to the finding: the
      gate header additionally retains the GENERAL consistency-not-truth caveat ("the doctored
      negatives ... prove detection of INCONSISTENT evidence; they prove nothing about TRUTHFUL
      provenance", `bench_gate.rs:68-69`), while the report's `residual:` line scopes that
      caveat only to the confusion MATRIX — so the published artifact under-discloses that the
      lock bytes and the selection manifest are ALSO producer-written and can also be forged
      consistently. Under-disclosure, by this phase's own stated standard, is as much a defect
      as over-claiming. It does NOT make any printed line false and does not claim more than
      the gate enforces.
    evidence_status: "reproduced (contract text and report text read and compared directly)"
  - finding: "WR-02 — the equivalence the whole cross-check rests on is asserted numerically only against rows a PREVIOUS build wrote"
    category: other
    reason: >-
      CONFIRMED and measured by the verifier. `assemble_quality_block` (bench_metrics.rs:269-313)
      computes from `(predicted, truth)` vectors; `quality_from_confusion_matrix` (:439) rebuilds
      those vectors from the counts and calls the SAME four surfaces. They share no call, so the
      equivalence rests on a permutation-invariance argument, not on shared code. A programmatic
      scan of every `#[test]` in `bench_metrics_tests.rs` found exactly ONE calling both —
      `bench_metrics_the_recomputation_authors_no_metric_arithmetic` — and that one is a
      SOURCE-TEXT assertion (`code.contains(required)`), not a numeric equality. The gate table's
      acceptance row is circular by construction (`synthetic_quality` builds its fixture BY
      CALLING the function under test). The only numeric cross-anchor is the 40-committed-row
      measurement, i.e. historical evidence. Not a gap: the property is measured at HEAD
      (40/40 bit-identical, max_abs_dev 0e0) and the positive control passes, and exactness is
      structural rather than lucky because every input is an integer count. It is a
      regression-DETECTION gap: a future drift in `assemble_quality_block` would surface only
      as `apr setfit bench report` refusing rows a fresh run just emitted.
    evidence_status: "reproduced (call-graph read; every test body scanned programmatically)"
  - finding: "WR-03 — the printed `verified:` line's trailing qualifier is self-contradictory as written"
    category: other
    reason: >-
      The line reads "RECOMPUTED from the committed lock bytes, the committed selection
      manifest, and the row's own confusion matrix rather than read off the rows"
      (`setfit_bench.rs:2420-2421`, rendered at `report.md:4-9`). The third source IS a row
      field, so "rather than read off the rows" is literally false of it. Weighed against this
      verifier's central charge — does any attestation claim MORE than the gate enforces? — the
      answer is NO: all three recomputations demonstrably fire (five independent negatives
      below), and the `residual:` line two lines down states that the matrix is producer-written.
      The defect is phrasing, on the one sentence a reader takes away, in a phase that exists
      because an attestation line was false. Recorded as advisory and routed to the human
      prohibition checkpoint rather than graded a gap.
    evidence_status: "reproduced (report.md read; all three recomputations proven live)"
  - finding: "WR-08 — `verify_cell`'s doc and inline comment both claim an ACTIVE-SCOPE refusal the code does not perform"
    category: other
    reason: >-
      CONFIRMED at `bench_gate.rs:1424-1426` ("`ExpectationSetMismatch` if `cell` is not in the
      ACTIVE expectation set") and `:1439-1442` ("It only refuses a REQUEST for a cell outside
      the active scope — including a second method's cell"), against the code at `:1443-1448`,
      which performs `manifest.payload.cells.iter().find(|e| e.cell() == cell)` — MANIFEST
      membership, not active-scope membership. Reachable-safe today only because
      `RunManifest::from_bytes` enforces the active expectation set, which is precisely the
      dependency the module forbids itself three lines earlier at `:1059-1061` ("this gate must
      not depend on which door its argument came through"). `verify_cell` is `pub fn`.
      Not a goal blocker — no shipped path can construct the offending manifest — but it is a
      false claim in a doc on a public function.
    evidence_status: "reproduced (doc and code read side by side)"
  - finding: "WR-07 + the probe is NOT RUNNABLE AT HEAD — the phase's only end-to-end proof is manual, un-.PHONY, and fails closed after any commit"
    category: other
    reason: >-
      MEASURED by the verifier, twice: `bash scripts/setfit_bench_gate_door_probe.sh` returns
      rc=1 at HEAD with "FAIL: scripts/apr_bin.sh refused to resolve an apr binary built from
      HEAD" — because `target/release/apr` was built at `a30a6ac73` and HEAD is `c07edb4ed`,
      two DOCS-ONLY commits later. The refusal is CORRECT (CLAUDE.md Verification Discipline
      rule 3, fail closed) and the two runs agreed exactly and left the tracked trees clean, so
      05-15's idempotency backstop truth is half-discharged. But the consequence stands: the
      `setfit-bench-door-probe` target is not in `.PHONY` and is a prerequisite of nothing
      (Makefile:2779-2790 argues this deliberately, to avoid a silently-skipping leg), and it
      requires a release rebuild after ANY commit, including a docs commit. The end-to-end
      green is therefore not reproducible on demand. THIS VERIFICATION DID NOT RELY ON IT — the
      substance was proven with five fixtures the verifier built itself, which is stronger
      evidence than replaying the executor's probe.
    evidence_status: "reproduced (probe run twice at HEAD; rc=1 both, identical verdict, no state left behind)"
  - finding: "WR-01 — `ordered_labels` now steers the recomputation, and `05-15-gate-input-surface.md`'s 'consumed by' column is stale for it and for `confusion_matrix`"
    category: other
    reason: >-
      Scoped precisely rather than accepted as stated. `quality_from_confusion_matrix`
      (bench_metrics.rs:443-485) uses `ordered_labels` for its LENGTH ONLY; the F_avg class
      selection is the fixed index constant `OFFICIAL_F_AVG_CLASSES`, so no label STRING
      steers any arithmetic. The residual is therefore narrow and semantic: a row whose labels
      are permuted while its matrix is untouched still passes, because published and recomputed
      metrics share the index convention — the row's label ASSIGNMENT is unattested against the
      contract's class-label map in `tweet-eval-stance-benchmark-v1.yaml`. The enumeration
      artifact records exactly this as "open rather than closed", which is honest; its
      "consumed by: no gate consumer" cells for `ordered_labels` and `confusion_matrix` are now
      stale (both ARE read by `verify_quality_closed_form`), though the class-(iii) assignment
      for `ordered_labels` remains correct since nothing checks it. Documentation staleness, not
      an over-claim.
    evidence_status: "reproduced (recomputation body read; label usage traced to arity only)"
  - finding: "Production calibration regime is still PROVISIONAL (4 of 40 cells measured) — carried forward unchanged from the 2026-09-08 verification"
    category: other
    reason: >-
      `contracts/setfit-train-lifecycle-v1.yaml` states it in-band and prominently; five of six
      parameter classes have no legal epsilon under the 2.x rule, and the two
      prospective-validation cells were not run. Unchanged by this round. Advisory because the
      disclosure is exemplary and the empirical per-run invariant held on all 40 cells.
    evidence_status: "self-disclosed in contract; no phase-goal impact observed; not re-measured this round"
  - finding: "CR-01 of the 2026-09-08 report (Makefile `.SHELLFLAGS` pipefail is dead) — carried forward, still out of phase scope"
    category: other
    reason: >-
      Line 29's `-o pipefail` is overwritten by line 57's `-e -c`. Authored outside phase 5
      (#2550, 2026-08-21 and Phase 2, 2026-08-09); no phase-5 commit touches either line. Every
      phase-5 Make floor uses `> log 2>&1; rc=$$?` redirection, never a pipe, so it reads
      cargo's real status. #2550 should be re-opened on its own.
    evidence_status: "reproduced on 2026-09-08; scope is pre-existing, off the phase-goal path"
coincidental_reliance_items:
  - truth: "A user can exactly recompute headline means, dispersion and uncertainty from all stored rows (EVAL-04, recomputation half)"
    reason: incidental-ordering
    harden: >-
      The recomputation holds because the committed rows' on-disk JSON key order happens to
      match the map order of the binary that sealed them. Nothing in the code enforces that an
      auditing canonicalizer preserve file order — and `bench_row.rs:39-44` documents the
      OPPOSITE rule (key-sorted, order-independent). Promote the actual canonicalization into a
      declared, gated precondition: either pin `preserve_order` explicitly and document the seal
      as order-preserving, or re-seal all 40 rows + 40 selection manifests + the run manifest
      under a genuinely order-independent canonicalization and assert it in a test that fails
      when the feature set changes. This is D-ITEM-05-17-A. ADVISORY ONLY — it changes no score
      and no status.
human_verification:
  - test: >-
      Decide the disposition of D-ITEM-05-17-A (the build-graph-dependent row seal) before this
      evidence set is published or cited anywhere outside the repo. Concretely: run
      `cargo tree -p aprender --features inference -e features -i serde_json | grep preserve_order`
      (expect hits via pmcp v2.19.3) and the same for `-p aprender-train --features setfit`
      (expect none), then decide whether to (a) pin and document the order-preserving seal, or
      (b) schedule the re-seal plan.
    expected: >-
      An explicit decision recorded as a plan or a deferred-item disposition, and the false
      claim at `bench_row.rs:39-44` corrected in the same change — a doc that states the
      opposite of the shipped behaviour is the exact defect class this phase was opened for.
    why_human: >-
      Architectural: the fix re-seals 40 rows, 40 selection manifests and the run manifest, or
      else commits the project to a feature pin. Neither is a verifier's call.
  - test: >-
      Resolve the eleven judgment-tier prohibitions declared across `05-15/16/17-PLAN.md`. Two
      need an explicit ruling rather than a check: (i) 05-15's "the sentence ... must be
      false-free on every run that emits it", against WR-03 — the `verified:` line's trailing
      "rather than read off the rows" is literally false of the third source it lists; (ii)
      05-17's "the report's disclosure must not overstate what the gate enforces", against
      WR-04 — the contract asserts a WORD-FOR-WORD cross-artifact identity that does not hold,
      and the report's residual drops the general consistency-not-truth caveat the gate header
      keeps. Read `report.md:1-17`, `bench_gate.rs:64-98` and
      `setfit-benchmark-claims-v1.yaml:628` side by side.
    expected: >-
      Each prohibition explicitly accepted or turned into a fix. Neither item makes a printed
      line claim MORE than the gate enforces — which is why this is a ruling and not a gap —
      but both are claim-language defects in a phase whose entire subject is claim language.
    why_human: >-
      Judgment-tier prohibitions, interactive mode (`config.json workflow` +
      `mode: interactive`): they belong to the end-of-phase human checkpoint by rule, and
      neither is decidable by grep.
  - test: >-
      Fix or consciously accept CR-01 before anyone runs
      `scripts/setfit_bench_gate_doctor.py` by hand. Reproduce first:
      `cd crates && python3 -c 'import os; print(os.path.realpath("benchmarks"))'` — it prints
      `<repo>/crates/benchmarks`, so the `startswith` guard at `:109-110` cannot fire for the
      real tree. The fix is to anchor the base on the script's own location
      (`os.path.dirname(os.path.abspath(__file__))`), compare with `os.path.commonpath` rather
      than `str.startswith`, and correct the comment, which currently claims the guard holds
      "whatever a caller passes".
    expected: >-
      The guard refuses `<repo>/benchmarks/tweeteval-stance` from every cwd, proven by a
      must-match/must-not-match case table run from at least two working directories
      (CLAUDE.md Verification Discipline rules 4 and 7).
    why_human: >-
      Destructive-hazard triage on a fixture tool. The verifier proved the shipped probe path is
      safe and did not exercise the unsafe path; whether to fix now or accept the footgun is a
      maintainer call.
  - test: >-
      Phase 05 has NO `05-SECURITY.md` while `.planning/config.json` sets
      `security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`, and all
      three gap plans declare `asvs_level: 1, block_on: high`. Run `/gsd secure-phase 05`.
    expected: >-
      A threat-mitigation record covering, at minimum, the path-traversal class 05-15 closed
      (`resolve_committed_evidence_path`), the evidence-substitution class 05-16 closed
      (transplanted / deleted selection manifest), the TOCTOU residual the resolver discloses at
      `bench_gate.rs:844-849` (containment checked, then the file opened), and the
      `MAX_CROSS_CHECK_ROWS` expansion cap that bounds the confusion-matrix recomputation.
    why_human: >-
      A security gate that never ran cannot be discharged by reading code. This phase's subject
      matter IS path traversal and evidence tampering, so the omission is material.
  - test: >-
      Re-run the shipped end-to-end door proof against a binary built from HEAD:
      `cargo build --release --bin apr --features setfit && make setfit-bench-door-probe`.
      Budget for the release build; note #2410 (600s harness stall) if driving it from an agent.
    expected: >-
      rc=0, with the positive control passing FIRST and then all four refusals (E, G, F, D).
      Also add `setfit-bench-door-probe` to `.PHONY` while there (WR-07).
    why_human: >-
      The verifier could not discharge this: the probe fails closed at HEAD because
      `target/release/apr` was built at `a30a6ac73`, two docs-only commits back. The SUBSTANCE
      was proven independently with five verifier-built fixtures, but 05-15's declared
      `verification: backstop` truth ("running the probe twice produces the same verdict") is
      only half-discharged — same verdict and no state left behind, but the verdict was the
      binary-pin refusal rather than the PASS.
---

# Phase 5: Benchmark and Claims Gate — Verification Report (RE-VERIFICATION)

**Phase Goal (AMENDED 2026-09-07, `05-CONTEXT.md` D-19):** Users can audit and recompute a
complete, selection-safe TweetEval evidence set for the verified SetFit APR across every
contracted shot and seed, with each cell bound to a recorded selection manifest so a second
method can later be paired against it without re-running this half.

**Verified:** 2026-09-12 · **Closing SHA:** `c07edb4ed` · **Status:** human_needed · **Score:** 5/5
**Re-verification:** YES — after the 05-15/16/17 gap round. Previous: `gaps_found`, 3/5, at
`4e80e48cc`.

## Binary pin (CLAUDE.md Verification Discipline rule 3)

`target/release/apr` reports `a30a6ac73`; HEAD is `c07edb4ed`. I did not run it blind and I did
not hardcode a path on faith — I proved the gap is empty for every surface under test:

```
git log --oneline a30a6ac73..HEAD   # -> 2 commits: docs(05) code review, chore(pv) index rebuild
git diff --stat a30a6ac73..HEAD -- crates/ contracts/ scripts/ Makefile   # -> NO OUTPUT
git status --porcelain -- crates/ contracts/ scripts/ Makefile benchmarks/
                                    # -> only 2 untracked files under the unrelated
                                    #    aprender-mcp-chronos-lambda/.pmcp/
```

Every `apr` invocation below is `/Users/guy/Development/machine-learning/aprender/target/release/apr`,
code-identical to HEAD. `pv` was invoked as `./target/debug/pv` — a bare `pv` returns 127 and
must not be read as a pass.

**One consequence of the same pin, reported rather than hidden:** it is why
`scripts/setfit_bench_gate_door_probe.sh` returns rc=1 at HEAD (`apr_bin.sh` refuses a binary
not built from HEAD). That is the probe working as designed. It is also why this verification
did NOT rest on the probe — see the five fixtures below, which I built myself.

## What this verification did NOT trust

Per the adversarial brief, I re-derived rather than accepted: the two graded gaps and the
advisory, every attestation line the tooling prints, the `serde_json/preserve_order` finding
(both halves), the code review's Critical, and four of its eight Warnings. The five refusal
fixtures are MINE — none is a replay of the executor's `setfit_bench_gate_door_probe.sh` cases,
which is deliberate: replaying a probe proves the probe, not the door.

## Goal Achievement

### Observable Truths

| # | Truth (ROADMAP Success Criterion) | Status | Evidence |
|---|-----------------------------------|--------|----------|
| 1 | Ordered predictions evaluable with official `F_avg`, per-class metrics, three-class macro-F1, MCC, confusion matrix, and validation-only calibration diagnostics bound to explicit ordered labels (EVAL-01) | ✓ VERIFIED | Previously verified; **strengthened** by 05-17. `verify_quality_closed_form` (`bench_gate.rs:1770`) now recomputes `f_avg`, `macro_f1`, `mcc`, all three per-class vectors and `n_test_rows` from the row's OWN `confusion_matrix` through `quality_from_confusion_matrix` (`bench_metrics.rs:439`), compared by `to_bits()` with NO epsilon, plus all five `*_bits` siblings. Wired into BOTH doors (`:1131`, `:1467`). **My own negatives, through the shipped binary:** `per_class_f1[1]` -> 0.9123456789 with every digest repaired -> rc=5 naming `per_class_f1[1]`, claimed vs recomputed bits both printed; `confusion_matrix[0][0] += 7` -> rc=5 naming `n_test_rows` 280 vs 287; a `mcc_bits` type violation -> rc=5 at the schema. Residual, scoped by measurement not assertion: `ordered_labels` feeds only the ARITY (`OFFICIAL_F_AVG_CLASSES` is a fixed index constant), so label STRINGS remain unattested — enumerated as open in `05-15-gate-input-surface.md:71`. See WR-01. |
| 2 | All 40 shot/seed cells runnable for SetFit, each cell recording the selection-manifest hash that would **bind** a second method to an identical sampled-ID set (EVAL-02, amended) | ✓ VERIFIED | **Gap 2 CLOSED.** `verify_selection_binding` (`:1683`) resolves `selections/s{shots}-seed{seed}/selection-manifest.json` via `selection_manifest_path(bench_dir, cell)` — a function taking `CellKey` + `Path` and **no `BenchRow`**, so a reviewer settles row-independence from the SIGNATURE alone — recomputes `semantic_hash` through `SelectionManifest::from_bytes`, then independently compares the manifest's own `payload.shots_per_class`/`root_seed` against the cell key. Called in `verify_run_scoped` step 6b (`:1119`) and `verify_cell` (`:1465`). **My own strongest negative:** I transplanted `s8-seed13`'s manifest into the `s8-seed41` slot AND doctored the row's `selection_manifest_hash` to match it AND repaired the row envelope + manifest `row_sha256` + manifest envelope — i.e. I made every digest agree — and the gate still returned rc=5: "declares `shots_per_class` = 8 and `root_seed` = 13, which is not this cell". The step-2 check is reachable and fires. The 40 manifests the prior report graded ORPHANED are now load-bearing. |
| 3 | One machine-readable row per method/shot/seed run carrying dataset/model revisions, selection lock, artifact hash, encoder-update evidence, backend/hardware identity, quality metrics, and consistently bounded resource measurements (EVAL-03) | ✓ VERIFIED | Regression check. The report regenerates from all 40 rows byte-identically at HEAD, which is only possible if every field the renderer consumes is still present in all 40; this round's diff removes no schema field, and `deny_unknown_fields` + non-`Option` fields make a dropped field a parse refusal rather than a silent gap (proven incidentally by my `mcc_bits` case, which the schema rejected before the gate). |
| 4 | Headline means, dispersion and uncertainty exactly recomputable from all stored rows, while any missing, selectively omitted, unmatched or post-test-selected cell invalidates the report (EVAL-04) | ✓ VERIFIED | **Gap 1 CLOSED.** (a) *Recomputation, re-proven at HEAD:* two consecutive `apr setfit bench report` runs are byte-identical to each other AND md5-identical to the committed `report.md` (`6fa674df21b1e7e96f7837ca6f7122ba` both), with `git status -- benchmarks/` clean afterwards — so no state is left behind. This also discharges 05-16's `verification: backstop` idempotency truth with direct evidence. (b) *Containment:* `resolve_committed_evidence_path` (`:857`) refuses empty / absolute / rooted / prefixed / `..`-carrying declarations in a TOTAL, filesystem-independent stage BEFORE any syscall, then refuses any target whose canonical form falls outside the canonicalized `bench_dir`; one helper serves the SetFit `lock_record_path` AND the deferred LoRA `candidate_ledger_path` arm (`:1519`, `:1567`), so a restored second arm inherits the fix. **My own negative:** `lock_record_path` = `../../outside/anywhere.json` with the row envelope and the manifest `row_sha256` + envelope repaired -> rc=5, `evidence_path_escape`, naming the cell, the kind, the offending component and why `Path::join` never resolves it. `grep -n 'bench_dir.join'` shows the only remaining raw joins are `ROWS_DIR` (gate-derived) — no row-supplied string reaches the filesystem except through the door. (c) *Invalidate-on-omission:* every one of my five negatives printed "The report has no partial-data mode: a missing, substituted, unmatched or post-test-selected cell invalidates the whole run (EVAL-04)" — no cell was skipped, downgraded or averaged out. See D-ITEM-05-17-A in Advisory for the durability finding this truth's *reason* rests on. |
| 5 | Training time, cold/warm latency, throughput with batch/warmup boundaries, peak memory, artifact size, calibration and classification quality comparable, measured from the same reloaded production artifacts (EVAL-05) | ✓ VERIFIED (narrow) | Regression check. The regenerated RESOURCE block is byte-identical, including each figure's measurement boundary and its `LOWER BOUND (sampled; can only understate)` rider, and the report still refuses to add or average the two mechanism classes. `reload_verified_run_from_apr` (`setfit_bench.rs:1237`) is untouched by this round's diff. **NARROW under D-19:** with one method measured, "comparable" reduces to comparability across shot levels; the report's own SCOPE block says so ("Read the absence as absence"). |

**Score: 5/5 truths verified** (0 present-but-behaviour-unverified, 0 overrides applied).
Previous: 3/5 with one FAILED and one PARTIAL. **All three graded items are genuinely closed** —
each re-proven by a fixture this verifier constructed, not by replaying the executor's probe.

### Adversarial check on the attestations (the brief's central charge)

This phase exists because `apr` printed "provenance was recomputed from the committed lock bytes
rather than read off the rows" on a run where that sentence was false. So: does every line the
tooling now prints hold on the run that prints it, and does anything claim more than the gate
enforces?

| Printed line | Claims | True of the run? |
|--------------|--------|------------------|
| `verified:` — "every cell of the contracted matrix. A missing, substituted, unmatched or post-test-selected cell would have REFUSED this report rather than shrunk it" | completeness against the CONTRACT-derived set, and no partial mode | ✓ Every one of my five negatives refused the WHOLE report and printed the no-partial-data paragraph |
| `verified:` — "...RECOMPUTED from the committed lock bytes" | the lock digest is recomputed from a committed file at a contained path | ✓ Fires (my `..` traversal), and now at a path the row cannot escape with |
| `verified:` — "...the committed selection manifest" | the pairing key is recomputed from a gate-derived path | ✓ Fires (my digest-repaired transplant) |
| `verified:` — "...and the row's own confusion matrix" | every published accuracy figure is recomputed in closed form | ✓ Fires (three of my negatives) |
| `verified:` — trailing "rather than read off the rows" | attaches to all three sources | ⚠️ **Literally false of the third** — the matrix IS a row field. Claims no verification that did not happen, and the `residual:` line immediately scopes it. WR-03; routed to the human prohibition checkpoint |
| `residual:` — three residuals | the matrix is producer-written; the two calibration diagnostics are not recomputable; `evidence_table_hash` / `apr_artifact_sha256` are claims about artifacts not carried | ✓ All three true. ⚠️ But it DROPS the general consistency-not-truth caveat that `bench_gate.rs:68-69` keeps — the lock bytes and the selection manifest are also producer-written. Under-disclosure, which this phase's own standard calls a defect. WR-04 |
| `SCOPE` — "ONE METHOD WAS MEASURED ... Nothing here states or implies any result about a second method. Read the absence as absence." | D-19 compliance | ✓ No SetFit-versus-LoRA figure anywhere in the report |
| contract `amended_4_0_0` — the three residual statements agree "WORD FOR WORD" | a cross-artifact identity | ✗ **FALSE.** Four items listed; the report names three, omits item (1)'s closure clause and item (4); wordings are paraphrases. `pv validate` cannot see it. WR-04 |

**Bottom line: nothing claims more than the gate enforces.** Two claim-language defects remain —
one self-contradictory qualifier and one false cross-artifact identity assertion in a provable
contract. Neither over-states a verification; both are routed to the human prohibition
checkpoint, because in a phase whose entire subject is claim language they should not be
absorbed silently into a pass.

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Paired SetFit-versus-LoRA delta, paired-t CIs over per-seed differences, and the 80-cell two-method expectation set | `D-ITEM-05-15` (deferred item, not a later milestone phase) | REQUIREMENTS.md EVAL-02/EVAL-04 amendment 2026-09-07; contract 4.0.0 retains the two-method design verbatim under `deferred_two_method_scope` / `no_selection_attestation: status: deferred`. Verified this round that the deferral is **inheritance-safe**: `resolve_committed_evidence_path` is applied to the LoRA `candidate_ledger_path` arm (`:1567`) and `selection_manifest_path` keys on `(shots, seed)` only — so a restored arm inherits both new enforcements instead of re-opening the holes |
| 2 | A contract-derived comparison for `ordered_labels` against `tweet-eval-stance-benchmark-v1.yaml`'s class-label map | not scheduled | `05-15-gate-input-surface.md:71` records it as "possible in principle and ... not taken in this round; recorded as open rather than closed". Honest enumeration; WR-01 |
| 3 | Per-row prediction artifact (would close the producer-written-matrix residual) | not scheduled | Named in all three residual statements as the artifact "no run writes today" |

### Advisory (Reproduced; None Blocking the Amended Goal)

| # | Finding | Category | Why Advisory |
|---|---------|----------|--------------|
| 1 | **D-ITEM-05-17-A** — the row seal is build-graph dependent; `bench_row.rs:39-44` states the opposite as fact | architectural | The most consequential thing here. I confirmed BOTH halves independently (`cargo tree`, and both digests recomputed in Python from the JSON alone). Advisory and not a gap for one measured reason: **auditability survives** — I reproduced the committed digest from the committed bytes outside the codebase entirely, by preserving the file's own key order. The doc that explains why is wrong, and one cargo resolution flips all 40 rows |
| 2 | **CR-01** — the fixture doctor's real-tree guard fails open from any non-root cwd | security | Reproduced. Not on the gate path; not reachable through the probe (proven: probe `cd`s to repo root, uses `mktemp`, and two runs left the tracked trees clean); git-recoverable. A live footgun against the phase's own deliverable |
| 3 | **WR-04** — the contract asserts a "WORD FOR WORD" identity that does not hold; plus the report drops a caveat the gate header keeps | other | Confirmed by reading all three artifacts. Under-disclosure on a published artifact, by this phase's own stated standard |
| 4 | **WR-02** — the equivalence the cross-check rests on is asserted numerically only against rows a previous build wrote | other | Confirmed by scanning every `#[test]` programmatically. Property holds and is measured at HEAD; the gap is regression DETECTION |
| 5 | **WR-03** — the `verified:` line's trailing qualifier is self-contradictory | other | Phrasing, on the sentence a reader takes away. Claims no verification that did not happen |
| 6 | **WR-08** — `verify_cell`'s doc claims an active-scope refusal the code does not perform | other | Confirmed. Reachable-safe only via the dependency the module forbids itself three lines earlier |
| 7 | **WR-07 + probe not runnable at HEAD** | other | Measured twice. The end-to-end green is not reproducible on demand; this verification did not rely on it |
| 8 | **WR-01** — `ordered_labels` steers arity only; the input-surface "consumed by" column is stale for two fields | other | Scoped by tracing the code rather than accepting the finding as stated. Documentation staleness |
| 9 | Production calibration regime still PROVISIONAL (4/40 cells) | other | Carried forward unchanged; disclosed in-band in the contract |
| 10 | Makefile `.SHELLFLAGS` pipefail dead (#2550) | other | Carried forward; authored outside phase 5; every phase-5 Make floor uses redirection, not a pipe |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/aprender-train/src/train/setfit/bench_gate.rs` | the EVAL-04 claims gate, with gap 1 + gap 2 + advisory 2 closed | ✓ VERIFIED (was ⚠️ HOLED) | +896 lines. `resolve_committed_evidence_path` (:857), `verify_selection_binding` (:1683), `verify_quality_closed_form` (:1770), `selection_manifest_path` (:237), four new typed error variants (`EvidencePathEscape`, `SelectionManifestMismatch`, `SelectionManifestCellMismatch`, `QualityCrossCheckMismatch`). All three checks wired into BOTH doors |
| `crates/aprender-train/src/train/setfit/bench_gate_tests.rs` | negatives at the ACTIVE 40-cell scope for each new bound | ✓ VERIFIED | +2035 lines; **55** `#[test]`, exactly the `assert_tests_ran` floor (Makefile:2716) — so the floor is at the true count and cannot be met vacuously. Was 38 |
| `crates/aprender-train/src/train/setfit/bench_metrics.rs` | `quality_from_confusion_matrix`, routing to the shipped surfaces | ✓ VERIFIED | +201 lines. Shape-then-total-then-expand, `saturating_add` with a `MAX_CROSS_CHECK_ROWS` cap (no panic, no unbounded allocation from a doctored `u64::MAX`), zero new metric arithmetic — it routes to `MultiClassMetrics::from_predictions_with_min_classes` / `f1_average_for_classes` / `matthews_corrcoef`, the same four surfaces `assemble_quality_block` uses |
| `crates/aprender-train/src/train/setfit/bench_metrics_tests.rs` | the 40-row measurement the acceptance band was chosen from | ✓ VERIFIED | +339 lines; **19** `#[test]` = the floor (Makefile:2741). The measurement asserts `rows.len() == 40` for non-vacuity, and its doc comment carries the `preserve_order` finding with both digests and the `pmcp v2.19.3` attribution — the workaround is routed around **visibly**, not silently |
| `crates/apr-cli/src/commands/setfit_bench.rs` | attestation + residual updated to exactly what the gate now enforces | ✓ VERIFIED (⚠️ see WR-03/WR-04) | +72 lines. `PROVENANCE_SOURCES` extended to three sources; `RESIDUAL_DISCLOSURE` narrowed from the retired unqualified concession to three precise residuals, with the retirement reasoned in the doc. `report.md` regenerated in-phase (byte-identical to a fresh run) |
| `crates/apr-cli/src/commands/setfit_bench_tests.rs` | disclosure tests | ✓ VERIFIED | +114 lines; **67** `#[test]` = the floor (Makefile:2769) |
| `contracts/setfit-benchmark-claims-v1.yaml` | 4.0.0, with `selection_binding_rule` and `quality_closed_form_crosscheck` | ✓ VERIFIED (⚠️ see WR-04) | `version: 4.0.0`; `selection_binding_rule` at :511, `quality_closed_form_crosscheck` at :555. **`./target/debug/pv validate` run by me: 0 errors, 0 warnings.** Deferred two-method scope retained verbatim |
| `benchmarks/tweeteval-stance/selections/*/selection-manifest.json` | 40 manifests, now load-bearing | ✓ VERIFIED (was ⚠️ ORPHANED) | The gate opens each at a gate-derived path and refuses a deleted, doctored or transplanted one |
| `benchmarks/tweeteval-stance/report.md` | recomputable claims report | ✓ VERIFIED | Regenerated twice at HEAD; md5 `6fa674df21b1e7e96f7837ca6f7122ba` == committed |
| `.planning/phases/.../05-15-gate-input-surface.md` | complete enumeration of the gate's row-supplied input surface | ⚠️ SUBSTANTIVE, PARTLY STALE | 27.8K, field-by-field with a trust class and a reason per field, and it pre-announces the 05-17 class moves. Its "consumed by" cells for `ordered_labels` and `confusion_matrix` are now stale (both ARE read by the gate). WR-01 |
| `scripts/setfit_bench_gate_door_probe.sh` | end-to-end proof through the shipped door | ⚠️ PRESENT, NOT RUNNABLE AT HEAD | 324 lines, positive-control-first, per-case slim copies, `cd "$REPO_ROOT"`, `mktemp` scratch, `rc` captured after the redirect not through a pipe. rc=1 at HEAD on the binary pin. WR-07 |
| `scripts/setfit_bench_gate_doctor.py` | fixture doctoring, extracted from the probe | ⚠️ HOLED | 173 lines. Proves its digest-repair scheme against the committed digest before using it — good. Its real-tree guard fails open from a non-root cwd. CR-01. Landed outside 05-15's declared `files_modified`, recorded as a justified deviation at `05-15-SUMMARY.md:300-303` — disclosed, not drift |
| `.planning/phases/.../05-SECURITY.md` | threat-mitigation record | ✗ MISSING | `security_enforcement: true`, `asvs_level: 1`, `block_on: high`. Routed to human verification, not graded a code gap |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| row `selection_manifest_hash` | `selections/*/selection-manifest.json` | `selection_manifest_path(bench_dir, cell)` -> `SelectionManifest::from_bytes` | ✓ **WIRED** (was ✗ NOT WIRED) | Verified behaviourally by my digest-repaired transplant. The path function takes no `BenchRow` |
| manifest `payload.shots_per_class` / `root_seed` | `CellKey` | direct comparison, `u64::from` widening the key rather than narrowing the manifest | ✓ WIRED | The second, independent statement of which cell drew the selection — reachable and firing |
| row `lock.lock_record_path` | `locks/*.lock.json` | `resolve_committed_evidence_path` -> `read_evidence` -> sha256 | ✓ **WIRED** (was ⚠️ PARTIAL) | Two-stage containment; my `..` traversal refused pre-syscall |
| row `evidence.lora.candidate_ledger_path` | `LEDGER_DIR` | the SAME helper (`:1567`) | ✓ WIRED | One door, both arms — the deferred scope inherits the fix |
| row `quality.confusion_matrix` + `ordered_labels` | published `f_avg` / `macro_f1` / `mcc` / per-class vectors / `n_test_rows` | `quality_from_confusion_matrix`, bit equality | ✓ WIRED | Three of my negatives fire; `ordered_labels` contributes arity only (WR-01) |
| `verify_quality_closed_form` | `assemble_quality_block` (the emitter) | permutation-invariance over integer counts, **not** a shared call | ⚠️ PARTIAL | Asserted numerically only against 40 rows a previous build wrote. WR-02 |
| `bench_gate` / `bench_metrics` negatives | CI + Make floors | `assert_tests_ran` 27 / 55 / 19 / 67 | ✓ WIRED | Each floor equals the true `#[test]` count at HEAD (verified by counting), so no new negative can be silently deselected |
| `setfit_bench_gate_door_probe.sh` | `apr setfit bench report` | `make setfit-bench-door-probe` | ⚠️ ORPHANED TARGET | Prerequisite of nothing, absent from `.PHONY`, and fails closed at HEAD. WR-07 |
| `bench_row.rs` canonical bytes | the 40 committed seals | `serde_json::Value` map ordering | ⚠️ **HOLLOW DOC** | The link works, by declaration order — which is the opposite of what the module header documents. D-ITEM-05-17-A |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `report.md` QUALITY table | `f_avg` mean / std / 95% CI | 40 row files -> `aggregate` -> frozen `t = 2.262157162798205`, df 9 | ✓ Non-degenerate and monotone in shots (0.4746 -> 0.5115 -> 0.5346 -> 0.5607); regenerates byte-identically | ✓ FLOWING |
| `report.md` QUALITY table | `macro_f1`, `mcc` means | same 40 rows, now cross-checked against each row's own confusion matrix | ✓ Published values are refused unless they follow from the counts | ✓ FLOWING |
| `report.md` RESOURCE tables | every figure + its mechanism class | 40 rows, per-host, never averaged across mechanism classes | ✓ Each figure carries its boundary and its `LOWER BOUND` rider where sampled | ✓ FLOWING |
| gate verdict | selection binding | 40 `selections/*/selection-manifest.json` opened at gate-derived paths | ✓ Deleting, doctoring or transplanting any one changes the verdict | ✓ FLOWING (was ✗ DISCONNECTED) |
| gate verdict | lock provenance | 40 `locks/*.lock.json` at contained paths | ✓ Escape, deletion and digest tamper all refuse | ✓ FLOWING |
| row seal | `semantic_hash` | `sha256(payload.to_canonical_bytes())` | ✓ Reproducible from the committed JSON alone (verifier did it in Python) — but only by preserving the FILE's key order, not by the key-sorted rule the doc states | ⚠️ FLOWING, DOC INVERTED |

### Behavioural Spot-Checks (all through the pinned shipped binary; fixtures built by the verifier)

| # | Behaviour | Command | Result | Status |
|---|-----------|---------|--------|--------|
| 0 | Positive control — undoctored committed tree verifies | `apr setfit bench report --bench-dir benchmarks/tweeteval-stance` | rc=0, full report, all three attestation sources named | ✓ PASS |
| 0b | Recomputation is exact and idempotent, no state left behind | two consecutive runs, `cmp`, `md5`, `git status -- benchmarks/` | byte-identical to each other and md5-identical to committed `report.md`; tree clean | ✓ PASS (discharges 05-16's `backstop` truth) |
| A | `..` traversal in `lock_record_path`, every digest repaired | isolated copy; `lock_record_path = ../../outside/anywhere.json` | rc=5, `evidence_path_escape`, names cell + kind + the `..` component + why `Path::join` never resolves it | ✓ PASS (gap 1) |
| B | Selection manifest transplanted from another cell **with the row hash doctored to match** and all digests repaired | isolated copy; `s8-seed13` manifest into the `s8-seed41` slot | rc=5, `selection_manifest_cell_mismatch`: "declares `shots_per_class` = 8 and `root_seed` = 13, which is not this cell" | ✓ PASS (gap 2, step-2 check reachable) |
| C1 | `per_class_f1[1]` doctored, digests repaired | isolated copy | rc=5, names `per_class_f1[1]`, prints claimed and recomputed bits | ✓ PASS (advisory 2) |
| C2 | `mcc_bits` type violation | isolated copy | rc=5 at the schema before the gate — `deny_unknown_fields` / typed schema is the outer door | ✓ PASS |
| C3 | `confusion_matrix[0][0] += 7`, digests repaired | isolated copy | rc=5, `n_test_rows` 280 vs 287 — reported FIRST, the fixed field order putting the denominator before its quotients | ✓ PASS (advisory 2) |
| D | Digest canonicalization, from the JSON alone, no Rust | Python: key-sorted vs file-order sha256 of `payload` | key-sorted `fafda648…`; file-order `1c54f3b4…` == the envelope's claim | ✓ PASS (auditability) / ✗ doc claim falsified |
| E | `preserve_order` in the two build graphs | `cargo tree -e features -i serde_json` for `-p aprender --features inference` and `-p aprender-train --features setfit` | present via `pmcp v2.19.3`; absent | ✗ doc claim falsified (D-ITEM-05-17-A) |
| F | CR-01 guard predicate from a non-root cwd | Python, `cwd=crates/` | `realpath("benchmarks")` -> `<repo>/crates/benchmarks`; guard returns False | ✗ FAIL (CR-01 confirmed) |
| G | Contract validity | `./target/debug/pv validate contracts/setfit-benchmark-claims-v1.yaml` | 0 errors, 0 warnings, rc=0 | ✓ PASS |
| H | Test existence (enumeration, not a suite run) | `#[test]` counts vs the Make floors | 27 / 55 / 19 / 67 — each floor equals the true count | ✓ PASS |
| I | Debt-marker gate | `TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER` across nine phase-modified artifacts | zero (only `mktemp`'s `XXXXXX` template) | ✓ PASS |
| — | Door probe at HEAD | `bash scripts/setfit_bench_gate_door_probe.sh` ×2 | rc=1 both, identical verdict, no state left behind — `apr_bin.sh` fail-closed on the binary pin | ? SKIP -> human (WR-07) |

Not re-run, per the orchestrator's #2410 stall warning: the release build, `make setfit-bench-tests`,
and the full workspace nextest suite. Their measurements are carried from the executor's run and
are corroborated here by the floor-equals-count check (H).

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EVAL-01 | 05-04, 05-08, 05-13, **05-17** | Ordered single-label predictions with the official TweetEval stance metrics | ✓ SATISFIED | Truth 1. 05-17 moved eight published figures from "trusted" to "recomputed in closed form"; three verifier negatives confirm. Residual: label VOCABULARY unattested (WR-01, enumerated as open) |
| EVAL-02 | 05-01, 05-02, 05-03, 05-06, 05-07, 05-12, 05-13, 05-14, **05-16** | Every 8/16/32/64-shot × ten-seed combination, selection-manifest-bound | ✓ SATISFIED | Truth 2. Was PARTIAL. The binding is now recomputed at a gate-derived path with an independent cell-key cross-check; verifier's digest-repaired transplant refused |
| EVAL-03 | 05-05, 05-09, 05-12, 05-13 | One machine-readable row per run with all named fields | ✓ SATISFIED | Truth 3, regression-checked via byte-identical regeneration |
| EVAL-04 | 05-04, 05-05, 05-10, 05-11, 05-13, **05-15** | Exact recomputation + invalidate on omission / substitution / post-test selection | ✓ SATISFIED | Truth 4. Was FAILED. Containment door closed the provenance hole; recomputation re-proven byte-identical AND reproduced from the JSON alone by the verifier. Durability caveat: D-ITEM-05-17-A |
| EVAL-05 | 05-06, 05-09, 05-11, 05-12, 05-13 | Comparable cost/quality figures from the same reloaded artifacts | ✓ SATISFIED (narrow) | Truth 5. "Comparable" reduces to across-shot-level under D-19; the report says so in its own SCOPE block |

**Orphan check: none.** `grep "| Phase 5 |" .planning/REQUIREMENTS.md` returns exactly EVAL-01..05,
and every one is claimed by at least one plan's `requirements:` frontmatter. No Phase 5 requirement
is unclaimed, and no plan claims a requirement outside the EVAL set.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/aprender-train/src/train/setfit/bench_row.rs` | 39-44 | Doc comment asserts the opposite of shipped behaviour, and reasons from it | ⚠️ Warning | The false premise ("no workspace crate enables `preserve_order`") is used to justify an auditability claim. Falsified two ways by the verifier. D-ITEM-05-17-A |
| `scripts/setfit_bench_gate_doctor.py` | 109-110 | cwd-relative containment base + `str.startswith`, with a comment claiming it holds "whatever a caller passes" | ⚠️ Warning (code review: Critical) | Fails open from any non-root cwd; would move the committed lock out of the real tree. Same defect class 05-15 closed in Rust. CR-01 |
| `contracts/setfit-benchmark-claims-v1.yaml` | 628 | A provable contract asserts an unverified cross-artifact identity ("WORD FOR WORD") | ⚠️ Warning | False as written; `pv validate` cannot see it. WR-04 |
| `crates/aprender-train/src/train/setfit/bench_gate.rs` | 1424-1426, 1439-1442 | Doc + inline comment describe a check the code does not perform | ⚠️ Warning | `pub fn`; reachable-safe only via the dependency the module forbids itself at :1059-1061. WR-08 |
| `crates/apr-cli/src/commands/setfit_bench.rs` | 2420-2421 | Printed attestation qualifier that is literally false of one of its three listed sources | ⚠️ Warning | Claims no verification that did not happen; resolved two lines later by `residual:`. WR-03 |
| `crates/aprender-train/src/train/setfit/bench_gate_tests.rs` | ~319-346 | Control fixture built by calling the function under test | ⚠️ Warning | The table's acceptance row is tautological w.r.t. the recomputation. WR-02 |
| `crates/aprender-train/src/train/setfit/bench_gate.rs` | ~1836-1844 | An error message attaches an "exact IEEE-754 bit equality" rider to an integer (`n_test_rows`) comparison | ℹ️ Info | Observed live in my case C3. Cosmetic over-specification in a refusal message |
| all nine phase-modified artifacts | — | `TBD` / `FIXME` / `XXX` / `TODO` / `HACK` / `PLACEHOLDER` | ✓ none | Debt-marker gate PASSES; the only `XXX` match is `mktemp`'s template |

No stub, no hollow prop, no empty implementation, no `return null` / `=> {}` pattern. Every
anti-pattern here is a **claim-language or documentation** defect, which is the class that matters
most in this particular phase and is why none of them is absorbed silently into the pass.

### Gaps Summary

**No gaps.** All three items the 2026-09-08 report graded are closed, and each closure was
re-proven by a fixture this verifier constructed rather than by replaying the executor's probe:

1. **Gap 1 (EVAL-04, was FAILED)** — closed at the CLASS level, not the probe level. One helper,
   two stages, applied to both the active and the deferred arm, refusing the whole shape family
   (empty, absolute, rooted, prefixed, `..`, and canonical-escape) before any syscall where it can.
2. **Gap 2 (EVAL-02, was PARTIAL)** — closed with the soundness argument visible in the type
   signature, plus a second independent check (the manifest's own cell key) that survives an
   attacker who has already made every digest agree. I confirmed that second check is reachable.
3. **Advisory 2 (EVAL-01)** — closed with no epsilon, because the band was MEASURED before it was
   chosen (40/40 bit-identical, max_abs_dev 0e0) and exactness is structural over integer counts.

**Status is `human_needed`, not `passed`,** and the reason is not hedging. Five items genuinely
need a human:

- **D-ITEM-05-17-A** is an architectural decision (re-seal 40 rows + 40 manifests + the run
  manifest, or pin the feature) that also requires correcting a false claim in shipped code. I
  measured it from both directions and it does not block today's goal — the auditability property
  survives, and I proved that by reproducing the committed digest from the committed bytes in
  Python — but the property holds for a reason the code documents backwards, which is exactly the
  reliance this phase should not be shipping on.
- **Eleven judgment-tier prohibitions** across the three gap plans are unresolved by rule
  (interactive mode routes them to the end-of-phase checkpoint), and two of them (05-15's
  "false-free" sentence, 05-17's "must not overstate") collide with WR-03 and WR-04 in ways a
  human should rule on rather than a verifier absorb.
- **CR-01** is a destructive footgun in a fixture tool, with a false comment, in the same defect
  class the phase just fixed.
- **No `05-SECURITY.md`** while `security_enforcement: true` and `block_on: high`, on a phase whose
  subject matter is path traversal and evidence tampering.
- **05-15's `verification: backstop` truth** is only half-discharged, because the probe fails
  closed at HEAD on the binary pin.

The engineering in this round is unusually strong — one door per defect class rather than one
patch per probe, a measured acceptance band, both new checks wired into both doors, refusal
messages that name the cell and teach the remedy, and a finding (D-ITEM-05-17-A) that the round
surfaced against its own interest and documented with the measurement attached. The remaining
defects are almost entirely in what the artifacts *say about themselves*, and in a phase whose
entire purpose is that claims must be recomputable rather than asserted, those are the right ones
to leave on the table for a human.

---

_Verified: 2026-09-12T02:03:33Z_
_Verifier: Claude (gsd-verifier) — re-verification after the 05-15/16/17 gap round_
