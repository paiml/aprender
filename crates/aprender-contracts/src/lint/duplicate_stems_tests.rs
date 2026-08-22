//! PV-DUP-001 tests, and the rename-invariance CASE TABLE.
//!
//! The load-bearing test here is `verdict_is_invariant_under_directory_rename`.
//! The defect it guards was falsified by renaming a directory and editing no file:
//! `mv contracts/aprender contracts/e-aprender` took `pv lint` from PASS to FAIL
//! (composition `edges_broken` 0 -> 5). A content-only mutation would have passed
//! and proved nothing, so the mutation here is a DIRECTORY NAME.

use super::*;
use crate::lint::{run_lint, GateDetail, GateExtra, LintConfig};

/// Directory names for the case table. They are chosen to straddle the two
/// orderings the old code could produce: names that sort BEFORE the top-level
/// duplicate and names that sort AFTER it, plus the literal `zzz-` form named in
/// the defect report. Under `read_dir` the order is a filename HASH, so it is not
/// predictable from the name at all — which is the point.
const DIR_CASES: &[&str] = &[
    "aprender",     // original
    "zzz-aprender", // the report's mutation: sorts after
    "e-aprender",   // sorts before `x-architecture-v1.yaml`; flipped the real corpus
    "A-aprender",   // uppercase: sorts before every lowercase name
    "000",          // digits: sorts before letters
    "zzz",          // sorts last
];

/// Upstream contract, variant WITH `guarantees` on `produce`.
fn upstream_with_guarantees() -> &'static str {
    "metadata:
  version: 1.0.0
  description: upstream
  references: [ref]
equations:
  produce:
    formula: y = f(x)
    guarantees:
      shapes:
        output: { dims: [batch, hidden] }
proof_obligations:
  - type: invariant
    property: p
falsification_tests:
  - id: F-001
    rule: r
    prediction: p
    if_fails: f
kani_harnesses:
  - id: K-001
    obligation: p
"
}

/// Same stem, variant WITHOUT `guarantees` — the divergence that decides the gate.
fn upstream_without_guarantees() -> &'static str {
    "metadata:
  version: 1.0.0
  description: upstream
  references: [ref]
equations:
  produce:
    formula: y = f(x)
proof_obligations:
  - type: invariant
    property: p
falsification_tests:
  - id: F-001
    rule: r
    prediction: p
    if_fails: f
kani_harnesses:
  - id: K-001
    obligation: p
"
}

fn downstream_assuming(from_contract: &str) -> String {
    format!(
        "metadata:
  version: 1.0.0
  description: downstream
  references: [ref]
  depends_on: [{from_contract}]
equations:
  consume:
    formula: z = g(y)
    assumes:
      from_contract: {from_contract}
      from_equation: produce
      shapes:
        output: {{ dims: [batch, hidden] }}
proof_obligations:
  - type: invariant
    property: p
falsification_tests:
  - id: F-002
    rule: r
    prediction: p
    if_fails: f
kani_harnesses:
  - id: K-002
    obligation: p
"
    )
}

/// Build `<root>/contracts` holding a divergent duplicate stem `x-architecture-v1`
/// (one copy at top level, one inside `subdir`) plus a downstream that assumes it.
fn build_tree(root: &std::path::Path, subdir: &str) -> std::path::PathBuf {
    let contracts = root.join("contracts");
    std::fs::create_dir_all(contracts.join(subdir)).unwrap();
    // Top-level copy: NO guarantees. Subdir copy: HAS guarantees.
    std::fs::write(
        contracts.join("x-architecture-v1.yaml"),
        upstream_without_guarantees(),
    )
    .unwrap();
    std::fs::write(
        contracts.join(subdir).join("x-architecture-v1.yaml"),
        upstream_with_guarantees(),
    )
    .unwrap();
    std::fs::write(
        contracts.join("downstream-v1.yaml"),
        downstream_assuming("x-architecture-v1"),
    )
    .unwrap();
    contracts
}

fn composition_of(report: &crate::lint::LintReport) -> (usize, usize, usize) {
    for g in &report.gates {
        if let GateDetail::Composition {
            edges_checked,
            edges_satisfied,
            edges_broken,
        } = &g.detail
        {
            return (*edges_checked, *edges_satisfied, *edges_broken);
        }
    }
    panic!("composition gate missing from report");
}

/// CASE TABLE: `mv contracts/X contracts/zzz-X` must not change the verdict.
///
/// Every row builds a byte-identical corpus differing ONLY in the name of the
/// directory holding one copy of a duplicate stem. The composition verdict and the
/// composition edge counts must be identical across every row.
#[test]
fn verdict_is_invariant_under_directory_rename() {
    let mut observed: Vec<(&str, bool, (usize, usize, usize))> = Vec::new();

    for subdir in DIR_CASES {
        let tmp = tempfile::tempdir().unwrap();
        let contracts = build_tree(tmp.path(), subdir);
        let mut config = LintConfig::new(&contracts, None, 0.0);
        config.no_cache = true;
        let report = run_lint(&config);
        let comp = report
            .gates
            .iter()
            .find(|g| g.name == "composition")
            .expect("composition gate");
        observed.push((subdir, comp.passed, composition_of(&report)));
    }

    let (first_name, first_passed, first_counts) = observed[0];
    for (name, passed, counts) in &observed {
        assert_eq!(
            (*passed, *counts),
            (first_passed, first_counts),
            "composition verdict moved when the directory was renamed \
             `{first_name}` -> `{name}`: {first_passed}/{first_counts:?} vs {passed}/{counts:?}. \
             No file content differs between these two trees."
        );
    }

    // And the verdict must be the DEFINED one: the ambiguous stem is refused, so
    // the edge is neither satisfied nor broken.
    assert!(first_passed, "composition must pass with a refused stem");
    assert_eq!(
        first_counts,
        (1, 0, 0),
        "1 edge checked, refused, not broken"
    );
}

/// The refusal must be visible, not silent: the ambiguous stem is reported.
#[test]
fn ambiguous_stem_is_reported_not_silently_resolved() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = build_tree(tmp.path(), "aprender");
    let mut config = LintConfig::new(&contracts, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);

    let gate = report
        .gates
        .iter()
        .find(|g| g.name == "duplicate-stems")
        .expect("duplicate-stems gate");
    // No baseline file in the temp tree => an unbaselined divergent stem => FAIL.
    assert!(!gate.passed, "an unbaselined divergent stem must fail");
    assert!(report
        .findings
        .iter()
        .any(|f| f.rule_id == "PV-DUP-001" && f.message.contains("x-architecture-v1")));
    assert!(report
        .findings
        .iter()
        .any(|f| f.rule_id == "COMPOSITION-001" && f.message.contains("AMBIGUOUS")));
}

/// Byte-identical copies are duplication, not ambiguity — the choice is invisible,
/// so they must NOT be refused and must NOT be reported.
#[test]
fn identical_copies_are_not_ambiguous() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = tmp.path().join("contracts");
    std::fs::create_dir_all(contracts.join("sub")).unwrap();
    std::fs::write(
        contracts.join("x-architecture-v1.yaml"),
        upstream_with_guarantees(),
    )
    .unwrap();
    std::fs::write(
        contracts.join("sub").join("x-architecture-v1.yaml"),
        upstream_with_guarantees(),
    )
    .unwrap();
    std::fs::write(
        contracts.join("downstream-v1.yaml"),
        downstream_assuming("x-architecture-v1"),
    )
    .unwrap();

    assert!(scan_duplicate_stems(&contracts).is_empty());

    let mut config = LintConfig::new(&contracts, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);
    let gate = report
        .gates
        .iter()
        .find(|g| g.name == "duplicate-stems")
        .expect("gate");
    assert!(gate.passed);
    // The edge resolves normally.
    assert_eq!(composition_of(&report), (1, 1, 0));
}

#[test]
fn scan_reports_paths_and_variant_count() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = build_tree(tmp.path(), "aprender");
    let dups = scan_duplicate_stems(&contracts);
    assert_eq!(dups.len(), 1);
    assert_eq!(dups[0].stem, "x-architecture-v1");
    assert_eq!(dups[0].variants, 2);
    assert_eq!(dups[0].paths.len(), 2);
}

#[test]
fn baseline_absorbs_known_divergence() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = build_tree(tmp.path(), "aprender");
    std::fs::create_dir_all(tmp.path().join("scripts")).unwrap();
    std::fs::write(
        tmp.path().join(BASELINE_REL_PATH),
        "# comment\n\nx-architecture-v1\n",
    )
    .unwrap();

    let mut config = LintConfig::new(&contracts, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);
    let gate = report
        .gates
        .iter()
        .find(|g| g.name == "duplicate-stems")
        .expect("gate");
    assert!(gate.passed, "baselined stem should not fail the gate");
}

/// The ratchet only turns one way: a baseline entry that no longer diverges fails.
#[test]
fn stale_baseline_entry_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = tmp.path().join("contracts");
    std::fs::create_dir_all(&contracts).unwrap();
    std::fs::write(
        contracts.join("x-architecture-v1.yaml"),
        upstream_with_guarantees(),
    )
    .unwrap();
    std::fs::create_dir_all(tmp.path().join("scripts")).unwrap();
    std::fs::write(tmp.path().join(BASELINE_REL_PATH), "x-architecture-v1\n").unwrap();

    let mut config = LintConfig::new(&contracts, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);
    let gate = report
        .gates
        .iter()
        .find(|g| g.name == "duplicate-stems")
        .expect("gate");
    assert!(!gate.passed);
    assert!(report.findings.iter().any(|f| f.rule_id == "PV-DUP-002"));
}

#[test]
fn missing_baseline_is_an_empty_baseline() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(read_baseline(tmp.path()).is_empty());
}

/// The real corpus: every divergent stem must be accounted for by the baseline,
/// and the baseline must contain nothing that has since been deduplicated.
#[test]
fn real_corpus_matches_its_baseline() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dups = scan_duplicate_stems(&root.join("contracts"));
    let found = ambiguous_stems(&dups);
    let baseline = read_baseline(&root);
    assert!(
        !baseline.is_empty(),
        "the ratchet baseline is REQUIRED; a missing {BASELINE_REL_PATH} silently \
         disarms PV-DUP-001"
    );
    let unbaselined: Vec<_> = found.difference(&baseline).collect();
    let stale: Vec<_> = baseline.difference(&found).collect();
    assert!(
        unbaselined.is_empty() && stale.is_empty(),
        "duplicate-stem baseline drift — new: {unbaselined:?}, stale: {stale:?}. \
         Deduplicate the stem; do NOT raise the baseline."
    );
}

/// Helper: pull the PV-DUP-001 census out of the gate's out-of-band channel.
#[allow(clippy::type_complexity)]
fn census_of(
    report: &crate::lint::LintReport,
) -> (usize, usize, Vec<String>, Vec<String>, Vec<String>) {
    let gate = report
        .gates
        .iter()
        .find(|g| g.name == "duplicate-stems")
        .expect("duplicate-stems gate");
    let Some(GateExtra::DuplicateStems {
        divergent,
        baselined,
        unbaselined,
        stale,
        divergent_stems,
    }) = gate.extra.as_ref()
    else {
        panic!("duplicate-stems gate carries no GateExtra::DuplicateStems payload");
    };
    (
        *divergent,
        *baselined,
        unbaselined.clone(),
        stale.clone(),
        divergent_stems.clone(),
    )
}

/// The report must NAME the ambiguous stems, not just count them.
///
/// `GateDetail` is frozen at the eight variants published in 0.3.1 (see its doc
/// comment), so this payload rides in `GateResult::extra`. That is an implementation
/// detail of WHERE it lives; this test fixes WHAT must be there, so a future move
/// cannot quietly downgrade the census to a number. A count alone is how 48
/// unresolvable contracts sat behind a green tick.
#[test]
fn the_gate_names_every_ambiguous_stem_with_its_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = build_tree(tmp.path(), "aprender");
    let mut config = LintConfig::new(&contracts, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);

    let (divergent, baselined, unbaselined, stale, stems) = census_of(&report);
    assert_eq!(divergent, 1);
    assert_eq!(baselined, 0);
    assert_eq!(unbaselined, vec!["x-architecture-v1".to_string()]);
    assert!(stale.is_empty());
    assert_eq!(
        stems.len(),
        1,
        "every divergent stem must appear: {stems:?}"
    );
    let line = &stems[0];
    assert!(line.contains("x-architecture-v1"), "{line}");
    assert!(line.contains("[2 variants]"), "{line}");
    // BOTH claiming paths, so a reader can go and delete one of them.
    assert_eq!(line.matches("x-architecture-v1.yaml").count(), 2, "{line}");
    assert!(line.contains("aprender"), "{line}");
}

/// The frozen `GateDetail` numbers must agree with the census, or the two channels
/// tell different stories to the 0.3.1 renderer and to ours.
#[test]
fn frozen_detail_numbers_agree_with_the_census() {
    let tmp = tempfile::tempdir().unwrap();
    let contracts = build_tree(tmp.path(), "aprender");
    let mut config = LintConfig::new(&contracts, None, 0.0);
    config.no_cache = true;
    let report = run_lint(&config);

    let (_, baselined, unbaselined, stale, _) = census_of(&report);
    let gate = report
        .gates
        .iter()
        .find(|g| g.name == "duplicate-stems")
        .expect("gate");
    let GateDetail::Validate {
        contracts: implicated,
        errors,
        warnings,
        error_messages,
    } = &gate.detail
    else {
        panic!("duplicate-stems must report through the frozen `Validate` variant");
    };
    assert_eq!(
        *implicated, 2,
        "two contract FILES claim the ambiguous stem"
    );
    assert_eq!(*errors, unbaselined.len() + stale.len());
    assert_eq!(*warnings, baselined);
    assert_eq!(error_messages.len(), *errors);
    assert!(error_messages[0].contains("x-architecture-v1"));
}

/// COMPATIBILITY LOCK — `GateDetail` is FROZEN at the eight variants published in
/// `provable-contracts` 0.3.1.
///
/// The 28 vendored 0.3.1 programs `match` this enum exhaustively, are sha256-frozen
/// (`check_facade_compat.sh` row R6) and are compiled by the `ci` job that the
/// required `gate` check depends on. A ninth variant is `error[E0004]` over there —
/// a slow failure in a different workspace, a different lockfile and a different
/// target dir. This match mirrors theirs so the same mistake fails HERE first, in
/// `cargo check -p aprender-contracts`, next to this comment. It is never called;
/// exhaustiveness is a compile-time property.
///
/// If you are reading this because you wanted a ninth variant: don't. Put it in
/// `GateExtra`, which is `#[non_exhaustive]` and which 0.3.1 never names.
#[allow(dead_code)]
fn gate_detail_vocabulary_is_frozen_at_the_0_3_1_variants(detail: &GateDetail) {
    match detail {
        GateDetail::Validate { .. }
        | GateDetail::Audit { .. }
        | GateDetail::Score { .. }
        | GateDetail::Verify { .. }
        | GateDetail::Enforce { .. }
        | GateDetail::ReverseCoverage { .. }
        | GateDetail::Composition { .. }
        | GateDetail::Skipped { .. } => {}
    }
}

/// The real corpus: the census must NAME every baselined stem, not just count them.
/// `real_corpus_matches_its_baseline` checks the SET; this checks the REPORT.
#[test]
fn real_corpus_census_names_every_baselined_stem() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dups = scan_duplicate_stems(&root.join("contracts"));
    let baseline = read_baseline(&root);
    let (result, _) = run_duplicate_stem_gate(&dups, &baseline);
    let Some(GateExtra::DuplicateStems {
        divergent,
        divergent_stems,
        ..
    }) = result.extra.as_ref()
    else {
        panic!("duplicate-stems gate carries no census payload");
    };
    assert_eq!(*divergent, baseline.len());
    assert_eq!(divergent_stems.len(), baseline.len());
    for stem in &baseline {
        let prefix = format!("{stem} [");
        assert!(
            divergent_stems.iter().any(|l| l.starts_with(&prefix)),
            "the report does not name baselined stem `{stem}`"
        );
    }
}
