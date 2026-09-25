//! PVL-001 EV-11 — the two `pv lint` ratchets, each case on a throwaway repo built in a tempdir.

use super::*;

const THEOREMS: &str = "crates/aprender-contracts-staging/lean/ProvableContracts/Theorems";

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().expect("has a parent")).expect("mkdir");
    std::fs::write(p, text).expect("write");
}

fn baseline(root: &Path, json: &str) {
    write(root, "contracts/lint-baseline.json", json);
}

/// A repo with two theorem modules, one of them named by a book page.
fn pairing_repo() -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        &format!("{THEOREMS}/Softmax/PartitionOfUnity.lean"),
        "theorem p : True := trivial\n",
    );
    write(
        t.path(),
        &format!("{THEOREMS}/Softmax/Bounds.lean"),
        "theorem b : True := trivial\n",
    );
    write(
        t.path(),
        "book/src/softmax.md",
        "The partition law is `ProvableContracts.Theorems.Softmax.PartitionOfUnity`.\n",
    );
    t
}

fn pairing(root: &Path) -> (GateResult, Vec<String>) {
    match run_theorem_pairing_gate(&root.join("contracts")) {
        RatchetOutcome::Ran { result, findings } => {
            (*result, findings.into_iter().map(|f| f.rule_id).collect())
        }
        RatchetOutcome::Declined(why) => panic!("expected a verdict, got a decline: {why}"),
    }
}

fn unpaired_of(r: &GateResult) -> (usize, Vec<String>) {
    match r.extra.as_ref() {
        Some(GateExtra::TheoremPairing {
            unpaired_theorem_modules,
            unpaired,
            ..
        }) => (*unpaired_theorem_modules, unpaired.clone()),
        other => panic!("expected TheoremPairing, got {other:?}"),
    }
}

// ── mentions_module: identifier boundaries ────────────────────────────────────────────────────────────────

#[test]
fn a_module_is_mentioned_only_on_identifier_boundaries() {
    let m = "P.T.Softmax.Partition";
    for (text, want) in [
        ("see P.T.Softmax.Partition here", true),
        ("`P.T.Softmax.Partition`", true),
        ("ends the sentence P.T.Softmax.Partition.", true),
        ("P.T.Softmax.Partition", true),
        ("(P.T.Softmax.Partition)", true),
        // a LONGER module is not this one
        ("P.T.Softmax.PartitionOfUnity", false),
        ("P.T.Softmax.Partition.Deeper", false),
        ("P.T.Softmax.Partition_2", false),
        ("P.T.Softmax.Partition'", false),
        // a PREFIXED name is not this one
        ("X.P.T.Softmax.Partition", false),
        ("XP.T.Softmax.Partition", false),
        // the file stem alone is not the module
        ("Partition", false),
        ("", false),
    ] {
        assert_eq!(mentions_module(text, m), want, "{text:?}");
    }
    // a bad first occurrence does not hide a good later one
    assert!(mentions_module(
        "P.T.Softmax.PartitionX and P.T.Softmax.Partition",
        m
    ));
}

#[test]
fn a_module_name_is_the_dotted_path_from_the_base() {
    let base = Path::new("/r/lean");
    assert_eq!(
        module_name(
            base,
            Path::new("/r/lean/ProvableContracts/Theorems/A/B.lean")
        )
        .as_deref(),
        Some("ProvableContracts.Theorems.A.B")
    );
    assert_eq!(module_name(base, Path::new("/elsewhere/A.lean")), None);
}

// ── the ratchet itself ────────────────────────────────────────────────────────────────────────────────────

#[test]
fn the_ratchet_rejects_a_rise_only() {
    let f = |b, n| ratchet_finding("PV-RAT-001", UNPAIRED_KEY, b, n, "fix").map(|f| f.rule_id);
    assert_eq!(f(Some(5), 4), None, "a fall passes");
    assert_eq!(f(Some(5), 5), None, "equal passes");
    assert_eq!(f(Some(5), 6).as_deref(), Some("PV-RAT-001"));
    assert_eq!(f(None, 1000), None, "no baseline: reported, not judged");
}

#[test]
fn a_baseline_that_is_not_a_non_negative_integer_is_no_baseline() {
    let t = tempfile::tempdir().expect("tempdir");
    let dir = t.path().join("contracts");
    for (json, want) in [
        (r#"{"unpaired_theorem_modules": 7}"#, Some(7)),
        (r#"{"unpaired_theorem_modules": -1}"#, None),
        (r#"{"unpaired_theorem_modules": "7"}"#, None),
        (r#"{"ont": {"unpaired_theorem_modules": 7}}"#, None),
        ("not json", None),
    ] {
        baseline(t.path(), json);
        assert_eq!(baseline_of(&dir, UNPAIRED_KEY), want, "{json}");
    }
}

// ── theorem-pairing over a repo ───────────────────────────────────────────────────────────────────────────

#[test]
fn theorem_pairing_counts_the_modules_no_book_page_names() {
    let t = pairing_repo();
    baseline(t.path(), r#"{"unpaired_theorem_modules": 1}"#);
    let (r, rules) = pairing(t.path());
    assert!(r.passed && !r.skipped, "at the baseline: pass");
    assert!(rules.is_empty());
    assert_eq!(
        unpaired_of(&r),
        (
            1,
            vec!["ProvableContracts.Theorems.Softmax.Bounds".to_string()]
        )
    );
}

/// The spec row's mutation, verbatim: "add an unpaired Theorem module → RED".
#[test]
fn adding_an_unpaired_theorem_module_is_red() {
    let t = pairing_repo();
    baseline(t.path(), r#"{"unpaired_theorem_modules": 1}"#);
    write(
        t.path(),
        &format!("{THEOREMS}/Gelu/Tanh.lean"),
        "theorem g : True := trivial\n",
    );
    let (r, rules) = pairing(t.path());
    assert!(!r.passed);
    assert_eq!(r.verdict, Verdict::from_gate(false, false));
    assert_eq!(rules, ["PV-RAT-001"]);
    assert_eq!(unpaired_of(&r).0, 2);
}

#[test]
fn naming_the_module_in_the_staging_book_pairs_it_too() {
    let t = pairing_repo();
    baseline(t.path(), r#"{"unpaired_theorem_modules": 0}"#);
    write(
        t.path(),
        "crates/aprender-contracts-staging/book/src/bounds.md",
        "ProvableContracts.Theorems.Softmax.Bounds\n",
    );
    let (r, rules) = pairing(t.path());
    assert!(r.passed, "{rules:?}");
    assert_eq!(unpaired_of(&r).0, 0);
}

#[test]
fn the_file_stem_or_a_longer_name_does_not_pair_a_module() {
    let t = pairing_repo();
    baseline(t.path(), r#"{"unpaired_theorem_modules": 1}"#);
    write(
        t.path(),
        "book/src/more.md",
        "Bounds, Softmax.Bounds and ProvableContracts.Theorems.Softmax.BoundsTight are all different.\n",
    );
    assert_eq!(unpaired_of(&pairing(t.path()).0).0, 1);
}

#[test]
fn no_baseline_is_reported_and_never_a_pass() {
    let t = pairing_repo();
    let (r, rules) = pairing(t.path());
    assert!(!r.passed && r.skipped);
    assert_eq!(r.verdict, Verdict::Unknown(Reason::Report));
    assert!(
        rules.is_empty(),
        "nothing to compare against, so no finding"
    );
    assert_eq!(unpaired_of(&r).0, 1, "the count is still reported");
}

#[test]
fn theorem_pairing_declines_when_nothing_was_measured() {
    // no Lean base at all
    let t = tempfile::tempdir().expect("tempdir");
    write(t.path(), "book/a.md", "x\n");
    assert!(matches!(
        run_theorem_pairing_gate(&t.path().join("contracts")),
        RatchetOutcome::Declined(_)
    ));
    // a base with no theorem module
    std::fs::create_dir_all(t.path().join(THEOREMS)).expect("mkdir");
    assert!(matches!(
        run_theorem_pairing_gate(&t.path().join("contracts")),
        RatchetOutcome::Declined(_)
    ));
    // theorem modules but no book page
    let t = pairing_repo();
    std::fs::remove_dir_all(t.path().join("book")).expect("rm book");
    match run_theorem_pairing_gate(&t.path().join("contracts")) {
        RatchetOutcome::Declined(why) => assert!(why.contains("no book page"), "{why}"),
        RatchetOutcome::Ran { .. } => panic!("no book is a decline, not 2 unpaired"),
    }
}

// ── depends-on-present over a corpus ──────────────────────────────────────────────────────────────────────

fn contract(kind: &str, depends_on: &str) -> String {
    format!(
        "metadata:\n  version: \"1.0.0\"\n  kind: {kind}\n  description: EV-11 fixture\n  depends_on: {depends_on}\n\
         equations:\n  identity:\n    formula: \"len(out) = len(x)\"\n"
    )
}

fn depends(root: &Path) -> (GateResult, Vec<String>) {
    match run_depends_on_present_gate(&root.join("contracts")) {
        RatchetOutcome::Ran { result, findings } => {
            (*result, findings.into_iter().map(|f| f.rule_id).collect())
        }
        RatchetOutcome::Declined(why) => panic!("expected a verdict, got a decline: {why}"),
    }
}

fn without_of(r: &GateResult) -> (usize, usize) {
    match r.extra.as_ref() {
        Some(GateExtra::DependsOnPresent {
            kernel_contracts,
            contracts_without_depends_on,
            ..
        }) => (*kernel_contracts, *contracts_without_depends_on),
        other => panic!("expected DependsOnPresent, got {other:?}"),
    }
}

fn depends_repo() -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        "contracts/a-v1.yaml",
        &contract("kernel", "[b-v1]"),
    );
    write(t.path(), "contracts/b-v1.yaml", &contract("kernel", "[]"));
    // neither of these is a kernel contract, so neither counts either way
    write(
        t.path(),
        "contracts/reg-v1.yaml",
        &contract("registry", "[]"),
    );
    write(
        t.path(),
        "contracts/pat-v1.yaml",
        &contract("pattern", "[]"),
    );
    t
}

#[test]
fn depends_on_counts_kernel_contracts_with_none() {
    let t = depends_repo();
    baseline(t.path(), r#"{"contracts_without_depends_on": 1}"#);
    let (r, rules) = depends(t.path());
    assert!(r.passed && !r.skipped, "{rules:?}");
    assert_eq!(
        without_of(&r),
        (2, 1),
        "registry and pattern are not kernel contracts"
    );
}

#[test]
fn adding_a_kernel_contract_with_no_depends_on_is_red() {
    let t = depends_repo();
    baseline(t.path(), r#"{"contracts_without_depends_on": 1}"#);
    write(t.path(), "contracts/c-v1.yaml", &contract("kernel", "[]"));
    let (r, rules) = depends(t.path());
    assert!(!r.passed);
    assert_eq!(rules, ["PV-RAT-002"]);
    assert_eq!(without_of(&r), (3, 2));
}

#[test]
fn depends_on_declines_with_no_kernel_contract() {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        "contracts/pat-v1.yaml",
        &contract("pattern", "[]"),
    );
    assert!(matches!(
        run_depends_on_present_gate(&t.path().join("contracts")),
        RatchetOutcome::Declined(_)
    ));
}

// ── the gates read the baseline and never write it ───────────────────────────────────────────────────────

#[test]
fn neither_gate_writes_the_baseline() {
    let t = pairing_repo();
    write(t.path(), "contracts/b-v1.yaml", &contract("kernel", "[]"));
    let json = "{\n  \"unpaired_theorem_modules\": 9,\n  \"contracts_without_depends_on\": 9\n}\n";
    baseline(t.path(), json);
    // both counts FELL (1 < 9): a gate that "helpfully" lowered the baseline would change the file
    assert!(pairing(t.path()).0.passed);
    assert!(depends(t.path()).0.passed);
    assert_eq!(
        std::fs::read_to_string(t.path().join("contracts/lint-baseline.json")).expect("read"),
        json
    );
}
