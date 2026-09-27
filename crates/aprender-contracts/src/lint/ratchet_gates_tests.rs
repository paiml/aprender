//! PVL-001 EV-11 — the two `pv lint` ratchets, each case on a throwaway repo built in a tempdir.

use super::*;
use crate::lint::comparand::with_comparand;

const THEOREMS: &str = "crates/aprender-contracts-staging/lean/ProvableContracts/Theorems";

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().expect("has a parent")).expect("mkdir");
    std::fs::write(p, text).expect("write");
}

/// A copy of `root` as it stands now: the comparand (merge-base) tree the edits made after it are judged against.
fn snapshot(root: &Path) -> tempfile::TempDir {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("mkdir");
        for entry in std::fs::read_dir(from).expect("read_dir") {
            let entry = entry.expect("entry");
            let dest = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy(&entry.path(), &dest);
            } else {
                std::fs::copy(entry.path(), dest).expect("copy");
            }
        }
    }
    let t = tempfile::tempdir().expect("tempdir");
    copy(root, t.path());
    t
}

/// Run `f` with `base` named as the comparand.
fn against<T>(base: &tempfile::TempDir, f: impl FnOnce() -> T) -> T {
    with_comparand(&base.path().join("contracts"), f)
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
fn the_baseline_is_the_comparand_measured_and_a_stored_number_is_none() {
    let t = pairing_repo();
    // the pre-#3569 stored shape: a number a PR could restamp
    write(t.path(), "contracts/lint-baseline.json", r#"{"unpaired_theorem_modules": 7}"#);
    assert_eq!(baseline_of(UNPAIRED_KEY), None, "no comparand: no baseline, whatever is stored");
    let base = snapshot(t.path());
    assert_eq!(against(&base, || baseline_of(UNPAIRED_KEY)), Some(1), "the comparand, measured");
    assert_eq!(against(&base, || baseline_of("formal_prose")), None, "not an EV-11 key");
    let empty = tempfile::tempdir().expect("tempdir");
    assert_eq!(against(&empty, || baseline_of(UNPAIRED_KEY)), None, "a comparand that declines is no baseline");
}

// ── theorem-pairing over a repo ───────────────────────────────────────────────────────────────────────────

#[test]
fn theorem_pairing_counts_the_modules_no_book_page_names() {
    let t = pairing_repo();
    let base = snapshot(t.path());
    let (r, rules) = against(&base, || pairing(t.path()));
    assert!(r.passed && !r.skipped, "unchanged against the comparand: pass");
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
    let base = snapshot(t.path());
    write(
        t.path(),
        &format!("{THEOREMS}/Gelu/Tanh.lean"),
        "theorem g : True := trivial\n",
    );
    let (r, rules) = against(&base, || pairing(t.path()));
    assert!(!r.passed);
    assert_eq!(r.verdict, Verdict::from_gate(false, false));
    assert_eq!(rules, ["PV-RAT-001"]);
    assert_eq!(unpaired_of(&r).0, 2);
}

#[test]
fn naming_the_module_in_the_staging_book_pairs_it_too() {
    let t = pairing_repo();
    let base = snapshot(t.path());
    write(
        t.path(),
        "crates/aprender-contracts-staging/book/src/bounds.md",
        "ProvableContracts.Theorems.Softmax.Bounds\n",
    );
    let (r, rules) = against(&base, || pairing(t.path()));
    assert!(r.passed, "a fall 1 -> 0 passes: {rules:?}");
    assert_eq!(unpaired_of(&r).0, 0);
}

#[test]
fn the_file_stem_or_a_longer_name_does_not_pair_a_module() {
    let t = pairing_repo();
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
    let base = snapshot(t.path());
    let (r, rules) = against(&base, || depends(t.path()));
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
    let base = snapshot(t.path());
    write(t.path(), "contracts/c-v1.yaml", &contract("kernel", "[]"));
    let (r, rules) = against(&base, || depends(t.path()));
    // and the same pair the other way round is a fall
    assert!(with_comparand(&t.path().join("contracts"), || depends(base.path())).0.passed);
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

// ── a stored number is neither read nor written (#3569) ──────────────────────────────────────────────────

#[test]
fn a_stored_count_moves_nothing_and_is_never_written() {
    let t = pairing_repo();
    write(t.path(), "contracts/b-v1.yaml", &contract("kernel", "[]"));
    let base = snapshot(t.path());
    // a stored 0 once made this unchanged head RED; a stored 9 would have hidden a rise
    let json = "{\n  \"unpaired_theorem_modules\": 0,\n  \"contracts_without_depends_on\": 0\n}\n";
    write(t.path(), "contracts/lint-baseline.json", json);
    assert!(against(&base, || pairing(t.path())).0.passed, "a hold, whatever is stored");
    assert!(against(&base, || depends(t.path())).0.passed, "a hold, whatever is stored");
    let (r, _) = pairing(t.path());
    assert!(!r.passed && r.skipped, "no comparand: reported, the stored 0 is not a baseline");
    assert_eq!(
        std::fs::read_to_string(t.path().join("contracts/lint-baseline.json")).expect("read"),
        json
    );
}

// ── proved-is-derived (EV-8a): the summary derives, the YAML only claims ──────────────────────────────────

const SUMMARY: &str = "crates/aprender-contracts-staging/discharge-summary.json";

/// A contract with one `lean.status: <status>` obligation on `theorem`.
fn claim(theorem: &str, status: &str) -> String {
    format!(
        "metadata:\n  version: \"1.0.0\"\n  description: EV-8a fixture\n\
         equations:\n  identity:\n    formula: \"y = x\"\n\
         proof_obligations:\n- type: invariant\n  property: p\n  lean:\n    theorem: {theorem}\n    status: {status}\n"
    )
}

/// A summary over one module listing `theorems`; green unless `leanchecker_exit` says otherwise.
fn summary_json(leanchecker_exit: &str, theorems: &[&str]) -> String {
    let list: Vec<String> = theorems.iter().map(|t| format!("\"{t}\"")).collect();
    format!(
        "{{\"tree_sha\": \"t\", \"toolchain\": \"l\", \"mathlib_rev\": \"m\", \"build_exit\": 0, \"lake_exit\": 0, \
         \"leanchecker_exit\": {leanchecker_exit}, \"axioms_ok\": true, \"escapes_ok\": true, \
         \"challenges_closed\": \"1/1\", \"modules\": [{{\"path\": \"P.lean\", \"blake3\": \"b\", \"theorems\": [{}]}}]}}",
        list.join(", ")
    )
}

/// Two proved claims (one named without the root namespace, one exactly) and one sorry, over a Lean base.
fn derived_repo() -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        &format!("{THEOREMS}/Softmax/P.lean"),
        "theorem p : True := trivial\n",
    );
    write(
        t.path(),
        "contracts/a-v1.yaml",
        &claim("Softmax.partition_of_unity", "proved"),
    );
    write(
        t.path(),
        "contracts/b-v1.yaml",
        &claim("ProvableContracts.Gelu.gelu_zero", "proved"),
    );
    write(
        t.path(),
        "contracts/c-v1.yaml",
        &claim("Relu.relu_nonneg", "sorry"),
    );
    t
}

fn derived_gate(root: &Path) -> (GateResult, Vec<String>) {
    match run_proved_is_derived_gate(&root.join("contracts")) {
        RatchetOutcome::Ran { result, findings } => {
            (*result, findings.into_iter().map(|f| f.rule_id).collect())
        }
        RatchetOutcome::Declined(why) => panic!("expected a verdict, got a decline: {why}"),
    }
}

/// (proved claims, underived, summary green)
fn underived_of(r: &GateResult) -> (usize, Vec<String>, bool) {
    match r.extra.as_ref() {
        Some(GateExtra::ProvedIsDerived {
            proved_claims,
            underived,
            summary_green,
            underived_proved_claims,
            ..
        }) => {
            assert_eq!(*underived_proved_claims, underived.len());
            (*proved_claims, underived.clone(), *summary_green)
        }
        other => panic!("expected ProvedIsDerived, got {other:?}"),
    }
}

#[test]
fn with_no_summary_every_proved_claim_is_underived() {
    let t = derived_repo();
    let base = snapshot(t.path());
    let (r, rules) = against(&base, || derived_gate(t.path()));
    let (claims, under, green) = underived_of(&r);
    assert_eq!(
        (claims, under.len(), green),
        (2, 2, false),
        "the sorry is no claim"
    );
    assert!(r.passed && rules.is_empty(), "2 -> 2: a hold");
}

#[test]
fn a_green_summary_derives_the_theorems_it_lists() {
    let t = derived_repo();
    let base = snapshot(t.path());
    write(
        t.path(),
        SUMMARY,
        &summary_json(
            "0",
            &[
                "ProvableContracts.Softmax.partition_of_unity",
                "ProvableContracts.Gelu.gelu_zero",
            ],
        ),
    );
    let (r, _) = against(&base, || derived_gate(t.path()));
    assert_eq!(underived_of(&r), (2, vec![], true));
    assert!(r.passed);
}

#[test]
fn a_summary_that_is_not_green_derives_nothing() {
    let t = derived_repo();
    let all = [
        "ProvableContracts.Softmax.partition_of_unity",
        "ProvableContracts.Gelu.gelu_zero",
    ];
    write(t.path(), SUMMARY, &summary_json("0", &all));
    let base = snapshot(t.path());
    assert!(against(&base, || derived_gate(t.path())).0.passed, "green: 0 underived");
    for bad in ["1", "124", "null"] {
        write(t.path(), SUMMARY, &summary_json(bad, &all));
        let (r, rules) = against(&base, || derived_gate(t.path()));
        let (_, under, green) = underived_of(&r);
        assert!(
            !green && under.len() == 2,
            "leanchecker_exit {bad}: {under:?}"
        );
        assert!(
            !r.passed && rules == ["PV-RAT-003"],
            "leanchecker_exit {bad} rose 0 -> 2"
        );
    }
    write(t.path(), SUMMARY, "{\"tree_sha\": ");
    let (r, _) = derived_gate(t.path());
    assert_eq!(
        underived_of(&r).1.len(),
        2,
        "an unreadable summary derives nothing"
    );
}

#[test]
fn a_theorem_the_summary_does_not_list_by_its_name_stays_underived() {
    let t = derived_repo();
    // the comparand derives gelu_zero only: one underived
    write(
        t.path(),
        SUMMARY,
        &summary_json("0", &["ProvableContracts.Gelu.gelu_zero"]),
    );
    let base = snapshot(t.path());
    // a suffix, a longer name and a different namespace are not the claim's theorem
    write(
        t.path(),
        SUMMARY,
        &summary_json(
            "0",
            &[
                "partition_of_unity",
                "X.Softmax.partition_of_unity",
                "ProvableContracts.Gelu.gelu_zero_x",
            ],
        ),
    );
    let (r, rules) = against(&base, || derived_gate(t.path()));
    let (_, under, _) = underived_of(&r);
    assert_eq!(
        under,
        [
            "a-v1: Softmax.partition_of_unity",
            "b-v1: ProvableContracts.Gelu.gelu_zero"
        ]
    );
    assert_eq!(rules, ["PV-RAT-003"], "1 -> 2");
}

#[test]
fn adding_a_proved_claim_no_summary_derives_is_red() {
    let t = derived_repo();
    let base = snapshot(t.path());
    assert!(against(&base, || derived_gate(t.path())).0.passed);
    write(
        t.path(),
        "contracts/d-v1.yaml",
        &claim("Silu.silu_zero", "proved"),
    );
    let (r, rules) = against(&base, || derived_gate(t.path()));
    assert!(!r.passed);
    assert_eq!(rules, ["PV-RAT-003"]);
    let GateDetail::Validate { error_messages, .. } = &r.detail else {
        panic!("expected a Validate detail, got {:?}", r.detail);
    };
    assert!(
        error_messages
            .iter()
            .any(|m| m.contains("underived_proved_claims rose 2 -> 3")),
        "{error_messages:?}"
    );
}

#[test]
fn proved_is_derived_without_a_baseline_is_reported_never_a_pass() {
    let t = derived_repo();
    let (r, rules) = derived_gate(t.path());
    assert!(!r.passed && r.skipped && rules.is_empty());
    assert_eq!(r.verdict, Verdict::Unknown(Reason::Report));
}

#[test]
fn proved_is_derived_declines_on_an_empty_corpus() {
    let t = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(t.path().join("contracts")).expect("mkdir");
    assert!(matches!(
        run_proved_is_derived_gate(&t.path().join("contracts")),
        RatchetOutcome::Declined(_)
    ));
}
