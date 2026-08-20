//! Tests for [`super`] — the inert-contract classifier and its ratchet.

use super::*;

/// The path to the repo's real `contracts/` tree, relative to this crate.
fn contracts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

/// The classifier's own case table must pass before it is allowed to judge
/// 1726 files. Verification Discipline #7.
#[test]
fn self_test_case_table_passes() {
    match self_test() {
        Ok(n) => assert!(n >= 15, "case table shrank to {n} rows"),
        Err(failures) => panic!("classifier case table failed:\n  {}", failures.join("\n  ")),
    }
}

/// THE RATCHET.
///
/// The number of contracts that assert something and ship no way to refute it
/// may never rise above [`INERT_BASELINE`]. Wired into CI by
/// `cargo nextest run --profile ci --workspace --lib` (`.github/workflows/ci.yml:289`,
/// job `workspace-test`, which is in `gate.needs` at ci.yml:798).
///
/// Non-vacuity is asserted first and separately: a run that walks nothing, or
/// parses nothing, or finds no falsifiable contract at all, is a FAILURE, not a
/// pass. Every gate in this repo that has ever been theater was one that could
/// pass on an empty measurement.
#[test]
fn inert_ratchet_holds() {
    let dir = contracts_dir();
    assert!(
        dir.is_dir(),
        "contracts/ not found at {} — the ratchet cannot pass by not looking",
        dir.display()
    );

    let report = classify_tree(&dir);

    // --- non-vacuity controls (each excludes a specific way to pass wrongly)
    assert!(
        report.walked >= WALK_FLOOR,
        "VACUOUS: walked only {} contracts (floor {WALK_FLOOR}) — the walker \
         found nothing, so the inert count below proves nothing",
        report.walked
    );
    assert!(
        report.count(Verdict::Falsifiable) > 500,
        "VACUOUS: only {} falsifiable contracts — classify() is answering \
         Inert/Catalog for everything",
        report.count(Verdict::Falsifiable)
    );
    assert!(
        report.count(Verdict::Catalog) > 100,
        "VACUOUS: only {} catalog contracts — the fair-exemption half of the \
         rule stopped firing",
        report.count(Verdict::Catalog)
    );
    assert!(
        report.probe_failures.is_empty(),
        "raw-YAML probe failed on {} file(s) — their dropped falsification \
         blocks are INVISIBLE to this ratchet, which is the exact silence it \
         exists to remove: {:?}",
        report.probe_failures.len(),
        report.probe_failures
    );
    assert_eq!(
        report.walked,
        report.contracts.len() + report.parse_failures + report.probe_failures.len(),
        "every walked file must be classified, or counted as a parse failure, \
         or counted as a probe failure — nothing may vanish"
    );

    // --- the ratchet itself
    let inert = report.inert_count();
    assert!(
        inert <= INERT_BASELINE,
        "INERT RATCHET: {inert} contracts assert something with no way to \
         refute it, above the pinned baseline of {INERT_BASELINE}.\n\
         Either give the new contract a `falsification_tests:` entry, or — if \
         it is a catalog — remove the claim field that made it a claim.\n\
         First offenders: {:?}",
        report
            .inert()
            .iter()
            .map(|c| format!("{} ({:?})", c.stem, c.reasons))
            .take(5)
            .collect::<Vec<_>>()
    );
}

/// Every bucket is populated on the real tree and the three partition it
/// exactly. A classifier that silently dropped files would break this.
#[test]
fn classification_partitions_the_tree() {
    let dir = contracts_dir();
    if !dir.is_dir() {
        return;
    }
    let report = classify_tree(&dir);
    let total = report.count(Verdict::Falsifiable)
        + report.count(Verdict::Catalog)
        + report.count(Verdict::Inert);
    assert_eq!(total, report.contracts.len(), "verdicts must partition");
    assert!(report.count(Verdict::Inert) > 0, "inert bucket empty");
}

/// MUTATION CONTROL, in-test: the same contract must flip verdict when — and
/// only when — the falsification block is spelled correctly. A check that has
/// only ever been observed green proves nothing, so the RED half is asserted
/// here rather than left to a manual run.
#[test]
fn misspelled_block_goes_inert_correct_spelling_does_not() {
    let broken = "metadata:\n  version: \"1.0.0\"\n  description: d\n  registry: true\n\
                  falsification:\n  - id: F-1\n    rule: r\n";
    let fixed = "metadata:\n  version: \"1.0.0\"\n  description: d\n  registry: true\n\
                 falsification_tests:\n  - id: F-1\n    rule: r\n";

    let c_broken: Contract = serde_yaml::from_str(broken).expect("broken fixture parses");
    let c_fixed: Contract = serde_yaml::from_str(fixed).expect("fixed fixture parses");

    let (v_broken, reasons) = classify(broken, &c_broken).expect("broken probes");
    let (v_fixed, _) = classify(fixed, &c_fixed).expect("fixed probes");

    assert_eq!(v_broken, Verdict::Inert, "misspelled block must read Inert");
    assert!(
        reasons.contains(&"falsification".to_string()),
        "the reason must name the dropped key, got {reasons:?}"
    );
    assert_eq!(
        v_fixed,
        Verdict::Falsifiable,
        "correctly spelled block must read Falsifiable — without this the \
         classifier could be answering Inert unconditionally"
    );
}

/// The 12 `beat-benchmark` contracts are exempt because they name a CI gate,
/// not because of their kind. Take the gate name away and the exemption is
/// unchanged (a bare `beat:` asserts nothing this module can check); add an
/// equation and it is revoked.
#[test]
fn beat_exemption_is_scoped_to_the_beat_block() {
    let bare = "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: beat-benchmark\n\
                beat:\n  incumbent: ollama\n  ci_gate_name: g\n";
    let with_eq = "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: beat-benchmark\n\
                   beat:\n  incumbent: ollama\n  ci_gate_name: g\nequations:\n  e:\n    formula: f\n";

    let c1: Contract = serde_yaml::from_str(bare).expect("parses");
    let c2: Contract = serde_yaml::from_str(with_eq).expect("parses");
    assert_eq!(classify(bare, &c1).expect("probes").0, Verdict::Catalog);
    assert_eq!(classify(with_eq, &c2).expect("probes").0, Verdict::Inert);
}

/// An empty or null dropped block is not a lost test — it must not inflate the
/// ratchet.
#[test]
fn empty_dropped_block_is_not_a_claim() {
    for yaml in [
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nfalsification: []\n",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nfalsification:\n",
        "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\nfalsification: \"  \"\n",
    ] {
        let c: Contract = serde_yaml::from_str(yaml).expect("parses");
        assert_eq!(
            classify(yaml, &c).expect("probes").0,
            Verdict::Catalog,
            "yaml: {yaml}"
        );
    }
}

/// `classify_tree` on an empty directory returns an empty report — and the
/// caller, not this function, is responsible for rejecting it. Documents that
/// the vacuity check lives at the gate.
#[test]
fn empty_tree_yields_zero_walked() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let report = classify_tree(tmp.path());
    assert_eq!(report.walked, 0);
    assert_eq!(report.inert_count(), 0);
}

/// The walker is `pv lint`'s walker: `binding.yaml` and the four skipped
/// directories must not appear in the population.
#[test]
fn walker_matches_pv_lint_population() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let minimal = "metadata:\n  version: \"1.0.0\"\n  description: d\n  kind: schema\n";
    std::fs::write(tmp.path().join("real-v1.yaml"), minimal).expect("write");
    std::fs::write(tmp.path().join("binding.yaml"), "crates: []\n").expect("write");
    std::fs::create_dir_all(tmp.path().join("kaizen")).expect("mkdir");
    std::fs::write(tmp.path().join("kaizen/skipped-v1.yaml"), minimal).expect("write");

    let report = classify_tree(tmp.path());
    let stems: Vec<_> = report.contracts.iter().map(|c| c.stem.as_str()).collect();
    assert_eq!(report.walked, 1, "walked {stems:?}");
    assert_eq!(stems, vec!["real-v1"]);
}

/// `Verdict` renders the strings the CLI and JSON output promise.
#[test]
fn verdict_display_is_stable() {
    assert_eq!(Verdict::Falsifiable.to_string(), "falsifiable");
    assert_eq!(Verdict::Catalog.to_string(), "catalog");
    assert_eq!(Verdict::Inert.to_string(), "inert");
}

/// Regression, named: the real file whose duplicate `subcommands` key defeated
/// the first draft. It must classify as Inert, with the dropped block named.
#[test]
fn apr_cli_commands_contract_is_inert_not_catalog() {
    let path = contracts_dir().join("apr-cli-commands-v1.yaml");
    if !path.is_file() {
        return;
    }
    let raw = std::fs::read_to_string(&path).expect("read");
    let contract: Contract = serde_yaml::from_str(&raw).expect("parses as Contract");
    assert!(
        contract.falsification_tests.is_empty(),
        "fixture assumption: this file's falsification entries are dropped by serde"
    );
    let (verdict, reasons) = classify(&raw, &contract).expect("probe must not fail");
    assert_eq!(verdict, Verdict::Inert);
    assert!(
        reasons.contains(&"falsification".to_string()),
        "got {reasons:?}"
    );
}

/// FALSIFY-README-008 (contracts/readme-claims-v1.yaml).
///
/// The README's falsification-coverage paragraph is the claim this whole module
/// was written to correct: it used to say "Every CLI command and kernel is
/// bound to a YAML contract with ... falsification tests", which was false for
/// 759 of 1726 files. The replacement states a CEILING, and a ceiling only
/// stays honest while it matches the ratchet. Both the prose number and the
/// runnable `pv inert contracts --max N` line must equal [`INERT_BASELINE`].
///
/// A prose claim nothing recomputes is exactly the drift readme-claims-v1
/// exists to reject; this is that contract's row for this claim.
#[test]
fn readme_inert_ceiling_matches_the_ratchet() {
    let readme = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
    if !readme.is_file() {
        return;
    }
    let text = std::fs::read_to_string(&readme).expect("read README.md");

    // The runnable command a reader can copy.
    let cmd = format!("pv inert contracts --max {INERT_BASELINE}");
    assert!(
        text.contains(&cmd),
        "README must show the exact ratchet command `{cmd}` — a documented \
         ceiling that does not match the enforced one is drift"
    );
    // No stale ceiling may survive alongside it.
    for stale in ["pv inert contracts --max ", "at most **"] {
        let occurrences = text.matches(stale).count();
        assert!(
            occurrences >= 1,
            "README lost its inert-ceiling claim (`{stale}` not found)"
        );
    }
    let claim = format!("at most **{INERT_BASELINE}**");
    assert!(
        text.contains(&claim),
        "README prose must state the ceiling as `{claim}`, matching \
         INERT_BASELINE — got a different number, which is the drift class \
         readme-claims-v1.yaml rejects"
    );

    // Non-vacuity: the retracted claim must be gone. Without this the test
    // would pass on a README that ALSO still carries the false sentence.
    assert!(
        !text.contains("Every CLI command and kernel is bound to a YAML contract"),
        "the retracted universal claim is still in README.md; it is false for \
         the {} contracts this module classifies as inert",
        INERT_BASELINE
    );
}
