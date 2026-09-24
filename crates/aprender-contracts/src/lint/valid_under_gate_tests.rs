//! ONT-7 — the rules of `metadata.valid_under`, one case per rule, against the real Σ.

use super::*;

fn sigma() -> Sigma {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/ontology.yaml"),
    )
    .expect("the repo's Σ is readable");
    Sigma::from_yaml(&text).expect("the repo's Σ parses")
}

fn rules_for(yaml: &str) -> Vec<String> {
    let v: serde_yaml::Value = serde_yaml::from_str(yaml).expect("case is YAML");
    let mut out = Vec::new();
    check_valid_under(&sigma(), &v, "case", Path::new("case.yaml"), &mut out);
    out.into_iter().map(|f| f.rule_id).collect()
}

#[test]
fn a_full_world_index_draws_nothing() {
    let y = "world: committed\ntoolchain: {rust: \"1.93\"}\nhost_class: [x86_64-linux]\nbackend: [cpu, cuda]\nfeatures: [cuda]\n";
    assert!(rules_for(y).is_empty());
}

#[test]
fn world_alone_is_enough() {
    assert!(rules_for("world: committed\n").is_empty());
}

#[test]
fn an_undeclared_world_is_pv_ont_014() {
    assert_eq!(rules_for("world: mars\n"), ["PV-ONT-014"]);
}

#[test]
fn a_non_string_world_is_pv_ont_014() {
    assert_eq!(rules_for("world: [committed]\n"), ["PV-ONT-014"]);
    // An OMITTED world is not this rule: it reads `committed` (see the Appendix B test below).
    assert!(rules_for("backend: [cpu]\n").is_empty());
}

#[test]
fn a_key_outside_the_closed_set_is_pv_ont_013() {
    assert_eq!(rules_for("world: committed\ngpu: true\n"), ["PV-ONT-013"]);
}

#[test]
fn a_non_mapping_is_pv_ont_013_and_nothing_else() {
    assert_eq!(rules_for("committed\n"), ["PV-ONT-013"]);
}

#[test]
fn malformed_qualifiers_are_pv_ont_015_each() {
    assert_eq!(rules_for("world: committed\nbackend: []\n"), ["PV-ONT-015"]);
    assert_eq!(
        rules_for("world: committed\nhost_class: [\"\"]\n"),
        ["PV-ONT-015"]
    );
    assert_eq!(
        rules_for("world: committed\nfeatures: cuda\n"),
        ["PV-ONT-015"]
    );
    assert_eq!(
        rules_for("world: committed\ntoolchain: {rust: 1}\n"),
        ["PV-ONT-015"]
    );
    assert_eq!(
        rules_for("world: committed\ntoolchain: {}\n"),
        ["PV-ONT-015"]
    );
}

#[test]
fn the_closed_set_is_exactly_the_five_keys() {
    assert_eq!(
        VALID_UNDER_KEYS,
        ["world", "toolchain", "host_class", "backend", "features"]
    );
}

// ── the omitted world and the empty block (#4076 re-review) ─────────────────────────────────────────────

#[test]
fn an_omitted_world_reads_committed_so_appendix_b_is_admitted() {
    let y = "toolchain: {rust: \"1.93\"}\nhost_class: [x86_64-linux]\nbackend: [cpu, cuda]\nfeatures: [cuda]\n";
    assert!(rules_for(y).is_empty(), "the spec's own example must pass");
    assert_eq!(DEFAULT_WORLD, "committed");
}

#[test]
fn an_omitted_world_with_no_committed_in_sigma_is_pv_ont_014() {
    let mut s = sigma();
    s.worlds.remove(DEFAULT_WORLD);
    let v: serde_yaml::Value = serde_yaml::from_str("backend: [cpu]\n").expect("yaml");
    let mut out = Vec::new();
    check_valid_under(&s, &v, "case", Path::new("case.yaml"), &mut out);
    assert_eq!(
        out.into_iter().map(|f| f.rule_id).collect::<Vec<_>>(),
        ["PV-ONT-014"]
    );
}

#[test]
fn an_empty_block_is_pv_ont_013() {
    assert_eq!(rules_for("{}\n"), ["PV-ONT-013"]);
}

// ── the ratchet ──────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn the_ratchet_rejects_a_rise_only() {
    assert!(ratchet_finding(Some(5), 4).is_none(), "a fall passes");
    assert!(ratchet_finding(Some(5), 5).is_none(), "equal passes");
    assert_eq!(
        ratchet_finding(Some(5), 6).map(|f| f.rule_id).as_deref(),
        Some("PV-ONT-016")
    );
    assert!(
        ratchet_finding(None, 1000).is_none(),
        "no baseline: reported, not judged"
    );
}

// ── the whole gate, over the committed fixtures (the lib-level witness the CI mutation lane runs) ─────────

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name)
}

fn ran(name: &str) -> (bool, Vec<String>, Option<GateExtra>) {
    match run_valid_under_gate(&fixture(name)) {
        ValidUnderOutcome::Ran { result, findings } => (
            result.passed,
            findings.into_iter().map(|f| f.rule_id).collect(),
            result.extra,
        ),
        other => panic!("{name}: expected a verdict, got {other:?}"),
    }
}

#[test]
fn every_fixture_draws_exactly_its_rule() {
    for (name, want) in [
        ("valid-under-ok", vec![]),
        ("valid-under-appendix-b", vec![]),
        ("valid-under-unknown-world", vec!["PV-ONT-014"]),
        ("valid-under-world-not-a-string", vec!["PV-ONT-014"]),
        ("valid-under-undeclared-key", vec!["PV-ONT-013"]),
        ("valid-under-empty", vec!["PV-ONT-013"]),
        (
            "valid-under-bad-qualifier",
            vec!["PV-ONT-015", "PV-ONT-015"],
        ),
        ("valid-under-ratchet-rise", vec!["PV-ONT-016"]),
        ("valid-under-nonkernel-bad", vec!["PV-ONT-014"]),
    ] {
        let (passed, rules, _) = ran(name);
        assert_eq!(rules, want, "{name}");
        assert_eq!(passed, want.is_empty(), "{name}");
    }
}

#[test]
fn the_census_counts_what_it_saw() {
    let (_, _, extra) = ran("valid-under-ratchet-rise");
    let Some(GateExtra::ValidUnder {
        kernel_contracts,
        contracts_without_valid_under,
        contracts_with_valid_under,
        baseline,
        ..
    }) = extra
    else {
        panic!("valid-under extra");
    };
    assert_eq!(
        (
            kernel_contracts,
            contracts_without_valid_under,
            contracts_with_valid_under,
            baseline
        ),
        (1, 1, 0, Some(0))
    );
    let (_, _, extra) = ran("valid-under-appendix-b");
    let Some(GateExtra::ValidUnder { by_world, .. }) = extra else {
        panic!("valid-under extra");
    };
    assert_eq!(
        by_world,
        ["committed=1"],
        "an omitted world is counted as committed"
    );
}

#[test]
fn a_corpus_that_measured_nothing_declines_and_no_sigma_declines() {
    assert!(matches!(
        run_valid_under_gate(&fixture("valid-under-no-kernels")),
        ValidUnderOutcome::NoKernels {
            contracts_checked: 1
        }
    ));
    assert!(matches!(
        run_valid_under_gate(&fixture("sigma-absent")),
        ValidUnderOutcome::NoSigma
    ));
}

#[test]
fn the_named_gate_dispatches_valid_under() {
    assert!(super::super::NAMED_GATES.contains(&"valid-under"));
    assert!(matches!(
        super::super::run_named_gate(&fixture("valid-under-ok"), "valid-under"),
        super::super::NamedGateOutcome::ValidUnder(ValidUnderOutcome::Ran { .. })
    ));
}
