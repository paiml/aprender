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
fn a_missing_or_non_string_world_is_pv_ont_014() {
    assert_eq!(rules_for("backend: [cpu]\n"), ["PV-ONT-014"]);
    assert_eq!(rules_for("world: [committed]\n"), ["PV-ONT-014"]);
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
