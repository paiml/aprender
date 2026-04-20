//! Falsification tests for CRUX-A-01 — Pull model by short name.
//!
//! Contract: contracts/crux-A-01-v1.yaml
//! This test closes FALSIFY-CRUX-A-01-002: `configs/aliases.yaml` ships with
//! the repo, parses as YAML mapping str→str, and contains the four canonical
//! short names required by the CRUX-A-01 invariants:
//!   {"llama3", "mistral", "phi3", "qwen2"} ⊆ keys(map)
//!
//! Other FALSIFY-CRUX-A-01-{001, 003, 004, 005} gates still need the
//! `apr pull --dry-run` + `apr registry aliases --json` surface and are
//! tracked by the M1 epic (GitHub #918).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .expect("CARGO_MANIFEST_DIR has repo root 2 ancestors up")
        .to_path_buf()
}

fn load_aliases() -> BTreeMap<String, String> {
    let path = repo_root().join("configs").join("aliases.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-002: cannot read {path:?}: {e}"));
    serde_yaml::from_str::<BTreeMap<String, String>>(&text)
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-002: aliases.yaml not a str→str map: {e}"))
}

#[test]
fn falsify_crux_a_01_002_aliases_yaml_present_and_parseable() {
    let map = load_aliases();
    assert!(
        !map.is_empty(),
        "FALSIFY-CRUX-A-01-002: aliases.yaml parsed empty"
    );
}

#[test]
fn falsify_crux_a_01_002_canonical_short_names_present() {
    let map = load_aliases();
    for canonical in ["llama3", "mistral", "phi3", "qwen2"] {
        assert!(
            map.contains_key(canonical),
            "FALSIFY-CRUX-A-01-002: canonical short name '{canonical}' missing from aliases.yaml"
        );
    }
}

#[test]
fn falsify_crux_a_01_002_every_value_has_known_scheme() {
    let map = load_aliases();
    for (name, url) in &map {
        assert!(
            url.starts_with("hf://") || url.starts_with("https://"),
            "FALSIFY-CRUX-A-01-002: alias '{name}' → '{url}' does not start with hf:// or https://"
        );
    }
}

#[test]
fn falsify_crux_a_01_002_resolution_deterministic() {
    let a = load_aliases();
    let b = load_aliases();
    assert_eq!(
        a, b,
        "FALSIFY-CRUX-A-01-002: two reads of aliases.yaml must yield byte-identical maps"
    );
}
