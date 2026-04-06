//! Integration tests for probar property test generation from contract YAML files.
//!
//! These tests verify that the full pipeline (parse → generate) produces
//! correct probar property tests mapping obligation types to test patterns.

use std::path::Path;

use provable_contracts::probar_gen::generate_probar_tests;
use provable_contracts::schema::parse_contract;

fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .canonicalize()
        .expect("contracts directory must exist")
}

fn all_contract_paths() -> Vec<std::path::PathBuf> {
    let dir = contracts_dir();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                && !path.file_name().unwrap().to_str().unwrap().starts_with('.')
            {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths
}

fn load_and_generate(path: &Path) -> String {
    let contract =
        parse_contract(path).unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    generate_probar_tests(&contract)
}

// --- Softmax: invariant and equivalence obligations ---

#[test]
fn softmax_generates_probar_module() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("#[cfg(test)]"));
    assert!(code.contains("mod probar_tests"));
}

#[test]
fn softmax_maps_invariant_obligations() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("Pattern: invariant"));
}

#[test]
fn softmax_generates_falsification_stubs() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("Falsification test stubs"));
}

// --- Matmul: bound obligations ---

#[test]
fn matmul_generates_bound_tests() {
    let code = load_and_generate(&contracts_dir().join("matmul-kernel-v1.yaml"));
    assert!(code.contains("#[cfg(test)]"));
    assert!(code.contains("Pattern: bound"));
}

// --- Attention: various obligation types ---

#[test]
fn attention_generates_probar_tests() {
    let code = load_and_generate(&contracts_dir().join("attention-kernel-v1.yaml"));
    assert!(code.contains("#[cfg(test)]"));
    assert!(code.contains("mod probar_tests"));
}

// --- RMSNorm: invariant and equivalence ---

#[test]
fn rmsnorm_generates_equivalence_tests() {
    let code = load_and_generate(&contracts_dir().join("rmsnorm-kernel-v1.yaml"));
    assert!(code.contains("#[cfg(test)]"));
}

// --- Cross-contract structural tests ---

#[test]
fn all_contracts_generate_valid_probar_output() {
    let paths = all_contract_paths();
    assert!(
        paths.len() >= 41,
        "Expected at least 41 contracts, found {}",
        paths.len()
    );
    for path in &paths {
        let code = load_and_generate(path);
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            code.contains("#[cfg(test)]"),
            "{name} should generate cfg(test) module"
        );
        assert!(
            code.contains("#[test]"),
            "{name} should contain test functions"
        );
    }
}

#[test]
fn contracts_with_obligations_generate_property_tests() {
    let dir = contracts_dir();
    let contracts = [
        "softmax-kernel-v1.yaml",
        "rmsnorm-kernel-v1.yaml",
        "matmul-kernel-v1.yaml",
        "attention-kernel-v1.yaml",
    ];

    for name in &contracts {
        let code = load_and_generate(&dir.join(name));
        assert!(
            code.contains("proof obligations"),
            "{name} should have property tests from proof obligations"
        );
        assert!(
            code.contains("// Pattern:"),
            "{name} should show pattern type"
        );
    }
}

#[test]
fn contracts_with_falsification_tests_generate_stubs() {
    let dir = contracts_dir();
    let contracts = ["softmax-kernel-v1.yaml", "rmsnorm-kernel-v1.yaml"];

    for name in &contracts {
        let code = load_and_generate(&dir.join(name));
        assert!(
            code.contains("Falsification test stubs"),
            "{name} should have falsification test stubs"
        );
    }
}
