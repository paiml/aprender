//! Integration tests for probar property test generation from contract YAML files.
//!
//! These tests verify that the full pipeline (parse → generate) produces
//! correct probar property tests mapping obligation types to test patterns.

// Test/example/bench binary: `.unwrap()` is idiomatic here; the lib`s
// cfg(test) allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use std::path::Path;

use provable_contracts::probar_gen::generate_probar_tests;
use provable_contracts::schema::{is_contract_yaml, parse_contract};

fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .canonicalize()
        .expect("contracts directory must exist")
}

/// Every contract document directly under `contracts/`.
///
/// Filtered by `provable_contracts::schema::is_contract_yaml`, the same
/// predicate `pv lint`'s walker uses. This file used to carry its own copy that
/// did not skip `contracts/binding.yaml` — a `BindingRegistry`, not a contract —
/// so it panicked on ``missing field `metadata` ``. That was the fourth copy of
/// one walker in the tree and the third to have the same bug.
fn all_contract_paths() -> Vec<std::path::PathBuf> {
    let dir = contracts_dir();
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("Cannot read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            is_contract_yaml(&path).then_some(path)
        })
        .collect();
    paths.sort();
    assert!(
        paths.len() > 100,
        "all_contract_paths() found only {} files under {} — a walker that finds \
         nothing passes every test in this file vacuously",
        paths.len(),
        dir.display()
    );
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
    // A contract with no proof obligations and no falsification tests has
    // nothing to generate a property test FROM, and the generator correctly
    // emits nothing. `contracts/apr-antigravity-parity-v1.yaml` is the live
    // example: its four checks sit under a top-level `falsification_conditions:`
    // key, so the typed contract carries zero obligations, zero falsification
    // tests and zero equations. Asserting unconditionally made this whole test
    // fail on the FIRST path it walked (alphabetically first), so the 1226
    // contracts behind it were never examined — and the target is dark, so the
    // red was never seen either.
    let mut generated = 0usize;
    for path in &paths {
        let contract = parse_contract(path)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
        let code = generate_probar_tests(&contract);
        let name = path.file_name().unwrap().to_str().unwrap();
        if contract.proof_obligations.is_empty() && contract.falsification_tests.is_empty() {
            assert!(
                code.trim().is_empty() || !code.contains("#[test]"),
                "{name} has nothing to generate from, so it must not emit tests"
            );
            continue;
        }
        assert!(
            code.contains("#[cfg(test)]"),
            "{name} should generate cfg(test) module"
        );
        assert!(
            code.contains("#[test]"),
            "{name} should contain test functions"
        );
        generated += 1;
    }
    assert!(
        generated >= 500,
        "only {generated} contracts generated probar tests — a `continue` that \
         swallows everything would make the loop above pass vacuously"
    );
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
