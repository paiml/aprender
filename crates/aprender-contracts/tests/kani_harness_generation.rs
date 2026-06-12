//! Integration tests for Kani harness generation from contract YAML files.
//!
//! These tests verify that the full pipeline (parse → generate) produces
//! correct Kani harnesses for each strategy: exhaustive, `stub_float`,
//! and compositional.

// Test/example/bench binary: `.unwrap()` is idiomatic here; the lib`s
// cfg(test) allow does not reach this separate crate.
#![allow(clippy::disallowed_methods)]

use std::path::Path;

use provable_contracts::kani_gen::generate_kani_harnesses;
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
    generate_kani_harnesses(&contract)
}

// --- Softmax: stub_float strategy ---

#[test]
fn softmax_generates_cfg_kani_module() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.starts_with("#[cfg(kani)]"));
    assert!(code.contains("mod verification"));
    assert!(code.contains("use super::*;"));
}

#[test]
fn softmax_generates_all_harnesses() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("fn verify_softmax_normalization()"));
    assert!(code.contains("fn verify_softmax_positivity()"));
    assert!(code.contains("fn verify_softmax_bounded()"));
}

#[test]
fn softmax_harnesses_have_kani_attributes() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert_eq!(code.matches("#[kani::proof]").count(), 12);
    assert!(code.contains("#[kani::solver(cadical)]"));
}

#[test]
fn softmax_harnesses_use_stub_float_strategy() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("Vec<f32>"));
    assert!(code.contains("is_finite()"));
    assert!(code.contains("stub_float"));
}

#[test]
fn softmax_harnesses_have_doc_comments() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("/// KANI-SM-001: Softmax sums to 1.0"));
    assert!(code.contains("/// Obligation: SM-INV-001"));
    assert!(code.contains("/// Strategy: stub_float"));
    assert!(code.contains("/// Bound: 8 elements"));
}

// --- RMSNorm: mixed strategies ---

#[test]
fn rmsnorm_generates_harnesses() {
    let code = load_and_generate(&contracts_dir().join("rmsnorm-kernel-v1.yaml"));
    assert!(code.starts_with("#[cfg(kani)]"));
    assert!(code.contains("#[kani::proof]"));
}

// --- Matmul: exhaustive strategy ---

#[test]
fn matmul_generates_exhaustive_harnesses() {
    let code = load_and_generate(&contracts_dir().join("matmul-kernel-v1.yaml"));
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("Vec<i32>"));
    assert!(code.contains("exhaustive"));
}

// --- Attention: stub_float strategy ---

#[test]
fn attention_generates_stub_float_harnesses() {
    let code = load_and_generate(&contracts_dir().join("attention-kernel-v1.yaml"));
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("stub_float"));
}

// --- Flash Attention: stub_float strategy ---

#[test]
fn flash_attention_generates_harnesses() {
    let code = load_and_generate(&contracts_dir().join("flash-attention-v1.yaml"));
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("mod verification"));
}

// --- Activation: exhaustive strategy ---

#[test]
fn activation_generates_exhaustive_harnesses() {
    let code = load_and_generate(&contracts_dir().join("activation-kernel-v1.yaml"));
    assert!(code.contains("#[kani::proof]"));
    assert!(code.contains("exhaustive"));
    assert!(code.contains("Vec<i32>"));
}

// --- RoPE: exhaustive strategy ---

#[test]
fn rope_generates_exhaustive_harnesses() {
    let code = load_and_generate(&contracts_dir().join("rope-kernel-v1.yaml"));
    assert!(code.contains("#[kani::proof]"));
}

// --- Cross-contract structural tests ---

#[test]
fn all_contracts_generate_valid_kani_output() {
    let paths = all_contract_paths();
    assert!(
        paths.len() >= 41,
        "Expected at least 41 contracts, found {}",
        paths.len()
    );
    for path in &paths {
        let code = load_and_generate(path);
        let name = path.file_name().unwrap().to_str().unwrap();
        // Contracts without kani_harnesses produce a comment instead
        if code.starts_with("// No Kani") {
            continue;
        }
        assert!(
            code.starts_with("#[cfg(kani)]"),
            "{name} should generate cfg(kani) module"
        );
        assert!(
            code.contains("#[kani::proof]"),
            "{name} should contain kani::proof"
        );
        assert!(
            code.contains("kani::any()"),
            "{name} should use symbolic inputs"
        );
        assert!(
            code.ends_with("}\n"),
            "{name} should end with closing brace"
        );
    }
}

#[test]
fn harness_count_matches_contract_definitions() {
    let dir = contracts_dir();
    let contracts_and_counts = [
        ("softmax-kernel-v1.yaml", 12),
        ("matmul-kernel-v1.yaml", 6),
        ("activation-kernel-v1.yaml", 8),
    ];

    for (name, expected_count) in &contracts_and_counts {
        let code = load_and_generate(&dir.join(name));
        let actual = code.matches("#[kani::proof]").count();
        assert_eq!(
            actual, *expected_count,
            "{name}: expected {expected_count} harnesses, found {actual}"
        );
    }
}

// --- Concrete playback test ---

#[test]
fn generated_harness_contains_unimplemented_marker() {
    let code = load_and_generate(&contracts_dir().join("softmax-kernel-v1.yaml"));
    assert!(code.contains("unimplemented!"));
}
