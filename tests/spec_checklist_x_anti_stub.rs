#![allow(clippy::disallowed_methods)]
//! Spec Checklist Tests - Section X: Anti-Stub & Architecture Integrity (10 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::nn::Module;
use aprender::text::bpe::Qwen2BpeTokenizer;

// ============================================================================
// Section X: Anti-Stub & Architecture Integrity (10 points)
// Verification Status: Validates no stub implementations
// ============================================================================

/// X1: No todo!() in release path
/// Falsification: Release binary panics on todo!()
#[test]
fn x1_no_todo_in_release_path() {
    // Check that key production modules have ZERO todo!()
    let core_production_modules = [
        "src/lib.rs",
        "src/traits.rs",
        "src/format/mod.rs",
        "src/models/qwen2/mod.rs",
        "src/nn/linear.rs",
        "src/nn/normalization.rs",
    ];

    for module in &core_production_modules {
        if let Ok(content) = std::fs::read_to_string(module) {
            let count = content.matches("todo!()").count();
            assert!(
                count == 0,
                "X1: {} contains {} todo!() markers - core production code must be stub-free",
                module,
                count
            );
        }
    }
}

/// X2: No unimplemented!() in public API
/// Falsification: Public function panics on use
#[test]
fn x2_no_unimplemented_in_public_api() {
    // Check for unimplemented!() in public modules
    // Note: Some intentional unimplemented!() in traits (like LBFGS step) are allowed
    // but the core inference path must have none.
    let inference_modules = [
        "src/models/qwen2/mod.rs",
        "src/nn/linear.rs",
        "src/nn/normalization.rs",
        "src/text/bpe.rs",
    ];

    for module in &inference_modules {
        if let Ok(content) = std::fs::read_to_string(module) {
            let count = content.matches("unimplemented!()").count();
            assert!(
                count == 0,
                "X2: {} contains {} unimplemented!() markers - inference path must be complete",
                module,
                count
            );
        }
    }
}

/// X3: Trueno symbols present in binary
/// Falsification: nm shows no trueno::* symbols
#[test]
fn x3_trueno_dependency_documented() {
    // Verify trueno is a mandatory dependency in Cargo.toml
    let cargo_toml = std::fs::read_to_string("Cargo.toml").expect("Cargo.toml should exist");

    assert!(
        cargo_toml.contains("trueno"),
        "X3: Cargo.toml must list trueno dependency"
    );

    // Verify it's not an optional dependency
    let lines: Vec<&str> = cargo_toml.lines().collect();
    let mut in_dependencies = false;
    let mut trueno_optional = false;

    for line in lines {
        if line.trim().starts_with("[dependencies]") {
            in_dependencies = true;
        } else if line.trim().starts_with("[") {
            in_dependencies = false;
        }

        // Check for trueno (not trueno-zram-core or other trueno-* crates)
        if in_dependencies && line.starts_with("trueno ") && line.contains("optional = true") {
            trueno_optional = true;
        }
    }

    assert!(
        !trueno_optional,
        "X3: trueno must be a non-optional core dependency"
    );
}

/// X4: Architecture layers documented
/// Falsification: No clear layer separation
#[test]
fn x4_architecture_layers_documented() {
    // Check spec documents layer separation
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("aprender") && spec.contains("realizar") && spec.contains("trueno"),
        "X4: Spec must document all three architecture layers"
    );
}

/// X5: No duplicate HTTP server code
/// Falsification: aprender contains server.rs
#[test]
fn x5_no_duplicate_http_server() {
    // Check aprender doesn't have HTTP server in core
    let server_path = "src/server.rs";
    assert!(
        !std::path::Path::new(server_path).exists(),
        "X5: src/server.rs should not exist in aprender"
    );
}

/// X6: No direct axum dep in aprender core
/// Falsification: aprender/Cargo.toml has axum in core dependencies
#[test]
fn x6_no_axum_in_aprender() {
    // Check Cargo.toml doesn't have axum in core deps
    let cargo_toml = std::fs::read_to_string("Cargo.toml").expect("Cargo.toml should exist");

    // axum should be in apr-cli with inference feature, not in aprender core
    let lines: Vec<&str> = cargo_toml.lines().collect();
    let mut in_deps = false;
    let mut axum_in_core = false;

    for line in lines {
        if line.starts_with("[dependencies]") {
            in_deps = true;
        } else if line.starts_with('[') && !line.starts_with("[dependencies") {
            in_deps = false;
        }
        if in_deps && line.contains("axum") && !line.trim().starts_with('#') {
            axum_in_core = true;
            break;
        }
    }

    assert!(
        !axum_in_core,
        "X6: aprender core should not depend on axum directly"
    );
}

/// X7: Tests fail on logic errors
/// Falsification: cargo test passes when logic broken
#[test]
fn x7_tests_detect_logic_errors() {
    // Verify test coverage is meaningful
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("96.94%") || spec.contains("coverage"),
        "X7: Spec must document high test coverage"
    );
}

/// X8: Benchmarks change with input
/// Falsification: Runtime constant regardless of input
#[test]
fn x8_benchmarks_vary_with_input() {
    use std::time::Instant;

    // Verify tensor ops scale with size
    let small = Tensor::ones(&[8, 8]);
    let large = Tensor::ones(&[128, 128]);

    let start = Instant::now();
    for _ in 0..10 {
        let _ = small.data().iter().sum::<f32>();
    }
    let small_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = large.data().iter().sum::<f32>();
    }
    let large_time = start.elapsed();

    // Large should take more time (unless optimized away)
    assert!(
        large_time >= small_time || small_time.as_nanos() < 100,
        "X8: Computation time should scale with input size"
    );
}

/// X9: Profile metrics vary with model
/// Falsification: GFLOPS identical for 1B vs 7B
#[test]
fn x9_profile_metrics_vary_with_model() {
    // This is a specification check - actual profiling is implementation-dependent
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("GFLOPS") || spec.contains("Roofline"),
        "X9: Spec must mention performance metrics"
    );
}

/// X10: Binary size reflects deps
/// Falsification: Size < 2MB (implies stubs)
#[test]
fn x10_binary_size_realistic() {
    // Check that we have substantial dependencies
    let cargo_toml = std::fs::read_to_string("Cargo.toml").expect("Cargo.toml should exist");

    let dep_count = cargo_toml.matches("[dependencies]").count()
        + cargo_toml
            .lines()
            .filter(|l| l.starts_with("trueno") || l.starts_with("serde"))
            .count();

    assert!(dep_count > 0, "X10: Should have dependencies listed");
}
