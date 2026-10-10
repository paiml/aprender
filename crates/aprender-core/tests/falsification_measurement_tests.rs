#![allow(clippy::disallowed_methods)]
//! M001-M010: Measurement Tools (cbtop) Falsification Tests
//!
//! Per spec: docs/specifications/archive/qwen2.5-coder-showcase-demo.md S9.4
//!
//! These tests verify the cbtop measurement tool works correctly.
//! Measurement tools must be accurate and reliable for optimization.
//!
//! FALSIFICATION: If measurement is unreliable, optimization is impossible.

use std::process::Command;

/// `cargo` for the nested `cargo run -p apr-cli` builds below, minus this test
/// target's `CONTRACT_*` vars. cargo hands a package's build-script `rustc-env`
/// vars to that package's test processes, so the nested build inherited
/// aprender-core's `CONTRACT_BINDING_SOURCE=binding.yaml`. apr-cli's build emits
/// no binding registry, so every `#[contract]` site in it then failed the #4368
/// registry check and apr-cli did not compile (#5001).
fn nested_cargo() -> Command {
    let mut cmd = Command::new("cargo");
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("CONTRACT_") {
            cmd.env_remove(key);
        }
    }
    cmd
}

include!("includes/falsification_measurement.rs");
include!("includes/falsification_measurement_scoring.rs");
