//! `apr mono` — Monorepo management commands.
//!
//! Subcommands for managing the APR-MONO workspace:
//! - `apr mono publish` — Publish workspace crates in topological order
//! - `apr mono shims` — Generate backward-compat shim crates
//! - `apr mono audit` — Verify workspace invariants (naming, layout, deps)
//! - `apr mono archive` — Generate archive commands for old repos
//!
//! Contract: contracts/publish-workspace-v1.yaml

use crate::error::CliError;
use clap::Subcommand;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::Command;

#[derive(Subcommand, Debug, Clone)]
pub enum MonoCommands {
    /// Publish workspace crates to crates.io in topological order
    Publish {
        /// Dry run — show what would be published without actually publishing
        #[arg(long, default_value_t = true)]
        dry_run: bool,
    },

    /// Generate backward-compat shim crates for old names
    Shims {
        /// Output directory for shim crates
        #[arg(long, default_value = "shims")]
        output: String,
    },

    /// Verify workspace invariants (FALSIFY-MONO-* and FALSIFY-BUILD-*)
    Audit,

    /// Generate GitHub API commands to archive old repos
    Archive {
        /// Actually execute the archive commands (default: print only)
        #[arg(long)]
        execute: bool,
    },
}

/// Topological publish order — leaves first, root last.
const PUBLISH_TIERS: &[(&str, &[&str])] = &[
    (
        "Tier 0: Zero-dep crates",
        &[
            "aprender-contracts-macros",
            "aprender-gemm-codegen",
            "aprender-quant",
            "aprender-rand",
            "aprender-fft",
            "aprender-sparse",
            "aprender-solve",
            "aprender-tensor",
            "aprender-image",
        ],
    ),
    (
        "Tier 1: Core compute",
        &[
            "aprender-gpu",
            "aprender-compute",
            "aprender-cuda-edge",
            "aprender-ptx-debug",
            "aprender-cupti",
            "aprender-cbtop",
            "aprender-cgp",
            "aprender-explain",
        ],
    ),
    (
        "Tier 2: Contracts + shared",
        &[
            "aprender-contracts",
            "aprender-contracts-cli",
            "aprender-profile-core",
            "aprender-profile",
        ],
    ),
    (
        "Tier 3: Data & Storage",
        &[
            "aprender-db",
            "aprender-graph",
            "aprender-rag",
            "aprender-data",
            "aprender-distribute",
        ],
    ),
    (
        "Tier 4: Visualization",
        &[
            "aprender-present-core",
            "aprender-present-terminal",
            "aprender-present-widgets",
            "aprender-present-layout",
            "aprender-present-yaml",
            "aprender-present-cli",
            "aprender-present-lib",
            "aprender-present-test",
            "aprender-present-test-macros",
            "aprender-viz",
        ],
    ),
    (
        "Tier 5: Test framework",
        &[
            "aprender-test-derive",
            "aprender-test-js-gen",
            "aprender-test-lib",
            "aprender-test-cli",
            "aprender-test-showcase",
        ],
    ),
    (
        "Tier 6: Compressed memory",
        &[
            "aprender-zram-core",
            "aprender-zram-adaptive",
            "aprender-zram",
            "aprender-zram-cli",
            "aprender-zram-generator",
        ],
    ),
    (
        "Tier 7: ML library",
        &[
            "aprender-core",
            "aprender-verify",
            "aprender-verify-ml",
            "aprender-simulate",
            "aprender-registry",
        ],
    ),
    (
        "Tier 8: Training",
        &[
            "aprender-train-common",
            "aprender-train-lora",
            "aprender-train-distill",
            "aprender-train-inspect",
            "aprender-train-shell",
            "aprender-train-bench",
            "aprender-train-wasm",
            "aprender-train",
        ],
    ),
    (
        "Tier 9: Serving + Orchestration",
        &["aprender-serve", "aprender-orchestrate"],
    ),
    ("Tier 10: CLI + Root", &["apr-cli", "aprender"]),
];

/// Old → New crate name mappings for shim generation.
const SHIM_MAP: &[(&str, &str, &str, &str)] = &[
    // (old_name, new_name, lib_name, shim_version)
    ("trueno", "aprender-compute", "trueno", "0.19.0"),
    ("trueno-gpu", "aprender-gpu", "trueno_gpu", "0.5.0"),
    ("trueno-quant", "aprender-quant", "trueno_quant", "0.2.0"),
    (
        "trueno-explain",
        "aprender-explain",
        "trueno_explain",
        "0.3.0",
    ),
    ("trueno-fft", "aprender-fft", "trueno_fft", "0.2.0"),
    ("trueno-sparse", "aprender-sparse", "trueno_sparse", "0.2.0"),
    ("trueno-solve", "aprender-solve", "trueno_solve", "0.2.0"),
    ("trueno-rand", "aprender-rand", "trueno_rand", "0.2.0"),
    ("trueno-image", "aprender-image", "trueno_image", "0.2.0"),
    ("trueno-tensor", "aprender-tensor", "trueno_tensor", "0.2.0"),
    (
        "trueno-cuda-edge",
        "aprender-cuda-edge",
        "trueno_cuda_edge",
        "0.2.0",
    ),
    ("trueno-db", "aprender-db", "trueno_db", "0.4.0"),
    ("trueno-graph", "aprender-graph", "trueno_graph", "0.2.0"),
    ("trueno-rag", "aprender-rag", "trueno_rag", "0.3.0"),
    ("trueno-viz", "aprender-viz", "trueno_viz", "0.3.0"),
    (
        "trueno-zram-core",
        "aprender-zram-core",
        "trueno_zram_core",
        "0.4.0",
    ),
    (
        "trueno-zram-adaptive",
        "aprender-zram-adaptive",
        "trueno_zram_adaptive",
        "0.4.0",
    ),
    ("cbtop", "aprender-cbtop", "cbtop", "0.2.0"),
    ("entrenar", "aprender-train", "entrenar", "0.8.0"),
    (
        "entrenar-common",
        "aprender-train-common",
        "entrenar_common",
        "0.2.0",
    ),
    (
        "entrenar-lora",
        "aprender-train-lora",
        "entrenar_lora",
        "0.4.0",
    ),
    ("realizar", "aprender-serve", "realizar", "0.9.0"),
    ("batuta", "aprender-orchestrate", "batuta", "0.8.0"),
    ("renacer", "aprender-profile", "renacer", "0.11.0"),
    (
        "renacer-core",
        "aprender-profile-core",
        "renacer_core",
        "0.2.0",
    ),
    ("certeza", "aprender-verify", "certeza", "0.2.0"),
    ("verificar", "aprender-verify-ml", "verificar", "0.6.0"),
    ("simular", "aprender-simulate", "simular", "0.4.0"),
    ("repartir", "aprender-distribute", "repartir", "2.1.0"),
    ("alimentar", "aprender-data", "alimentar", "0.3.0"),
    ("pacha", "aprender-registry", "pacha", "0.3.0"),
    (
        "provable-contracts",
        "aprender-contracts",
        "provable_contracts",
        "0.4.0",
    ),
    (
        "provable-contracts-macros",
        "aprender-contracts-macros",
        "provable_contracts_macros",
        "0.4.0",
    ),
    (
        "presentar-core",
        "aprender-present-core",
        "presentar_core",
        "0.4.0",
    ),
    (
        "presentar-terminal",
        "aprender-present-terminal",
        "presentar_terminal",
        "0.4.0",
    ),
    ("jugar-probar", "aprender-test-lib", "jugar_probar", "1.1.0"),
];

/// Repos to archive after migration.
const ARCHIVE_REPOS: &[(&str, &str)] = &[
    ("paiml/trueno", "crates/aprender-compute/"),
    ("paiml/entrenar", "crates/aprender-train/"),
    ("paiml/realizar", "crates/aprender-serve/"),
    ("paiml/Batuta", "crates/aprender-orchestrate/"),
    ("paiml/provable-contracts", "crates/aprender-contracts/"),
    ("paiml/presentar", "crates/aprender-present-*/"),
    ("paiml/renacer", "crates/aprender-profile/"),
    ("paiml/certeza", "crates/aprender-verify/"),
    ("paiml/trueno-db", "crates/aprender-db/"),
    ("paiml/trueno-graph", "crates/aprender-graph/"),
    ("paiml/trueno-rag", "crates/aprender-rag/"),
    ("paiml/trueno-viz", "crates/aprender-viz/"),
    ("paiml/trueno-zram", "crates/aprender-zram/"),
    ("paiml/alimentar", "crates/aprender-data/"),
    ("paiml/simular", "crates/aprender-simulate/"),
    ("paiml/verificar", "crates/aprender-verify-ml/"),
    ("paiml/repartir", "crates/aprender-distribute/"),
    ("paiml/probar", "crates/aprender-test-*/"),
    ("paiml/pacha", "crates/aprender-registry/"),
    ("paiml/batuta-common", "crates/aprender-orchestrate/"),
];

pub fn run(command: MonoCommands) -> Result<(), CliError> {
    match command {
        MonoCommands::Publish { dry_run } => run_publish(dry_run),
        MonoCommands::Shims { output } => run_shims(&output),
        MonoCommands::Audit => run_audit(),
        MonoCommands::Archive { execute } => run_archive(execute),
    }
}

fn run_publish(dry_run: bool) -> Result<(), CliError> {
    if dry_run {
        println!("=== DRY RUN — showing topological publish order ===\n");
    }

    let mut total = 0;
    for (tier_name, crates) in PUBLISH_TIERS {
        println!("--- {tier_name} ---");
        for krate in *crates {
            if dry_run {
                // Verify the crate can be packaged
                let status = Command::new("cargo")
                    .args(["package", "-p", krate, "--allow-dirty", "--list"])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
                let ok = status.map_or(false, |s| s.success());
                let marker = if ok { "OK" } else { "FAIL" };
                println!("  [{marker}] cargo publish -p {krate}");
            } else {
                println!("  Publishing: {krate}");
                let status = Command::new("cargo")
                    .args(["publish", "-p", krate, "--allow-dirty"])
                    .status()
                    .map_err(|e| CliError::Aprender(format!("cargo publish failed: {e}")))?;
                if !status.success() {
                    return Err(CliError::Aprender(format!(
                        "FALSIFY-PUB-001: cargo publish -p {krate} failed (exit {})",
                        status.code().unwrap_or(-1)
                    )));
                }
                // Wait for crates.io index propagation
                std::thread::sleep(std::time::Duration::from_secs(15));
            }
            total += 1;
        }
    }

    println!("\n{total} crates in topological order.");
    if dry_run {
        println!("Run without --dry-run to publish for real.");
    }
    Ok(())
}

fn run_shims(output_dir: &str) -> Result<(), CliError> {
    let output = Path::new(output_dir);
    std::fs::create_dir_all(output)
        .map_err(|e| CliError::Aprender(format!("failed to create {output_dir}: {e}")))?;

    let mut count = 0;
    for (old_name, new_name, lib_name, shim_version) in SHIM_MAP {
        let shim_path = output.join(old_name);
        std::fs::create_dir_all(shim_path.join("src"))
            .map_err(|e| CliError::Aprender(format!("mkdir failed: {e}")))?;

        // Cargo.toml
        let cargo_toml = format!(
            r#"[package]
name = "{old_name}"
version = "{shim_version}"
edition = "2021"
license = "MIT"
description = "DEPRECATED: Use {new_name} instead. Re-exports {new_name} for backward compatibility."
repository = "https://github.com/paiml/aprender"
keywords = ["deprecated", "moved", "aprender"]

[dependencies]
{new_name} = "0.29"
"#
        );
        std::fs::write(shim_path.join("Cargo.toml"), cargo_toml)
            .map_err(|e| CliError::Aprender(format!("write failed: {e}")))?;

        // lib.rs
        let lib_rs = format!(
            r#"//! `{old_name}` has moved to `{new_name}`.
//!
//! This crate re-exports `{new_name}` for backward compatibility.
//! New code should depend on `{new_name}` directly.

pub use {lib_name}::*;
"#
        );
        std::fs::write(shim_path.join("src/lib.rs"), lib_rs)
            .map_err(|e| CliError::Aprender(format!("write failed: {e}")))?;

        count += 1;
    }

    println!("Generated {count} shim crates in {output_dir}/");
    println!("To publish: for d in {output_dir}/*/; do (cd \"$d\" && cargo publish); done");
    Ok(())
}

fn run_audit() -> Result<(), CliError> {
    println!("Running monorepo invariant checks...\n");

    // 1. Check workspace member count
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .map_err(|e| CliError::Aprender(format!("cargo metadata failed: {e}")))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let member_count = stdout.matches("\"source\":null").count();
    println!(
        "[{}] Workspace members: {member_count} (min: 60)",
        if member_count >= 60 { "PASS" } else { "FAIL" }
    );

    // 2. Check for non-aprender names
    let has_old_names = stdout.contains("\"name\":\"trueno\"")
        || stdout.contains("\"name\":\"realizar\"")
        || stdout.contains("\"name\":\"entrenar\"")
        || stdout.contains("\"name\":\"batuta\"");
    println!(
        "[{}] No old package names",
        if !has_old_names { "PASS" } else { "FAIL" }
    );

    // 3. Check no [patch.crates-io] in root Cargo.toml
    let root_toml = std::fs::read_to_string("Cargo.toml").unwrap_or_default();
    let has_patch = root_toml.contains("[patch.crates-io]");
    println!(
        "[{}] No [patch.crates-io] in root Cargo.toml",
        if !has_patch { "PASS" } else { "FAIL" }
    );

    // 4. Run integration tests
    println!("\nRunning integration tests...");
    let test_status = Command::new("cargo")
        .args([
            "test",
            "--test",
            "monorepo_invariants",
            "-p",
            "aprender-core",
        ])
        .status()
        .map_err(|e| CliError::Aprender(format!("test failed: {e}")))?;
    println!(
        "[{}] monorepo_invariants tests",
        if test_status.success() {
            "PASS"
        } else {
            "FAIL"
        }
    );

    let cli_test = Command::new("cargo")
        .args(["test", "--test", "cli_commands", "-p", "apr-cli"])
        .status()
        .map_err(|e| CliError::Aprender(format!("test failed: {e}")))?;
    println!(
        "[{}] cli_commands tests",
        if cli_test.success() { "PASS" } else { "FAIL" }
    );

    if !test_status.success() || !cli_test.success() {
        return Err(CliError::Aprender(
            "Audit failed — see test output above".into(),
        ));
    }

    println!("\nAll checks passed.");
    Ok(())
}

fn run_archive(execute: bool) -> Result<(), CliError> {
    println!("=== APR-MONO Phase 6: Archive Old Repositories ===\n");

    if !execute {
        println!("DRY RUN — showing commands. Use --execute to run them.\n");
    }

    for (repo, new_location) in ARCHIVE_REPOS {
        let cmd = format!("gh api -X PATCH repos/{repo} -f archived=true");
        if execute {
            println!("Archiving: {repo} → {new_location}");
            let status = Command::new("gh")
                .args([
                    "api",
                    "-X",
                    "PATCH",
                    &format!("repos/{repo}"),
                    "-f",
                    "archived=true",
                ])
                .status()
                .map_err(|e| CliError::Aprender(format!("gh api failed: {e}")))?;
            if !status.success() {
                eprintln!("  WARNING: Failed to archive {repo}");
            }
        } else {
            println!("  {cmd}");
            println!("    → {new_location}");
        }
    }

    println!("\n{} repos to archive.", ARCHIVE_REPOS.len());
    Ok(())
}
