//! Build script for simular
//! Captures build environment for reproducibility
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    // Capture build metadata for reproducibility verification
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=rust-toolchain.toml");

    // Embed version information
    if let Ok(version) = std::env::var("CARGO_PKG_VERSION") {
        println!("cargo:rustc-env=SIMULAR_VERSION={version}");
    }

    // Capture git hash for reproducibility
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
    {
        if let Ok(hash) = String::from_utf8(output.stdout) {
            println!("cargo:rustc-env=GIT_HASH={}", hash.trim());
        }
    }

    // Capture build timestamp (ISO 8601)
    println!(
        "cargo:rustc-env=BUILD_TIMESTAMP={}",
        chrono_lite_timestamp()
    );

    // Emit contract assertions from YAML
    emit_contract_assertions();
}

/// Simple ISO 8601 timestamp without external crate
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Approximate UTC timestamp (not leap-second accurate, but sufficient)
    format!("{secs}")
}

#[derive(Deserialize, Default)]
struct ContractYaml {
    #[serde(default)]
    equations: BTreeMap<String, EquationYaml>,
}

#[derive(Deserialize, Default)]
struct EquationYaml {
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    postconditions: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    lean_theorem: Option<String>,
}

fn emit_contract_assertions() {
    let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
    if !contracts_dir.exists() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(&contracts_dir) else {
        return;
    };
    let mut total_pre = 0usize;
    let mut total_post = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(contract) = serde_yaml::from_str::<ContractYaml>(&content) else {
            continue;
        };
        let stem_upper = stem.to_uppercase().replace('-', "_");
        for (eq_name, equation) in &contract.equations {
            let eq_upper = eq_name.to_uppercase().replace('-', "_");
            let key = format!("CONTRACT_{stem_upper}_{eq_upper}");
            let pre_count = equation.preconditions.len();
            if pre_count > 0 {
                println!("cargo:rustc-env={key}_PRE_COUNT={pre_count}");
                for (i, pre) in equation.preconditions.iter().enumerate() {
                    println!("cargo:rustc-env={key}_PRE_{i}={pre}");
                }
                total_pre += pre_count;
            }
            let post_count = equation.postconditions.len();
            if post_count > 0 {
                println!("cargo:rustc-env={key}_POST_COUNT={post_count}");
                for (i, post) in equation.postconditions.iter().enumerate() {
                    println!("cargo:rustc-env={key}_POST_{i}={post}");
                }
                total_post += post_count;
            }
        }
    }
    println!("cargo:warning=[contract] Assertions: {total_pre} preconditions, {total_post} postconditions from YAML");

    // ── provable-contracts binding enforcement (AllImplemented) ──
    {
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("provable-contracts/contracts/simular/binding.yaml");

        println!("cargo:rerun-if-changed={}", binding_path.display());

        if binding_path.exists() {
            #[derive(serde::Deserialize)]
            struct BF {
                #[allow(dead_code)]
                version: String,
                bindings: Vec<B>,
            }
            #[derive(serde::Deserialize)]
            struct B {
                contract: String,
                equation: String,
                status: String,
            }

            if let Ok(yaml) = std::fs::read_to_string(&binding_path) {
                if let Ok(bf) = serde_yaml::from_str::<BF>(&yaml) {
                    let (mut imp, mut gaps) = (0u32, Vec::new());
                    for b in &bf.bindings {
                        let var = format!(
                            "CONTRACT_{}_{}",
                            b.contract
                                .trim_end_matches(".yaml")
                                .to_uppercase()
                                .replace('-', "_"),
                            b.equation.to_uppercase().replace('-', "_")
                        );
                        println!("cargo:rustc-env={var}={}", b.status);
                        if b.status == "implemented" {
                            imp += 1;
                        } else {
                            gaps.push(var.clone());
                        }
                    }
                    let total = u32::try_from(bf.bindings.len()).unwrap_or(u32::MAX);
                    println!("cargo:warning=[contract] AllImplemented: {imp}/{total} implemented, {} gaps", gaps.len());
                    if !gaps.is_empty() {
                        for g in &gaps {
                            println!("cargo:warning=[contract] UNALLOWED GAP: {g}");
                        }
                    }
                    assert!(
                        gaps.is_empty(),
                        "[contract] AllImplemented: {} gap(s). Fix bindings or update status.",
                        gaps.len()
                    );
                }
            }
        }
    }
}
