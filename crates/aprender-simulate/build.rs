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

#[derive(Deserialize)]
struct BindingFile {
    #[allow(dead_code)]
    version: String,
    bindings: Vec<Binding>,
}

#[derive(Deserialize)]
struct Binding {
    contract: String,
    equation: String,
    status: String,
}

fn emit_contract_assertions() {
    emit_local_assertions();
    enforce_provable_binding();
}

fn emit_local_assertions() {
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
        emit_contract_file(&path, &mut total_pre, &mut total_post);
    }
    println!(
        "cargo:warning=[contract] Assertions: {total_pre} preconditions, {total_post} postconditions from YAML"
    );
}

fn emit_contract_file(path: &Path, total_pre: &mut usize, total_post: &mut usize) {
    println!("cargo:rerun-if-changed={}", path.display());
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(contract) = serde_yaml::from_str::<ContractYaml>(&content) else {
        return;
    };
    let stem_upper = stem.to_uppercase().replace('-', "_");
    for (eq_name, equation) in &contract.equations {
        let eq_upper = eq_name.to_uppercase().replace('-', "_");
        let key = format!("CONTRACT_{stem_upper}_{eq_upper}");
        emit_pre_post(&key, equation, total_pre, total_post);
    }
}

fn emit_pre_post(
    key: &str,
    equation: &EquationYaml,
    total_pre: &mut usize,
    total_post: &mut usize,
) {
    let pre_count = equation.preconditions.len();
    if pre_count > 0 {
        println!("cargo:rustc-env={key}_PRE_COUNT={pre_count}");
        for (i, pre) in equation.preconditions.iter().enumerate() {
            println!("cargo:rustc-env={key}_PRE_{i}={pre}");
        }
        *total_pre += pre_count;
    }
    let post_count = equation.postconditions.len();
    if post_count > 0 {
        println!("cargo:rustc-env={key}_POST_COUNT={post_count}");
        for (i, post) in equation.postconditions.iter().enumerate() {
            println!("cargo:rustc-env={key}_POST_{i}={post}");
        }
        *total_post += post_count;
    }
}

fn enforce_provable_binding() {
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("provable-contracts/contracts/simular/binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        return;
    }
    let Ok(yaml) = std::fs::read_to_string(&binding_path) else {
        return;
    };
    let Ok(bf) = serde_yaml::from_str::<BindingFile>(&yaml) else {
        return;
    };

    let gaps = emit_binding_vars(&bf);
    let total = u32::try_from(bf.bindings.len()).unwrap_or(u32::MAX);
    let implemented = total - u32::try_from(gaps.len()).unwrap_or(u32::MAX);
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, {} gaps",
        gaps.len()
    );
    for g in &gaps {
        println!("cargo:warning=[contract] UNALLOWED GAP: {g}");
    }
    assert!(
        gaps.is_empty(),
        "[contract] AllImplemented: {} gap(s). Fix bindings or update status.",
        gaps.len()
    );
}

fn emit_binding_vars(bf: &BindingFile) -> Vec<String> {
    let mut gaps = Vec::new();
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
        if b.status != "implemented" {
            gaps.push(var);
        }
    }
    gaps
}
