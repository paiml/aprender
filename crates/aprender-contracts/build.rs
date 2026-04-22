// build.rs — provable-contracts binding + PRE/POST enforcement (L1)
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct BindingFile {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    target_crate: String,
    bindings: Vec<Binding>,
}

#[derive(Deserialize)]
struct Binding {
    contract: String,
    equation: String,
    status: String,
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
}

fn main() {
    let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("contracts");
    emit_binding_env(&contracts_dir);
    emit_pre_post_env(&contracts_dir);
}

fn emit_binding_env(contracts_dir: &Path) {
    let mut total_bindings = 0u32;
    let mut total_implemented = 0u32;
    let mut found_any = false;

    if let Ok(entries) = std::fs::read_dir(contracts_dir) {
        for entry in entries.flatten() {
            let binding_path = entry.path().join("binding.yaml");
            if !binding_path.exists() {
                continue;
            }
            println!("cargo:rerun-if-changed={}", binding_path.display());
            let Some(bindings) = load_binding_file(&binding_path) else {
                continue;
            };
            found_any = true;
            emit_binding_vars(&bindings, &mut total_bindings, &mut total_implemented);
        }
    }

    if found_any {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
        println!("cargo:rustc-env=CONTRACT_TOTAL={total_bindings}");
        println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={total_implemented}");
    } else {
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
    }
}

fn load_binding_file(binding_path: &Path) -> Option<BindingFile> {
    let yaml = std::fs::read_to_string(binding_path).ok()?;
    serde_yaml::from_str(&yaml).ok()
}

fn emit_binding_vars(bindings: &BindingFile, total: &mut u32, implemented: &mut u32) {
    for b in &bindings.bindings {
        let stem = b
            .contract
            .trim_end_matches(".yaml")
            .to_uppercase()
            .replace('-', "_");
        let eq = b.equation.to_uppercase().replace('-', "_");
        let var = format!("CONTRACT_{stem}_{eq}");
        println!("cargo:rustc-env={var}={}", b.status);
        *total += 1;
        if b.status == "implemented" {
            *implemented += 1;
        }
    }
}

fn emit_pre_post_env(contracts_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(contracts_dir) else {
        return;
    };
    let (mut total_pre, mut total_post) = (0usize, 0usize);
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_contract_yaml(&path) {
            continue;
        }
        emit_contract_file_pre_post(&path, &mut total_pre, &mut total_post);
    }
    println!(
        "cargo:warning=[contract] Assertions: {total_pre} preconditions, {total_post} postconditions from YAML"
    );
}

fn is_contract_yaml(path: &Path) -> bool {
    if path.extension().and_then(|x| x.to_str()) != Some("yaml") {
        return false;
    }
    !path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().contains("binding"))
}

fn emit_contract_file_pre_post(path: &PathBuf, total_pre: &mut usize, total_post: &mut usize) {
    println!("cargo:rerun-if-changed={}", path.display());
    let stem = path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("x")
        .to_uppercase()
        .replace('-', "_");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(parsed) = serde_yaml::from_str::<ContractYaml>(&contents) else {
        return;
    };
    for (eq_name, eq) in &parsed.equations {
        let key = format!(
            "CONTRACT_{}_{}",
            stem,
            eq_name.to_uppercase().replace('-', "_")
        );
        emit_pre_post_lists(&key, eq, total_pre, total_post);
    }
}

fn emit_pre_post_lists(
    key: &str,
    eq: &EquationYaml,
    total_pre: &mut usize,
    total_post: &mut usize,
) {
    if !eq.preconditions.is_empty() {
        println!("cargo:rustc-env={key}_PRE_COUNT={}", eq.preconditions.len());
        for (i, v) in eq.preconditions.iter().enumerate() {
            println!("cargo:rustc-env={key}_PRE_{i}={v}");
        }
        *total_pre += eq.preconditions.len();
    }
    if !eq.postconditions.is_empty() {
        println!(
            "cargo:rustc-env={key}_POST_COUNT={}",
            eq.postconditions.len()
        );
        for (i, v) in eq.postconditions.iter().enumerate() {
            println!("cargo:rustc-env={key}_POST_{i}={v}");
        }
        *total_post += eq.postconditions.len();
    }
}
