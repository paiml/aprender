// build.rs — Read contracts/*.yaml and provable-contracts binding.yaml,
// emit CONTRACT_* env vars for compile-time assertion checking.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(serde::Deserialize, Default)]
struct ContractYaml {
    #[serde(default)]
    equations: BTreeMap<String, EquationYaml>,
}

#[derive(serde::Deserialize, Default)]
struct EquationYaml {
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    postconditions: Vec<String>,
}

#[derive(serde::Deserialize)]
struct BindingFile {
    #[allow(dead_code)]
    version: String,
    #[allow(dead_code)]
    target_crate: String,
    bindings: Vec<Binding>,
}

#[derive(serde::Deserialize)]
struct Binding {
    contract: String,
    equation: String,
    status: String,
}

fn main() {
    emit_local_contract_assertions();
    emit_presentar_binding_env();
}

fn emit_local_contract_assertions() {
    let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
    let Ok(entries) = std::fs::read_dir(&contracts_dir) else {
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
    let Ok(parsed) = serde_yaml_ng::from_str::<ContractYaml>(&contents) else {
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

fn emit_presentar_binding_env() {
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("presentar")
        .join("binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        println!(
            "cargo:warning=[contract] provable-contracts binding.yaml not found at {}; skipping",
            binding_path.display()
        );
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    }

    let Ok(content) = std::fs::read_to_string(&binding_path) else {
        println!("cargo:warning=[contract] Failed to read binding.yaml");
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    };
    let Ok(bf) = serde_yaml_ng::from_str::<BindingFile>(&content) else {
        println!("cargo:warning=[contract] Failed to parse binding.yaml");
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    };

    emit_binding_counts_and_vars(&bf);
    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
}

fn emit_binding_counts_and_vars(bf: &BindingFile) {
    let mut implemented = 0u32;
    let mut partial = 0u32;
    let mut not_implemented = 0u32;
    for b in &bf.bindings {
        let var = env_var_name(&b.contract, &b.equation);
        println!("cargo:rustc-env={var}={}", b.status);
        match b.status.as_str() {
            "implemented" => implemented += 1,
            "partial" => partial += 1,
            _ => not_implemented += 1,
        }
    }
    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps"
    );
    if not_implemented > 0 {
        println!(
            "cargo:warning=[contract] AllImplemented: {not_implemented} \
             binding(s) are not_implemented — implement before next release"
        );
    }
}

fn env_var_name(contract: &str, equation: &str) -> String {
    let stem = contract
        .trim_end_matches(".yaml")
        .trim_end_matches(".yml")
        .to_uppercase()
        .replace('-', "_");
    let eq = equation.to_uppercase().replace('-', "_");
    format!("CONTRACT_{stem}_{eq}")
}
