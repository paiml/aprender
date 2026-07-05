// build.rs — Read provable-contracts binding.yaml and set CONTRACT_* env vars
//
// Policy: AllImplemented. Emits warnings for partial/not_implemented bindings
// but does NOT fail the build. Entrenar has 5 known gaps (GPU wait queue,
// QLora learning_rate_scaling) tracked via paiml/provable-contracts#11.
//
// The env vars follow the pattern:
//   CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>
//
// Example:
//   CONTRACT_LEARNING_RATE_SCHEDULES_V1_COSINE_WARMUP=implemented
//
// These are consumed at compile time by the #[contract] proc macro.

use std::path::Path;

use serde::Deserialize;

/// Minimal subset of the binding.yaml schema.
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
    #[serde(default)]
    notes: Option<String>,
}

/// Convert a contract filename + equation into a canonical env var name.
///
/// `"learning-rate-schedules-v1.yaml"` + `"cosine_warmup"` → `"CONTRACT_LEARNING_RATE_SCHEDULES_V1_COSINE_WARMUP"`
fn env_var_name(contract: &str, equation: &str) -> String {
    let stem = contract
        .trim_end_matches(".yaml")
        .trim_end_matches(".yml")
        .to_uppercase()
        .replace('-', "_");
    let eq = equation.to_uppercase().replace('-', "_");
    format!("CONTRACT_{stem}_{eq}")
}

/// Rank status values for deduplication: `implemented` > `partial` > `not_implemented`.
fn status_rank(s: &str) -> u8 {
    match s {
        "implemented" => 2,
        "partial" => 1,
        _ => 0,
    }
}

#[derive(serde::Deserialize, Default)]
struct ContractYaml {
    #[serde(default)]
    equations: std::collections::BTreeMap<String, EquationYaml>,
}

#[derive(serde::Deserialize, Default)]
struct EquationYaml {
    #[serde(default)]
    preconditions: Vec<String>,
    #[serde(default)]
    postconditions: Vec<String>,
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"__has_embedding_contract\"))");

    enforce_entrenar_binding();
    emit_local_contract_assertions();
}

fn enforce_entrenar_binding() {
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("entrenar")
        .join("binding.yaml");
    println!("cargo:rerun-if-changed={}", binding_path.display());

    let Some(bindings) = load_binding_file(&binding_path) else {
        return;
    };

    let deduped = dedupe_bindings(&bindings);
    let (implemented, partial, not_implemented, gaps) = emit_binding_vars(&bindings, &deduped);

    report_and_enforce_gaps(&bindings, implemented, partial, not_implemented, &gaps);
}

/// Load + parse the provable-contracts binding.yaml, emitting warnings + a
/// sentinel env var when the file is absent or malformed.
fn load_binding_file(binding_path: &Path) -> Option<BindingFile> {
    if !binding_path.exists() {
        println!(
            "cargo:warning=provable-contracts binding.yaml not found at {}; \
             CONTRACT_* env vars will not be set (CI/crates.io build)",
            binding_path.display()
        );
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return None;
    }

    let yaml_content = match std::fs::read_to_string(binding_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=Failed to read binding.yaml: {e}; \
                 CONTRACT_* env vars will not be set"
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return None;
        }
    };

    match serde_yaml_ng::from_str::<BindingFile>(&yaml_content) {
        Ok(b) => Some(b),
        Err(e) => {
            println!(
                "cargo:warning=Failed to parse binding.yaml: {e}; \
                 CONTRACT_* env vars will not be set"
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            None
        }
    }
}

/// Keep the best status (implemented > partial > not_implemented) per env var.
fn dedupe_bindings(bindings: &BindingFile) -> std::collections::HashMap<String, String> {
    let mut seen = std::collections::HashMap::<String, String>::new();
    for binding in &bindings.bindings {
        let var_name = env_var_name(&binding.contract, &binding.equation);
        let new_rank = status_rank(&binding.status);
        let dominated =
            seen.get(&var_name).is_some_and(|existing| status_rank(existing) >= new_rank);
        if !dominated {
            seen.insert(var_name, binding.status.clone());
        }
    }
    seen
}

/// Emit per-binding env vars + status-annotated warnings. Returns
/// (implemented, partial, not_implemented, gaps).
fn emit_binding_vars(
    bindings: &BindingFile,
    deduped: &std::collections::HashMap<String, String>,
) -> (u32, u32, u32, Vec<String>) {
    let mut implemented = 0u32;
    let mut partial = 0u32;
    let mut not_implemented = 0u32;
    let mut gaps: Vec<String> = Vec::new();

    let mut keys: Vec<_> = deduped.keys().cloned().collect();
    keys.sort();

    for var_name in &keys {
        let status = &deduped[var_name];
        println!("cargo:rustc-env={var_name}={status}");
        match status.as_str() {
            "implemented" => implemented += 1,
            "partial" => {
                partial += 1;
                emit_binding_note(bindings, var_name, "PARTIAL");
            }
            "not_implemented" => {
                not_implemented += 1;
                gaps.push(var_name.clone());
                emit_binding_note(bindings, var_name, "GAP");
            }
            other => {
                println!("cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}");
            }
        }
    }

    (implemented, partial, not_implemented, gaps)
}

fn emit_binding_note(bindings: &BindingFile, var_name: &str, label: &str) {
    let note = bindings
        .bindings
        .iter()
        .find(|b| env_var_name(&b.contract, &b.equation) == var_name)
        .and_then(|b| b.notes.as_deref())
        .unwrap_or("");
    println!("cargo:warning=[contract] {label}: {var_name} — {note}");
}

fn report_and_enforce_gaps(
    bindings: &BindingFile,
    implemented: u32,
    partial: u32,
    not_implemented: u32,
    gaps: &[String],
) {
    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps"
    );

    if not_implemented > 0 {
        for gap in gaps {
            println!("cargo:warning=[contract] UNALLOWED GAP: {gap}");
        }
        panic!(
            "[contract] AllImplemented policy violation: {not_implemented} binding(s) are \
             not_implemented. Fix: implement the binding or update binding.yaml status."
        );
    }

    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!("cargo:rustc-env=CONTRACT_BINDING_VERSION={}", bindings.version);
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
    println!("cargo:rustc-env=CONTRACT_PARTIAL={partial}");
    println!("cargo:rustc-env=CONTRACT_GAPS={not_implemented}");
}

/// Phase 2: PRE/POST env vars for the #[contract] proc macro.
fn emit_local_contract_assertions() {
    let cdir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
    let Ok(entries) = std::fs::read_dir(&cdir) else {
        return;
    };

    let (mut total_pre, mut total_post) = (0usize, 0usize);
    for entry in entries.flatten() {
        let path = entry.path();
        if !is_contract_yaml(&path) {
            continue;
        }
        emit_contract_file(&path, &mut total_pre, &mut total_post);
    }

    println!(
        "cargo:warning=[contract] Assertions: {total_pre} preconditions, {total_post} postconditions from YAML"
    );
}

fn is_contract_yaml(path: &Path) -> bool {
    if path.extension().and_then(|x| x.to_str()) != Some("yaml") {
        return false;
    }
    !path.file_name().is_some_and(|n| n.to_string_lossy().contains("binding"))
}

fn emit_contract_file(path: &Path, total_pre: &mut usize, total_post: &mut usize) {
    println!("cargo:rerun-if-changed={}", path.display());
    let stem =
        path.file_stem().and_then(|x| x.to_str()).unwrap_or("x").to_uppercase().replace('-', "_");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(parsed) = serde_yaml_ng::from_str::<ContractYaml>(&contents) else {
        return;
    };
    for (eq_name, eq) in &parsed.equations {
        let key = format!("CONTRACT_{}_{}", stem, eq_name.to_uppercase().replace('-', "_"));
        emit_pre_post(&key, eq, total_pre, total_post);
    }
}

fn emit_pre_post(key: &str, eq: &EquationYaml, total_pre: &mut usize, total_post: &mut usize) {
    if !eq.preconditions.is_empty() {
        println!("cargo:rustc-env={key}_PRE_COUNT={}", eq.preconditions.len());
        for (i, v) in eq.preconditions.iter().enumerate() {
            println!("cargo:rustc-env={key}_PRE_{i}={v}");
        }
        *total_pre += eq.preconditions.len();
    }
    if !eq.postconditions.is_empty() {
        println!("cargo:rustc-env={key}_POST_COUNT={}", eq.postconditions.len());
        for (i, v) in eq.postconditions.iter().enumerate() {
            println!("cargo:rustc-env={key}_POST_{i}={v}");
        }
        *total_post += eq.postconditions.len();
    }
}
