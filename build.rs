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

fn main() {
    // Declare __has_embedding_contract as a known cfg value (used for fallback macros)
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"__has_embedding_contract\"))");

    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("entrenar")
        .join("binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        println!(
            "cargo:warning=provable-contracts binding.yaml not found at {}; \
             CONTRACT_* env vars will not be set (CI/crates.io build)",
            binding_path.display()
        );
        println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
        return;
    }

    let yaml_content = match std::fs::read_to_string(&binding_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=Failed to read binding.yaml: {e}; \
                 CONTRACT_* env vars will not be set"
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        }
    };

    let bindings: BindingFile = match serde_yaml_ng::from_str(&yaml_content) {
        Ok(b) => b,
        Err(e) => {
            println!(
                "cargo:warning=Failed to parse binding.yaml: {e}; \
                 CONTRACT_* env vars will not be set"
            );
            println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=none");
            return;
        }
    };

    // Deduplicate bindings, keeping the best status
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

    let mut implemented = 0u32;
    let mut partial = 0u32;
    let mut not_implemented = 0u32;
    let mut gaps: Vec<String> = Vec::new();

    let mut keys: Vec<_> = seen.keys().cloned().collect();
    keys.sort();

    for var_name in &keys {
        let status = &seen[var_name];
        println!("cargo:rustc-env={var_name}={status}");

        match status.as_str() {
            "implemented" => implemented += 1,
            "partial" => {
                partial += 1;
                let note = bindings
                    .bindings
                    .iter()
                    .find(|b| env_var_name(&b.contract, &b.equation) == *var_name)
                    .and_then(|b| b.notes.as_deref())
                    .unwrap_or("");
                println!("cargo:warning=[contract] PARTIAL: {var_name} — {note}");
            }
            "not_implemented" => {
                not_implemented += 1;
                gaps.push(var_name.clone());
                let note = bindings
                    .bindings
                    .iter()
                    .find(|b| env_var_name(&b.contract, &b.equation) == *var_name)
                    .and_then(|b| b.notes.as_deref())
                    .unwrap_or("");
                println!("cargo:warning=[contract] GAP: {var_name} — {note}");
            }
            other => {
                println!("cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}");
            }
        }
    }

    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps"
    );

    // AllImplemented policy: fail build on any not_implemented gap
    if not_implemented > 0 {
        for gap in &gaps {
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

    // Phase 2: contract PRE/POST env vars for #[contract] proc macro
    {
        let cdir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        if let Ok(es) = std::fs::read_dir(&cdir) {
            #[derive(serde::Deserialize, Default)]
            struct CY {
                #[serde(default)]
                equations: std::collections::BTreeMap<String, EY>,
            }
            #[derive(serde::Deserialize, Default)]
            struct EY {
                #[serde(default)]
                preconditions: Vec<String>,
                #[serde(default)]
                postconditions: Vec<String>,
            }
            let (mut tp, mut tq) = (0, 0);
            for e in es.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("yaml") {
                    continue;
                }
                if p.file_name().is_some_and(|n| n.to_string_lossy().contains("binding")) {
                    continue;
                }
                println!("cargo:rerun-if-changed={}", p.display());
                let s = p
                    .file_stem()
                    .and_then(|x| x.to_str())
                    .unwrap_or("x")
                    .to_uppercase()
                    .replace('-', "_");
                if let Ok(c) = std::fs::read_to_string(&p) {
                    if let Ok(y) = serde_yaml_ng::from_str::<CY>(&c) {
                        for (n, eq) in &y.equations {
                            let k =
                                format!("CONTRACT_{}_{}", s, n.to_uppercase().replace('-', "_"));
                            if !eq.preconditions.is_empty() {
                                println!(
                                    "cargo:rustc-env={k}_PRE_COUNT={}",
                                    eq.preconditions.len()
                                );
                                for (i, v) in eq.preconditions.iter().enumerate() {
                                    println!("cargo:rustc-env={k}_PRE_{i}={v}");
                                }
                                tp += eq.preconditions.len();
                            }
                            if !eq.postconditions.is_empty() {
                                println!(
                                    "cargo:rustc-env={k}_POST_COUNT={}",
                                    eq.postconditions.len()
                                );
                                for (i, v) in eq.postconditions.iter().enumerate() {
                                    println!("cargo:rustc-env={k}_POST_{i}={v}");
                                }
                                tq += eq.postconditions.len();
                            }
                        }
                    }
                }
            }
            println!("cargo:warning=[contract] Assertions: {tp} preconditions, {tq} postconditions from YAML");
        }
    }
}
