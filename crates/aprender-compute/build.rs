// build.rs — Read provable-contracts binding.yaml and set CONTRACT_* env vars
//
// Policy: AllImplemented. Trueno has 38/38 bindings implemented (100%),
// so any gap is a build failure.
//
// The env vars follow the pattern:
//   CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>
//
// Example:
//   CONTRACT_BATCHED_MATMUL_ZERO_COPY_V1_BATCHED_MATMUL=implemented
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
fn env_var_name(contract: &str, equation: &str) -> String {
    let stem = contract
        .trim_end_matches(".yaml")
        .trim_end_matches(".yml")
        .to_uppercase()
        .replace('-', "_");
    let eq = equation.to_uppercase().replace('-', "_");
    format!("CONTRACT_{stem}_{eq}")
}

/// Rank status values for deduplication.
fn status_rank(s: &str) -> u8 {
    match s {
        "implemented" => 2,
        "partial" => 1,
        _ => 0,
    }
}

fn main() {
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("..").join("..").join("provable-contracts")
        .join("contracts")
        .join("trueno")
        .join("binding.yaml");

    println!("cargo:rerun-if-changed={}", binding_path.display());

    // Ensure generated_contracts.rs exists (empty stub for crates.io builds)
    let gen_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("generated_contracts.rs");
    if !gen_path.exists() {
        std::fs::write(
            &gen_path,
            "// Auto-generated stub — pv codegen populates this when provable-contracts is available\n",
        )
        .ok();
    }

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
            }
            other => {
                println!("cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}");
            }
        }
    }

    // AllImplemented policy: fail on any gaps
    if !gaps.is_empty() {
        for gap in &gaps {
            println!("cargo:warning=[contract] UNALLOWED GAP: {gap}");
        }
        panic!(
            "[contract] AllImplemented policy violation: {} binding(s) are \
             `not_implemented`:\n  {}\n\
             Fix: implement the binding, or switch to WarnOnGaps policy.",
            gaps.len(),
            gaps.join("\n  ")
        );
    }

    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps"
    );

    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!("cargo:rustc-env=CONTRACT_BINDING_VERSION={}", bindings.version);
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
    println!("cargo:rustc-env=CONTRACT_PARTIAL={partial}");
    println!("cargo:rustc-env=CONTRACT_GAPS={not_implemented}");
    // CB-1204: emit_contract marker for pmat comply pipeline detection
    println!("cargo:rustc-env=CONTRACT_PRE_COUNT={implemented}");
}
