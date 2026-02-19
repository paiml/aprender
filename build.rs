//! Build script for aprender (PMAT-250 + provable-contracts)
//!
//! Two responsibilities:
//!
//! 1. **Model family contracts** (PMAT-250): Reads YAML files from
//!    `contracts/model-families/` and generates Rust code with compiled-in
//!    family data. This catches YAML/Rust contract drift at build time.
//!
//! 2. **Provable-contracts bindings**: Reads `binding.yaml` from the sibling
//!    `provable-contracts` repo and sets `CONTRACT_*` env vars consumed by the
//!    `#[contract]` proc macro. Policy: AllImplemented (Step 6.7) — any
//!    `not_implemented` binding NOT in ALLOWED_GAPS fails the build.
//!
//! The env vars follow the pattern:
//!   `CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>`
//!
//! Example:
//!   `CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented`

use serde::Deserialize;
use std::fs;
use std::path::Path;

// ============================================================================
// Provable-Contracts Binding Types (serde-based)
// ============================================================================

/// Minimal subset of the binding.yaml schema — just enough to parse what we need.
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
/// "softmax-kernel-v1.yaml" + "softmax" -> "CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX"
fn binding_env_var_name(contract: &str, equation: &str) -> String {
    let stem = contract
        .trim_end_matches(".yaml")
        .trim_end_matches(".yml")
        .to_uppercase()
        .replace('-', "_");
    let eq = equation.to_uppercase().replace('-', "_");
    format!("CONTRACT_{stem}_{eq}")
}

/// Contracts allowed to remain `not_implemented` without failing the build.
/// Each entry is `(contract_stem, equation)`. Any `not_implemented` binding
/// NOT in this list triggers a hard build failure (AllImplemented policy).
const ALLOWED_GAPS: &[(&str, &str)] = &[
    // SSM/Mamba kernel — no implementation yet (GH-278 tracks Gated Delta Net)
    ("ssm-kernel-v1", "ssm_discretize"),
    ("ssm-kernel-v1", "ssm_scan"),
    ("ssm-kernel-v1", "selective_gate"),
];

/// Returns true if the new status is dominated by what we already have.
fn is_status_dominated(existing: Option<&str>, new: &str) -> bool {
    match (existing, new) {
        (None, _) => false,                        // first time
        (Some("implemented"), _) => true,          // already best
        (Some("partial"), "implemented") => false, // upgrade
        (Some("partial"), _) => true,              // keep partial
        (Some("not_implemented"), "not_implemented") => true,
        (Some("not_implemented"), _) => false, // upgrade
        _ => false,
    }
}

/// De-duplicate bindings, keeping the best status for each (contract, equation).
/// Returns (env_var → status, env_var → (contract_stem, equation)).
fn dedup_bindings(
    bindings: &[Binding],
) -> (
    std::collections::HashMap<String, String>,
    std::collections::HashMap<String, (String, String)>,
) {
    let mut seen = std::collections::HashMap::<String, String>::new();
    let mut seen_raw = std::collections::HashMap::<String, (String, String)>::new();

    for binding in bindings {
        let var_name = binding_env_var_name(&binding.contract, &binding.equation);

        if is_status_dominated(seen.get(&var_name).map(|s| s.as_str()), &binding.status) {
            continue;
        }

        seen.insert(var_name.clone(), binding.status.clone());
        let contract_stem = binding
            .contract
            .trim_end_matches(".yaml")
            .trim_end_matches(".yml")
            .to_string();
        seen_raw.insert(var_name, (contract_stem, binding.equation.clone()));
    }

    (seen, seen_raw)
}

/// Check if a (contract_stem, equation) gap is in the ALLOWED_GAPS list.
fn is_gap_allowed(contract_stem: &str, equation: &str) -> bool {
    ALLOWED_GAPS
        .iter()
        .any(|(c, e)| *c == contract_stem && *e == equation)
}

/// Enforce AllImplemented policy: panic if any unallowed gaps exist.
fn enforce_all_implemented(unallowed_gaps: &[String]) {
    if unallowed_gaps.is_empty() {
        return;
    }
    for gap in unallowed_gaps {
        println!("cargo:warning=[contract] UNALLOWED GAP: {gap}");
    }
    panic!(
        "[contract] AllImplemented policy violation: {} binding(s) are \
         `not_implemented` but not in ALLOWED_GAPS:\n  {}\n\
         Fix: implement the binding in binding.yaml, or add to ALLOWED_GAPS \
         in build.rs with a tracking issue.",
        unallowed_gaps.len(),
        unallowed_gaps.join("\n  ")
    );
}

/// Read provable-contracts binding.yaml and set CONTRACT_* env vars.
///
/// Policy: AllImplemented (Step 6.7). Any `not_implemented` binding not in
/// `ALLOWED_GAPS` fails the build. This ensures all algorithm contracts
/// have working implementations before code compiles.
fn emit_provable_contract_bindings() {
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("aprender")
        .join("binding.yaml");

    // Always tell Cargo to re-run if the file appears or changes
    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        // Graceful fallback: CI/crates.io builds won't have the sibling repo.
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

    let bindings: BindingFile = match serde_yaml::from_str(&yaml_content) {
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

    let (seen, seen_raw) = dedup_bindings(&bindings.bindings);

    let mut implemented = 0u32;
    let mut partial = 0u32;
    let mut not_implemented = 0u32;
    let mut unallowed_gaps: Vec<String> = Vec::new();

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
                    .find(|b| &binding_env_var_name(&b.contract, &b.equation) == var_name)
                    .and_then(|b| b.notes.as_deref())
                    .unwrap_or("");
                println!("cargo:warning=[contract] PARTIAL: {var_name} — {note}");
            }
            "not_implemented" => {
                not_implemented += 1;
                if let Some((ref stem, ref eq)) = seen_raw.get(var_name) {
                    if !is_gap_allowed(stem, eq) {
                        unallowed_gaps.push(format!("{var_name} ({stem}.yaml / {eq})"));
                    }
                }
            }
            other => {
                println!("cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}");
            }
        }
    }

    enforce_all_implemented(&unallowed_gaps);

    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} allowed gaps"
    );

    // Set metadata env vars for the proc macro
    println!("cargo:rustc-env=CONTRACT_BINDING_SOURCE=binding.yaml");
    println!(
        "cargo:rustc-env=CONTRACT_BINDING_VERSION={}",
        bindings.version
    );
    println!("cargo:rustc-env=CONTRACT_TOTAL={total}");
    println!("cargo:rustc-env=CONTRACT_IMPLEMENTED={implemented}");
    println!("cargo:rustc-env=CONTRACT_PARTIAL={partial}");
    println!("cargo:rustc-env=CONTRACT_GAPS={not_implemented}");
}

fn main() {
    // ── Phase 1: Provable-contracts binding.yaml → CONTRACT_* env vars ──
    emit_provable_contract_bindings();

    // ── Phase 2: Model family contracts → generated Rust code ──
    let families_dir = Path::new("contracts/model-families");

    // Tell Cargo to re-run if any YAML changes
    println!("cargo:rerun-if-changed=contracts/model-families");

    if !families_dir.exists() {
        // No contracts directory — generate empty registry
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_path = Path::new(&out_dir).join("model_families_generated.rs");
        fs::write(
            &out_path,
            "// No contracts/model-families/ directory found at build time\n\
             // Using empty generated registry\n\
             \n\
             /// Known model family names (generated at build time)\n\
             pub const KNOWN_FAMILIES: &[&str] = &[];\n",
        )
        .expect("write generated code");
        return;
    }

    let mut families: Vec<FamilyData> = Vec::new();

    let entries = fs::read_dir(families_dir).expect("read contracts/model-families");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();

        // Skip non-YAML and _-prefixed files
        if path.extension().map_or(true, |e| e != "yaml") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or(true, |s| s.starts_with('_'))
        {
            continue;
        }

        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

        let family = parse_family_yaml(&content, &path);
        families.push(family);
    }

    // Sort by family name for deterministic output
    families.sort_by(|a, b| a.family.cmp(&b.family));

    // Generate Rust source
    let generated = generate_rust(&families);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join("model_families_generated.rs");
    fs::write(&out_path, generated).expect("write generated code");
}


include!("build_parsing.rs");
include!("build_codegen.rs");
