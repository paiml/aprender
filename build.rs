// build.rs — Read provable-contracts binding.yaml and set CONTRACT_* env vars
//
// Policy: AllImplemented (Phase 5, PMAT-353). Build FAILS on any gaps.
// Upgraded from WarnOnGaps: 212/212 bindings implemented, 0 gaps.
//
// The env vars follow the pattern:
//   CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>
//
// Example:
//   CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
//   CONTRACT_SILU_KERNEL_V1_SILU=not_implemented
//
// These are consumed at compile time by the #[contract] proc macro to gate
// code generation and emit compile_error! for missing implementations.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use serde::Deserialize;

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
    #[serde(default)]
    function: Option<String>,
}

/// Convert a contract filename + equation into a canonical env var name.
///
/// `"softmax-kernel-v1.yaml"` + `"softmax"` → `"CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX"`
fn env_var_name(contract: &str, equation: &str) -> String {
    let stem = contract
        .trim_end_matches(".yaml")
        .trim_end_matches(".yml")
        .to_uppercase()
        .replace('-', "_");
    let eq = equation.to_uppercase().replace('-', "_");
    format!("CONTRACT_{stem}_{eq}")
}

// ─────────────────────────────────────────────────────────────────────────────
// PMAT-228: Architecture requirements YAML → Rust codegen
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal schema for architecture-requirements-v1.yaml.
#[derive(Deserialize)]
struct ArchRequirements {
    weight_roles: BTreeMap<String, WeightRoleDef>,
    role_sets: BTreeMap<String, RoleSetDef>,
    constraint_matrix: BTreeMap<String, ConstraintCell>,
}

#[derive(Deserialize)]
struct WeightRoleDef {
    description: String,
    field_name: String,
    #[allow(dead_code)]
    category: String,
}

#[derive(Deserialize)]
struct RoleSetDef {
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    count: usize,
    roles: Vec<String>,
}

#[derive(Deserialize)]
struct ConstraintCell {
    has_qk_norm: bool,
    has_bias: bool,
    required_sets: Vec<String>,
    total_required: usize,
}

/// Convert YAML role name to Rust enum variant name.
///
/// Special cases preserve backward compatibility with existing codebase:
/// - `gate_proj` → `FfnGate`, `up_proj` → `FfnUp`, `down_proj` → `FfnDown`
/// - Everything else: `snake_case` → `PascalCase` (e.g. `attn_q_norm` → `AttnQNorm`)
fn role_to_variant(s: &str) -> String {
    match s {
        "gate_proj" => "FfnGate".to_string(),
        "up_proj" => "FfnUp".to_string(),
        "down_proj" => "FfnDown".to_string(),
        _ => s
            .split('_')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(c) => {
                        let upper: String = c.to_uppercase().collect();
                        format!("{upper}{}", chars.as_str())
                    },
                    None => String::new(),
                }
            })
            .collect(),
    }
}

/// Generate the `arch_requirements.rs` content from YAML data.
fn generate_arch_requirements(req: &ArchRequirements) -> String {
    let mut out = String::with_capacity(4096);

    // Header — use // not //! since this is included via include!()
    out.push_str(
        "// Per-architecture required weight roles.\n\
         //\n\
         // AUTO-GENERATED from architecture-requirements-v1.yaml by build.rs — DO NOT EDIT.\n\
         // See: provable-contracts/contracts/architecture-requirements-v1.yaml\n\
         //\n\
         // UCBD §4 / GH-279: Compile-time enforcement that every loader\n\
         // provides all tensors required by the target architecture.\n\
         \n\
         use crate::gguf::ArchConstraints;\n\
         \n",
    );

    // WeightRole enum
    out.push_str(
        "/// Weight roles that may be required for a transformer layer.\n\
         /// Each architecture requires a subset of these.\n\
         #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]\n\
         pub enum WeightRole {\n",
    );
    for (name, def) in &req.weight_roles {
        let _ = writeln!(out, "    /// {}", def.description);
        let _ = writeln!(out, "    {},", role_to_variant(name));
    }
    out.push_str("}\n\n");

    // field_name() method
    out.push_str(
        "impl WeightRole {\n\
         \x20   /// Returns the field name prefix as it appears in `IndexedLayerWeights`.\n\
         \x20   #[must_use]\n\
         \x20   pub const fn field_name(&self) -> &'static str {\n\
         \x20       match self {\n",
    );
    for (name, def) in &req.weight_roles {
        let _ = writeln!(
            out,
            "            Self::{} => \"{}\",",
            role_to_variant(name),
            def.field_name
        );
    }
    out.push_str("        }\n    }\n}\n\n");

    // Const arrays for each constraint cell
    // Sort cells by (has_qk_norm, has_bias) for deterministic output
    let mut cells: Vec<_> = req.constraint_matrix.iter().collect();
    cells.sort_by_key(|(_, c)| (c.has_qk_norm, c.has_bias));

    for (cell_name, cell) in &cells {
        let const_name = cell_const_name(cell_name);
        let roles = expand_role_sets(&cell.required_sets, &req.role_sets);
        let _ = writeln!(
            out,
            "/// Roles for constraint cell: {cell_name} (has_qk_norm={}, has_bias={}).",
            cell.has_qk_norm, cell.has_bias
        );
        let _ = writeln!(out, "const {const_name}: &[WeightRole] = &[");
        for role in &roles {
            let _ = writeln!(out, "    WeightRole::{},", role_to_variant(role));
        }
        out.push_str("];\n\n");

        // Compile-time assertion on count
        let _ = writeln!(
            out,
            "const _: () = assert!({const_name}.len() == {}, \"YAML declares {} roles\");\n",
            cell.total_required, cell.total_required
        );
    }

    // required_roles() function
    out.push_str(
        "/// Returns the required weight roles for a given architecture.\n\
         ///\n\
         /// Exhaustive match on `(has_qk_norm, has_bias)` — adding a new architecture\n\
         /// combination without updating this function will still match one of the\n\
         /// four arms, but the contract test FALSIFY-ARCH-001 will catch mismatches.\n\
         #[must_use]\n\
         pub fn required_roles(arch: &ArchConstraints) -> &'static [WeightRole] {\n\
         \x20   match (arch.has_qk_norm, arch.has_bias) {\n",
    );
    for (cell_name, cell) in &cells {
        let const_name = cell_const_name(cell_name);
        let _ = writeln!(
            out,
            "        ({}, {}) => {const_name},",
            cell.has_qk_norm, cell.has_bias
        );
    }
    out.push_str("    }\n}\n");

    out
}

/// Convert YAML cell name to a Rust const name.
///
/// `"no_qk_norm_no_bias"` → `"ROLES_NO_QK_NORM_NO_BIAS"`
fn cell_const_name(cell_name: &str) -> String {
    format!("ROLES_{}", cell_name.to_uppercase())
}

/// Expand a list of role set names into a flat list of role names.
fn expand_role_sets(set_names: &[String], sets: &BTreeMap<String, RoleSetDef>) -> Vec<String> {
    let mut roles = Vec::new();
    for set_name in set_names {
        if let Some(set) = sets.get(set_name) {
            roles.extend(set.roles.iter().cloned());
        }
    }
    roles
}

fn main() {
    // Phase 1: Architecture requirements codegen (PMAT-228)
    generate_arch_requirements_file();

    // Phase 1b: Architecture constraints codegen (GH-323)
    generate_arch_constraints_file();

    // Phase 1c: Tensor names codegen (GH-311)
    generate_tensor_names_file();

    // Phase 2: Contract binding env vars
    emit_contract_bindings();

    // Phase 3: Contract assertion env vars from per-equation YAML
    emit_contract_assertions();

    // Phase 4: Ensure generated_contracts.rs exists (PMAT-157: publish fix).
    // This file is gitignored (auto-generated by `pv codegen`). When building
    // from a crates.io tarball, it won't exist. Create an empty stub so
    // `mod generated_contracts;` in lib.rs compiles.
    let contracts_path = Path::new("src/generated_contracts.rs");
    if !contracts_path.exists() {
        std::fs::write(
            contracts_path,
            "// Stub — generated_contracts.rs not available in crates.io build.\n\
             // Run `pv codegen contracts/ -o src/generated_contracts.rs` to generate.\n\
             macro_rules! contract_pre_matmul { () => {{}}; ($i:expr) => {{ let _ = &$i; }}; }\n\
             macro_rules! contract_pre_prefill_phase { () => {{}}; ($i:expr) => {{ let _ = &$i; }}; }\n",
        )
        .expect("create stub generated_contracts.rs");
    }
}

/// Phase 2: Read binding.yaml and emit CONTRACT_* env vars for the proc macro.
fn emit_contract_bindings() {
    // Re-run if binding.yaml changes
    let binding_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("realizar")
        .join("binding.yaml");

    // Always tell Cargo to re-run if the file appears or changes
    println!("cargo:rerun-if-changed={}", binding_path.display());

    if !binding_path.exists() {
        // Graceful fallback: CI/crates.io builds won't have the sibling repo.
        // The #[contract] proc macro should handle missing env vars gracefully.
        println!(
            "cargo:warning=provable-contracts binding.yaml not found at {}; \
             CONTRACT_* env vars will not be set (CI/crates.io build)",
            binding_path.display()
        );
        // Set a sentinel so the proc macro knows we ran but had no file
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
        },
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
        },
    };

    let seen = dedup_bindings(&bindings);
    let (implemented, partial, not_implemented) = emit_binding_env_vars(&seen, &bindings);

    let total = implemented + partial + not_implemented;
    println!(
        "cargo:warning=[contract] AllImplemented: {implemented}/{total} implemented, \
         {partial} partial, {not_implemented} gaps"
    );

    // PMAT-353: AllImplemented policy — build fails on any gaps
    assert!(
        not_implemented == 0,
        "[contract] AllImplemented policy violation: {not_implemented} binding(s) \
         not_implemented. Check warnings above for GAP entries.\n\
         Fix: implement the binding, or mark as partial with notes."
    );

    // ── Layer 2: Verify bound functions exist in source ──
    {
        let mut expected: std::collections::HashSet<String> = std::collections::HashSet::new();
        for b in &bindings.bindings {
            if b.status == "implemented" {
                if let Some(ref func) = b.function {
                    let short = func.rsplit("::").next().unwrap_or(func).to_lowercase();
                    expected.insert(short);
                }
            }
        }
        if !expected.is_empty() {
            let mut found: std::collections::HashSet<String> = std::collections::HashSet::new();
            fn scan_rs(dir: &std::path::Path, found: &mut std::collections::HashSet<String>) {
                let Ok(entries) = std::fs::read_dir(dir) else {
                    return;
                };
                for e in entries.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if n != "target" && n != ".git" {
                            scan_rs(&p, found);
                        }
                    } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                        if let Ok(c) = std::fs::read_to_string(&p) {
                            for line in c.lines() {
                                let t = line.trim();
                                if t.starts_with("pub fn ")
                                    || t.starts_with("pub async fn ")
                                    || t.starts_with("pub(crate) fn ")
                                {
                                    let part = t
                                        .trim_start_matches("pub async fn ")
                                        .trim_start_matches("pub(crate) fn ")
                                        .trim_start_matches("pub fn ");
                                    let name = part
                                        .split('(')
                                        .next()
                                        .unwrap_or("")
                                        .split('<')
                                        .next()
                                        .unwrap_or("")
                                        .trim()
                                        .to_lowercase();
                                    if !name.is_empty() {
                                        found.insert(name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            scan_rs(std::path::Path::new("src"), &mut found);
            scan_rs(std::path::Path::new("crates"), &mut found);
            let missing: Vec<_> = expected
                .iter()
                .filter(|n| !found.contains(n.as_str()))
                .collect();
            if !missing.is_empty() {
                println!("cargo:warning=[contract] L2: {} bound function(s) not found in source (soft warning)", missing.len());
            }
        }
    }

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

/// Rank status values for deduplication: `implemented` > `partial` > `not_implemented`.
fn status_rank(s: &str) -> u8 {
    match s {
        "implemented" => 2,
        "partial" => 1,
        _ => 0,
    }
}

/// De-duplicate bindings, keeping the best status for each (contract, equation) pair.
fn dedup_bindings(bindings: &BindingFile) -> std::collections::HashMap<String, String> {
    let mut seen = std::collections::HashMap::<String, String>::new();

    for binding in &bindings.bindings {
        let var_name = env_var_name(&binding.contract, &binding.equation);
        let new_rank = status_rank(&binding.status);

        let dominated = seen
            .get(&var_name)
            .is_some_and(|existing| status_rank(existing) >= new_rank);

        if !dominated {
            seen.insert(var_name, binding.status.clone());
        }
    }

    seen
}

/// Emit rustc-env and cargo:warning lines for each binding; returns (implemented, partial, gaps).
fn emit_binding_env_vars(
    seen: &std::collections::HashMap<String, String>,
    bindings: &BindingFile,
) -> (u32, u32, u32) {
    let mut implemented = 0u32;
    let mut partial = 0u32;
    let mut not_implemented = 0u32;

    let mut keys: Vec<_> = seen.keys().cloned().collect();
    keys.sort();

    for var_name in &keys {
        let status = &seen[var_name];
        println!("cargo:rustc-env={var_name}={status}");

        match status.as_str() {
            "implemented" => implemented += 1,
            "partial" => {
                partial += 1;
                let note = find_binding_note(bindings, var_name);
                println!("cargo:warning=[contract] PARTIAL: {var_name} — {note}");
            },
            "not_implemented" => {
                not_implemented += 1;
                let note = find_binding_note(bindings, var_name);
                println!("cargo:warning=[contract] GAP: {var_name} — {note}");
            },
            other => {
                println!("cargo:warning=[contract] UNKNOWN STATUS '{other}': {var_name}");
            },
        }
    }

    (implemented, partial, not_implemented)
}

/// Find the note for a binding by its var name.
fn find_binding_note<'a>(bindings: &'a BindingFile, var_name: &str) -> &'a str {
    bindings
        .bindings
        .iter()
        .find(|b| env_var_name(&b.contract, &b.equation) == var_name)
        .and_then(|b| b.notes.as_deref())
        .unwrap_or("")
}

/// PMAT-228: Read architecture-requirements-v1.yaml and generate `arch_requirements.rs`.
fn generate_arch_requirements_file() {
    let yaml_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("architecture-requirements-v1.yaml");

    println!("cargo:rerun-if-changed={}", yaml_path.display());

    if !yaml_path.exists() {
        // Graceful fallback for CI/crates.io — write a stub generated file
        println!(
            "cargo:warning=[PMAT-228] architecture-requirements-v1.yaml not found; \
             using hand-written arch_requirements.rs fallback"
        );
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_path = Path::new(&out_dir).join("arch_requirements_generated.rs");
        std::fs::write(&out_path, include_str!("src/arch_requirements_fallback.rs"))
            .expect("Failed to write fallback arch_requirements_generated.rs");
        return;
    }

    let yaml_content = match std::fs::read_to_string(&yaml_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=[PMAT-228] Failed to read architecture-requirements-v1.yaml: {e}"
            );
            return;
        },
    };

    let req: ArchRequirements = match serde_yaml_ng::from_str(&yaml_content) {
        Ok(r) => r,
        Err(e) => {
            println!(
                "cargo:warning=[PMAT-228] Failed to parse architecture-requirements-v1.yaml: {e}"
            );
            return;
        },
    };

    let generated = generate_arch_requirements(&req);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join("arch_requirements_generated.rs");
    std::fs::write(&out_path, generated).expect("Failed to write generated arch_requirements.rs");

    println!(
        "cargo:warning=[PMAT-228] Generated arch_requirements from YAML ({} weight roles, {} constraint cells)",
        req.weight_roles.len(),
        req.constraint_matrix.len()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GH-323: Architecture constraints YAML → Rust codegen
// ─────────────────────────────────────────────────────────────────────────────

/// Schema for arch-constraints-v1.yaml.
#[derive(Deserialize)]
struct ArchConstraintsContract {
    architectures: BTreeMap<String, ArchEntry>,
    default: ArchEntry,
}

#[derive(Deserialize)]
struct ArchEntry {
    #[serde(default)]
    aliases: Vec<String>,
    norm_type: String,
    activation: String,
    positional_encoding: String,
    mlp_type: String,
    weight_layout: String,
    has_bias: bool,
    tied_embeddings: bool,
    has_qk_norm: bool,
    default_eps: f64,
}

/// GH-323: Read arch-constraints-v1.yaml and generate `arch_constraints_generated.rs`.
fn generate_arch_constraints_file() {
    let yaml_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("arch-constraints-v1.yaml");

    println!("cargo:rerun-if-changed={}", yaml_path.display());

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join("arch_constraints_generated.rs");

    if !yaml_path.exists() {
        println!(
            "cargo:warning=[GH-323] arch-constraints-v1.yaml not found; \
             using fallback arch_constraints.rs"
        );
        std::fs::write(
            &out_path,
            include_str!("src/gguf/arch_constraints_fallback.rs"),
        )
        .expect("Failed to write fallback arch_constraints_generated.rs");
        return;
    }

    let yaml_content = match std::fs::read_to_string(&yaml_path) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "cargo:warning=[GH-323] Failed to read arch-constraints-v1.yaml: {e}; \
                 using fallback"
            );
            std::fs::write(
                &out_path,
                include_str!("src/gguf/arch_constraints_fallback.rs"),
            )
            .expect("Failed to write fallback arch_constraints_generated.rs");
            return;
        },
    };

    let contract: ArchConstraintsContract = match serde_yaml_ng::from_str(&yaml_content) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "cargo:warning=[GH-323] Failed to parse arch-constraints-v1.yaml: {e}; \
                 using fallback"
            );
            std::fs::write(
                &out_path,
                include_str!("src/gguf/arch_constraints_fallback.rs"),
            )
            .expect("Failed to write fallback arch_constraints_generated.rs");
            return;
        },
    };

    let generated = generate_arch_constraints(&contract);
    std::fs::write(&out_path, &generated).expect("Failed to write arch_constraints_generated.rs");

    println!(
        "cargo:warning=[GH-323] Generated arch_constraints from YAML ({} architectures)",
        contract.architectures.len()
    );
}

/// Map YAML enum string to Rust enum variant path.
fn map_norm_type(s: &str) -> &'static str {
    match s {
        "LayerNorm" => "NormType::LayerNorm",
        "RmsNorm" | "rmsnorm" => "NormType::RmsNorm",
        _ => panic!("Unknown norm_type in YAML: {s}"),
    }
}

fn map_activation(s: &str) -> &'static str {
    match s {
        "Gelu" | "gelu" => "Activation::Gelu",
        "Silu" | "silu" => "Activation::Silu",
        _ => panic!("Unknown activation in YAML: {s}"),
    }
}

fn map_positional_encoding(s: &str) -> &'static str {
    match s {
        "Absolute" | "absolute" => "PositionalEncoding::Absolute",
        "Rope" | "rope" => "PositionalEncoding::Rope",
        "Alibi" | "alibi" => "PositionalEncoding::Alibi",
        "Relative" | "relative" => "PositionalEncoding::Relative",
        "None" | "none" => "PositionalEncoding::None",
        _ => panic!("Unknown positional_encoding in YAML: {s}"),
    }
}

fn map_mlp_type(s: &str) -> &'static str {
    match s {
        "GeluMlp" | "gelu_mlp" => "MlpType::GeluMlp",
        "SwiGlu" | "swiglu" => "MlpType::SwiGlu",
        "GatedMlp" | "gated_mlp" => "MlpType::GatedMlp",
        _ => panic!("Unknown mlp_type in YAML: {s}"),
    }
}

fn map_weight_layout(s: &str) -> &'static str {
    match s {
        "Linear" | "linear" => "WeightLayout::Linear",
        "Conv1D" | "conv1d" => "WeightLayout::Conv1D",
        _ => panic!("Unknown weight_layout in YAML: {s}"),
    }
}

/// Format an f64 epsilon as a Rust f32 literal.
///
/// Uses exact comparison since these are well-known constant values
/// parsed from YAML, not computed results.
#[allow(clippy::float_cmp)]
fn format_eps(eps: f64) -> String {
    if eps == 1e-12 {
        "1e-12".to_string()
    } else if eps == 1e-6 {
        "1e-6".to_string()
    } else if eps == 1e-5 {
        "1e-5".to_string()
    } else {
        format!("{eps:e}")
    }
}

/// Emit a single architecture match arm into `out`.
fn emit_arch_arm(out: &mut String, comment: &str, pattern: &str, entry: &ArchEntry) {
    let _ = write!(
        out,
        "        // {comment}\n\
         \x20       {pattern} => ArchConstraints {{\n\
         \x20           norm_type: {},\n\
         \x20           activation: {},\n\
         \x20           positional_encoding: {},\n\
         \x20           mlp_type: {},\n\
         \x20           weight_layout: {},\n\
         \x20           has_bias: {},\n\
         \x20           tied_embeddings: {},\n\
         \x20           has_qk_norm: {},\n\
         \x20           default_eps: {},\n\
         \x20       }},\n",
        map_norm_type(&entry.norm_type),
        map_activation(&entry.activation),
        map_positional_encoding(&entry.positional_encoding),
        map_mlp_type(&entry.mlp_type),
        map_weight_layout(&entry.weight_layout),
        entry.has_bias,
        entry.tied_embeddings,
        entry.has_qk_norm,
        format_eps(entry.default_eps),
    );
}

/// Generate the `from_architecture()` match body from YAML data.
fn generate_arch_constraints(contract: &ArchConstraintsContract) -> String {
    let mut out = String::with_capacity(8192);

    out.push_str(
        "// Per-architecture inference constraints.\n\
         //\n\
         // AUTO-GENERATED from arch-constraints-v1.yaml by build.rs — DO NOT EDIT.\n\
         // See: provable-contracts/contracts/arch-constraints-v1.yaml\n\
         //\n\
         // GH-323: Compile-time enforcement of architecture constraints.\n\
         \n",
    );

    // Generate the from_architecture_generated() function
    out.push_str(
        "/// Look up architecture constraints from the GGUF `general.architecture` value.\n\
         ///\n\
         /// AUTO-GENERATED from arch-constraints-v1.yaml — DO NOT EDIT.\n\
         /// Maps architecture names and aliases to their contract-defined behavior.\n\
         /// Unknown architectures fall back to LLaMA-like defaults.\n\
         #[must_use]\n\
         fn from_architecture_generated(arch: &str) -> ArchConstraints {\n\
         \x20   match arch {\n",
    );

    // Generate match arms for each architecture
    for (name, entry) in &contract.architectures {
        // Build pattern: "name" | "alias1" | "alias2"
        let mut patterns = vec![format!("\"{name}\"")];
        for alias in &entry.aliases {
            patterns.push(format!("\"{alias}\""));
        }
        let pattern = patterns.join(" | ");
        emit_arch_arm(&mut out, &format!("{name}.yaml"), &pattern, entry);
    }

    // Default arm
    emit_arch_arm(
        &mut out,
        "Default: LLaMA-like (most common pattern in modern LLMs)",
        "_",
        &contract.default,
    );

    out.push_str("    }\n}\n");

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// GH-311: Tensor names YAML → Rust codegen
// ─────────────────────────────────────────────────────────────────────────────

/// GH-311: Write `tensor_names_generated.rs` (fallback for now, YAML codegen planned).
fn generate_tensor_names_file() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join("tensor_names_generated.rs");

    // For now, always use the fallback file.
    // YAML-to-codegen pipeline from tensor-names-v1.yaml is planned (GH-311).
    let yaml_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("provable-contracts")
        .join("contracts")
        .join("tensor-names-v1.yaml");

    println!("cargo:rerun-if-changed={}", yaml_path.display());

    std::fs::write(&out_path, include_str!("src/tensor_names_fallback.rs"))
        .expect("Failed to write tensor_names_generated.rs");
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Contract assertion env vars from per-equation YAML
// ─────────────────────────────────────────────────────────────────────────────

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

fn emit_contract_assertions() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
    if !dir.exists() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let (mut tp, mut tq) = (0usize, 0usize);
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", p.display());
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("x");
        let Ok(c) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(y): Result<ContractYaml, _> = serde_yaml_ng::from_str(&c) else {
            continue;
        };
        let su = stem.to_uppercase().replace('-', "_");
        for (eq, e) in &y.equations {
            let eu = eq.to_uppercase().replace('-', "_");
            let k = format!("CONTRACT_{su}_{eu}");
            if !e.preconditions.is_empty() {
                println!("cargo:rustc-env={k}_PRE_COUNT={}", e.preconditions.len());
                for (i, v) in e.preconditions.iter().enumerate() {
                    println!("cargo:rustc-env={k}_PRE_{i}={v}");
                }
                tp += e.preconditions.len();
            }
            if !e.postconditions.is_empty() {
                println!("cargo:rustc-env={k}_POST_COUNT={}", e.postconditions.len());
                for (i, v) in e.postconditions.iter().enumerate() {
                    println!("cargo:rustc-env={k}_POST_{i}={v}");
                }
                tq += e.postconditions.len();
            }
        }
    }
    println!(
        "cargo:warning=[contract] Assertions: {tp} preconditions, {tq} postconditions from YAML"
    );
}
