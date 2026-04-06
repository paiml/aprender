//! Build script helper for consuming crates.
//!
//! Consuming crates (realizar, aprender, trueno, entrenar) use this module
//! in their `build.rs` to:
//!
//! 1. Read `binding.yaml` and extract all implemented bindings
//! 2. Set `CONTRACT_<NAME>_<EQ>=bound` env vars for each binding
//! 3. Fail the build if any binding has status `not_implemented`
//!
//! ## Usage in build.rs
//!
//! ```rust,ignore
//! // build.rs
//! fn main() {
//!     provable_contracts::build_helper::verify_bindings(
//!         "../provable-contracts/contracts/aprender/binding.yaml",
//!         BindingPolicy::AllImplemented,
//!     );
//! }
//! ```

use std::path::Path;

use crate::binding::{BindingRegistry, ImplStatus};

/// Policy for handling unimplemented bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingPolicy {
    /// All bindings must have status `implemented`. Any `partial` or
    /// `not_implemented` binding is a compile error.
    AllImplemented,

    /// Only `implemented` bindings get env vars. `partial` and
    /// `not_implemented` are warnings (printed to cargo stderr).
    WarnOnGaps,

    /// Tiered: `not_implemented` is an error, `partial` is a warning.
    TieredEnforcement,
}

/// Result of binding verification.
#[derive(Debug)]
pub struct VerifyResult {
    /// Number of bindings that got env vars set.
    pub bound_count: usize,
    /// Number of partial bindings (warnings).
    pub partial_count: usize,
    /// Number of not-implemented bindings (errors or warnings depending on policy).
    pub not_implemented_count: usize,
}

/// Read `binding.yaml` and set `CONTRACT_*` env vars for `#[contract]` macros.
///
/// Call this from your `build.rs`. It:
/// 1. Parses the binding YAML
/// 2. For each `implemented` binding, emits `cargo:rustc-env=CONTRACT_<KEY>=bound`
/// 3. Enforces the given policy for gaps
///
/// # Panics
///
/// Panics (failing the build) if:
/// - The binding YAML cannot be read or parsed
/// - Policy is `AllImplemented` and any binding is not `implemented`
/// - Policy is `TieredEnforcement` and any binding is `not_implemented`
#[allow(clippy::too_many_lines)]
pub fn verify_bindings(binding_yaml_path: &str, policy: BindingPolicy) -> VerifyResult {
    let path = Path::new(binding_yaml_path);

    // Rerun build.rs if binding.yaml changes
    println!("cargo:rerun-if-changed={binding_yaml_path}");

    // Also rerun if the contracts directory changes
    if let Some(parent) = path.parent() {
        if let Some(grandparent) = parent.parent() {
            println!("cargo:rerun-if-changed={}", grandparent.display());
        }
    }

    let yaml_content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "CONTRACT BUILD ERROR: Cannot read binding YAML at '{}': {e}\n\
             Hint: Ensure provable-contracts is checked out as a sibling directory.",
            path.display()
        );
    });

    let registry: BindingRegistry = serde_yaml::from_str(&yaml_content).unwrap_or_else(|e| {
        panic!(
            "CONTRACT BUILD ERROR: Cannot parse binding YAML at '{}': {e}",
            path.display()
        );
    });

    let mut result = VerifyResult {
        bound_count: 0,
        partial_count: 0,
        not_implemented_count: 0,
    };

    for binding in &registry.bindings {
        let env_key = make_env_key(&binding.contract, &binding.equation);

        match binding.status {
            ImplStatus::Implemented => {
                println!("cargo:rustc-env={env_key}=bound");
                result.bound_count += 1;
            }
            ImplStatus::Partial => {
                result.partial_count += 1;
                match policy {
                    BindingPolicy::AllImplemented => {
                        panic!(
                            "CONTRACT BUILD ERROR: Binding {}.{} has status 'partial'. \
                             Policy requires all bindings to be 'implemented'.\n\
                             Module: {}\n\
                             See: unified-contract-by-design.md §10",
                            binding.contract,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("(unknown)"),
                        );
                    }
                    BindingPolicy::WarnOnGaps | BindingPolicy::TieredEnforcement => {
                        println!(
                            "cargo:warning=CONTRACT: partial binding {}.{} ({})",
                            binding.contract,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("?"),
                        );
                        // Still set env var for partial — the function exists, just incomplete
                        println!("cargo:rustc-env={env_key}=partial");
                    }
                }
            }
            ImplStatus::NotImplemented => {
                result.not_implemented_count += 1;
                match policy {
                    BindingPolicy::AllImplemented | BindingPolicy::TieredEnforcement => {
                        panic!(
                            "CONTRACT BUILD ERROR: Binding {}.{} has status 'not_implemented'. \
                             All bindings must be implemented.\n\
                             Equation: {}\n\
                             Target: {}\n\
                             See: unified-contract-by-design.md §10",
                            binding.contract,
                            binding.equation,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("(unassigned)"),
                        );
                    }
                    BindingPolicy::WarnOnGaps => {
                        println!(
                            "cargo:warning=CONTRACT: not_implemented binding {}.{} ({})",
                            binding.contract,
                            binding.equation,
                            binding.module_path.as_deref().unwrap_or("?"),
                        );
                    }
                }
            }
            ImplStatus::Pending => {
                result.not_implemented_count += 1;
                println!(
                    "cargo:warning=CONTRACT: pending binding {}.{} ({})",
                    binding.contract,
                    binding.equation,
                    binding.module_path.as_deref().unwrap_or("?"),
                );
            }
        }
    }

    println!(
        "cargo:warning=CONTRACT: {}/{} bindings bound ({} partial, {} not_implemented)",
        result.bound_count,
        registry.bindings.len(),
        result.partial_count,
        result.not_implemented_count,
    );

    result
}

/// Verify that functions named in binding.yaml actually exist in the crate source.
///
/// Scans `src_dir` for `pub fn <name>` declarations and checks that every
/// `function` field in the binding registry has a matching source function.
/// Returns the names of missing functions. If the list is non-empty and
/// `hard_fail` is true, panics to fail the build.
///
/// This closes the "ghost binding" gap where `status: implemented` passes
/// build.rs but the actual function doesn't exist (renamed, deleted, typo).
///
/// # Example
///
/// ```rust,ignore
/// // build.rs
/// let missing = provable_contracts::build_helper::verify_source_functions(
///     "../provable-contracts/contracts/aprender/binding.yaml",
///     "src/",
///     true, // hard fail
/// );
/// ```
pub fn verify_source_functions(
    binding_yaml_path: &str,
    src_dir: &str,
    hard_fail: bool,
) -> Vec<String> {
    let path = Path::new(binding_yaml_path);
    let Ok(yaml_content) = std::fs::read_to_string(path) else {
        println!("cargo:warning=verify_source_functions: cannot read {binding_yaml_path}");
        return vec![];
    };
    let Ok(registry) = serde_yaml::from_str::<BindingRegistry>(&yaml_content) else {
        println!("cargo:warning=verify_source_functions: cannot parse {binding_yaml_path}");
        return vec![];
    };

    // Collect all function names from bindings
    let mut expected_fns: std::collections::HashSet<String> = std::collections::HashSet::new();
    for b in &registry.bindings {
        if b.status != ImplStatus::Implemented {
            continue;
        }
        if let Some(ref func) = b.function {
            // Extract just the function name (after last ::)
            let short = func.rsplit("::").next().unwrap_or(func);
            expected_fns.insert(short.to_lowercase());
        }
    }

    if expected_fns.is_empty() {
        return vec![];
    }

    // Scan source files for pub fn declarations
    let mut found_fns: std::collections::HashSet<String> = std::collections::HashSet::new();
    let src = Path::new(src_dir);
    if src.exists() {
        scan_source_fns(src, &mut found_fns);
    }
    // Also check crates/ subdirectory
    let crates_dir = Path::new("crates");
    if crates_dir.exists() {
        scan_source_fns(crates_dir, &mut found_fns);
    }

    let mut missing: Vec<String> = expected_fns
        .iter()
        .filter(|name| !found_fns.contains(name.as_str()))
        .cloned()
        .collect();
    missing.sort();

    if !missing.is_empty() {
        let count = missing.len();
        let sample: Vec<_> = missing.iter().take(10).collect();
        let msg = format!(
            "[contract] verify_source_functions: {count} bound function(s) not found in source: {}{}",
            sample
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            if count > 10 {
                format!(" (and {} more)", count - 10)
            } else {
                String::new()
            },
        );

        if hard_fail {
            panic!("{msg}");
        } else {
            println!("cargo:warning={msg}");
        }
    }

    missing
}

/// Recursively scan a directory for `pub fn` declarations.
fn scan_source_fns(dir: &Path, found: &mut std::collections::HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name != "target" && name != ".git" {
                scan_source_fns(&path, found);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("pub(crate) fn ")
                    {
                        let fn_part = trimmed
                            .trim_start_matches("pub async fn ")
                            .trim_start_matches("pub(crate) fn ")
                            .trim_start_matches("pub fn ");
                        let fn_name = fn_part
                            .split('(')
                            .next()
                            .unwrap_or("")
                            .split('<')
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_lowercase();
                        if !fn_name.is_empty() {
                            found.insert(fn_name);
                        }
                    }
                }
            }
        }
    }
}

/// Generate the env var key from contract name and equation name.
///
/// Same convention as `provable-contracts-macros::make_env_key`.
fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::all)]
    use super::*;
    /// Helper: write YAML content to a temp file and return the path string.
    fn write_temp_yaml(content: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("binding.yaml");
        std::fs::write(&path, content).expect("write yaml");
        let path_str = path.to_str().unwrap().to_string();
        (dir, path_str)
    }

    /// Minimal valid binding YAML with all-implemented bindings.
    fn yaml_all_implemented() -> String {
        r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    module_path: "test::softmax"
    function: my_softmax
    status: implemented
  - contract: relu-v1.yaml
    equation: relu
    module_path: "test::relu"
    function: my_relu
    status: implemented
"#
        .to_string()
    }

    /// Binding YAML with a partial binding.
    fn yaml_with_partial() -> String {
        r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    module_path: "test::softmax"
    function: my_softmax
    status: implemented
  - contract: relu-v1.yaml
    equation: relu
    module_path: "test::relu"
    function: my_relu
    status: partial
"#
        .to_string()
    }

    /// Binding YAML with a `not_implemented` binding.
    fn yaml_with_not_implemented() -> String {
        r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    module_path: "test::softmax"
    function: my_softmax
    status: implemented
  - contract: gelu-v1.yaml
    equation: gelu
    status: not_implemented
"#
        .to_string()
    }

    /// Binding YAML with a pending binding.
    fn yaml_with_pending() -> String {
        r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    module_path: "test::softmax"
    function: my_softmax
    status: implemented
  - contract: silu-v1.yaml
    equation: silu
    module_path: "test::silu"
    status: pending
"#
        .to_string()
    }

    /// Binding YAML with mixed statuses (implemented + partial + `not_implemented` + pending).
    fn yaml_mixed() -> String {
        r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    module_path: "test::softmax"
    function: my_softmax
    status: implemented
  - contract: relu-v1.yaml
    equation: relu
    module_path: "test::relu"
    function: my_relu
    status: partial
  - contract: gelu-v1.yaml
    equation: gelu
    status: not_implemented
  - contract: silu-v1.yaml
    equation: silu
    status: pending
"#
        .to_string()
    }

    // ── make_env_key tests ──

    #[test]
    fn test_make_env_key_matches_macro_convention() {
        assert_eq!(
            make_env_key("rmsnorm-kernel-v1", "rmsnorm"),
            "CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"
        );
        assert_eq!(
            make_env_key("gated-delta-net-v1", "decay"),
            "CONTRACT_GATED_DELTA_NET_V1_DECAY"
        );
    }

    #[test]
    fn make_env_key_with_yaml_extension() {
        assert_eq!(
            make_env_key("softmax-kernel-v1.yaml", "softmax"),
            "CONTRACT_SOFTMAX_KERNEL_V1_YAML_SOFTMAX"
        );
    }

    #[test]
    fn make_env_key_dots_replaced() {
        assert_eq!(
            make_env_key("my.contract.v1", "eq.1"),
            "CONTRACT_MY_CONTRACT_V1_EQ_1"
        );
    }

    // ── VerifyResult tests ──

    #[test]
    fn test_verify_result_defaults() {
        let r = VerifyResult {
            bound_count: 0,
            partial_count: 0,
            not_implemented_count: 0,
        };
        assert_eq!(r.bound_count, 0);
        assert_eq!(r.partial_count, 0);
        assert_eq!(r.not_implemented_count, 0);
    }

    #[test]
    fn verify_result_debug_display() {
        let r = VerifyResult {
            bound_count: 5,
            partial_count: 2,
            not_implemented_count: 1,
        };
        let dbg = format!("{r:?}");
        assert!(dbg.contains("bound_count: 5"));
        assert!(dbg.contains("partial_count: 2"));
        assert!(dbg.contains("not_implemented_count: 1"));
    }

    // ── BindingPolicy tests ──

    #[test]
    fn binding_policy_debug() {
        assert_eq!(
            format!("{:?}", BindingPolicy::AllImplemented),
            "AllImplemented"
        );
        assert_eq!(
            format!("{:?}", BindingPolicy::TieredEnforcement),
            "TieredEnforcement"
        );
        assert_eq!(format!("{:?}", BindingPolicy::WarnOnGaps), "WarnOnGaps");
    }

    #[test]
    fn binding_policy_clone_eq() {
        let a = BindingPolicy::AllImplemented;
        let b = a;
        assert_eq!(a, b);

        let c = BindingPolicy::WarnOnGaps;
        assert_ne!(a, c);
    }

    // ── verify_bindings: success paths ──

    #[test]
    fn verify_bindings_all_implemented_all_ok() {
        let (_dir, path) = write_temp_yaml(&yaml_all_implemented());
        let result = verify_bindings(&path, BindingPolicy::AllImplemented);
        assert_eq!(result.bound_count, 2);
        assert_eq!(result.partial_count, 0);
        assert_eq!(result.not_implemented_count, 0);
    }

    #[test]
    fn verify_bindings_warn_on_gaps_all_implemented() {
        let (_dir, path) = write_temp_yaml(&yaml_all_implemented());
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.bound_count, 2);
        assert_eq!(result.partial_count, 0);
        assert_eq!(result.not_implemented_count, 0);
    }

    #[test]
    fn verify_bindings_tiered_all_implemented() {
        let (_dir, path) = write_temp_yaml(&yaml_all_implemented());
        let result = verify_bindings(&path, BindingPolicy::TieredEnforcement);
        assert_eq!(result.bound_count, 2);
        assert_eq!(result.partial_count, 0);
        assert_eq!(result.not_implemented_count, 0);
    }

    // ── verify_bindings: WarnOnGaps with partial ──

    #[test]
    fn verify_bindings_warn_on_gaps_partial() {
        let (_dir, path) = write_temp_yaml(&yaml_with_partial());
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.partial_count, 1);
        assert_eq!(result.not_implemented_count, 0);
    }

    // ── verify_bindings: WarnOnGaps with not_implemented ──

    #[test]
    fn verify_bindings_warn_on_gaps_not_implemented() {
        let (_dir, path) = write_temp_yaml(&yaml_with_not_implemented());
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.partial_count, 0);
        assert_eq!(result.not_implemented_count, 1);
    }

    // ── verify_bindings: WarnOnGaps with mixed statuses ──

    #[test]
    fn verify_bindings_warn_on_gaps_mixed() {
        let (_dir, path) = write_temp_yaml(&yaml_mixed());
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.partial_count, 1);
        // not_implemented + pending both count as not_implemented_count
        assert_eq!(result.not_implemented_count, 2);
    }

    // ── verify_bindings: TieredEnforcement with partial (warns, doesn't panic) ──

    #[test]
    fn verify_bindings_tiered_partial_warns_but_ok() {
        let (_dir, path) = write_temp_yaml(&yaml_with_partial());
        let result = verify_bindings(&path, BindingPolicy::TieredEnforcement);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.partial_count, 1);
        assert_eq!(result.not_implemented_count, 0);
    }

    // ── verify_bindings: pending status handling ──

    #[test]
    fn verify_bindings_pending_status_warns() {
        let (_dir, path) = write_temp_yaml(&yaml_with_pending());
        let result = verify_bindings(&path, BindingPolicy::AllImplemented);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.not_implemented_count, 1); // pending counts here
    }

    #[test]
    fn verify_bindings_pending_with_tiered() {
        let (_dir, path) = write_temp_yaml(&yaml_with_pending());
        let result = verify_bindings(&path, BindingPolicy::TieredEnforcement);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.not_implemented_count, 1);
    }

    #[test]
    fn verify_bindings_pending_with_warn_on_gaps() {
        let (_dir, path) = write_temp_yaml(&yaml_with_pending());
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.bound_count, 1);
        assert_eq!(result.not_implemented_count, 1);
    }

    // ── verify_bindings: panic paths ──

    #[test]
    #[should_panic(expected = "CONTRACT BUILD ERROR")]
    fn verify_bindings_all_implemented_panics_on_partial() {
        let (_dir, path) = write_temp_yaml(&yaml_with_partial());
        verify_bindings(&path, BindingPolicy::AllImplemented);
    }

    #[test]
    #[should_panic(expected = "CONTRACT BUILD ERROR")]
    fn verify_bindings_all_implemented_panics_on_not_implemented() {
        let (_dir, path) = write_temp_yaml(&yaml_with_not_implemented());
        verify_bindings(&path, BindingPolicy::AllImplemented);
    }

    #[test]
    #[should_panic(expected = "CONTRACT BUILD ERROR")]
    fn verify_bindings_tiered_panics_on_not_implemented() {
        let (_dir, path) = write_temp_yaml(&yaml_with_not_implemented());
        verify_bindings(&path, BindingPolicy::TieredEnforcement);
    }

    #[test]
    #[should_panic(expected = "Cannot read binding YAML")]
    fn verify_bindings_panics_on_missing_yaml() {
        verify_bindings("/nonexistent/path/binding.yaml", BindingPolicy::WarnOnGaps);
    }

    #[test]
    #[should_panic(expected = "Cannot parse binding YAML")]
    fn verify_bindings_panics_on_invalid_yaml() {
        let (_dir, path) = write_temp_yaml("not: [valid: {{yaml");
        verify_bindings(&path, BindingPolicy::AllImplemented);
    }

    // ── verify_bindings: empty bindings list ──

    #[test]
    fn verify_bindings_empty_bindings() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings: []
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        let result = verify_bindings(&path, BindingPolicy::AllImplemented);
        assert_eq!(result.bound_count, 0);
        assert_eq!(result.partial_count, 0);
        assert_eq!(result.not_implemented_count, 0);
    }

    // ── verify_bindings with real binding file ──

    #[test]
    fn verify_bindings_warn_on_gaps_real_file() {
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let result = verify_bindings(binding_path.to_str().unwrap(), BindingPolicy::WarnOnGaps);
        assert!(
            result.bound_count > 0,
            "Should have some implemented bindings"
        );
    }

    // ── verify_bindings: rerun-if-changed with parent directories ──

    #[test]
    fn verify_bindings_handles_nested_path() {
        // Tests the parent/grandparent rerun-if-changed logic
        let dir = tempfile::tempdir().expect("create tempdir");
        let nested = dir.path().join("contracts").join("crate");
        std::fs::create_dir_all(&nested).expect("create nested dirs");
        let yaml_path = nested.join("binding.yaml");
        std::fs::write(&yaml_path, &yaml_all_implemented()).expect("write yaml");
        let result = verify_bindings(yaml_path.to_str().unwrap(), BindingPolicy::AllImplemented);
        assert_eq!(result.bound_count, 2);
    }

    #[test]
    fn verify_bindings_partial_no_module_path() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: relu-v1.yaml
    equation: relu
    status: partial
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        // WarnOnGaps should show "(?" for missing module_path
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.partial_count, 1);
    }

    #[test]
    fn verify_bindings_not_implemented_no_module_path() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: relu-v1.yaml
    equation: relu
    status: not_implemented
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.not_implemented_count, 1);
    }

    #[test]
    fn verify_bindings_pending_no_module_path() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: relu-v1.yaml
    equation: relu
    status: pending
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        let result = verify_bindings(&path, BindingPolicy::WarnOnGaps);
        assert_eq!(result.not_implemented_count, 1);
    }

    // ── verify_source_functions tests ──

    #[test]
    fn verify_source_functions_missing_yaml() {
        let missing = verify_source_functions("/nonexistent/binding.yaml", "src/", false);
        assert!(missing.is_empty());
    }

    #[test]
    fn verify_source_functions_invalid_yaml() {
        let (_dir, path) = write_temp_yaml("not: [valid: {{yaml");
        let missing = verify_source_functions(&path, "src/", false);
        assert!(missing.is_empty());
    }

    #[test]
    fn verify_source_functions_no_implemented_bindings() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: gelu-v1.yaml
    equation: gelu
    status: not_implemented
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        let missing = verify_source_functions(&path, "src/", false);
        assert!(missing.is_empty());
    }

    #[test]
    fn verify_source_functions_all_found() {
        let dir = tempfile::tempdir().expect("create tempdir");

        // Write binding YAML with a function we'll create in source
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    function: "test_crate::my_softmax"
    status: implemented
  - contract: relu-v1.yaml
    equation: relu
    function: "test_crate::my_relu"
    status: implemented
"#;
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");

        // Write source files with matching pub fn declarations
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        std::fs::write(
            src_dir.join("lib.rs"),
            "pub fn my_softmax(x: &[f32]) -> Vec<f32> { vec![] }\npub fn my_relu(x: f32) -> f32 { x }\n",
        )
        .expect("write lib.rs");

        let missing = verify_source_functions(
            yaml_path.to_str().unwrap(),
            src_dir.to_str().unwrap(),
            false,
        );
        assert!(
            missing.is_empty(),
            "All functions should be found: {missing:?}"
        );
    }

    #[test]
    fn verify_source_functions_some_missing() {
        let dir = tempfile::tempdir().expect("create tempdir");

        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    function: "test_crate::my_softmax"
    status: implemented
  - contract: silu-v1.yaml
    equation: silu
    function: "test_crate::my_silu"
    status: implemented
"#;
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");

        // Only provide my_softmax, not my_silu
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        std::fs::write(
            src_dir.join("lib.rs"),
            "pub fn my_softmax(x: &[f32]) -> Vec<f32> { vec![] }\n",
        )
        .expect("write lib.rs");

        let missing = verify_source_functions(
            yaml_path.to_str().unwrap(),
            src_dir.to_str().unwrap(),
            false,
        );
        assert_eq!(missing.len(), 1);
        assert!(missing.contains(&"my_silu".to_string()));
    }

    #[test]
    #[should_panic(expected = "verify_source_functions")]
    fn verify_source_functions_hard_fail_panics() {
        let dir = tempfile::tempdir().expect("create tempdir");

        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: missing-v1.yaml
    equation: missing
    function: "test_crate::nonexistent_function"
    status: implemented
"#;
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");

        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        std::fs::write(src_dir.join("lib.rs"), "pub fn other() {}\n").expect("write lib.rs");

        verify_source_functions(
            yaml_path.to_str().unwrap(),
            src_dir.to_str().unwrap(),
            true, // hard_fail = true
        );
    }

    #[test]
    fn verify_source_functions_nonexistent_src_dir() {
        let dir = tempfile::tempdir().expect("create tempdir");

        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    function: "test_crate::softmax"
    status: implemented
"#;
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");

        // Point to a src_dir that doesn't exist
        let missing = verify_source_functions(
            yaml_path.to_str().unwrap(),
            "/tmp/nonexistent_src_dir_test_provable",
            false,
        );
        // Function should be missing since src_dir doesn't exist
        assert!(!missing.is_empty());
    }

    #[test]
    fn verify_source_functions_skips_not_implemented() {
        let dir = tempfile::tempdir().expect("create tempdir");

        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: gelu-v1.yaml
    equation: gelu
    function: "test_crate::gelu"
    status: not_implemented
  - contract: silu-v1.yaml
    equation: silu
    function: "test_crate::silu"
    status: partial
"#;
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");

        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        std::fs::write(src_dir.join("lib.rs"), "").expect("write empty lib.rs");

        // Only implemented bindings are checked
        let missing = verify_source_functions(
            yaml_path.to_str().unwrap(),
            src_dir.to_str().unwrap(),
            false,
        );
        assert!(
            missing.is_empty(),
            "Non-implemented bindings should be skipped"
        );
    }

    #[test]
    fn verify_source_functions_no_function_field() {
        let dir = tempfile::tempdir().expect("create tempdir");

        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: softmax-v1.yaml
    equation: softmax
    status: implemented
"#;
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, yaml).expect("write yaml");

        // No function field means nothing to check
        let missing = verify_source_functions(yaml_path.to_str().unwrap(), "/tmp/whatever", false);
        assert!(missing.is_empty());
    }

    #[test]
    fn verify_source_functions_many_missing_truncates() {
        let dir = tempfile::tempdir().expect("create tempdir");

        // Create > 10 implemented bindings so the output truncation triggers
        let mut bindings = String::new();
        for i in 0..15 {
            bindings.push_str(&format!(
                r#"  - contract: fn{i}-v1.yaml
    equation: fn{i}
    function: "test_crate::missing_fn_{i}"
    status: implemented
"#
            ));
        }
        let yaml = format!(
            r#"
version: "1.0.0"
target_crate: test_crate
bindings:
{bindings}"#
        );
        let yaml_path = dir.path().join("binding.yaml");
        std::fs::write(&yaml_path, &yaml).expect("write yaml");

        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).expect("create src dir");
        std::fs::write(src_dir.join("lib.rs"), "").expect("write empty lib.rs");

        let missing = verify_source_functions(
            yaml_path.to_str().unwrap(),
            src_dir.to_str().unwrap(),
            false,
        );
        assert_eq!(missing.len(), 15);
    }

    // ── scan_source_fns tests ──

    #[test]
    fn scan_source_fns_empty_dir() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.is_empty());
    }

    #[test]
    fn scan_source_fns_finds_pub_fn() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub fn my_function(x: i32) -> i32 { x }\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("my_function"));
    }

    #[test]
    fn scan_source_fns_finds_pub_async_fn() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub async fn fetch_data() -> Vec<u8> { vec![] }\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("fetch_data"));
    }

    #[test]
    fn scan_source_fns_finds_pub_crate_fn() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub(crate) fn internal_helper() {}\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("internal_helper"));
    }

    #[test]
    fn scan_source_fns_ignores_private_fn() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(dir.path().join("lib.rs"), "fn private_function() {}\n").expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.is_empty());
    }

    #[test]
    fn scan_source_fns_skips_non_rs_files() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(
            dir.path().join("data.txt"),
            "pub fn should_be_ignored() {}\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.is_empty());
    }

    #[test]
    fn scan_source_fns_recursive() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let subdir = dir.path().join("submod");
        std::fs::create_dir_all(&subdir).expect("create subdir");
        std::fs::write(subdir.join("inner.rs"), "pub fn nested_function() {}\n").expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("nested_function"));
    }

    #[test]
    fn scan_source_fns_skips_target_dir() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let target_dir = dir.path().join("target");
        std::fs::create_dir_all(&target_dir).expect("create target dir");
        std::fs::write(
            target_dir.join("generated.rs"),
            "pub fn should_be_skipped() {}\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.is_empty());
    }

    #[test]
    fn scan_source_fns_skips_git_dir() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let git_dir = dir.path().join(".git");
        std::fs::create_dir_all(&git_dir).expect("create .git dir");
        std::fs::write(git_dir.join("hook.rs"), "pub fn should_be_skipped() {}\n").expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.is_empty());
    }

    #[test]
    fn scan_source_fns_handles_generic_fn() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub fn generic_fn<T: Clone>(x: T) -> T { x }\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("generic_fn"));
    }

    #[test]
    fn scan_source_fns_case_insensitive() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(dir.path().join("lib.rs"), "pub fn MyMixedCase() {}\n").expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("mymixedcase"), "should be lowercased");
    }

    #[test]
    fn scan_source_fns_multiple_fns_one_file() {
        let dir = tempfile::tempdir().expect("create tempdir");
        std::fs::write(
            dir.path().join("lib.rs"),
            "pub fn alpha() {}\npub fn beta() {}\npub async fn gamma() {}\n",
        )
        .expect("write");
        let mut found = std::collections::HashSet::new();
        scan_source_fns(dir.path(), &mut found);
        assert!(found.contains("alpha"));
        assert!(found.contains("beta"));
        assert!(found.contains("gamma"));
        assert_eq!(found.len(), 3);
    }

    #[test]
    fn scan_source_fns_nonexistent_dir() {
        let mut found = std::collections::HashSet::new();
        scan_source_fns(
            Path::new("/tmp/nonexistent_dir_test_provable_xyz"),
            &mut found,
        );
        assert!(found.is_empty());
    }

    // ── verify_bindings: partial with module_path set ──

    #[test]
    #[should_panic(expected = "status 'partial'")]
    fn verify_bindings_all_implemented_partial_with_module_path() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: relu-v1.yaml
    equation: relu
    module_path: "test::relu"
    function: my_relu
    status: partial
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        verify_bindings(&path, BindingPolicy::AllImplemented);
    }

    #[test]
    #[should_panic(expected = "status 'not_implemented'")]
    fn verify_bindings_all_implemented_not_impl_with_module_path() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: relu-v1.yaml
    equation: relu
    module_path: "test::relu"
    function: my_relu
    status: not_implemented
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        verify_bindings(&path, BindingPolicy::AllImplemented);
    }

    #[test]
    #[should_panic(expected = "status 'not_implemented'")]
    fn verify_bindings_tiered_not_impl_with_module_path() {
        let yaml = r#"
version: "1.0.0"
target_crate: test_crate
bindings:
  - contract: relu-v1.yaml
    equation: relu
    module_path: "test::relu"
    function: my_relu
    status: not_implemented
"#;
        let (_dir, path) = write_temp_yaml(yaml);
        verify_bindings(&path, BindingPolicy::TieredEnforcement);
    }

    // ── verify_bindings: path with no parent (edge case) ──

    #[test]
    #[should_panic(expected = "Cannot read binding YAML")]
    fn verify_bindings_bare_filename() {
        // A bare filename with no directory separators
        verify_bindings("nonexistent.yaml", BindingPolicy::WarnOnGaps);
    }
}
