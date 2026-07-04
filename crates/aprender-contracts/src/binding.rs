//! Binding registry — maps contract equations to implementations.
//!
//! A `BindingRegistry` connects kernel contract equations (defined in
//! YAML) to the actual Rust functions that implement them in a target
//! crate (e.g. aprender). This enables:
//!
//! - **Audit**: `pv audit --binding` reports which obligations have
//!   implementations and which are gaps.
//! - **Wired tests**: `pv probar --binding` generates property tests
//!   that call real functions instead of `unimplemented!()`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ContractError;

/// Top-level binding registry parsed from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingRegistry {
    pub version: String,
    pub target_crate: String,
    /// Developer-declared critical path functions (Section 28).
    /// CD2 completeness = `critical_path` entries with bindings / len.
    #[serde(default)]
    pub critical_path: Vec<String>,
    #[serde(default)]
    pub bindings: Vec<KernelBinding>,
}

/// A single binding: one contract equation mapped to one implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelBinding {
    /// Contract YAML filename (e.g. "softmax-kernel-v1.yaml").
    pub contract: String,
    /// Equation name within the contract (e.g. "softmax").
    pub equation: String,
    /// Full Rust module path (e.g. `aprender::nn::functional::softmax`).
    #[serde(default)]
    pub module_path: Option<String>,
    /// Function or method name.
    #[serde(default)]
    pub function: Option<String>,
    /// Full Rust signature string.
    #[serde(default)]
    pub signature: Option<String>,
    /// Implementation status.
    pub status: ImplStatus,
    /// Free-form notes.
    #[serde(default)]
    pub notes: Option<String>,
}

/// Implementation status of a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplStatus {
    /// Fully implemented and ready for use.
    Implemented,
    /// Partially implemented with known gaps.
    Partial,
    /// Not yet implemented.
    NotImplemented,
    /// Planned but not started — skipped by enforcement checks.
    Pending,
}

/// Display implementation status as a `snake_case` string
impl std::fmt::Display for ImplStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Implemented => "implemented",
            Self::Partial => "partial",
            Self::NotImplemented => "not_implemented",
            Self::Pending => "pending",
        };
        write!(f, "{s}")
    }
}

/// Parse a binding registry YAML file.
///
/// # Errors
///
/// Returns [`ContractError::Io`] if the file cannot be read,
/// or [`ContractError::Yaml`] if the YAML is malformed.
pub fn parse_binding(path: &Path) -> Result<BindingRegistry, ContractError> {
    let content = std::fs::read_to_string(path)?;
    parse_binding_str(&content)
}

/// Parse a binding registry from a YAML string.
pub fn parse_binding_str(yaml: &str) -> Result<BindingRegistry, ContractError> {
    let registry: BindingRegistry = serde_yaml::from_str(yaml)?;
    Ok(registry)
}

/// Normalize a contract identifier by stripping `.yaml`/`.yml` extension.
///
/// Both binding entries (`contract: foo-v1.yaml`) and file stems (`foo-v1`)
/// are normalized to the bare stem so comparisons work regardless of whether
/// the caller used a filename or stem.
pub fn normalize_contract_id(id: &str) -> &str {
    id.strip_suffix(".yaml")
        .or_else(|| id.strip_suffix(".yml"))
        .unwrap_or(id)
}

impl BindingRegistry {
    /// Find all bindings matching a contract (normalizes both sides).
    pub fn bindings_for(&self, contract_id: &str) -> Vec<&KernelBinding> {
        let needle = normalize_contract_id(contract_id);
        self.bindings
            .iter()
            .filter(|b| normalize_contract_id(&b.contract) == needle)
            .collect()
    }

    /// Find a specific binding by contract + equation (normalizes contract).
    pub fn find_binding(&self, contract_id: &str, equation: &str) -> Option<&KernelBinding> {
        let needle = normalize_contract_id(contract_id);
        self.bindings
            .iter()
            .find(|b| normalize_contract_id(&b.contract) == needle && b.equation == equation)
    }

    /// L5 verification: return a copy of this registry in which every binding
    /// marked `implemented` whose `function` is NOT actually defined in the
    /// source tree under `source_root` is downgraded to `not_implemented`.
    ///
    /// This turns the L5 predicate "all bindings **verified** as implemented"
    /// (see [`crate::proof_status`]) into a fact instead of a self-declared YAML
    /// flag: a binding only survives as `implemented` if a real `fn <function>`
    /// exists in source. Rename or delete the function and the binding is
    /// downgraded, dropping the contract below L5 — the check is falsifiable.
    ///
    /// The source tree is scanned once (all `.rs` files, skipping build/vcs
    /// dirs) and the resulting function-name set is reused for every binding.
    #[must_use]
    pub fn verified(&self, source_root: &Path) -> BindingRegistry {
        let fn_names = collect_fn_names(source_root);
        let bindings = self
            .bindings
            .iter()
            .map(|b| {
                let mut b = b.clone();
                if b.status == ImplStatus::Implemented && !b.function_defined_in(&fn_names) {
                    b.status = ImplStatus::NotImplemented;
                }
                b
            })
            .collect();
        BindingRegistry {
            version: self.version.clone(),
            target_crate: self.target_crate.clone(),
            critical_path: self.critical_path.clone(),
            bindings,
        }
    }
}

impl KernelBinding {
    /// True when this binding's `function` is present in the given set of
    /// function names discovered in source. A binding with no `function` field
    /// cannot be verified and returns `false`.
    #[must_use]
    pub fn function_defined_in(&self, fn_names: &std::collections::HashSet<String>) -> bool {
        self.function
            .as_deref()
            .is_some_and(|f| fn_names.contains(f))
    }
}

/// Collect every Rust function name (`fn <name>`) defined in `.rs` files under
/// `root`, skipping `target/`, `.git/`, `.lake/`, and `node_modules/`. Used by
/// [`BindingRegistry::verified`] to check bindings point to real code.
#[must_use]
pub fn collect_fn_names(root: &Path) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skip = matches!(
                    path.file_name().and_then(|n| n.to_str()),
                    Some("target" | ".git" | ".lake" | "node_modules")
                );
                if !skip {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|e| e == "rs") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    extract_fn_names(&content, &mut names);
                }
            }
        }
    }
    names
}

/// Extract `fn <name>` identifiers from a Rust source string into `names`.
fn extract_fn_names(content: &str, names: &mut std::collections::HashSet<String>) {
    for line in content.lines() {
        let mut rest = line;
        while let Some(pos) = rest.find("fn ") {
            // Require `fn ` to start a word (preceded by start/space) to avoid
            // matching identifiers like `my_fn `.
            let ok_boundary = pos == 0
                || rest[..pos]
                    .chars()
                    .next_back()
                    .is_some_and(|c| !c.is_alphanumeric() && c != '_');
            let after = &rest[pos + 3..];
            if ok_boundary {
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    names.insert(name);
                }
            }
            rest = after;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_binding() {
        let yaml = r#"
version: "1.0.0"
target_crate: aprender
bindings: []
"#;
        let reg = parse_binding_str(yaml).unwrap();
        assert_eq!(reg.version, "1.0.0");
        assert_eq!(reg.target_crate, "aprender");
        assert!(reg.bindings.is_empty());
    }

    #[test]
    fn parse_binding_with_entries() {
        let yaml = r#"
version: "1.0.0"
target_crate: aprender
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    module_path: "aprender::nn::functional::softmax"
    function: softmax
    signature: "fn softmax(x: &Tensor, dim: i32) -> Tensor"
    status: implemented
  - contract: activation-kernel-v1.yaml
    equation: silu
    status: not_implemented
    notes: "Not yet available"
"#;
        let reg = parse_binding_str(yaml).unwrap();
        assert_eq!(reg.bindings.len(), 2);
        assert_eq!(reg.bindings[0].equation, "softmax");
        assert_eq!(reg.bindings[0].status, ImplStatus::Implemented);
        assert!(reg.bindings[0].module_path.is_some());
        assert_eq!(reg.bindings[1].equation, "silu");
        assert_eq!(reg.bindings[1].status, ImplStatus::NotImplemented);
        assert!(reg.bindings[1].module_path.is_none());
    }

    #[test]
    fn parse_partial_status() {
        let yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: partial
    notes: "Only scalar path"
"#;
        let reg = parse_binding_str(yaml).unwrap();
        assert_eq!(reg.bindings[0].status, ImplStatus::Partial);
    }

    #[test]
    fn impl_status_display() {
        assert_eq!(ImplStatus::Implemented.to_string(), "implemented");
        assert_eq!(ImplStatus::Partial.to_string(), "partial");
        assert_eq!(ImplStatus::NotImplemented.to_string(), "not_implemented");
        assert_eq!(ImplStatus::Pending.to_string(), "pending");
    }

    #[test]
    fn parse_invalid_binding_yaml() {
        let result = parse_binding_str("not: [valid: {{");
        assert!(result.is_err());
    }

    #[test]
    fn parse_binding_from_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let reg = parse_binding(&path).unwrap();
        assert_eq!(reg.target_crate, "aprender");
        assert!(!reg.bindings.is_empty());
    }

    #[test]
    fn parse_binding_nonexistent_file() {
        let result = parse_binding(std::path::Path::new("/nonexistent/binding.yaml"));
        assert!(result.is_err());
    }

    // ── L5 binding-verification feature ──

    #[test]
    fn extract_fn_names_finds_definitions() {
        let mut names = std::collections::HashSet::new();
        extract_fn_names(
            "pub fn to_anthropic(m: &Message) -> Value {\n  async fn helper() {}\n",
            &mut names,
        );
        assert!(names.contains("to_anthropic"));
        assert!(names.contains("helper"));
    }

    #[test]
    fn extract_fn_names_respects_word_boundary() {
        let mut names = std::collections::HashSet::new();
        // `my_fn foo` must NOT register `foo` (the `fn ` is inside `my_fn `).
        extract_fn_names("let my_fn foo = 1;", &mut names);
        assert!(!names.contains("foo"));
    }

    #[test]
    fn function_defined_in_checks_membership() {
        let names: std::collections::HashSet<String> =
            ["to_anthropic".to_string()].into_iter().collect();
        let bound = KernelBinding {
            contract: "c-v1.yaml".into(),
            equation: "e".into(),
            module_path: None,
            function: Some("to_anthropic".into()),
            signature: None,
            status: ImplStatus::Implemented,
            notes: None,
        };
        assert!(bound.function_defined_in(&names));

        let missing = KernelBinding {
            function: Some("does_not_exist".into()),
            ..bound.clone()
        };
        assert!(!missing.function_defined_in(&names));

        // No function field → cannot be verified.
        let no_fn = KernelBinding {
            function: None,
            ..bound
        };
        assert!(!no_fn.function_defined_in(&names));
    }

    #[test]
    fn verified_downgrades_phantom_implemented_bindings() {
        // A temp source tree with exactly one real function.
        let dir = std::env::temp_dir().join(format!("bindver_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("lib.rs"), "pub fn real_one() {}\n").unwrap();

        let reg = BindingRegistry {
            version: "1.0.0".into(),
            target_crate: "t".into(),
            critical_path: vec![],
            bindings: vec![
                KernelBinding {
                    contract: "c-v1.yaml".into(),
                    equation: "a".into(),
                    module_path: None,
                    function: Some("real_one".into()),
                    signature: None,
                    status: ImplStatus::Implemented,
                    notes: None,
                },
                KernelBinding {
                    contract: "c-v1.yaml".into(),
                    equation: "b".into(),
                    module_path: None,
                    function: Some("phantom".into()),
                    signature: None,
                    status: ImplStatus::Implemented,
                    notes: None,
                },
            ],
        };

        let v = reg.verified(&dir);
        // Real fn stays implemented; phantom is downgraded.
        assert_eq!(v.bindings[0].status, ImplStatus::Implemented);
        assert_eq!(v.bindings[1].status, ImplStatus::NotImplemented);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
