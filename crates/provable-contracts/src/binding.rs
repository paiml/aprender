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
}
