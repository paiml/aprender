//! Contract Verification Gap Analysis (BH-26)
//!
//! Analyzes provable-contracts binding registries and contract YAML files
//! to find verification gaps: unimplemented bindings, partial bindings,
//! and contracts with insufficient proof obligation coverage.

use super::{DefectCategory, Finding, FindingEvidence, FindingSeverity, HuntMode};
use serde::Deserialize;
use std::path::Path;

// ============================================================================
// Serde types mirroring provable-contracts format (no cross-crate dep)
// ============================================================================

#[derive(Deserialize)]
struct BindingRegistry {
    target_crate: String,
    bindings: Vec<KernelBinding>,
}

#[derive(Deserialize)]
struct KernelBinding {
    contract: String,
    equation: String,
    status: String,
    notes: Option<String>,
    module_path: Option<String>,
}

#[derive(Deserialize)]
struct ContractFile {
    metadata: ContractMetadata,
    #[serde(default)]
    proof_obligations: Vec<ProofObligation>,
    #[serde(default)]
    falsification_tests: Vec<FalsificationTest>,
}

#[derive(Deserialize)]
struct ContractMetadata {
    version: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize)]
struct ProofObligation {
    property: Option<String>,
    #[serde(rename = "type")]
    obligation_type: Option<String>,
}

#[derive(Deserialize)]
struct FalsificationTest {
    name: Option<String>,
}

// ============================================================================
// Public API
// ============================================================================

/// Discover the provable-contracts directory.
///
/// Checks explicit path first, then auto-discovers `../provable-contracts/contracts/`.
pub fn discover_contracts_dir(
    project_path: &Path,
    explicit_path: Option<&Path>,
) -> Option<std::path::PathBuf> {
    if let Some(p) = explicit_path {
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    // Auto-discover in parent directory (canonicalize to resolve ".")
    let resolved = project_path.canonicalize().ok()?;
    let parent = resolved.parent()?;
    let auto_path = parent.join("provable-contracts").join("contracts");
    if auto_path.is_dir() {
        Some(auto_path)
    } else {
        None
    }
}

/// Analyze contract verification gaps.
///
/// Produces `BH-CONTRACT-NNNN` findings for:
/// 1. Bindings with status `not_implemented` or `partial`
/// 2. Contract YAMLs with no binding reference
/// 3. Contracts where <50% of proof obligations have falsification tests
pub fn analyze_contract_gaps(contracts_dir: &Path, _project_path: &Path) -> Vec<Finding> {
    contract_pre_analyze!(contracts_dir);
    let mut findings = Vec::new();
    let mut finding_id = 0u32;

    // Collect bound contract names from all binding registries
    let mut bound_contracts: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Check 1: Binding gap analysis
    let binding_pattern = format!("{}/**/binding.yaml", contracts_dir.display());
    if let Ok(entries) = glob::glob(&binding_pattern) {
        for entry in entries.flatten() {
            analyze_binding_file(&entry, &mut findings, &mut finding_id, &mut bound_contracts);
        }
    }

    // Check 2: Unbound contracts
    let contract_pattern = format!("{}/*.yaml", contracts_dir.display());
    if let Ok(entries) = glob::glob(&contract_pattern) {
        for entry in entries.flatten() {
            let file_name = entry.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if file_name == "binding.yaml" || !file_name.ends_with(".yaml") {
                continue;
            }
            if !bound_contracts.contains(file_name) {
                finding_id += 1;
                findings.push(
                    Finding::new(
                        format!("BH-CONTRACT-{:04}", finding_id),
                        &entry,
                        1,
                        format!("Unbound contract: {}", file_name),
                    )
                    .with_description(
                        "Contract YAML has no binding reference in any binding.yaml registry",
                    )
                    .with_severity(FindingSeverity::Medium)
                    .with_category(DefectCategory::ContractGap)
                    .with_suspiciousness(0.5)
                    .with_discovered_by(HuntMode::Analyze)
                    .with_evidence(FindingEvidence::contract_binding(file_name, "none", "unbound")),
                );
            }

            // Check 3: Proof obligation coverage
            analyze_obligation_coverage(&entry, file_name, &mut findings, &mut finding_id);
        }
    }

    findings
}

// ============================================================================
// Internal helpers
// ============================================================================

fn analyze_binding_file(
    path: &Path,
    findings: &mut Vec<Finding>,
    finding_id: &mut u32,
    bound_contracts: &mut std::collections::HashSet<String>,
) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(registry) = serde_yaml_ng::from_str::<BindingRegistry>(&content) else {
        return;
    };

    for binding in &registry.bindings {
        bound_contracts.insert(binding.contract.clone());

        let (severity, suspiciousness, desc) = match binding.status.as_str() {
            "not_implemented" => (
                FindingSeverity::High,
                0.8,
                format!(
                    "Binding `{}::{}` has no implementation{}",
                    binding.contract,
                    binding.equation,
                    binding.notes.as_deref().map(|n| format!(" — {}", n)).unwrap_or_default()
                ),
            ),
            "partial" => (
                FindingSeverity::Medium,
                0.6,
                format!(
                    "Binding `{}::{}` is partially implemented{}",
                    binding.contract,
                    binding.equation,
                    binding.notes.as_deref().map(|n| format!(" — {}", n)).unwrap_or_default()
                ),
            ),
            _ => continue,
        };

        *finding_id += 1;
        findings.push(
            Finding::new(
                format!("BH-CONTRACT-{:04}", finding_id),
                path,
                1,
                format!(
                    "Contract gap: {} — {} ({})",
                    binding.contract, binding.equation, binding.status
                ),
            )
            .with_description(desc)
            .with_severity(severity)
            .with_category(DefectCategory::ContractGap)
            .with_suspiciousness(suspiciousness)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::contract_binding(
                &binding.contract,
                &binding.equation,
                &binding.status,
            )),
        );
    }
}

fn analyze_obligation_coverage(
    path: &Path,
    file_name: &str,
    findings: &mut Vec<Finding>,
    finding_id: &mut u32,
) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(contract) = serde_yaml_ng::from_str::<ContractFile>(&content) else {
        return;
    };

    let total_obligations = contract.proof_obligations.len();
    let total_tests = contract.falsification_tests.len();
    if total_obligations == 0 {
        return;
    }

    let coverage_ratio = total_tests as f64 / total_obligations as f64;
    if coverage_ratio < 0.5 {
        *finding_id += 1;
        findings.push(
            Finding::new(
                format!("BH-CONTRACT-{:04}", finding_id),
                path,
                1,
                format!(
                    "Low obligation coverage: {} ({}/{})",
                    file_name, total_tests, total_obligations
                ),
            )
            .with_description(format!(
                "Only {:.0}% of proof obligations have falsification tests",
                coverage_ratio * 100.0
            ))
            .with_severity(FindingSeverity::Low)
            .with_category(DefectCategory::ContractGap)
            .with_suspiciousness(0.4)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::contract_binding(
                file_name,
                "obligations",
                format!("{}/{}", total_tests, total_obligations),
            )),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Filter findings by title substring — reduces repeated filter chains.
    fn by_title<'a>(findings: &'a [Finding], pattern: &str) -> Vec<&'a Finding> {
        findings.iter().filter(|f| f.title.contains(pattern)).collect()
    }

    #[test]
    fn test_parse_binding_registry() {
        let yaml = r#"
version: "1.0.0"
target_crate: aprender
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    status: implemented
    module_path: "aprender::nn::softmax"
  - contract: matmul-kernel-v1.yaml
    equation: matmul
    status: not_implemented
    notes: "No public function"
"#;
        let registry: BindingRegistry =
            serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(registry.target_crate, "aprender");
        assert_eq!(registry.bindings.len(), 2);
        assert_eq!(registry.bindings[0].status, "implemented");
        assert_eq!(registry.bindings[1].status, "not_implemented");
    }

    #[test]
    fn test_analyze_bindings_not_implemented() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let crate_dir = dir.path().join("aprender");
        std::fs::create_dir_all(&crate_dir).expect("mkdir failed");
        let binding_path = crate_dir.join("binding.yaml");
        {
            let mut f = std::fs::File::create(&binding_path).expect("file open failed");
            write!(
                f,
                r#"
target_crate: aprender
bindings:
  - contract: matmul-kernel-v1.yaml
    equation: matmul
    status: not_implemented
    notes: "Missing"
"#
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let not_impl = by_title(&findings, "not_implemented");
        assert!(!not_impl.is_empty());
        assert_eq!(not_impl[0].severity, FindingSeverity::High);
        assert!((not_impl[0].suspiciousness - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_analyze_bindings_partial() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let crate_dir = dir.path().join("test_crate");
        std::fs::create_dir_all(&crate_dir).expect("mkdir failed");
        let binding_path = crate_dir.join("binding.yaml");
        {
            let mut f = std::fs::File::create(&binding_path).expect("file open failed");
            write!(
                f,
                r#"
target_crate: test_crate
bindings:
  - contract: attn-kernel-v1.yaml
    equation: attention
    status: partial
    notes: "Only supports 2D"
"#
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let partial = by_title(&findings, "partial");
        assert!(!partial.is_empty());
        assert_eq!(partial[0].severity, FindingSeverity::Medium);
        assert!((partial[0].suspiciousness - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_discover_explicit_path() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let contracts = dir.path().join("my-contracts");
        std::fs::create_dir_all(&contracts).expect("mkdir failed");

        let result = discover_contracts_dir(dir.path(), Some(&contracts));
        assert!(result.is_some());
        assert_eq!(result.expect("operation failed"), contracts);
    }

    #[test]
    fn test_discover_explicit_path_missing() {
        let outer = tempfile::tempdir().expect("tempdir creation failed");
        std::fs::create_dir(outer.path().join("p")).expect("mkdir failed");
        let inner = outer.path().join("p");
        assert!(discover_contracts_dir(&inner, Some(&inner.join("x"))).is_none());
    }

    #[test]
    fn test_unbound_contract_detection() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        // Create a contract YAML with no binding
        let contract_path = dir.path().join("orphan-kernel-v1.yaml");
        {
            let mut f = std::fs::File::create(&contract_path).expect("file open failed");
            write!(
                f,
                r#"
metadata:
  version: "1.0.0"
  description: "Orphan kernel"
proof_obligations: []
falsification_tests: []
"#
            )
            .expect("unexpected failure");
        }
        // No binding.yaml exists, so orphan-kernel-v1.yaml is unbound
        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let unbound = by_title(&findings, "Unbound");
        assert!(!unbound.is_empty());
        assert_eq!(unbound[0].severity, FindingSeverity::Medium);
    }

    #[test]
    fn test_obligation_coverage_low() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let contract_path = dir.path().join("test-kernel-v1.yaml");
        {
            let mut f = std::fs::File::create(&contract_path).expect("file open failed");
            write!(
                f,
                r#"
metadata:
  version: "1.0.0"
  description: "Test"
proof_obligations:
  - type: invariant
    property: "shape"
  - type: associativity
    property: "assoc"
  - type: linearity
    property: "linear"
  - type: equivalence
    property: "simd"
falsification_tests:
  - name: "test_shape"
"#
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let low_cov = by_title(&findings, "Low obligation coverage");
        assert!(!low_cov.is_empty());
        assert_eq!(low_cov[0].severity, FindingSeverity::Low);
    }

    // ===== Falsification tests =====

    #[test]
    fn test_falsify_malformed_binding_yaml() {
        // Malformed YAML → gracefully ignored (0 findings from that file)
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let crate_dir = dir.path().join("broken");
        std::fs::create_dir_all(&crate_dir).expect("mkdir failed");
        std::fs::write(crate_dir.join("binding.yaml"), "{{{{not valid yaml at all!!!!")
            .expect("unexpected failure");

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        // Should not panic, just skip the malformed file
        let binding_findings = by_title(&findings, "Contract gap:");
        assert_eq!(binding_findings.len(), 0);
    }

    #[test]
    fn test_falsify_malformed_contract_yaml() {
        // Contract YAML with invalid structure → no obligation findings
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        std::fs::write(dir.path().join("bad-kernel-v1.yaml"), "not: a: valid: contract: [")
            .expect("unexpected failure");

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        // Should get unbound finding but no obligation crash
        let unbound = by_title(&findings, "Unbound");
        assert_eq!(unbound.len(), 1);
        let obligation = by_title(&findings, "obligation");
        assert_eq!(obligation.len(), 0);
    }

    #[test]
    fn test_falsify_empty_bindings_list() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let crate_dir = dir.path().join("empty");
        std::fs::create_dir_all(&crate_dir).expect("mkdir failed");
        {
            let mut f =
                std::fs::File::create(crate_dir.join("binding.yaml")).expect("file open failed");
            write!(f, "target_crate: empty\nbindings: []\n").expect("write failed");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let binding_findings = by_title(&findings, "Contract gap:");
        assert_eq!(binding_findings.len(), 0);
    }

    #[test]
    fn test_falsify_obligation_coverage_exact_50pct() {
        // Exactly 50% coverage → should NOT trigger (threshold is <50%)
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let contract_path = dir.path().join("exact50-kernel-v1.yaml");
        {
            let mut f = std::fs::File::create(&contract_path).expect("file open failed");
            write!(
                f,
                r#"
metadata:
  version: "1.0.0"
  description: "Boundary test"
proof_obligations:
  - type: invariant
    property: "shape"
  - type: associativity
    property: "assoc"
falsification_tests:
  - name: "test_shape"
"#
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let low_cov = by_title(&findings, "Low obligation coverage");
        assert_eq!(low_cov.len(), 0, "50% is at threshold, not below");
    }

    #[test]
    fn test_falsify_obligation_coverage_zero_obligations() {
        // 0 obligations → should NOT trigger (early return)
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let contract_path = dir.path().join("noobs-kernel-v1.yaml");
        {
            let mut f = std::fs::File::create(&contract_path).expect("file open failed");
            write!(
                f,
                r#"
metadata:
  version: "1.0.0"
  description: "No obligations"
proof_obligations: []
falsification_tests:
  - name: "test_something"
"#
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let low_cov = by_title(&findings, "Low obligation coverage");
        assert_eq!(low_cov.len(), 0, "0 obligations → no coverage finding");
    }

    #[test]
    fn test_falsify_bound_contract_still_gets_obligation_check() {
        // Bound contract with low obligation coverage → BOTH bound + low coverage
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        // Create contract with low obligation coverage
        let contract_path = dir.path().join("matmul-kernel-v1.yaml");
        {
            let mut f = std::fs::File::create(&contract_path).expect("file open failed");
            write!(
                f,
                r#"
metadata:
  version: "1.0.0"
  description: "Matmul"
proof_obligations:
  - type: invariant
    property: "shape"
  - type: associativity
    property: "assoc"
  - type: commutativity
    property: "commute"
falsification_tests: []
"#
            )
            .expect("unexpected failure");
        }
        // Create binding that references this contract
        let crate_dir = dir.path().join("test_crate");
        std::fs::create_dir_all(&crate_dir).expect("mkdir failed");
        {
            let mut f =
                std::fs::File::create(crate_dir.join("binding.yaml")).expect("file open failed");
            write!(
                f,
                "target_crate: test_crate\nbindings:\n  - contract: matmul-kernel-v1.yaml\n    equation: matmul\n    status: implemented\n"
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        // Should NOT be unbound (it has a binding)
        let unbound = by_title(&findings, "Unbound");
        assert_eq!(unbound.len(), 0, "Bound contract should not be flagged as unbound");
        // SHOULD still get low obligation coverage
        let low_cov = by_title(&findings, "Low obligation coverage");
        assert_eq!(low_cov.len(), 1, "Bound contract should still get obligation check");
    }

    #[test]
    fn test_falsify_discover_nonexistent_parent() {
        let result = discover_contracts_dir(Path::new("/nonexistent/path/xyz"), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_falsify_implemented_bindings_not_flagged() {
        // Bindings with status "implemented" → 0 findings
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let crate_dir = dir.path().join("good_crate");
        std::fs::create_dir_all(&crate_dir).expect("mkdir failed");
        {
            let mut f =
                std::fs::File::create(crate_dir.join("binding.yaml")).expect("file open failed");
            write!(
                f,
                r#"
target_crate: good_crate
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    status: implemented
  - contract: matmul-kernel-v1.yaml
    equation: matmul
    status: implemented
"#
            )
            .expect("unexpected failure");
        }

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let gaps = by_title(&findings, "Contract gap:");
        assert_eq!(gaps.len(), 0, "Implemented bindings should not be flagged");
    }

    #[test]
    fn test_contract_findings_suspiciousness_values() {
        // Verify suspiciousness values are set correctly for min_suspiciousness filtering
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let binding_dir = dir.path().join("aprender");
        std::fs::create_dir_all(&binding_dir).expect("mkdir failed");
        std::fs::write(
            binding_dir.join("binding.yaml"),
            r#"
target_crate: aprender
bindings:
  - contract: kernel-v1.yaml
    equation: eq1
    status: not_implemented
  - contract: kernel-v2.yaml
    equation: eq2
    status: partial
"#,
        )
        .expect("unexpected failure");

        let findings = analyze_contract_gaps(dir.path(), dir.path());
        let not_impl = by_title(&findings, "not_implemented");
        let partial = by_title(&findings, "partial");

        assert!(!not_impl.is_empty(), "Should find not_implemented binding");
        assert!(!partial.is_empty(), "Should find partial binding");

        // not_implemented = High severity → suspiciousness 0.8
        assert!(
            (not_impl[0].suspiciousness - 0.8).abs() < 0.01,
            "not_implemented should have 0.8 suspiciousness, got {}",
            not_impl[0].suspiciousness
        );
        // partial = Medium severity → suspiciousness 0.6
        assert!(
            (partial[0].suspiciousness - 0.6).abs() < 0.01,
            "partial should have 0.6 suspiciousness, got {}",
            partial[0].suspiciousness
        );
    }
}
