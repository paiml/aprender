//! Cross-contract obligation coverage report.
//!
//! Analyzes a set of contracts (optionally with a binding registry)
//! to compute how many proof obligations are backed by falsification
//! tests, Kani harnesses, and implementation bindings.

use std::collections::BTreeMap;

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::Contract;

/// Coverage report across multiple contracts.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Per-contract coverage details.
    pub contracts: Vec<ContractCoverage>,
    /// Aggregate totals.
    pub totals: CoverageTotals,
}

/// Coverage details for a single contract.
#[derive(Debug, Clone)]
pub struct ContractCoverage {
    /// Contract stem (filename without `.yaml`).
    pub stem: String,
    /// Total equations.
    pub equations: usize,
    /// Total proof obligations.
    pub obligations: usize,
    /// Obligations covered by at least one falsification test.
    pub falsification_covered: usize,
    /// Obligations covered by at least one Kani harness.
    pub kani_covered: usize,
    /// Number of equations with an implemented binding.
    pub binding_implemented: usize,
    /// Number of equations with partial binding.
    pub binding_partial: usize,
    /// Number of equations with no binding.
    pub binding_missing: usize,
    /// Per-obligation-type breakdown.
    pub by_type: BTreeMap<String, usize>,
}

/// Aggregate totals across all contracts.
#[derive(Debug, Clone)]
pub struct CoverageTotals {
    /// Number of contracts analyzed.
    pub contracts: usize,
    /// Total equations across all contracts.
    pub equations: usize,
    /// Total proof obligations across all contracts.
    pub obligations: usize,
    /// Total falsification tests across all contracts.
    pub falsification_tests: usize,
    /// Total Kani harnesses across all contracts.
    pub kani_harnesses: usize,
    /// Equations with implemented bindings.
    pub binding_implemented: usize,
    /// Equations with partial bindings.
    pub binding_partial: usize,
    /// Equations with no binding entry.
    pub binding_missing: usize,
}

/// Compute a coverage report for a set of contracts.
///
/// If a `binding` registry is provided, binding coverage is
/// included. Otherwise binding fields are zero.
pub fn coverage_report(
    contracts: &[(String, &Contract)],
    binding: Option<&BindingRegistry>,
) -> CoverageReport {
    let mut results = Vec::new();
    let mut totals = CoverageTotals {
        contracts: contracts.len(),
        equations: 0,
        obligations: 0,
        falsification_tests: 0,
        kani_harnesses: 0,
        binding_implemented: 0,
        binding_partial: 0,
        binding_missing: 0,
    };

    for (stem, contract) in contracts {
        let contract_file = format!("{stem}.yaml");

        let equations = contract.equations.len();
        let obligations = contract.proof_obligations.len();
        let ft_count = contract.falsification_tests.len();
        let kh_count = contract.kani_harnesses.len();

        // Count obligation types
        let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
        for ob in &contract.proof_obligations {
            *by_type.entry(ob.obligation_type.to_string()).or_default() += 1;
        }

        // Falsification coverage: count obligations that have at
        // least one falsification test (heuristic: if any tests
        // exist, all obligations get some coverage).
        let falsification_covered = if ft_count > 0 { obligations } else { 0 };

        // Kani coverage: each harness covers one obligation
        let kani_covered = kh_count.min(obligations);

        // Binding coverage
        let (binding_implemented, binding_partial, binding_missing) = if let Some(reg) = binding {
            count_binding_coverage(&contract_file, contract, reg)
        } else {
            (0, 0, equations)
        };

        totals.equations += equations;
        totals.obligations += obligations;
        totals.falsification_tests += ft_count;
        totals.kani_harnesses += kh_count;
        totals.binding_implemented += binding_implemented;
        totals.binding_partial += binding_partial;
        totals.binding_missing += binding_missing;

        results.push(ContractCoverage {
            stem: stem.clone(),
            equations,
            obligations,
            falsification_covered,
            kani_covered,
            binding_implemented,
            binding_partial,
            binding_missing,
            by_type,
        });
    }

    CoverageReport {
        contracts: results,
        totals,
    }
}

/// Count implemented, partial, and missing bindings for a single contract
fn count_binding_coverage(
    contract_file: &str,
    contract: &Contract,
    binding: &BindingRegistry,
) -> (usize, usize, usize) {
    let mut implemented = 0usize;
    let mut partial = 0usize;
    let mut missing = 0usize;

    for eq_name in contract.equations.keys() {
        let status = binding
            .find_binding(contract_file, eq_name)
            .map(|b| b.status);

        match status {
            Some(ImplStatus::Implemented) => implemented += 1,
            Some(ImplStatus::Partial) => partial += 1,
            Some(ImplStatus::NotImplemented) | None => missing += 1,
            Some(ImplStatus::Pending) => {} // Pending: excluded from denominator
        }
    }

    (implemented, partial, missing)
}

/// Compute overall obligation coverage percentage.
///
/// An obligation is "covered" if it has both a falsification test
/// and a Kani harness (or at least one of them).
pub fn overall_percentage(report: &CoverageReport) -> f64 {
    if report.totals.obligations == 0 {
        return 100.0;
    }
    let covered: usize = report
        .contracts
        .iter()
        .map(|c| c.falsification_covered.max(c.kani_covered))
        .sum();
    #[allow(clippy::cast_precision_loss)]
    let pct = (covered as f64 / report.totals.obligations as f64) * 100.0;
    pct
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    fn contract_with_obs(n_obligations: usize, n_ft: usize, n_kani: usize) -> Contract {
        let mut yaml = String::from(
            r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
"#,
        );
        for i in 0..n_obligations {
            yaml.push_str(&format!(
                "  - type: invariant\n    property: \"prop {i}\"\n"
            ));
        }
        yaml.push_str("falsification_tests:\n");
        for i in 0..n_ft {
            yaml.push_str(&format!(
                "  - id: FT-{i:03}\n    rule: \"r\"\n    \
                 prediction: \"p\"\n    if_fails: \"f\"\n"
            ));
        }
        yaml.push_str("kani_harnesses:\n");
        for i in 0..n_kani {
            yaml.push_str(&format!(
                "  - id: KH-{i:03}\n    obligation: OBL-{i:03}\n    \
                 bound: 16\n"
            ));
        }
        parse_contract_str(&yaml).unwrap()
    }

    #[test]
    fn empty_contracts() {
        let report = coverage_report(&[], None);
        assert_eq!(report.totals.contracts, 0);
        assert!((overall_percentage(&report) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn single_contract_full_coverage() {
        let c = contract_with_obs(3, 3, 3);
        let report = coverage_report(&[("test".to_string(), &c)], None);
        assert_eq!(report.totals.obligations, 3);
        assert_eq!(report.totals.falsification_tests, 3);
        assert_eq!(report.totals.kani_harnesses, 3);
        assert_eq!(report.contracts[0].falsification_covered, 3);
        assert_eq!(report.contracts[0].kani_covered, 3);
        assert!((overall_percentage(&report) - 100.0).abs() < 0.01);
    }

    #[test]
    fn no_tests_zero_coverage() {
        let c = contract_with_obs(5, 0, 0);
        let report = coverage_report(&[("test".to_string(), &c)], None);
        assert_eq!(report.contracts[0].falsification_covered, 0);
        assert_eq!(report.contracts[0].kani_covered, 0);
        assert!((overall_percentage(&report) - 0.0).abs() < 0.01);
    }

    #[test]
    fn multiple_contracts() {
        let c1 = contract_with_obs(2, 2, 1);
        let c2 = contract_with_obs(3, 0, 3);
        let report = coverage_report(&[("a".to_string(), &c1), ("b".to_string(), &c2)], None);
        assert_eq!(report.totals.contracts, 2);
        assert_eq!(report.totals.obligations, 5);
        assert_eq!(report.totals.equations, 2);
    }

    #[test]
    fn binding_coverage_implemented() {
        let c = contract_with_obs(2, 2, 2);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
        )
        .unwrap();
        let report = coverage_report(&[("test".to_string(), &c)], Some(&binding));
        assert_eq!(report.contracts[0].binding_implemented, 1);
        assert_eq!(report.contracts[0].binding_missing, 0);
        assert_eq!(report.totals.binding_implemented, 1);
    }

    #[test]
    fn binding_coverage_partial() {
        let c = contract_with_obs(1, 1, 1);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: partial
"#,
        )
        .unwrap();
        let report = coverage_report(&[("test".to_string(), &c)], Some(&binding));
        assert_eq!(report.contracts[0].binding_partial, 1);
        assert_eq!(report.totals.binding_partial, 1);
    }

    #[test]
    fn binding_missing_when_no_entry() {
        let c = contract_with_obs(1, 1, 1);
        let binding = crate::binding::parse_binding_str(
            r#"
version: "1.0.0"
target_crate: test
bindings: []
"#,
        )
        .unwrap();
        let report = coverage_report(&[("test".to_string(), &c)], Some(&binding));
        assert_eq!(report.contracts[0].binding_missing, 1);
    }

    #[test]
    fn obligation_type_breakdown() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "p1"
  - type: bound
    property: "p2"
  - type: invariant
    property: "p3"
falsification_tests: []
"#;
        let c = parse_contract_str(yaml).unwrap();
        let report = coverage_report(&[("test".to_string(), &c)], None);
        let by_type = &report.contracts[0].by_type;
        assert_eq!(by_type.get("invariant"), Some(&2));
        assert_eq!(by_type.get("bound"), Some(&1));
    }

    #[test]
    fn kani_covered_capped_at_obligations() {
        // More harnesses than obligations
        let c = contract_with_obs(2, 0, 10);
        let report = coverage_report(&[("test".to_string(), &c)], None);
        assert_eq!(report.contracts[0].kani_covered, 2);
    }

    #[test]
    fn no_binding_defaults_to_missing() {
        let c = contract_with_obs(1, 1, 1);
        let report = coverage_report(&[("test".to_string(), &c)], None);
        assert_eq!(report.contracts[0].binding_missing, 1);
    }
}
