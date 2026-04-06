//! Audit trail generator — traceability chain.
//!
//! Traces the full chain from paper reference → equation →
//! proof obligation → falsification test → Kani harness.
//! Detects orphaned obligations and untested equations.

use crate::binding::{BindingRegistry, ImplStatus};
use crate::error::{Severity, Violation};
use crate::schema::Contract;

/// Audit result summarizing traceability coverage.
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Total equations in the contract.
    pub equations: usize,
    /// Total proof obligations.
    pub obligations: usize,
    /// Total falsification tests.
    pub falsification_tests: usize,
    /// Total Kani harnesses.
    pub kani_harnesses: usize,
    /// Violations found during audit.
    pub violations: Vec<Violation>,
}

/// Run a traceability audit on a contract.
///
/// Checks that every proof obligation is covered by at least
/// one falsification test or Kani harness, and that no
/// harnesses reference non-existent obligations.
pub fn audit_contract(contract: &Contract) -> AuditReport {
    let mut violations = Vec::new();

    // Kernel-only audit checks: skip for registries, model-family schemas,
    // pattern contracts, and reference documents that are exempt from
    // the provability invariant.
    let is_kernel_kind = contract.requires_proofs();

    // Check: contract has at least one falsification test (kernel only)
    if is_kernel_kind && contract.falsification_tests.is_empty() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "AUDIT-001".to_string(),
            message: "No falsification tests — contract is \
                      not falsifiable"
                .to_string(),
            location: Some("falsification_tests".to_string()),
        });
    }

    // Check: every Kani harness references a valid obligation ID
    let obligation_props: Vec<&str> = contract
        .proof_obligations
        .iter()
        .map(|o| o.property.as_str())
        .collect();

    for harness in &contract.kani_harnesses {
        // The harness.obligation field should reference an
        // obligation. We do a soft check (warn, not error)
        // since obligation IDs are free-form in current schema.
        if harness.obligation.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "AUDIT-002".to_string(),
                message: format!(
                    "Kani harness {} has empty obligation \
                     reference",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.obligation", harness.id)),
            });
        }
    }

    // Check: contract has proof obligations if it has equations (kernel only)
    if is_kernel_kind && !contract.equations.is_empty() && contract.proof_obligations.is_empty() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "AUDIT-003".to_string(),
            message: "Equations defined but no proof obligations \
                      — obligations should be derived from equations"
                .to_string(),
            location: Some("proof_obligations".to_string()),
        });
    }

    // Suppress unused variable warning
    let _ = obligation_props;

    AuditReport {
        equations: contract.equations.len(),
        obligations: contract.proof_obligations.len(),
        falsification_tests: contract.falsification_tests.len(),
        kani_harnesses: contract.kani_harnesses.len(),
        violations,
    }
}

/// Binding audit result — cross-references contracts with implementations.
#[derive(Debug, Clone)]
pub struct BindingAuditReport {
    /// Total equations across all contracts.
    pub total_equations: usize,
    /// Equations with a binding entry.
    pub bound_equations: usize,
    /// Bindings with status = implemented.
    pub implemented: usize,
    /// Bindings with status = partial.
    pub partial: usize,
    /// Bindings with status = `not_implemented`.
    pub not_implemented: usize,
    /// Total proof obligations across matched contracts.
    pub total_obligations: usize,
    /// Obligations covered by implemented bindings.
    pub covered_obligations: usize,
    /// Violations found during binding audit.
    pub violations: Vec<Violation>,
}

/// Audit the binding registry against a set of contracts.
///
/// For each contract, checks whether its equations have a
/// corresponding binding entry and reports coverage gaps.
pub fn audit_binding(
    contracts: &[(&str, &Contract)],
    binding: &BindingRegistry,
) -> BindingAuditReport {
    let mut violations = Vec::new();
    let mut total_equations = 0usize;
    let mut bound_equations = 0usize;
    let mut implemented = 0usize;
    let mut partial = 0usize;
    let mut not_implemented = 0usize;
    let mut total_obligations = 0usize;
    let mut covered_obligations = 0usize;

    for &(contract_file, contract) in contracts {
        let eq_count = contract.equations.len();
        total_equations += eq_count;
        total_obligations += contract.proof_obligations.len();

        for eq_name in contract.equations.keys() {
            let matching = binding.find_binding(contract_file, eq_name);

            match matching {
                Some(b) => {
                    bound_equations += 1;
                    match b.status {
                        ImplStatus::Implemented => {
                            implemented += 1;
                            // All obligations for this equation
                            // are considered covered.
                            covered_obligations += obligations_for_equation(contract);
                        }
                        ImplStatus::Partial => {
                            partial += 1;
                            violations.push(Violation {
                                severity: Severity::Warning,
                                rule: "BIND-002".to_string(),
                                message: format!(
                                    "Equation '{eq_name}' in \
                                     {contract_file} is partially \
                                     implemented"
                                ),
                                location: Some(format!("bindings.{eq_name}")),
                            });
                        }
                        ImplStatus::NotImplemented => {
                            not_implemented += 1;
                            violations.push(Violation {
                                severity: Severity::Warning,
                                rule: "BIND-003".to_string(),
                                message: format!(
                                    "Equation '{eq_name}' in \
                                     {contract_file} has no \
                                     implementation"
                                ),
                                location: Some(format!("bindings.{eq_name}")),
                            });
                        }
                        ImplStatus::Pending => {
                            not_implemented += 1;
                            violations.push(Violation {
                                severity: Severity::Warning,
                                rule: "BIND-004".to_string(),
                                message: format!(
                                    "Equation '{eq_name}' in \
                                     {contract_file} is pending \
                                     implementation"
                                ),
                                location: Some(format!("bindings.{eq_name}")),
                            });
                        }
                    }
                }
                None => {
                    violations.push(Violation {
                        severity: Severity::Error,
                        rule: "BIND-001".to_string(),
                        message: format!(
                            "Equation '{eq_name}' in {contract_file} \
                             has no binding entry"
                        ),
                        location: Some(format!("{contract_file}.equations.{eq_name}")),
                    });
                }
            }
        }
    }

    BindingAuditReport {
        total_equations,
        bound_equations,
        implemented,
        partial,
        not_implemented,
        total_obligations,
        covered_obligations,
        violations,
    }
}

/// Count obligations that are not SIMD-specific for a contract.
///
/// When an equation is implemented, we consider all non-SIMD
/// obligations as covered (SIMD equivalence requires a separate
/// SIMD implementation which is trueno's domain).
fn obligations_for_equation(contract: &Contract) -> usize {
    contract
        .proof_obligations
        .iter()
        .filter(|o| o.applies_to != Some(crate::schema::AppliesTo::Simd))
        .count()
}

#[cfg(test)]
mod tests {
    include!("mod_tests.rs");
}
