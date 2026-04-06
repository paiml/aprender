use std::collections::HashSet;

use crate::error::{Severity, Violation};
use crate::schema::types::{Contract, ContractKind};

/// Validate a parsed contract for completeness and consistency.
///
/// Returns a list of violations. If any violation has
/// [`Severity::Error`], the contract is considered invalid.
///
/// Validation is kind-aware: non-kernel contracts (registries, model-family
/// schemas, reference documents) are validated only for metadata consistency;
/// the provability invariant, equations, and proof/kani/falsification checks
/// only apply to `ContractKind::Kernel`.
pub fn validate_contract(contract: &Contract) -> Vec<Violation> {
    let mut violations = Vec::new();

    validate_metadata(contract, &mut violations);

    // Kernel-only checks: these enforce the provability invariant and
    // require equations + proof obligations + tests + Kani harnesses.
    if contract.kind() == ContractKind::Kernel && !contract.is_registry() {
        validate_equations(contract, &mut violations);
        validate_provability_invariant(contract, &mut violations);
        validate_proof_obligations(contract, &mut violations);
        validate_falsification_tests(contract, &mut violations);
        validate_kani_harnesses(contract, &mut violations);
        validate_qa_gate(contract, &mut violations);
    } else {
        // Non-kernel kinds (registry, model-family, schema): still validate
        // any proof obligations/falsification/kani data that IS present, so
        // mistakes are caught even on exempt contracts.
        validate_proof_obligations(contract, &mut violations);
        validate_falsification_tests(contract, &mut violations);
        validate_kani_harnesses(contract, &mut violations);
    }

    violations
}

/// Enforce the provability invariant: kernel contracts (non-registry) MUST have
/// `proof_obligations`, `falsification_tests`, and `kani_harnesses`.
fn validate_provability_invariant(contract: &Contract, violations: &mut Vec<Violation>) {
    for v in contract.provability_violations() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "PROVABILITY-001".to_string(),
            message: v,
            location: None,
        });
    }
}

fn validate_metadata(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.metadata.references.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-001".to_string(),
            message: "metadata.references must not be empty — \
                      every contract must cite its source paper(s)"
                .to_string(),
            location: Some("metadata.references".to_string()),
        });
    }

    if contract.metadata.version.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-002".to_string(),
            message: "metadata.version must not be empty".to_string(),
            location: Some("metadata.version".to_string()),
        });
    }
}

fn validate_equations(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.equations.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-003".to_string(),
            message: "equations must contain at least one equation".to_string(),
            location: Some("equations".to_string()),
        });
    }

    for (name, eq) in &contract.equations {
        if eq.formula.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-004".to_string(),
                message: format!("equations.{name}.formula must not be empty"),
                location: Some(format!("equations.{name}.formula")),
            });
        }
    }
}

fn validate_proof_obligations(contract: &Contract, violations: &mut Vec<Violation>) {
    use crate::schema::types::ObligationType;

    let mut seen_ids = HashSet::new();
    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        if ob.property.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-005".to_string(),
                message: format!("proof_obligations[{i}].property must not be empty"),
                location: Some(format!("proof_obligations[{i}].property")),
            });
        }
        if let Some(ref formal) = ob.formal {
            if !seen_ids.insert(formal.clone()) {
                violations.push(Violation {
                    severity: Severity::Warning,
                    rule: "SCHEMA-006".to_string(),
                    message: format!("Duplicate formal predicate: {formal}"),
                    location: Some(format!("proof_obligations[{i}].formal")),
                });
            }
        }

        // DbC field/type constraints
        if ob.requires.is_some() && ob.obligation_type != ObligationType::Postcondition {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-014".to_string(),
                message: format!(
                    "proof_obligations[{i}].requires is only valid on \
                     postcondition obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].requires")),
            });
        }

        if ob.applies_to_phase.is_some()
            && ob.obligation_type != ObligationType::LoopInvariant
            && ob.obligation_type != ObligationType::LoopVariant
        {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-015".to_string(),
                message: format!(
                    "proof_obligations[{i}].applies_to_phase is only valid on \
                     loop_invariant or loop_variant obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].applies_to_phase")),
            });
        }

        if ob.parent_contract.is_some() && ob.obligation_type != ObligationType::Subcontract {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-016".to_string(),
                message: format!(
                    "proof_obligations[{i}].parent_contract is only valid on \
                     subcontract obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].parent_contract")),
            });
        }

        // Subcontract parent_contract must be in depends_on
        if let Some(ref parent) = ob.parent_contract {
            if ob.obligation_type == ObligationType::Subcontract
                && !contract.metadata.depends_on.contains(parent)
            {
                violations.push(Violation {
                    severity: Severity::Error,
                    rule: "SCHEMA-017".to_string(),
                    message: format!(
                        "proof_obligations[{i}].parent_contract \"{parent}\" \
                         must be listed in metadata.depends_on"
                    ),
                    location: Some(format!("proof_obligations[{i}].parent_contract")),
                });
            }
        }
    }
}

fn validate_falsification_tests(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for test in &contract.falsification_tests {
        if !ids.insert(&test.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-007".to_string(),
                message: format!("Duplicate falsification test ID: {}", test.id),
                location: Some(format!("falsification_tests.{}", test.id)),
            });
        }
        if test.prediction.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-008".to_string(),
                message: format!(
                    "falsification_tests.{}.prediction must not be empty — \
                     every test must make a falsifiable prediction",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.prediction", test.id)),
            });
        }
        if test.if_fails.is_empty() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-009".to_string(),
                message: format!(
                    "falsification_tests.{}.if_fails is empty — \
                     should describe root cause diagnosis",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.if_fails", test.id)),
            });
        }
    }
}

fn validate_kani_harnesses(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for harness in &contract.kani_harnesses {
        if !ids.insert(&harness.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-010".to_string(),
                message: format!("Duplicate Kani harness ID: {}", harness.id),
                location: Some(format!("kani_harnesses.{}", harness.id)),
            });
        }
        if harness.obligation.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-011".to_string(),
                message: format!(
                    "kani_harnesses.{}.obligation must not be empty — \
                     every harness must reference a proof obligation",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.obligation", harness.id)),
            });
        }
        if harness.bound.is_none() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-012".to_string(),
                message: format!(
                    "kani_harnesses.{}.bound not specified — \
                     Kani requires an unwind bound",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.bound", harness.id)),
            });
        }
    }
}

fn validate_qa_gate(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.qa_gate.is_none() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "SCHEMA-013".to_string(),
            message: "No qa_gate defined — contract should define a \
                      certeza quality gate"
                .to_string(),
            location: Some("qa_gate".to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    include!("validator_tests.rs");
}
