//! Proof status tracking for kernel contracts.

use super::family::yaml_str;
use super::kernel_ops::kernel_ops_for_class;
use super::{KernelClass, KERNEL_CONTRACTS};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProofLevel {
    Proven,
    Tested,
    Documented,
    Unknown,
}

impl ProofLevel {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Proven => "\u{2713}",
            Self::Tested => "\u{25c9}",
            Self::Documented => "\u{25cb}",
            Self::Unknown => "?",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Proven => "Proven",
            Self::Tested => "Tested",
            Self::Documented => "Documented",
            Self::Unknown => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ContractProof {
    pub contract: String,
    pub level: ProofLevel,
    pub evidence: String,
}

/// Determine proof level for a contract by inspecting its embedded YAML.
pub fn proof_status_for_contract(contract_name: &str) -> ContractProof {
    let yaml_text = KERNEL_CONTRACTS
        .iter()
        .find(|(name, _)| *name == contract_name)
        .map(|(_, text)| *text);

    let Some(text) = yaml_text else {
        return ContractProof {
            contract: contract_name.to_string(),
            level: ProofLevel::Unknown,
            evidence: "No contract YAML found".to_string(),
        };
    };

    // Check for kani harnesses or extensive test references
    let has_kani = text.contains("kani_harness") || text.contains("kani:");
    let has_falsification = text.contains("FALSIFY-") || text.contains("falsification:");
    let has_tests_file = text.contains("tests_file:");
    let has_qa_gate = text.contains("qa_gate:");

    let (level, evidence) = if has_kani {
        (
            ProofLevel::Proven,
            "Kani harness + contract tests".to_string(),
        )
    } else if has_falsification && has_tests_file {
        (
            ProofLevel::Tested,
            format!(
                "Falsification tests in {}",
                yaml_str(text, "tests_file").unwrap_or_default()
            ),
        )
    } else if has_qa_gate || has_falsification {
        (ProofLevel::Tested, "Contract tests".to_string())
    } else {
        (
            ProofLevel::Documented,
            "Contract specification exists".to_string(),
        )
    };

    ContractProof {
        contract: contract_name.to_string(),
        level,
        evidence,
    }
}

/// Get proof status for all kernel ops in a class.
pub fn proof_status_for_class(class: KernelClass) -> Vec<ContractProof> {
    let ops = kernel_ops_for_class(class);
    let mut seen = Vec::new();
    let mut proofs = Vec::new();

    for op in &ops {
        if !seen.contains(&op.contract) {
            seen.push(op.contract);
            proofs.push(proof_status_for_contract(op.contract));
        }
    }

    proofs
}
