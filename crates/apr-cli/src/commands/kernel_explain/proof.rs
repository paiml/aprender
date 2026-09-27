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

    let (level, evidence) = classify(text);

    ContractProof {
        contract: contract_name.to_string(),
        level,
        evidence,
    }
}

fn classify(text: &str) -> (ProofLevel, String) {
    // #3890. Every arm below reads the contract TEXT, so each one can only report
    // what the contract CLAIMS. That is fine for `Documented` and `Tested`, whose
    // wording now says "declares". It is not fine for `Proven`, which asserted that
    // a property had been established.
    //
    // WHAT WAS WRONG: `Proven` was reached by `text.contains("kani_harness")`. A
    // substring decided a verification claim, and NOTHING in this repository runs
    // kani — no workflow, script or Makefile target invokes `cargo kani`; `pv kani`
    // GENERATES harnesses; CI tests harness GENERATION. contracts/rope-kernel-v1.yaml
    // declares eight harnesses that nothing executes. So `apr kernel-explain` told a
    // user a kernel was Proven on the strength of a string.
    //
    // WHY IT IS NOT MERELY THEORETICAL: the six kernel contracts with a valid root
    // sibling would each have flipped Tested -> Proven the moment anyone reconciled
    // them to satisfy `pv validate`. The truncated shipped copies were, by accident,
    // the only thing keeping that claim away from users.
    //
    // THE RULE NOW: `Proven` requires a VERIFICATION RECORD -- evidence that a run
    // happened -- not a declaration that a harness exists. `Proven` is deliberately
    // NOT deleted: the day verification actually runs and records itself, this
    // becomes reachable again with no code change. A level that is deleted has to be
    // re-invented; one that is grounded just starts working.
    let has_kani_decl = text.contains("kani_harness") || text.contains("kani:");
    let verification = verification_record(text);
    let has_falsification = text.contains("FALSIFY-") || text.contains("falsification:");
    let has_tests_file = text.contains("tests_file:");
    let has_qa_gate = text.contains("qa_gate:");

    let (level, evidence) = if has_kani_decl && verification.is_some() {
        (
            ProofLevel::Proven,
            format!(
                "Kani verification recorded: {}",
                verification.unwrap_or_default()
            ),
        )
    } else if has_falsification && has_tests_file {
        // The wording says "declares" because this function cannot check that the
        // file exists: it runs on a user's machine with only the embedded YAML. And
        // the claim is not hypothetical -- every `tests_file:` in the shipped kernel
        // contracts names a path that does not exist in this tree (5 of 5, all
        // pre-monorepo: `trueno/src/...`, `src/autograd/...`, `tests/contracts/...`).
        // `contract_tests_files_exist` in tests.rs is the gate that catches that
        // drift where it CAN be checked -- in the repository, at test time.
        (
            ProofLevel::Tested,
            format!(
                "Contract declares falsification tests in {}",
                yaml_str(text, "tests_file").unwrap_or_default()
            ),
        )
    } else if has_qa_gate || has_falsification {
        (ProofLevel::Tested, "Contract declares tests".to_string())
    } else if has_kani_decl {
        (
            ProofLevel::Documented,
            "Declares Kani harnesses; no verification record".to_string(),
        )
    } else {
        (
            ProofLevel::Documented,
            "Contract specification exists".to_string(),
        )
    };

    (level, evidence)
}

/// The classifier, driven by contract TEXT — so the rule can be tested on planted
/// inputs rather than only on the nine embedded contracts (#3890).
pub fn proof_level_of_text(text: &str) -> (ProofLevel, String) {
    classify(text)
}

/// The verification RECORD a contract must carry to be reported `Proven` (#3890).
///
/// A declaration that a harness exists is not evidence that it ran. This looks for a
/// record of an actual verification run and returns its value, or `None`.
///
/// No contract in this repository carries one today, which is why `Proven` is
/// currently unreachable — correctly, since nothing runs kani. Note what does NOT
/// count: `verification_summary` carries `l3_kani_proved: 8` and `l4_lean_proved: 4`,
/// which are COUNTS OF CLAIMS, not a record that anything executed. Accepting those
/// would reproduce the defect one field over.
fn verification_record(text: &str) -> Option<String> {
    yaml_str(text, "kani_verified_at").or_else(|| yaml_str(text, "verification_run"))
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
