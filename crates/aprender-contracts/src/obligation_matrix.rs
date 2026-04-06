//! Per-obligation verification matrix.
//!
//! Shows L2/L3/L4 status for each proof obligation across all contracts.

use serde::{Deserialize, Serialize};

use crate::proof_status::ProofLevel;
use crate::schema::Contract;

/// Verification status for a single obligation across all proof levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationStatus {
    /// Human-readable obligation property name
    pub property: String,
    /// Obligation type (invariant, bound, equivalence, etc.)
    pub obligation_type: String,
    /// Whether at least one falsification test covers this obligation
    pub l2_tested: bool,
    /// Whether at least one Kani harness covers this obligation
    pub l3_kani: bool,
    /// Whether the obligation has a Lean proof with status "proved"
    pub l4_lean: bool,
    /// Highest achieved level for this specific obligation
    pub max_level: ProofLevel,
}

/// Per-contract obligation verification matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractObligationMatrix {
    /// Contract file stem (e.g. "softmax-kernel-v1")
    pub stem: String,
    /// Per-obligation verification status entries
    pub obligations: Vec<ObligationStatus>,
}

/// Build per-obligation verification matrices for a list of contracts.
///
/// For each contract, determines per-obligation coverage:
/// - **L2**: A falsification test covers this obligation (index-based or rule match)
/// - **L3**: A Kani harness references this obligation (property match)
/// - **L4**: The obligation has a `lean` field with `status: proved`
pub fn obligation_matrix(contracts: &[(String, &Contract)]) -> Vec<ContractObligationMatrix> {
    contracts
        .iter()
        .map(|(stem, contract)| {
            let obligations = contract
                .proof_obligations
                .iter()
                .enumerate()
                .map(|(idx, ob)| {
                    let prop_lower = ob.property.to_lowercase();

                    // L2: check if any falsification test covers this obligation.
                    // Strategy: index-based (obligation i covered if i < test count),
                    // plus rule-text matching against the property name.
                    let l2_tested = idx < contract.falsification_tests.len()
                        || contract.falsification_tests.iter().any(|ft| {
                            let rule_lower = ft.rule.to_lowercase();
                            prop_lower.contains(&rule_lower) || rule_lower.contains(&prop_lower)
                        });

                    // L3: check if any Kani harness covers this obligation.
                    // Match by property text overlap or by harness property containing
                    // key words from the obligation property.
                    let l3_kani = contract.kani_harnesses.iter().any(|kh| {
                        if let Some(ref kh_prop) = kh.property {
                            let kh_lower = kh_prop.to_lowercase();
                            property_words_match(&prop_lower, &kh_lower)
                        } else {
                            false
                        }
                    });

                    // L4: obligation has lean proof with status proved
                    let l4_lean = ob
                        .lean
                        .as_ref()
                        .is_some_and(|lp| lp.status == crate::schema::LeanStatus::Proved);

                    let max_level = if l4_lean {
                        ProofLevel::L4
                    } else if l3_kani {
                        ProofLevel::L3
                    } else if l2_tested {
                        ProofLevel::L2
                    } else {
                        ProofLevel::L1
                    };

                    ObligationStatus {
                        property: ob.property.clone(),
                        obligation_type: ob.obligation_type.to_string(),
                        l2_tested,
                        l3_kani,
                        l4_lean,
                        max_level,
                    }
                })
                .collect();

            ContractObligationMatrix {
                stem: stem.clone(),
                obligations,
            }
        })
        .collect()
}

/// Format per-obligation matrices as a human-readable table.
pub fn format_obligation_table(matrices: &[ContractObligationMatrix]) -> String {
    let mut out = String::new();

    out.push_str("\nObligation Status Matrix\n");
    out.push_str("========================\n");

    for matrix in matrices {
        if matrix.obligations.is_empty() {
            continue;
        }

        // Compute max property width (capped at 40 for readability)
        let max_prop_width = matrix
            .obligations
            .iter()
            .map(|o| o.property.len())
            .max()
            .unwrap_or(20)
            .min(40);

        out.push_str(&format!("Contract: {}\n", matrix.stem));
        out.push_str(&format!(
            "  {:<width$} | L2 Test | L3 Kani | L4 Lean | Status\n",
            "Obligation",
            width = max_prop_width
        ));
        out.push_str(&format!("  {}\n", "-".repeat(max_prop_width + 39)));

        for ob in &matrix.obligations {
            let check = "\u{2713}";
            let cross = "\u{2717}";
            let l2 = if ob.l2_tested { check } else { cross };
            let l3 = if ob.l3_kani { check } else { cross };
            let l4 = if ob.l4_lean { check } else { cross };
            let prop_display = truncate(&ob.property, max_prop_width);
            out.push_str(&format!(
                "  {:<width$} |    {}    |    {}    |    {}    | {}\n",
                prop_display,
                l2,
                l3,
                l4,
                ob.max_level,
                width = max_prop_width
            ));
        }

        out.push('\n');
    }

    out
}

/// Check whether two property descriptions share significant words.
///
/// Splits both strings into words (>= 3 chars, excluding stop words) and
pub fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

/// returns true if at least one non-trivial word overlaps.
pub fn property_words_match(a: &str, b: &str) -> bool {
    let stop_words: &[&str] = &[
        "the", "for", "and", "all", "are", "with", "from", "that", "this", "has", "not", "any",
        "each", "into",
    ];

    let words_a: Vec<&str> = a
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !stop_words.contains(w))
        .collect();

    words_a.iter().any(|wa| {
        b.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3 && !stop_words.contains(w))
            .any(|wb| wa == &wb)
    })
}

// Tests for obligation_matrix are in proof_status_tests.rs
// (included via proof_status.rs #[path] directive)
