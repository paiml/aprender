//! Per-obligation verification matrix.
//!
//! Shows L2/L3/L4 status for each proof obligation across all contracts.

use serde::{Deserialize, Serialize};

use crate::proof_status::ProofLevel;
use crate::schema::Contract;

/// What the L2 column knows about ONE obligation (#3347).
///
/// Three values, not two, because "no test covers this" and "nothing in this
/// contract says which test covers what" are different facts and only one of
/// them is a finding. The old column had no way to say the second, so it said
/// the first -- as a tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum L2Status {
    /// A falsification test in this contract CITES this obligation.
    Tested,
    /// This contract's obligation-to-test links resolve, and none names this
    /// obligation -- or the contract ships no falsification test at all.
    Untested,
    /// Nothing readable links any test to any obligation here, so whether this
    /// obligation is tested was not measured. An unread window is Unknown,
    /// never a tick and never a failure.
    Unknown,
}

impl L2Status {
    /// `true` only for [`Tested`](L2Status::Tested) -- an Unknown is not a pass.
    #[must_use]
    pub fn is_tested(self) -> bool {
        matches!(self, Self::Tested)
    }

    /// The table cell for this verdict.
    #[must_use]
    pub fn mark(self) -> &'static str {
        match self {
            Self::Tested => "\u{2713}",
            Self::Untested => "\u{2717}",
            Self::Unknown => "?",
        }
    }
}

/// Verification status for a single obligation across all proof levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationStatus {
    /// Human-readable obligation property name
    pub property: String,
    /// Obligation type (invariant, bound, equivalence, etc.)
    pub obligation_type: String,
    /// Whether a falsification test is LINKED to this obligation -- and
    /// whether that question could be answered at all. Replaces the old
    /// `l2_tested: bool`, which could not distinguish the two (#3347).
    pub l2: L2Status,
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

/// Which obligations of one contract a falsification test actually cites.
///
/// `resolved` counts citations that named something findable. Zero of them is
/// the difference between "this contract says nothing is tested" and "this
/// contract says nothing at all", and only the second is Unknown.
struct LinkTally {
    /// Per obligation index: some test cites it.
    cited: Vec<bool>,
    /// Citations that resolved to a real obligation / a real test.
    resolved: usize,
}

/// Position of the obligation a test citation names: its `id`, else its exact
/// `property` text. Both spellings occur in `contracts/` (12 and 6 entries).
fn obligation_index(contract: &Contract, token: &str) -> Option<usize> {
    contract.proof_obligations.iter().position(|ob| {
        ob.id.as_deref().is_some_and(|id| id.trim() == token) || ob.property.trim() == token
    })
}

/// Whether a `discharged_by` token names a falsification test that EXISTS.
///
/// `falsification_tests[N]` resolves only when `N` is in range -- an
/// out-of-range index is a dangling citation, not a proof. A token naming a
/// kani harness resolves here as false: kani is the L3 column.
fn cites_existing_test(contract: &Contract, token: &str) -> bool {
    if let Some(inner) = token
        .strip_prefix("falsification_tests[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return inner
            .trim()
            .parse::<usize>()
            .is_ok_and(|i| i < contract.falsification_tests.len());
    }
    contract
        .falsification_tests
        .iter()
        .any(|ft| ft.id.trim() == token)
}

/// Resolve every obligation-to-test link this contract declares, in both
/// spellings: `falsification_tests[].obligation` (alias `binds_to`) and
/// `proof_obligations[].discharged_by`.
fn tally_links(contract: &Contract) -> LinkTally {
    let mut tally = LinkTally {
        cited: vec![false; contract.proof_obligations.len()],
        resolved: 0,
    };

    for ft in &contract.falsification_tests {
        let Some(citation) = ft.obligation.as_ref() else {
            continue;
        };
        for token in citation.targets() {
            if let Some(idx) = obligation_index(contract, token) {
                tally.cited[idx] = true;
                tally.resolved += 1;
            }
        }
    }

    for (idx, ob) in contract.proof_obligations.iter().enumerate() {
        let Some(citation) = ob.discharged_by.as_ref() else {
            continue;
        };
        for token in citation.targets() {
            if cites_existing_test(contract, token) {
                tally.cited[idx] = true;
                tally.resolved += 1;
            }
        }
    }

    tally
}

/// The L2 verdict for one obligation. See [`L2Status`].
fn l2_verdict(contract: &Contract, tally: &LinkTally, idx: usize) -> L2Status {
    // No falsification test exists, so no test covers this. That is a reading,
    // not an unread window.
    if contract.falsification_tests.is_empty() {
        return L2Status::Untested;
    }
    // Tests exist but nothing links any of them to any obligation -- or every
    // link this contract declares dangles. Either way the window is unread.
    if tally.resolved == 0 {
        return L2Status::Unknown;
    }
    if tally.cited.get(idx).copied().unwrap_or(false) {
        L2Status::Tested
    } else {
        L2Status::Untested
    }
}

/// Build per-obligation verification matrices for a list of contracts.
///
/// For each contract, determines per-obligation coverage:
/// - **L2**: a falsification test is LINKED to this obligation -- see
///   [`l2_verdict`] and [`L2Status`]
/// - **L3**: A Kani harness references this obligation (property match)
/// - **L4**: The obligation has a `lean` field with `status: proved`
///
/// # The L2 column used to tick on a count (#3347)
///
/// It read `idx < falsification_tests.len()`: obligation 3 was "tested"
/// because the contract had at least 4 tests, whoever those tests were about.
/// All 7 obligations of `qwen35-e2e-verification-v1` showed a tick before a
/// single test existed. The fallback -- a substring match between the
/// obligation's `property` and a test's `rule` prose -- is gone too: an
/// inferred overlap of two English sentences is not a claim that the test
/// proves the obligation, and it ticked on words like "count".
///
/// Measured consequence over `contracts/`, and it is the point rather than a
/// regression: L2 ticks fall from 3,573 of 3,753 obligation rows to the
/// handful the corpus actually binds. The rest report `?`.
pub fn obligation_matrix(contracts: &[(String, &Contract)]) -> Vec<ContractObligationMatrix> {
    contracts
        .iter()
        .map(|(stem, contract)| {
            let tally = tally_links(contract);
            let obligations = contract
                .proof_obligations
                .iter()
                .enumerate()
                .map(|(idx, ob)| {
                    let prop_lower = ob.property.to_lowercase();

                    let l2 = l2_verdict(contract, &tally, idx);

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
                    } else if l2.is_tested() {
                        ProofLevel::L2
                    } else {
                        ProofLevel::L1
                    };

                    ObligationStatus {
                        property: ob.property.clone(),
                        obligation_type: ob.obligation_type.to_string(),
                        l2,
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
            let l2 = ob.l2.mark();
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

/// Truncate `s` to at most `max` BYTES, cutting on a char boundary.
///
/// #3338: this sliced `&s[..max]` and `pv proof-status contracts/ --table`
/// panicked on the real corpus — `byte index 40 is not a char boundary; it is
/// inside '∈'`. The column width is a byte count (`property.len()`), so the
/// cut lands mid-char for any property holding a multi-byte char near it;
/// eight contracts in `contracts/` do. The budget stays a byte budget (the
/// table is laid out in bytes) and the cut walks back to the nearest boundary.
pub fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Check whether two property descriptions share significant words.
///
/// Splits both strings into words (>= 3 chars, excluding stop words) and
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
