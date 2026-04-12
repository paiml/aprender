//! Popperian Falsification Scoring
//!
//! Implements scientific scoring based on Karl Popper's falsification methodology.
//!
//! ## Popperian Principles
//!
//! 1. **Falsifiability**: A theory must be testable and potentially refutable
//! 2. **Corroboration**: Surviving rigorous testing increases confidence
//! 3. **Severity**: Harder tests provide stronger evidence
//! 4. **Reproducibility**: Results must be independently verifiable

use apr_qa_runner::{EvidenceCollector, Outcome};
use serde::{Deserialize, Serialize};

/// A gate classification rule: a predicate on the gate ID mapped to a value.
struct GateRule<T: Clone> {
    /// How to match the gate ID
    matcher: GateMatcher,
    /// Value to return when matched
    value: T,
}

/// How a gate rule matches against a gate ID string.
enum GateMatcher {
    /// Match if gate_id.contains(pattern)
    Contains(&'static str),
    /// Match if gate_id.starts_with(prefix)
    StartsWith(&'static str),
}

/// Classify a gate ID by scanning rules in order, returning the first match or a default.
fn classify_gate<T: Clone>(gate_id: &str, rules: &[GateRule<T>], default: T) -> T {
    for rule in rules {
        let matched = match &rule.matcher {
            GateMatcher::Contains(pat) => gate_id.contains(pat),
            GateMatcher::StartsWith(prefix) => gate_id.starts_with(prefix),
        };
        if matched {
            return rule.value.clone();
        }
    }
    default
}

/// Severity classification rules (checked in priority order).
const SEVERITY_RULES: &[GateRule<u8>] = &[
    GateRule {
        matcher: GateMatcher::Contains("-P0-"),
        value: 5,
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G0-"),
        value: 5,
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G1-"),
        value: 5,
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G2-"),
        value: 5,
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G3-"),
        value: 5,
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G4-"),
        value: 5,
    },
    GateRule {
        matcher: GateMatcher::Contains("-P1-"),
        value: 4,
    },
    GateRule {
        matcher: GateMatcher::Contains("-P2-"),
        value: 3,
    },
    GateRule {
        matcher: GateMatcher::Contains("EDGE"),
        value: 3,
    },
    GateRule {
        matcher: GateMatcher::Contains("STAB"),
        value: 3,
    },
    GateRule {
        matcher: GateMatcher::Contains("PERF"),
        value: 2,
    },
];

/// Hypothesis classification rules (checked in priority order).
const HYPOTHESIS_RULES: &[GateRule<&str>] = &[
    // Gateway checks (G0-G4) — checked before category keywords
    GateRule {
        matcher: GateMatcher::StartsWith("G0-"),
        value: "Model metadata and tensor layout are internally consistent",
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G1-"),
        value: "Model loads successfully within the timeout budget",
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G2-"),
        value: "Basic inference produces output without error",
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G3-"),
        value: "Model runs without crashes or panics",
    },
    GateRule {
        matcher: GateMatcher::StartsWith("G4-"),
        value: "Model output is not garbage (no layout or dtype corruption)",
    },
    // Category-level hypotheses
    GateRule {
        matcher: GateMatcher::Contains("QUAL"),
        value: "Model produces valid output",
    },
    GateRule {
        matcher: GateMatcher::Contains("PERF"),
        value: "Model meets performance requirements",
    },
    GateRule {
        matcher: GateMatcher::Contains("STAB"),
        value: "Model is stable under stress",
    },
    GateRule {
        matcher: GateMatcher::Contains("COMP"),
        value: "Model is compatible with configuration",
    },
    GateRule {
        matcher: GateMatcher::Contains("EDGE"),
        value: "Model handles edge cases correctly",
    },
    GateRule {
        matcher: GateMatcher::Contains("REGR"),
        value: "Model behavior is consistent",
    },
];

/// Popperian score with falsification details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopperianScore {
    /// Model identifier
    pub model_id: String,
    /// Total hypotheses tested
    pub hypotheses_tested: usize,
    /// Hypotheses not falsified (corroborated)
    pub corroborated: usize,
    /// Hypotheses falsified
    pub falsified: usize,
    /// Inconclusive tests (timeout, skip)
    pub inconclusive: usize,
    /// Corroboration ratio (0.0 - 1.0)
    pub corroboration_ratio: f64,
    /// Severity-weighted score (accounts for test difficulty)
    pub severity_weighted_score: f64,
    /// Confidence level (0.0 - 1.0)
    pub confidence_level: f64,
    /// Reproducibility index (based on seed consistency)
    pub reproducibility_index: f64,
    /// Black swan events (rare, high-impact failures)
    pub black_swan_count: usize,
    /// Falsification details
    pub falsifications: Vec<FalsificationDetail>,
}

impl PopperianScore {
    /// Check if the model has strong corroboration
    #[must_use]
    pub fn is_strongly_corroborated(&self) -> bool {
        self.corroboration_ratio >= 0.95 && self.black_swan_count == 0
    }

    /// Check if black swan events were detected
    #[must_use]
    pub fn has_black_swans(&self) -> bool {
        self.black_swan_count > 0
    }

    /// Get falsification summary
    #[must_use]
    pub fn falsification_summary(&self) -> String {
        if self.hypotheses_tested == 0 {
            "No hypotheses tested".to_string()
        } else if self.falsified == 0 {
            "No falsifications - strongly corroborated".to_string()
        } else {
            format!(
                "{} of {} hypotheses falsified ({:.1}%)",
                self.falsified,
                self.hypotheses_tested,
                (self.falsified as f64 / self.hypotheses_tested as f64) * 100.0
            )
        }
    }
}

/// Detail about a specific falsification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationDetail {
    /// Gate ID that was falsified
    pub gate_id: String,
    /// Hypothesis that was falsified
    pub hypothesis: String,
    /// Evidence of falsification
    pub evidence: String,
    /// Severity (1-5, 5 being most severe)
    pub severity: u8,
    /// Is this a black swan event?
    pub is_black_swan: bool,
    /// Reproducibility (how many times this was observed)
    pub occurrence_count: usize,
}

/// Popperian score calculator
#[derive(Debug, Default)]
pub struct PopperianCalculator {
    /// Weight for high-severity tests
    severity_weights: [f64; 5],
}

impl PopperianCalculator {
    /// Create a new calculator with default severity weights
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Higher severity tests contribute more to confidence
            severity_weights: [1.0, 1.5, 2.0, 2.5, 3.0],
        }
    }

    /// Calculate Popperian score from evidence
    #[must_use]
    pub fn calculate(&self, model_id: &str, evidence: &EvidenceCollector) -> PopperianScore {
        let all_evidence = evidence.all();

        let mut inconclusive = 0;
        let mut severity_total = 0.0;
        let mut severity_passed = 0.0;
        let mut falsifications = Vec::new();
        let mut black_swan_count = 0;

        // Track unique gate_ids for consistent hypothesis counting.
        // Both corroborated and falsified counts must use the same unit
        // (unique gate_ids = hypotheses) to avoid ratio distortion.
        let mut corroborated_gates: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Group failures by gate_id for reproducibility analysis
        let mut failure_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for e in all_evidence {
            let severity = Self::determine_severity(&e.gate_id);
            let weight = self.severity_weights[severity.saturating_sub(1) as usize];
            severity_total += weight;

            match e.outcome {
                Outcome::Corroborated => {
                    corroborated_gates.insert(e.gate_id.clone());
                    severity_passed += weight;
                }
                Outcome::Falsified | Outcome::Crashed => {
                    *failure_counts.entry(e.gate_id.clone()).or_insert(0) += 1;

                    // Black swan: crash or severe unexpected failure
                    let is_black_swan = e.outcome == Outcome::Crashed || severity >= 4;
                    if is_black_swan {
                        black_swan_count += 1;
                    }

                    falsifications.push(FalsificationDetail {
                        gate_id: e.gate_id.clone(),
                        hypothesis: Self::gate_to_hypothesis(&e.gate_id),
                        evidence: e.reason.clone(),
                        severity,
                        is_black_swan,
                        occurrence_count: 1, // Will be updated later
                    });
                }
                Outcome::Skipped | Outcome::Timeout => {
                    inconclusive += 1;
                }
            }
        }

        // Update occurrence counts
        for falsification in &mut falsifications {
            if let Some(&count) = failure_counts.get(&falsification.gate_id) {
                falsification.occurrence_count = count;
            }
        }

        // Deduplicate falsifications (keep highest severity per gate)
        falsifications.sort_by(|a, b| a.gate_id.cmp(&b.gate_id).then(b.severity.cmp(&a.severity)));
        falsifications.dedup_by(|a, b| a.gate_id == b.gate_id);

        // Both corroborated and falsified use unique gate_ids (hypotheses)
        // to ensure the ratio uses consistent counting units.
        let falsified = falsifications.len();
        // Remove gate_ids that were also falsified (mixed results → falsified wins)
        let corroborated = corroborated_gates
            .iter()
            .filter(|g| !failure_counts.contains_key(g.as_str()))
            .count();
        let hypotheses_tested = corroborated + falsified;
        let corroboration_ratio = if hypotheses_tested > 0 {
            corroborated as f64 / hypotheses_tested as f64
        } else {
            0.0
        };

        let severity_weighted_score = if severity_total > 0.0 {
            severity_passed / severity_total
        } else {
            0.0
        };

        // Confidence level based on sample size and consistency
        let confidence_level = self.calculate_confidence(hypotheses_tested, corroboration_ratio);

        // Reproducibility based on failure consistency
        let reproducibility_index =
            self.calculate_reproducibility(&failure_counts, all_evidence.len());

        PopperianScore {
            model_id: model_id.to_string(),
            hypotheses_tested,
            corroborated,
            falsified,
            inconclusive,
            corroboration_ratio,
            severity_weighted_score,
            confidence_level,
            reproducibility_index,
            black_swan_count,
            falsifications,
        }
    }

    /// Determine severity from gate ID using data-driven rules.
    fn determine_severity(gate_id: &str) -> u8 {
        classify_gate(gate_id, SEVERITY_RULES, 1)
    }

    /// Convert gate ID to human-readable hypothesis using data-driven rules.
    fn gate_to_hypothesis(gate_id: &str) -> String {
        let hypothesis = classify_gate(gate_id, HYPOTHESIS_RULES, "");
        if hypothesis.is_empty() {
            format!("Hypothesis for {gate_id}")
        } else {
            hypothesis.to_string()
        }
    }

    /// Calculate confidence level
    fn calculate_confidence(&self, n: usize, ratio: f64) -> f64 {
        if n == 0 {
            return 0.0;
        }

        // Wilson score interval lower bound approximation
        // Provides conservative confidence estimate
        let z = 1.96; // 95% confidence
        let n_f = n as f64;
        let denominator = 1.0 + z * z / n_f;
        let center = ratio + z * z / (2.0 * n_f);
        let spread = z * ((ratio * (1.0 - ratio) / n_f) + (z * z / (4.0 * n_f * n_f))).sqrt();

        ((center - spread) / denominator).clamp(0.0, 1.0)
    }

    /// Calculate reproducibility index
    fn calculate_reproducibility(
        &self,
        failure_counts: &std::collections::HashMap<String, usize>,
        total_tests: usize,
    ) -> f64 {
        if total_tests == 0 || failure_counts.is_empty() {
            return 1.0; // No failures = perfectly reproducible (trivially)
        }

        // Count consistent failures (appeared more than once)
        let consistent_failures: usize = failure_counts.values().filter(|&&count| count > 1).sum();

        let total_failures: usize = failure_counts.values().sum();

        if total_failures == 0 {
            1.0
        } else {
            // Higher ratio of consistent failures = more reproducible
            (consistent_failures as f64 / total_failures as f64).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
#[path = "popperian_tests.rs"]
mod popperian_tests;
