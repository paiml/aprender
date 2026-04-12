//! Model Qualification Score (MQS) Calculator
//!
//! Implements Toyota-style gateway checks and Popperian falsification scoring.
//!
//! ## Scoring System
//!
//! - **Raw score**: 0-1000 points across 6 categories
//! - **Normalized score**: 0-100 (logarithmic scaling, 100 is extremely hard)
//! - **Gateway checks**: G1-G4 failures zero the entire score
//!
//! ## Categories (1000 raw points total)
//!
//! | Category | Points | Description |
//! |----------|--------|-------------|
//! | QUAL     | 200    | Basic quality (loads, responds) |
//! | PERF     | 150    | Performance metrics |
//! | STAB     | 200    | Stability under stress |
//! | COMP     | 150    | Compatibility (formats, backends) |
//! | EDGE     | 150    | Edge case handling |
//! | REGR     | 150    | Regression resistance |

use apr_qa_runner::{Evidence, EvidenceCollector, Outcome};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::proof_status::ProofBonus;

/// Gateway check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayResult {
    /// Gateway ID (G1, G2, G3, G4)
    pub id: String,
    /// Whether the gateway passed
    pub passed: bool,
    /// Description of the check
    pub description: String,
    /// Failure reason (if any)
    pub failure_reason: Option<String>,
}

impl GatewayResult {
    /// Create a passed gateway result
    #[must_use]
    pub fn passed(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            passed: true,
            description: description.into(),
            failure_reason: None,
        }
    }

    /// Create a failed gateway result
    #[must_use]
    pub fn failed(
        id: impl Into<String>,
        description: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            passed: false,
            description: description.into(),
            failure_reason: Some(reason.into()),
        }
    }
}

/// MQS category scores
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryScores {
    /// Quality score (0-200)
    pub qual: u32,
    /// Performance score (0-150)
    pub perf: u32,
    /// Stability score (0-200)
    pub stab: u32,
    /// Compatibility score (0-150)
    pub comp: u32,
    /// Edge case score (0-150)
    pub edge: u32,
    /// Regression score (0-150)
    pub regr: u32,
}

impl CategoryScores {
    /// Maximum points per category
    pub const MAX_QUAL: u32 = 200;
    /// Maximum performance points
    pub const MAX_PERF: u32 = 150;
    /// Maximum stability points
    pub const MAX_STAB: u32 = 200;
    /// Maximum compatibility points
    pub const MAX_COMP: u32 = 150;
    /// Maximum edge case points
    pub const MAX_EDGE: u32 = 150;
    /// Maximum regression points
    pub const MAX_REGR: u32 = 150;
    /// Total maximum raw score
    pub const MAX_TOTAL: u32 = 1000;

    /// Calculate total raw score
    #[must_use]
    pub fn total(&self) -> u32 {
        self.qual + self.perf + self.stab + self.comp + self.edge + self.regr
    }

    /// Get category breakdown as HashMap
    #[must_use]
    pub fn breakdown(&self) -> HashMap<String, (u32, u32)> {
        let mut map = HashMap::new();
        map.insert("QUAL".to_string(), (self.qual, Self::MAX_QUAL));
        map.insert("PERF".to_string(), (self.perf, Self::MAX_PERF));
        map.insert("STAB".to_string(), (self.stab, Self::MAX_STAB));
        map.insert("COMP".to_string(), (self.comp, Self::MAX_COMP));
        map.insert("EDGE".to_string(), (self.edge, Self::MAX_EDGE));
        map.insert("REGR".to_string(), (self.regr, Self::MAX_REGR));
        map
    }
}

/// Final MQS score with all details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqsScore {
    /// Model identifier
    pub model_id: String,
    /// Raw score (0-1000)
    pub raw_score: u32,
    /// Normalized score (0-100)
    pub normalized_score: f64,
    /// Letter grade (A+, A, B, C, D, F)
    pub grade: String,
    /// Gateway results
    pub gateways: Vec<GatewayResult>,
    /// Whether all gateways passed
    pub gateways_passed: bool,
    /// Category breakdown
    pub categories: CategoryScores,
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Penalty deductions applied
    pub penalties: Vec<Penalty>,
    /// Total penalty points deducted
    pub total_penalty: u32,
    /// Proof bonus from provable-contracts integration (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_bonus: Option<ProofBonus>,
}

impl MqsScore {
    /// Check if model qualifies (normalized score >= 70)
    #[must_use]
    pub fn qualifies(&self) -> bool {
        self.gateways_passed && self.normalized_score >= 70.0
    }

    /// Check if model is production-ready (normalized score >= 90)
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.gateways_passed && self.normalized_score >= 90.0
    }

    /// Classify model into a deployment risk tier.
    ///
    /// Risk tiers consider gateway status, score, crash count, and penalty
    /// severity to produce a single risk classification for deployment decisions.
    ///
    /// Returns one of: "MINIMAL", "LOW", "MODERATE", "ELEVATED", "HIGH",
    /// "VERY HIGH", "CRITICAL", or "BLOCKED".
    /// Risk tier thresholds: (min_score, max_penalty, label).
    const RISK_TIERS: &[(f64, u32, &'static str)] = &[
        (95.0, 0, "MINIMAL"),
        (90.0, 20, "LOW"),
        (80.0, 50, "MODERATE"),
        (70.0, 100, "ELEVATED"),
        (60.0, u32::MAX, "HIGH"),
        (40.0, u32::MAX, "VERY HIGH"),
    ];

    #[must_use]
    /// Returns the risk tier label based on score and penalty thresholds.
    pub fn risk_tier(&self) -> &'static str {
        if !self.gateways_passed {
            return "BLOCKED";
        }
        Self::RISK_TIERS
            .iter()
            .find(|&&(score, penalty, _)| {
                self.normalized_score >= score && self.total_penalty <= penalty
            })
            .map_or("CRITICAL", |&(_, _, label)| label)
    }
}

/// Penalty applied to score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Penalty {
    /// Penalty code
    pub code: String,
    /// Description
    pub description: String,
    /// Points deducted
    pub points: u32,
}

/// MQS Calculator
#[derive(Debug)]
pub struct MqsCalculator {
    /// Minimum tests required per category
    #[allow(dead_code)]
    min_tests_per_category: usize,
    /// Optional proof bonus from provable-contracts
    proof_bonus: Option<ProofBonus>,
}

impl Default for MqsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl MqsCalculator {
    /// Create a new calculator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_tests_per_category: 10,
            proof_bonus: None,
        }
    }

    /// Set proof bonus from provable-contracts integration.
    ///
    /// Bonus points are added to the raw score before normalization.
    /// The normalization denominator becomes `MAX_TOTAL + 50` when present.
    /// Gateway failures still zero everything — bonus cannot override.
    #[must_use]
    pub fn with_proof_bonus(mut self, bonus: ProofBonus) -> Self {
        self.proof_bonus = Some(bonus);
        self
    }

    /// Calculate MQS from evidence
    ///
    /// # Errors
    ///
    /// Returns an error if score calculation fails.
    pub fn calculate(&self, model_id: &str, evidence: &EvidenceCollector) -> Result<MqsScore> {
        let all_evidence = evidence.all();

        // No evidence means no qualification — cannot certify what was never tested
        if all_evidence.is_empty() {
            return Ok(MqsScore {
                model_id: model_id.to_string(),
                raw_score: 0,
                normalized_score: 0.0,
                grade: "F".to_string(),
                gateways: vec![],
                gateways_passed: false,
                categories: CategoryScores::default(),
                total_tests: 0,
                tests_passed: 0,
                tests_failed: 0,
                penalties: vec![Penalty {
                    code: "NO_EVIDENCE".to_string(),
                    description: "No test evidence — cannot qualify untested model".to_string(),
                    points: 1000,
                }],
                total_penalty: 1000,
                proof_bonus: self.proof_bonus.clone(),
            });
        }

        // Run gateway checks
        let gateways = self.check_gateways(all_evidence);
        let gateways_passed = gateways.iter().all(|g| g.passed);

        // If gateways fail, score is zero — bonus cannot override
        if !gateways_passed {
            return Ok(MqsScore {
                model_id: model_id.to_string(),
                raw_score: 0,
                normalized_score: 0.0,
                grade: "F".to_string(),
                gateways,
                gateways_passed: false,
                categories: CategoryScores::default(),
                total_tests: all_evidence.len(),
                tests_passed: evidence.pass_count(),
                tests_failed: evidence.fail_count(),
                penalties: vec![Penalty {
                    code: "GATEWAY".to_string(),
                    description: "Gateway check failed - score zeroed".to_string(),
                    points: 1000,
                }],
                total_penalty: 1000,
                proof_bonus: self.proof_bonus.clone(),
            });
        }

        // Calculate category scores
        let categories = self.calculate_categories(all_evidence);
        let mut penalties = Vec::new();
        let mut total_penalty: u32 = 0;

        // Apply penalties
        let crash_count = all_evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Crashed)
            .count();
        if crash_count > 0 {
            let penalty = (crash_count as u32) * 20;
            penalties.push(Penalty {
                code: "CRASH".to_string(),
                description: format!("{crash_count} crash(es) detected"),
                points: penalty,
            });
            total_penalty += penalty;
        }

        let timeout_count = all_evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Timeout)
            .count();
        if timeout_count > 0 {
            let penalty = (timeout_count as u32) * 10;
            penalties.push(Penalty {
                code: "TIMEOUT".to_string(),
                description: format!("{timeout_count} timeout(s) detected"),
                points: penalty,
            });
            total_penalty += penalty;
        }

        // Calculate max possible score (with or without proof bonus)
        // When proof bonus is present, denominator expands to MAX_TOTAL + 50
        let max_possible = if self.proof_bonus.is_some() {
            CategoryScores::MAX_TOTAL + crate::proof_status::MAX_PROOF_BONUS
        } else {
            CategoryScores::MAX_TOTAL
        };

        // Calculate raw score with penalties + proof bonus (capped at MAX_PROOF_BONUS)
        // Cap at max_possible to prevent overflow from corrupted evidence
        let bonus_points = self.proof_bonus.as_ref().map_or(0, |b| {
            b.bonus_points.min(crate::proof_status::MAX_PROOF_BONUS)
        });
        let raw_score = categories
            .total()
            .saturating_sub(total_penalty)
            .saturating_add(bonus_points)
            .min(max_possible);

        // Normalize to 0-100 using logarithmic scaling
        // This makes 100/100 extremely difficult to achieve
        let normalized = self.normalize_score_with_max(raw_score, categories.total(), max_possible);

        let grade = Self::calculate_grade(normalized);

        Ok(MqsScore {
            model_id: model_id.to_string(),
            raw_score,
            normalized_score: normalized,
            grade,
            gateways,
            gateways_passed: true,
            categories,
            total_tests: all_evidence.len(),
            tests_passed: evidence.pass_count(),
            tests_failed: evidence.fail_count(),
            penalties,
            total_penalty,
            proof_bonus: self.proof_bonus.clone(),
        })
    }
}

// Gateway checks, category scoring, normalization, and grading
include!("mqs_gateways.rs");

#[cfg(test)]
#[path = "mqs_tests.rs"]
mod tests;
