//! Scoring types: contract scores, codebase scores, grades.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Custom scoring weights for the 5 contract dimensions.
/// Must sum to 1.0 (auto-normalized if not).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub spec_depth: f64,
    pub falsification: f64,
    pub kani: f64,
    pub lean: f64,
    pub binding: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            spec_depth: 0.25,
            falsification: 0.25,
            kani: 0.25,
            lean: 0.05,
            binding: 0.20,
        }
    }
}

impl ScoringWeights {
    /// Normalize weights so they sum to 1.0.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let total = self.spec_depth + self.falsification + self.kani + self.lean + self.binding;
        if total == 0.0 || (total - 1.0).abs() < 1e-9 {
            return self.clone();
        }
        Self {
            spec_depth: self.spec_depth / total,
            falsification: self.falsification / total,
            kani: self.kani / total,
            lean: self.lean / total,
            binding: self.binding / total,
        }
    }
}

/// A single probe within a scoring dimension.
///
/// Each probe represents one contributing factor to a dimension's score,
/// e.g. "equation `softmax` has domain" or "obligation `finite` has harness KANI-001".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreProbe {
    /// Which scoring dimension this probe belongs to (e.g. `spec_depth`, `kani`).
    pub dimension: String,
    /// Human-readable name of the probed item (e.g. equation name, obligation property).
    pub probe: String,
    /// Whether this probe passed (true) or failed (false).
    pub outcome: bool,
    /// Detail string explaining the result (e.g. harness ID or "(no harness)").
    pub detail: String,
}

/// Score for a single contract across 5 dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractScore {
    pub stem: String,
    pub spec_depth: f64,
    pub falsification_coverage: f64,
    pub kani_coverage: f64,
    pub lean_coverage: f64,
    pub binding_coverage: f64,
    pub composite: f64,
    pub grade: Grade,
    /// Per-dimension probe breakdown showing what contributed to each score.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probes: Vec<ScoreProbe>,
}

/// Score for a codebase that consumes contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseScore {
    pub path: String,
    pub contract_coverage: f64,
    pub binding_completeness: f64,
    pub mean_contract_score: f64,
    pub proof_depth_dist: f64,
    pub drift: f64,
    /// D6: Reverse coverage (0.0-1.0) — fraction of pub fns with bindings.
    /// Set via `pv coverage --reverse` on the consumer crate.
    #[serde(default)]
    pub reverse_coverage: f64,
    /// D7: Mutation testing score (0.0-1.0) — mutants killed / mutants tested.
    /// Populated by certeza or cargo-mutants results.
    #[serde(default)]
    pub mutation_testing: f64,
    /// D8: CI pipeline depth (0.0-1.0) — fraction of CI stages present.
    /// Populated by GitHub Actions audit.
    #[serde(default)]
    pub ci_pipeline_depth: f64,
    /// D9: Proof freshness (0.0-1.0) — decay based on days since last Kani/Lean run.
    /// Populated from CI timestamps.
    #[serde(default)]
    pub proof_freshness: f64,
    /// D10: Defect patterns (0.0-1.0) — inverse defect density from git history.
    /// Populated by org-intelligence-plugin analysis.
    #[serde(default)]
    pub defect_patterns: f64,
    pub composite: f64,
    pub grade: Grade,
    pub top_gaps: Vec<ScoringGap>,
}

/// A gap identified by the scoring system.
///
/// Impact is computed per spec Section 4:
/// `impact = (1.0 - obligation_coverage) * dependency_fanout * tier_weight`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringGap {
    pub contract: String,
    pub dimension: String,
    pub current: f64,
    pub target: f64,
    pub impact: f64,
    pub action: String,
}

/// Letter grade A-F.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    A,
    B,
    C,
    D,
    F,
}

impl Grade {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.90 {
            Grade::A
        } else if score >= 0.75 {
            Grade::B
        } else if score >= 0.60 {
            Grade::C
        } else if score >= 0.40 {
            Grade::D
        } else {
            Grade::F
        }
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Grade::A => write!(f, "A"),
            Grade::B => write!(f, "B"),
            Grade::C => write!(f, "C"),
            Grade::D => write!(f, "D"),
            Grade::F => write!(f, "F"),
        }
    }
}

impl fmt::Display for CodebaseScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Codebase: {} — {:.2} (Grade {})",
            self.path, self.composite, self.grade
        )?;
        writeln!(
            f,
            "  Coverage: {:.0}% | Binding: {:.0}% | MeanScore: {:.2} | ProofDepth: {:.2} | Drift: {:.2}",
            self.contract_coverage * 100.0,
            self.binding_completeness * 100.0,
            self.mean_contract_score,
            self.proof_depth_dist,
            self.drift,
        )?;
        if !self.top_gaps.is_empty() {
            writeln!(f, "  Top gaps:")?;
            for gap in &self.top_gaps {
                writeln!(
                    f,
                    "    {}: {} ({:.2} -> {:.2}, impact: {:.2}) — {}",
                    gap.contract, gap.dimension, gap.current, gap.target, gap.impact, gap.action
                )?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for ContractScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} — {:.2} (Grade {})",
            self.stem, self.composite, self.grade
        )?;
        writeln!(
            f,
            "  Spec: {:.2} | Falsify: {:.2} | Kani: {:.2} | Lean: {:.2} | Bind: {:.2}",
            self.spec_depth,
            self.falsification_coverage,
            self.kani_coverage,
            self.lean_coverage,
            self.binding_coverage
        )
    }
}
