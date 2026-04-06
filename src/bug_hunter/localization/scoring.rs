//! Scoring types for fault localization.
//!
//! Contains `ScoredLocation`, `TestCoverage`, `SpectrumData` (SBFL),
//! and `MutationData` (MBFL) used by the multi-channel localizer.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::bug_hunter::types::{ChannelWeights, SbflFormula};

/// Location with multi-channel scores.
#[derive(Debug, Clone)]
pub struct ScoredLocation {
    pub file: PathBuf,
    pub line: usize,
    /// SBFL spectrum score
    pub spectrum_score: f64,
    /// MBFL mutation score
    pub mutation_score: f64,
    /// Static analysis score
    pub static_score: f64,
    /// Semantic similarity score
    pub semantic_score: f64,
    /// PMAT quality score (BH-21)
    pub quality_score: f64,
    /// Final combined score
    pub final_score: f64,
}

impl ScoredLocation {
    pub fn new(file: PathBuf, line: usize) -> Self {
        Self {
            file,
            line,
            spectrum_score: 0.0,
            mutation_score: 0.0,
            static_score: 0.0,
            semantic_score: 0.0,
            quality_score: 0.0,
            final_score: 0.0,
        }
    }

    /// Compute final score using channel weights (BH-19, BH-21)
    pub fn compute_final_score(&mut self, weights: &ChannelWeights) {
        self.final_score = weights.combine(
            self.spectrum_score,
            self.mutation_score,
            self.static_score,
            self.semantic_score,
            self.quality_score,
        );
    }
}

/// Coverage data for a single test.
#[derive(Debug, Clone)]
pub struct TestCoverage {
    pub test_name: String,
    pub passed: bool,
    /// Lines executed: (file, line) -> execution count
    pub executed_lines: HashMap<(PathBuf, usize), usize>,
}

/// SBFL spectrum data.
#[derive(Debug, Default)]
pub struct SpectrumData {
    /// Lines covered by failing tests
    pub failed_coverage: HashMap<(PathBuf, usize), usize>,
    /// Lines covered by passing tests
    pub passed_coverage: HashMap<(PathBuf, usize), usize>,
    /// Total failing tests
    pub total_failed: usize,
    /// Total passing tests
    pub total_passed: usize,
}

impl SpectrumData {
    /// Compute SBFL score for a location using the specified formula.
    pub fn compute_score(&self, file: &Path, line: usize, formula: SbflFormula) -> f64 {
        let key = (file.to_path_buf(), line);
        let ef = *self.failed_coverage.get(&key).unwrap_or(&0) as f64;
        let ep = *self.passed_coverage.get(&key).unwrap_or(&0) as f64;
        let nf = (self.total_failed as f64) - ef;

        match formula {
            SbflFormula::Tarantula => sbfl_tarantula(ef, ep, self.total_failed, self.total_passed),
            SbflFormula::Ochiai => sbfl_ochiai(ef, ep, nf),
            SbflFormula::DStar2 => sbfl_dstar(ef, ep, nf, 2),
            SbflFormula::DStar3 => sbfl_dstar(ef, ep, nf, 3),
        }
    }
}

fn sbfl_tarantula(ef: f64, ep: f64, total_failed: usize, total_passed: usize) -> f64 {
    let fail_ratio = if total_failed > 0 { ef / total_failed as f64 } else { 0.0 };
    let pass_ratio = if total_passed > 0 { ep / total_passed as f64 } else { 0.0 };
    let sum = fail_ratio + pass_ratio;
    if sum > 0.0 {
        fail_ratio / sum
    } else {
        0.0
    }
}

fn sbfl_ochiai(ef: f64, ep: f64, nf: f64) -> f64 {
    let denom = ((ef + nf) * (ef + ep)).sqrt();
    if denom > 0.0 {
        ef / denom
    } else {
        0.0
    }
}

fn sbfl_dstar(ef: f64, ep: f64, nf: f64, power: u32) -> f64 {
    let denom = ep + nf;
    if denom > 0.0 {
        ef.powi(power as i32) / denom
    } else if ef > 0.0 {
        f64::MAX
    } else {
        0.0
    }
}

/// Mutation data for MBFL (BH-16).
#[derive(Debug, Default)]
pub struct MutationData {
    /// Mutants at each location: (file, line) -> (total_mutants, killed_by_failing_tests)
    pub mutants: HashMap<(PathBuf, usize), (usize, usize)>,
}

impl MutationData {
    /// Compute MBFL score for a location.
    /// Score = |killed_by_failing| / |total_mutants|
    pub fn compute_score(&self, file: &Path, line: usize) -> f64 {
        let key = (file.to_path_buf(), line);
        if let Some((total, killed)) = self.mutants.get(&key) {
            if *total > 0 {
                return *killed as f64 / *total as f64;
            }
        }
        0.0
    }
}
