//! Multi-channel fault localizer (BH-19).
//!
//! Combines SBFL spectrum, MBFL mutation, static analysis,
//! and semantic similarity channels for fault localization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::bug_hunter::types::{ChannelWeights, LocalizationStrategy, SbflFormula};

use super::scoring::{MutationData, ScoredLocation, SpectrumData, TestCoverage};

/// Multi-channel fault localizer (BH-19).
pub struct MultiChannelLocalizer {
    pub strategy: LocalizationStrategy,
    pub weights: ChannelWeights,
    pub sbfl_formula: SbflFormula,
    pub spectrum_data: SpectrumData,
    pub mutation_data: MutationData,
    /// Static analysis findings mapped to locations
    pub static_findings: HashMap<(PathBuf, usize), f64>,
    /// Error message for semantic matching
    pub error_message: Option<String>,
}

impl MultiChannelLocalizer {
    pub fn new(strategy: LocalizationStrategy, weights: ChannelWeights) -> Self {
        Self {
            strategy,
            weights,
            sbfl_formula: SbflFormula::Ochiai,
            spectrum_data: SpectrumData::default(),
            mutation_data: MutationData::default(),
            static_findings: HashMap::new(),
            error_message: None,
        }
    }

    /// Add spectrum data from coverage.
    pub fn add_coverage(&mut self, coverage: &[TestCoverage]) {
        for test in coverage {
            if test.passed {
                self.spectrum_data.total_passed += 1;
                for (loc, count) in &test.executed_lines {
                    *self.spectrum_data.passed_coverage.entry(loc.clone()).or_insert(0) += count;
                }
            } else {
                self.spectrum_data.total_failed += 1;
                for (loc, count) in &test.executed_lines {
                    *self.spectrum_data.failed_coverage.entry(loc.clone()).or_insert(0) += count;
                }
            }
        }
    }

    /// Add static analysis finding.
    pub fn add_static_finding(&mut self, file: &Path, line: usize, score: f64) {
        self.static_findings.insert((file.to_path_buf(), line), score);
    }

    /// Set error message for semantic matching.
    pub fn set_error_message(&mut self, msg: &str) {
        self.error_message = Some(msg.to_string());
    }

    /// Compute semantic similarity score for a location.
    pub(crate) fn compute_semantic_score(&self, _file: &Path, line: usize, content: &str) -> f64 {
        let Some(ref error_msg) = self.error_message else {
            return 0.0;
        };

        // Simple keyword matching (could be enhanced with embeddings)
        let error_lower = error_msg.to_lowercase();
        let error_words: Vec<&str> =
            error_lower.split_whitespace().filter(|w| w.len() > 3).collect();

        let line_content = content.lines().nth(line.saturating_sub(1)).unwrap_or("");
        let line_lower = line_content.to_lowercase();

        let matches = error_words.iter().filter(|w| line_lower.contains(*w)).count();

        if error_words.is_empty() {
            0.0
        } else {
            (matches as f64 / error_words.len() as f64).min(1.0)
        }
    }

    /// Localize faults using the configured strategy.
    pub fn localize(&self, project_path: &Path) -> Vec<ScoredLocation> {
        let mut locations: HashMap<(PathBuf, usize), ScoredLocation> = HashMap::new();

        // Collect all locations from all channels
        for key in self.spectrum_data.failed_coverage.keys() {
            locations
                .entry(key.clone())
                .or_insert_with(|| ScoredLocation::new(key.0.clone(), key.1));
        }
        for key in self.mutation_data.mutants.keys() {
            locations
                .entry(key.clone())
                .or_insert_with(|| ScoredLocation::new(key.0.clone(), key.1));
        }
        for key in self.static_findings.keys() {
            locations
                .entry(key.clone())
                .or_insert_with(|| ScoredLocation::new(key.0.clone(), key.1));
        }

        // Compute channel scores for each location
        let mut result: Vec<ScoredLocation> = locations
            .into_iter()
            .map(|(key, mut loc)| {
                // Spectrum score (SBFL)
                loc.spectrum_score =
                    self.spectrum_data.compute_score(&key.0, key.1, self.sbfl_formula);

                // Mutation score (MBFL)
                loc.mutation_score = self.mutation_data.compute_score(&key.0, key.1);

                // Static analysis score
                loc.static_score = *self.static_findings.get(&key).unwrap_or(&0.0);

                // Semantic score (if error message available)
                if self.error_message.is_some() {
                    let file_path = project_path.join(&key.0);
                    if let Ok(content) = std::fs::read_to_string(&file_path) {
                        loc.semantic_score = self.compute_semantic_score(&key.0, key.1, &content);
                    }
                }

                // Compute final score based on strategy
                match self.strategy {
                    LocalizationStrategy::Sbfl => {
                        loc.final_score = loc.spectrum_score;
                    }
                    LocalizationStrategy::Mbfl => {
                        loc.final_score = loc.mutation_score;
                    }
                    LocalizationStrategy::Causal => {
                        // Causal uses modified spectrum with intervention estimation
                        // For now, use spectrum as approximation
                        loc.final_score = loc.spectrum_score;
                    }
                    LocalizationStrategy::MultiChannel | LocalizationStrategy::Hybrid => {
                        loc.compute_final_score(&self.weights);
                    }
                }

                loc
            })
            .collect();

        // Sort by final score (descending)
        result.sort_by(|a, b| {
            b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        result
    }
}
