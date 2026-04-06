//! Core implementation of the CITL trainer

#![allow(clippy::field_reassign_with_default)]

use super::super::stats::Session;
use super::super::{
    CITLConfig, CompilationOutcome, DecisionStats, DecisionTrace, ErrorCorrelation, SourceSpan,
    SuspiciousDecision,
};
use super::types::DecisionCITL;
use crate::citl::{DecisionPatternStore, FixPattern};
use std::collections::HashMap;

impl DecisionCITL {
    /// Create a new CITL trainer with default configuration
    pub fn new() -> Result<Self, crate::Error> {
        Self::with_config(CITLConfig::default())
    }

    /// Create a new CITL trainer with custom configuration
    pub fn with_config(config: CITLConfig) -> Result<Self, crate::Error> {
        Ok(Self {
            pattern_store: DecisionPatternStore::new()?,
            success_sessions: Vec::new(),
            failed_sessions: Vec::new(),
            decision_stats: HashMap::new(),
            config,
            session_counter: 0,
        })
    }

    /// Ingest a compilation session
    ///
    /// # Arguments
    ///
    /// * `traces` - Decision traces from the compilation
    /// * `outcome` - The compilation outcome
    /// * `fix_diff` - Optional fix diff if the error was fixed
    pub fn ingest_session(
        &mut self,
        traces: Vec<DecisionTrace>,
        outcome: CompilationOutcome,
        fix_diff: Option<String>,
    ) -> Result<(), crate::Error> {
        self.session_counter += 1;
        let session_id = format!("session_{}", self.session_counter);

        let session = Session {
            id: session_id,
            decisions: traces.clone(),
            outcome: outcome.clone(),
            fix_diff: fix_diff.clone(),
        };

        // Update decision statistics
        let is_success = outcome.is_success();
        for trace in &traces {
            let stats = self.decision_stats.entry(trace.decision_type.clone()).or_default();
            if is_success {
                stats.success_count += 1;
            } else {
                stats.fail_count += 1;
            }
        }

        // Update totals
        for stats in self.decision_stats.values_mut() {
            if is_success {
                stats.total_success += 1;
            } else {
                stats.total_fail += 1;
            }
        }

        // Store session
        if is_success {
            self.success_sessions.push(session);
        } else {
            // If we have a fix diff, create a pattern
            if let Some(diff) = fix_diff {
                for error_code in outcome.error_codes() {
                    let decisions: Vec<String> =
                        traces.iter().map(|t| t.decision_type.clone()).collect();

                    let pattern = FixPattern::new(error_code, &diff).with_decisions(decisions);

                    self.pattern_store.index_fix(pattern)?;
                }
            }
            self.failed_sessions.push(session);
        }

        Ok(())
    }

    /// Correlate an error with decisions that may have caused it
    ///
    /// # Arguments
    ///
    /// * `error_code` - The error code to analyze
    /// * `error_span` - The source span where the error occurred
    ///
    /// # Returns
    ///
    /// Error correlation with suspicious decisions and fix suggestions
    pub fn correlate_error(
        &self,
        error_code: &str,
        error_span: &SourceSpan,
    ) -> Result<ErrorCorrelation, crate::Error> {
        // Find decisions that overlap with the error span
        let overlapping_decisions = self.find_overlapping_decisions(error_span);

        // Calculate suspiciousness scores using Tarantula
        let mut suspicious: Vec<SuspiciousDecision> = overlapping_decisions
            .into_iter()
            .map(|d| {
                let score = self
                    .decision_stats
                    .get(&d.decision_type)
                    .map_or(0.0, DecisionStats::tarantula_score);

                SuspiciousDecision::new(
                    d.clone(),
                    score,
                    format!(
                        "Decision '{}' overlaps error span with suspiciousness {:.2}",
                        d.decision_type, score
                    ),
                )
            })
            .filter(|s| s.suspiciousness >= self.config.min_suspiciousness)
            .collect();

        // Sort by suspiciousness
        suspicious.sort_by(|a, b| {
            b.suspiciousness.partial_cmp(&a.suspiciousness).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build dependency chain if enabled
        if self.config.enable_dependency_graph && !suspicious.is_empty() {
            suspicious = self.expand_with_dependencies(suspicious);
        }

        // Get fix suggestions from pattern store
        let decision_context: Vec<String> =
            suspicious.iter().map(|s| s.decision.decision_type.clone()).collect();

        let fix_suggestions = self.pattern_store.suggest_fix(
            error_code,
            &decision_context,
            self.config.max_suggestions,
        )?;

        Ok(ErrorCorrelation {
            error_code: error_code.to_string(),
            error_span: error_span.clone(),
            suspicious_decisions: suspicious,
            fix_suggestions,
        })
    }
}
