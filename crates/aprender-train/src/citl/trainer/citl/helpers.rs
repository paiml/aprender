//! Helper methods for the CITL trainer

use super::super::{DecisionStats, DecisionTrace, SourceSpan, SuspiciousDecision};
use super::types::DecisionCITL;
use crate::citl::DecisionPatternStore;
use std::collections::{HashMap, HashSet};

impl DecisionCITL {
    /// Find decisions whose spans overlap with the given span
    pub(crate) fn find_overlapping_decisions(&self, span: &SourceSpan) -> Vec<&DecisionTrace> {
        let mut result = Vec::new();

        for session in &self.failed_sessions {
            for decision in &session.decisions {
                if let Some(decision_span) = &decision.span {
                    if decision_span.overlaps(span) {
                        result.push(decision);
                    }
                }
            }
        }

        result
    }

    /// Search failed sessions for a decision by ID and return its suspiciousness score.
    fn find_dependency_score(&self, dep_id: &str) -> Option<(DecisionTrace, f32)> {
        for session in &self.failed_sessions {
            for decision in &session.decisions {
                if decision.id != dep_id {
                    continue;
                }
                let score = self
                    .decision_stats
                    .get(&decision.decision_type)
                    .map_or(0.0, DecisionStats::tarantula_score);
                // Reduce suspiciousness for indirect dependencies
                let adjusted = score * 0.8;
                return Some((decision.clone(), adjusted));
            }
        }
        None
    }

    /// Expand suspicious decisions with their dependencies
    pub(crate) fn expand_with_dependencies(
        &self,
        mut suspicious: Vec<SuspiciousDecision>,
    ) -> Vec<SuspiciousDecision> {
        let mut seen: HashSet<String> = suspicious.iter().map(|s| s.decision.id.clone()).collect();
        let mut i = 0;

        while i < suspicious.len() {
            let deps = suspicious[i].decision.depends_on.clone();

            for dep_id in deps {
                if seen.contains(&dep_id) {
                    continue;
                }
                if let Some((decision, adjusted_score)) = self.find_dependency_score(&dep_id) {
                    if adjusted_score >= self.config.min_suspiciousness {
                        suspicious.push(SuspiciousDecision::new(
                            decision,
                            adjusted_score,
                            "Dependency of suspicious decision (indirect)",
                        ));
                        seen.insert(dep_id);
                    }
                }
            }

            i += 1;
        }

        // Re-sort after adding dependencies
        suspicious.sort_by(|a, b| {
            b.suspiciousness.partial_cmp(&a.suspiciousness).unwrap_or(std::cmp::Ordering::Equal)
        });

        suspicious
    }

    /// Get the pattern store
    #[must_use]
    pub fn pattern_store(&self) -> &DecisionPatternStore {
        &self.pattern_store
    }

    /// Get mutable pattern store
    pub fn pattern_store_mut(&mut self) -> &mut DecisionPatternStore {
        &mut self.pattern_store
    }

    /// Get the number of successful sessions
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.success_sessions.len()
    }

    /// Get the number of failed sessions
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failed_sessions.len()
    }

    /// Get the total number of sessions
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.success_sessions.len() + self.failed_sessions.len()
    }

    /// Get decision statistics
    #[must_use]
    pub fn decision_stats(&self) -> &HashMap<String, DecisionStats> {
        &self.decision_stats
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &super::super::CITLConfig {
        &self.config
    }

    /// Get top suspicious decision types (by Tarantula score)
    #[must_use]
    pub fn top_suspicious_types(&self, k: usize) -> Vec<(&str, f32)> {
        let mut scores: Vec<_> =
            self.decision_stats.iter().map(|(t, s)| (t.as_str(), s.tarantula_score())).collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Group decisions by source file
    #[must_use]
    pub fn decisions_by_file(&self) -> HashMap<String, Vec<&DecisionTrace>> {
        let mut by_file: HashMap<String, Vec<&DecisionTrace>> = HashMap::new();

        for session in self.failed_sessions.iter().chain(self.success_sessions.iter()) {
            for decision in &session.decisions {
                if let Some(span) = &decision.span {
                    by_file.entry(span.file.clone()).or_default().push(decision);
                }
            }
        }

        by_file
    }

    /// Build a dependency graph for all decisions
    #[must_use]
    pub fn build_dependency_graph(&self) -> HashMap<String, Vec<String>> {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for session in self.failed_sessions.iter().chain(self.success_sessions.iter()) {
            for decision in &session.decisions {
                graph.entry(decision.id.clone()).or_default().extend(decision.depends_on.clone());
            }
        }

        graph
    }

    /// Find the root causes (decisions with no dependencies that are suspicious)
    #[must_use]
    pub fn find_root_causes(&self, error_span: &SourceSpan) -> Vec<&DecisionTrace> {
        let overlapping = self.find_overlapping_decisions(error_span);
        let _graph = self.build_dependency_graph();

        // Find decisions that are depended upon but don't depend on others in the set
        let all_ids: HashSet<_> = overlapping.iter().map(|d| &d.id).collect();

        overlapping
            .into_iter()
            .filter(|d| {
                // Check if this decision's dependencies are outside the suspicious set
                d.depends_on.iter().all(|dep| !all_ids.contains(dep))
            })
            .collect()
    }
}
