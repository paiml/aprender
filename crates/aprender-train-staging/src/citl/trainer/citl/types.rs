//! Type definitions for the CITL trainer

use super::super::stats::Session;
use super::super::{CITLConfig, DecisionStats};
use crate::citl::DecisionPatternStore;
use std::collections::HashMap;

/// Compiler-in-the-Loop (CITL) trainer
///
/// Correlates compiler decision traces with compilation outcomes
/// to identify error-contributing decisions and suggest fixes.
///
/// # Example
///
/// ```ignore
/// use entrenar::citl::{DecisionCITL, DecisionTrace, CompilationOutcome, SourceSpan};
///
/// let mut trainer = DecisionCITL::new()?;
///
/// // Ingest a failed session
/// let traces = vec![
///     DecisionTrace::new("d1", "type_inference", "Inferred type i32")
///         .with_span(SourceSpan::line("main.rs", 5)),
/// ];
/// let outcome = CompilationOutcome::failure(
///     vec!["E0308".to_string()],
///     vec![SourceSpan::line("main.rs", 5)],
///     vec!["mismatched types".to_string()],
/// );
///
/// trainer.ingest_session(traces, outcome, None)?;
///
/// // Correlate errors
/// let correlations = trainer.correlate_error("E0308", &SourceSpan::line("main.rs", 5))?;
/// ```
pub struct DecisionCITL {
    /// Pattern store for fix suggestions
    pub(crate) pattern_store: DecisionPatternStore,
    /// Sessions indexed by outcome type
    pub(crate) success_sessions: Vec<Session>,
    pub(crate) failed_sessions: Vec<Session>,
    /// Decision frequency in successful vs failed sessions (for Tarantula)
    pub(crate) decision_stats: HashMap<String, DecisionStats>,
    /// Configuration
    pub(crate) config: CITLConfig,
    /// Session counter
    pub(crate) session_counter: u64,
}

impl std::fmt::Debug for DecisionCITL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionCITL")
            .field("success_sessions", &self.success_sessions.len())
            .field("failed_sessions", &self.failed_sessions.len())
            .field("decision_types", &self.decision_stats.len())
            .field("patterns", &self.pattern_store.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
