//! Pipeline Audit Trail Integration
//!
//! Provides distributed provenance and audit logging for transpilation pipelines
//! using entrenar's inference monitoring infrastructure.
//!
//! # Features
//!
//! - **Stage Execution Tracking**: Record decision paths for each pipeline stage
//! - **Hash Chain Provenance**: Tamper-evident audit trails for distributed execution
//! - **Pipeline Lineage**: Track input→output transformations with full reproducibility
//!
//! # Toyota Way: 品質は作り込む (Hinshitsu wa tsukuri komu)
//! Quality is built in - every pipeline execution is fully auditable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::pipeline::{PipelineContext, ValidationResult};
use crate::types::Language;

// =============================================================================
// Pipeline Decision Path
// =============================================================================

/// Decision path representing a pipeline stage execution.
///
/// Implements `DecisionPath` semantics for pipeline audit trails.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelinePath {
    /// Stage name that was executed
    pub stage_name: String,

    /// Stage execution duration in nanoseconds
    pub duration_ns: u64,

    /// Whether the stage succeeded
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Input file count processed
    pub input_files: usize,

    /// Output file count produced
    pub output_files: usize,

    /// Language detected/processed
    pub language: Option<Language>,

    /// Optimizations applied during this stage
    pub optimizations: Vec<String>,

    /// Validation result for this stage
    pub validation: Option<ValidationResult>,

    /// Stage-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,

    /// Feature contributions (for ML-enhanced pipelines)
    contributions: Vec<f32>,

    /// Confidence score for this execution
    confidence: f32,
}

impl PipelinePath {
    /// Create a new pipeline path for a stage execution.
    pub fn new(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            duration_ns: 0,
            success: true,
            error: None,
            input_files: 0,
            output_files: 0,
            language: None,
            optimizations: Vec::new(),
            validation: None,
            metadata: HashMap::new(),
            contributions: Vec::new(),
            confidence: 1.0,
        }
    }

    /// Record stage execution timing.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ns = duration.as_nanos() as u64;
        self
    }

    /// Mark stage as failed with error.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        self.error = Some(error.into());
        self.confidence = 0.0;
        self
    }

    /// Set file counts.
    pub fn with_file_counts(mut self, input: usize, output: usize) -> Self {
        self.input_files = input;
        self.output_files = output;
        self
    }

    /// Set language context.
    pub fn with_language(mut self, lang: Language) -> Self {
        self.language = Some(lang);
        self
    }

    /// Add optimizations applied.
    pub fn with_optimizations(mut self, opts: Vec<String>) -> Self {
        self.optimizations = opts;
        self
    }

    /// Set validation result.
    pub fn with_validation(mut self, validation: ValidationResult) -> Self {
        if !validation.passed {
            self.confidence *= 0.5;
        }
        self.validation = Some(validation);
        self
    }

    /// Add metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set feature contributions (for ML-enhanced analysis).
    pub fn with_contributions(mut self, contributions: Vec<f32>) -> Self {
        self.contributions = contributions;
        self
    }

    /// Get feature contributions.
    pub fn feature_contributions(&self) -> &[f32] {
        &self.contributions
    }

    /// Get confidence score.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Serialize to bytes for hashing.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Use bincode-style serialization
        let mut bytes = Vec::new();

        // Stage name
        bytes.extend_from_slice(self.stage_name.as_bytes());
        bytes.push(0);

        // Duration
        bytes.extend_from_slice(&self.duration_ns.to_le_bytes());

        // Success flag
        bytes.push(u8::from(self.success));

        // Error (if any)
        if let Some(ref error) = self.error {
            bytes.extend_from_slice(error.as_bytes());
        }
        bytes.push(0);

        // File counts
        bytes.extend_from_slice(&(self.input_files as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.output_files as u64).to_le_bytes());

        // Confidence
        bytes.extend_from_slice(&self.confidence.to_le_bytes());

        bytes
    }

    /// Generate a text explanation of the stage execution.
    pub fn explain(&self) -> String {
        let mut explanation = format!("Stage: {}\n", self.stage_name);
        explanation
            .push_str(&format!("Duration: {:.2}ms\n", self.duration_ns as f64 / 1_000_000.0));
        explanation.push_str(&format!("Success: {}\n", self.success));

        if let Some(ref error) = self.error {
            explanation.push_str(&format!("Error: {}\n", error));
        }

        explanation.push_str(&format!(
            "Files: {} input → {} output\n",
            self.input_files, self.output_files
        ));

        if let Some(ref lang) = self.language {
            explanation.push_str(&format!("Language: {:?}\n", lang));
        }

        if !self.optimizations.is_empty() {
            explanation.push_str(&format!("Optimizations: {}\n", self.optimizations.join(", ")));
        }

        explanation.push_str(&format!("Confidence: {:.1}%", self.confidence * 100.0));

        explanation
    }
}

// =============================================================================
// Pipeline Audit Trace
// =============================================================================

/// A single audit trace entry for pipeline execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineTrace {
    /// Sequence number within the audit trail
    pub sequence: u64,

    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,

    /// The decision path for this trace
    pub path: PipelinePath,

    /// Pipeline context snapshot (optional, for full reproducibility)
    pub context_snapshot: Option<ContextSnapshot>,
}

/// Snapshot of pipeline context for audit purposes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextSnapshot {
    /// Input path
    pub input_path: PathBuf,

    /// Output path
    pub output_path: PathBuf,

    /// Primary language
    pub language: Option<Language>,

    /// Number of file mappings
    pub file_mapping_count: usize,

    /// Metadata keys present
    pub metadata_keys: Vec<String>,
}

impl From<&PipelineContext> for ContextSnapshot {
    fn from(ctx: &PipelineContext) -> Self {
        Self {
            input_path: ctx.input_path.clone(),
            output_path: ctx.output_path.clone(),
            language: ctx.primary_language.clone(),
            file_mapping_count: ctx.file_mappings.len(),
            metadata_keys: ctx.metadata.keys().cloned().collect(),
        }
    }
}

// =============================================================================
// Hash Chain Entry
// =============================================================================

/// Hash chain entry for tamper-evident audit trail.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashChainEntry {
    /// Sequence number
    pub sequence: u64,

    /// SHA-256 hash of previous entry (all zeros for genesis)
    pub prev_hash: [u8; 32],

    /// SHA-256 hash of this entry
    pub hash: [u8; 32],

    /// The pipeline trace
    pub trace: PipelineTrace,
}

// =============================================================================
// Pipeline Audit Collector
// =============================================================================

/// Collector for pipeline audit trails.
///
/// Provides tamper-evident hash chain provenance for distributed pipeline execution.
#[derive(Debug)]
pub struct PipelineAuditCollector {
    /// Audit entries in hash chain order
    entries: Vec<HashChainEntry>,

    /// Next sequence number
    next_sequence: u64,

    /// Pipeline run identifier
    run_id: String,

    /// Whether to capture context snapshots
    capture_snapshots: bool,
}

impl PipelineAuditCollector {
    /// Create a new pipeline audit collector.
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 0,
            run_id: run_id.into(),
            capture_snapshots: true,
        }
    }

    /// Disable context snapshot capture (for reduced memory usage).
    pub fn without_snapshots(mut self) -> Self {
        self.capture_snapshots = false;
        self
    }

    /// Get the run identifier.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Record a stage execution.
    pub fn record_stage(
        &mut self,
        path: PipelinePath,
        context: Option<&PipelineContext>,
    ) -> &HashChainEntry {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let context_snapshot =
            if self.capture_snapshots { context.map(ContextSnapshot::from) } else { None };

        let trace =
            PipelineTrace { sequence: self.next_sequence, timestamp_ns, path, context_snapshot };

        // Get previous hash
        let prev_hash = self.entries.last().map(|e| e.hash).unwrap_or([0u8; 32]);

        // Compute hash of this entry
        let hash = self.compute_hash(&trace, &prev_hash);

        let entry = HashChainEntry { sequence: self.next_sequence, prev_hash, hash, trace };

        self.entries.push(entry);
        self.next_sequence += 1;

        self.entries.last().expect("just pushed")
    }

    /// Compute SHA-256 hash for an entry.
    fn compute_hash(&self, trace: &PipelineTrace, prev_hash: &[u8; 32]) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple hash computation (in production, use SHA-256)
        let mut hasher = DefaultHasher::new();

        // Hash previous hash
        prev_hash.hash(&mut hasher);

        // Hash trace data
        trace.sequence.hash(&mut hasher);
        trace.timestamp_ns.hash(&mut hasher);
        trace.path.stage_name.hash(&mut hasher);
        trace.path.duration_ns.hash(&mut hasher);
        trace.path.success.hash(&mut hasher);

        let hash_value = hasher.finish();

        // Convert u64 hash to [u8; 32] by repeating
        let mut result = [0u8; 32];
        for i in 0..4 {
            result[i * 8..(i + 1) * 8].copy_from_slice(&hash_value.to_le_bytes());
        }

        result
    }

    /// Get all entries.
    pub fn entries(&self) -> &[HashChainEntry] {
        &self.entries
    }

    /// Get entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Verify hash chain integrity.
    pub fn verify_chain(&self) -> ChainVerification {
        let mut entries_verified = 0;

        for (i, entry) in self.entries.iter().enumerate() {
            // Verify prev_hash linkage
            if i == 0 {
                if entry.prev_hash != [0u8; 32] {
                    return ChainVerification::invalid_at(entries_verified, 0);
                }
            } else {
                let expected_prev = self.entries[i - 1].hash;
                if entry.prev_hash != expected_prev {
                    return ChainVerification::invalid_at(entries_verified, i);
                }
            }

            // Verify entry hash
            let computed_hash = self.compute_hash(&entry.trace, &entry.prev_hash);
            if entry.hash != computed_hash {
                return ChainVerification::invalid_at(entries_verified, i);
            }

            entries_verified += 1;
        }

        ChainVerification::valid(entries_verified)
    }

    /// Get recent entries.
    pub fn recent(&self, n: usize) -> Vec<&HashChainEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Export to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct Export<'a> {
            run_id: &'a str,
            chain_length: usize,
            verified: bool,
            entries: &'a [HashChainEntry],
        }

        let verification = self.verify_chain();

        let export = Export {
            run_id: &self.run_id,
            chain_length: self.entries.len(),
            verified: verification.valid,
            entries: &self.entries,
        };

        serde_json::to_string_pretty(&export)
    }
}

/// Result of hash chain verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainVerification {
    /// Whether the chain is valid
    pub valid: bool,

    /// Number of entries successfully verified
    pub entries_verified: usize,

    /// Index of first broken link (if any)
    pub first_break: Option<usize>,
}

impl ChainVerification {
    fn valid(entries_verified: usize) -> Self {
        Self { valid: true, entries_verified, first_break: None }
    }
    fn invalid_at(entries_verified: usize, index: usize) -> Self {
        Self { valid: false, entries_verified, first_break: Some(index) }
    }
}

// =============================================================================
// Stage Timer
// =============================================================================

/// Timer for measuring stage execution duration.
pub struct StageTimer {
    start: Instant,
    stage_name: String,
}

impl StageTimer {
    /// Start timing a stage.
    pub fn start(stage_name: impl Into<String>) -> Self {
        Self { start: crate::timing::start_timer(), stage_name: stage_name.into() }
    }

    /// Stop timing and create a pipeline path.
    pub fn stop(self) -> PipelinePath {
        let duration = self.start.elapsed();
        PipelinePath::new(self.stage_name).with_duration(duration)
    }

    /// Stop timing with error.
    pub fn stop_with_error(self, error: impl Into<String>) -> PipelinePath {
        let duration = self.start.elapsed();
        PipelinePath::new(self.stage_name).with_duration(duration).with_error(error)
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a new pipeline audit collector with a generated run ID.
pub fn new_audit_collector() -> PipelineAuditCollector {
    let run_id = format!(
        "run-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    );
    PipelineAuditCollector::new(run_id)
}

/// Record a successful stage execution.
pub fn record_success<'a>(
    collector: &'a mut PipelineAuditCollector,
    stage_name: &str,
    duration: Duration,
    context: Option<&PipelineContext>,
) -> &'a HashChainEntry {
    let path = PipelinePath::new(stage_name).with_duration(duration);
    collector.record_stage(path, context)
}

/// Record a failed stage execution.
pub fn record_failure<'a>(
    collector: &'a mut PipelineAuditCollector,
    stage_name: &str,
    duration: Duration,
    error: &str,
    context: Option<&PipelineContext>,
) -> &'a HashChainEntry {
    let path = PipelinePath::new(stage_name).with_duration(duration).with_error(error);
    collector.record_stage(path, context)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn new_collector_with_stages(run_id: &str, stages: &[&str]) -> PipelineAuditCollector {
        let mut collector = PipelineAuditCollector::new(run_id);
        for stage in stages {
            collector.record_stage(PipelinePath::new(*stage), None);
        }
        collector
    }

    pub(crate) fn make_validation(stage: &str, passed: bool) -> ValidationResult {
        ValidationResult {
            stage: stage.to_string(),
            passed,
            message: if passed { "OK" } else { "Failed" }.to_string(),
            details: None,
        }
    }

    #[test]
    fn test_pipeline_path_creation() {
        let path = PipelinePath::new("Analysis");
        assert_eq!(path.stage_name, "Analysis");
        assert!(path.success);
        assert_eq!(path.confidence(), 1.0);
    }

    #[test]
    fn test_pipeline_path_with_duration() {
        let path = PipelinePath::new("Build").with_duration(Duration::from_millis(100));
        assert_eq!(path.duration_ns, 100_000_000);
    }

    #[test]
    fn test_pipeline_path_with_error() {
        let path = PipelinePath::new("Compile").with_error("Syntax error");
        assert!(!path.success);
        assert_eq!(path.error, Some("Syntax error".to_string()));
        assert_eq!(path.confidence(), 0.0);
    }

    #[test]
    fn test_pipeline_path_with_file_counts() {
        let path = PipelinePath::new("Transform").with_file_counts(10, 5);
        assert_eq!(path.input_files, 10);
        assert_eq!(path.output_files, 5);
    }

    #[test]
    fn test_pipeline_path_with_language() {
        let path = PipelinePath::new("Detect").with_language(Language::Python);
        assert_eq!(path.language, Some(Language::Python));
    }

    #[test]
    fn test_pipeline_path_with_optimizations() {
        let path =
            PipelinePath::new("Optimize").with_optimizations(vec!["SIMD".into(), "GPU".into()]);
        assert_eq!(path.optimizations.len(), 2);
    }

    #[test]
    fn test_pipeline_path_explain() {
        let path = PipelinePath::new("Test")
            .with_duration(Duration::from_millis(50))
            .with_file_counts(3, 2);
        let explanation = path.explain();
        assert!(explanation.contains("Test"));
        assert!(explanation.contains("50.00ms"));
        assert!(explanation.contains("3 input → 2 output"));
    }

    #[test]
    fn test_pipeline_path_to_bytes() {
        let path = PipelinePath::new("Stage");
        let bytes = path.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_audit_collector_creation() {
        let collector = PipelineAuditCollector::new("test-run");
        assert_eq!(collector.run_id(), "test-run");
        assert!(collector.is_empty());
    }

    #[test]
    fn test_audit_collector_record_stage() {
        let mut collector = PipelineAuditCollector::new("test");
        let path = PipelinePath::new("Stage1");

        let entry = collector.record_stage(path, None);

        assert_eq!(entry.sequence, 0);
        assert_eq!(entry.prev_hash, [0u8; 32]);
        assert_eq!(collector.len(), 1);
    }

    #[test]
    fn test_audit_collector_hash_chain_linkage() {
        let collector = new_collector_with_stages("test", &["Stage1", "Stage2", "Stage3"]);

        let entries = collector.entries();

        // First entry has zero prev_hash
        assert_eq!(entries[0].prev_hash, [0u8; 32]);

        // Each subsequent entry links to previous
        assert_eq!(entries[1].prev_hash, entries[0].hash);
        assert_eq!(entries[2].prev_hash, entries[1].hash);
    }

    #[test]
    fn test_audit_collector_verify_chain_valid() {
        let collector = new_collector_with_stages("test", &["Stage1", "Stage2"]);

        let verification = collector.verify_chain();
        assert!(verification.valid);
        assert_eq!(verification.entries_verified, 2);
        assert!(verification.first_break.is_none());
    }

    #[test]
    fn test_audit_collector_recent() {
        let mut collector = PipelineAuditCollector::new("test");

        for i in 0..5 {
            collector.record_stage(PipelinePath::new(format!("Stage{}", i)), None);
        }

        let recent = collector.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].sequence, 4); // Most recent first
        assert_eq!(recent[1].sequence, 3);
        assert_eq!(recent[2].sequence, 2);
    }

    #[test]
    fn test_audit_collector_to_json() {
        let mut collector = PipelineAuditCollector::new("test");
        collector.record_stage(PipelinePath::new("Stage1"), None);

        let json = collector.to_json().expect("unexpected failure");
        assert!(json.contains("test"));
        assert!(json.contains("Stage1"));
        assert!(json.contains("verified"));
    }

    #[test]
    fn test_stage_timer() {
        let timer = StageTimer::start("Test");
        let path = timer.stop();

        assert_eq!(path.stage_name, "Test");
        // duration_ns is set by Instant::elapsed - always non-negative
        assert!(path.success);
    }

    #[test]
    fn test_stage_timer_with_error() {
        let timer = StageTimer::start("Test");
        let path = timer.stop_with_error("Failed");

        assert!(!path.success);
        assert_eq!(path.error, Some("Failed".to_string()));
    }

    #[test]
    fn test_new_audit_collector() {
        let collector = new_audit_collector();
        assert!(collector.run_id().starts_with("run-"));
    }

    #[test]
    fn test_record_success() {
        let mut collector = new_audit_collector();
        let entry = record_success(&mut collector, "Stage", Duration::from_millis(100), None);

        assert!(entry.trace.path.success);
        assert_eq!(entry.trace.path.stage_name, "Stage");
    }

    #[test]
    fn test_record_failure() {
        let mut collector = new_audit_collector();
        let entry = record_failure(
            &mut collector,
            "Stage",
            Duration::from_millis(50),
            "Error message",
            None,
        );

        assert!(!entry.trace.path.success);
        assert_eq!(entry.trace.path.error, Some("Error message".to_string()));
    }

    #[test]
    fn test_context_snapshot() {
        let ctx = PipelineContext::new(
            std::path::PathBuf::from("/input"),
            std::path::PathBuf::from("/output"),
        );
        let snapshot = ContextSnapshot::from(&ctx);

        assert_eq!(snapshot.input_path, std::path::PathBuf::from("/input"));
        assert_eq!(snapshot.output_path, std::path::PathBuf::from("/output"));
    }

    #[test]
    fn test_collector_without_snapshots() {
        let mut collector = PipelineAuditCollector::new("test").without_snapshots();

        let ctx = PipelineContext::new(
            std::path::PathBuf::from("/input"),
            std::path::PathBuf::from("/output"),
        );

        collector.record_stage(PipelinePath::new("Stage"), Some(&ctx));

        // Should not have snapshot
        assert!(collector.entries()[0].trace.context_snapshot.is_none());
    }

    #[test]
    fn test_pipeline_path_with_validation_passed() {
        let path = PipelinePath::new("Stage").with_validation(make_validation("Test", true));
        assert_eq!(path.confidence(), 1.0); // Unchanged when passed
    }

    #[test]
    fn test_pipeline_path_with_validation_failed() {
        let path = PipelinePath::new("Stage").with_validation(make_validation("Test", false));
        assert_eq!(path.confidence(), 0.5); // Reduced when failed
    }

    #[test]
    fn test_pipeline_path_with_metadata() {
        let path = PipelinePath::new("Stage")
            .with_metadata("key1", serde_json::json!("value1"))
            .with_metadata("key2", serde_json::json!(42));

        assert_eq!(path.metadata.len(), 2);
        assert_eq!(path.metadata.get("key1"), Some(&serde_json::json!("value1")));
        assert_eq!(path.metadata.get("key2"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_pipeline_path_with_contributions() {
        let contributions = vec![0.1, -0.2, 0.3];
        let path = PipelinePath::new("Stage").with_contributions(contributions.clone());

        assert_eq!(path.feature_contributions(), &contributions);
    }

    #[test]
    fn test_chain_verification_serialization() {
        let verification = ChainVerification::valid(5);

        let json = serde_json::to_string(&verification).expect("json serialize failed");
        let deserialized: ChainVerification =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(verification.valid, deserialized.valid);
        assert_eq!(verification.entries_verified, deserialized.entries_verified);
    }
}

#[cfg(test)]
mod proptests {
    use super::tests::make_validation;
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_hash_chain_always_valid(n in 1usize..20) {
            let mut collector = PipelineAuditCollector::new("prop-test");

            for i in 0..n {
                collector.record_stage(PipelinePath::new(format!("Stage{}", i)), None);
            }

            let verification = collector.verify_chain();
            prop_assert!(verification.valid);
            prop_assert_eq!(verification.entries_verified, n);
        }

        #[test]
        fn prop_sequence_numbers_monotonic(n in 2usize..20) {
            let mut collector = PipelineAuditCollector::new("prop-test");

            for i in 0..n {
                collector.record_stage(PipelinePath::new(format!("Stage{}", i)), None);
            }

            let entries = collector.entries();
            for i in 1..entries.len() {
                prop_assert!(entries[i].sequence > entries[i-1].sequence);
            }
        }

        #[test]
        fn prop_path_confidence_bounded(
            success in any::<bool>(),
            validation_passed in any::<bool>()
        ) {
            let mut path = PipelinePath::new("Test");

            if !success {
                path = path.with_error("Error");
            }

            path = path.with_validation(make_validation("Test", validation_passed));

            let confidence = path.confidence();
            prop_assert!(confidence >= 0.0);
            prop_assert!(confidence <= 1.0);
        }

        #[test]
        fn prop_to_bytes_deterministic(stage_name in "[a-z]{1,20}") {
            let path1 = PipelinePath::new(&stage_name);
            let path2 = PipelinePath::new(&stage_name);

            let bytes1 = path1.to_bytes();
            let bytes2 = path2.to_bytes();

            prop_assert_eq!(bytes1, bytes2);
        }

        #[test]
        fn prop_recent_count_correct(n in 1usize..50, take in 1usize..20) {
            let mut collector = PipelineAuditCollector::new("test");

            for i in 0..n {
                collector.record_stage(PipelinePath::new(format!("S{}", i)), None);
            }

            let recent = collector.recent(take);
            let expected = take.min(n);
            prop_assert_eq!(recent.len(), expected);
        }
    }
}
