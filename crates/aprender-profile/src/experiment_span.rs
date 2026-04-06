//! Experiment Span Types (REN-001)
//!
//! Provides span types and metadata structures for ML experiment tracking.
//! Integrates with entrenar experiment tracking (v1.8.0 §5).
//!
//! # Cross-Project Integration
//!
//! This module supports ENT-EPIC-001 by allowing entrenar's Run to create
//! Renacer spans for syscall correlation during ML training.
//!
//! # Example
//!
//! ```
//! use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan, SpanType};
//! use std::collections::HashMap;
//!
//! let mut metrics = HashMap::new();
//! metrics.insert("accuracy".to_string(), 0.95);
//!
//! let metadata = ExperimentMetadata {
//!     model_name: "gpt-2".to_string(),
//!     epoch: Some(10),
//!     step: Some(1000),
//!     loss: Some(0.0025),
//!     metrics,
//! };
//!
//! let span = ExperimentSpan::new_experiment("training_step", metadata);
//! assert_eq!(span.span_type, SpanType::Experiment);
//! ```

use crate::span_record::{SpanKind, SpanRecord, StatusCode};
use crate::unified_trace::UnifiedTrace;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Span type classification for different tracing contexts
///
/// Extends beyond OpenTelemetry `SpanKind` to include domain-specific types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SpanType {
    /// System call span (read, write, etc.)
    #[default]
    Syscall,

    /// GPU operation span (kernel launch, memory transfer)
    Gpu,

    /// ML experiment span (training step, evaluation)
    Experiment,
}

/// Metadata for ML experiment tracking spans
///
/// Contains structured data about an ML training run, epoch, or step.
/// Follows entrenar experiment tracking spec v1.8.0 §5.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExperimentMetadata {
    /// Model name or identifier (e.g., "gpt-2", "bert-base")
    #[serde(default)]
    pub model_name: String,

    /// Current training epoch (None if not applicable)
    #[serde(default)]
    pub epoch: Option<u32>,

    /// Current training step (None if not applicable)
    #[serde(default)]
    pub step: Option<u64>,

    /// Current loss value (None if not computed yet)
    #[serde(default)]
    pub loss: Option<f64>,

    /// Additional metrics (accuracy, `f1_score`, perplexity, etc.)
    #[serde(default)]
    pub metrics: HashMap<String, f64>,
}

impl ExperimentMetadata {
    /// Serialize metadata to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize metadata from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Convert to attribute map for `SpanRecord`
    pub fn to_attributes(&self) -> HashMap<String, String> {
        let mut attrs = HashMap::new();

        attrs.insert("experiment.model_name".to_string(), self.model_name.clone());

        if let Some(epoch) = self.epoch {
            attrs.insert("experiment.epoch".to_string(), epoch.to_string());
        }

        if let Some(step) = self.step {
            attrs.insert("experiment.step".to_string(), step.to_string());
        }

        if let Some(loss) = self.loss {
            attrs.insert("experiment.loss".to_string(), loss.to_string());
        }

        for (key, value) in &self.metrics {
            attrs.insert(format!("experiment.metrics.{key}"), value.to_string());
        }

        attrs
    }
}

/// Experiment span for ML training tracking
///
/// Wraps experiment metadata with span timing and identification.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSpan {
    /// W3C Trace Context trace ID (128-bit)
    pub trace_id: [u8; 16],

    /// W3C Trace Context span ID (64-bit)
    pub span_id: [u8; 8],

    /// Parent span ID (if this is a child span)
    pub parent_span_id: Option<[u8; 8]>,

    /// Human-readable span name
    pub name: String,

    /// Span type classification
    pub span_type: SpanType,

    /// Experiment metadata
    pub metadata: ExperimentMetadata,

    /// Start time in nanoseconds since UNIX epoch
    pub start_time_nanos: u64,

    /// End time in nanoseconds (0 if span not finished)
    pub end_time_nanos: u64,

    /// Lamport logical clock timestamp
    pub logical_clock: u64,
}

impl ExperimentSpan {
    /// Create a new experiment span with the given name and metadata
    ///
    /// Automatically generates `trace_id`, `span_id`, and sets `start_time`.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable span name (e.g., "`training_step`", "evaluation")
    /// * `metadata` - Experiment metadata with model info, epoch, loss, etc.
    ///
    /// # Example
    ///
    /// ```
    /// use renacer::experiment_span::{ExperimentMetadata, ExperimentSpan};
    ///
    /// let metadata = ExperimentMetadata {
    ///     model_name: "bert".to_string(),
    ///     epoch: Some(5),
    ///     ..Default::default()
    /// };
    ///
    /// let span = ExperimentSpan::new_experiment("training", metadata);
    /// assert_eq!(span.name, "training");
    /// ```
    pub fn new_experiment(name: &str, metadata: ExperimentMetadata) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

        ExperimentSpan {
            trace_id: generate_trace_id(),
            span_id: generate_span_id(),
            parent_span_id: None,
            name: name.to_string(),
            span_type: SpanType::Experiment,
            metadata,
            start_time_nanos: now.as_nanos() as u64,
            end_time_nanos: 0,
            logical_clock: 0,
        }
    }

    /// Create experiment span with specific parent context
    pub fn new_experiment_with_parent(
        name: &str,
        metadata: ExperimentMetadata,
        trace_id: [u8; 16],
        parent_span_id: [u8; 8],
    ) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

        ExperimentSpan {
            trace_id,
            span_id: generate_span_id(),
            parent_span_id: Some(parent_span_id),
            name: name.to_string(),
            span_type: SpanType::Experiment,
            metadata,
            start_time_nanos: now.as_nanos() as u64,
            end_time_nanos: 0,
            logical_clock: 0,
        }
    }

    /// End the span and set duration
    pub fn end(&mut self) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        self.end_time_nanos = now.as_nanos() as u64;
    }

    /// Convert to `SpanRecord` for storage in trueno-db
    pub fn to_span_record(&self) -> SpanRecord {
        let attributes = self.metadata.to_attributes();

        let mut resource = HashMap::new();
        resource.insert("service.name".to_string(), "renacer".to_string());
        resource.insert("span.type".to_string(), "experiment".to_string());

        SpanRecord::new(
            self.trace_id,
            self.span_id,
            self.parent_span_id,
            self.name.clone(),
            SpanKind::Internal,
            self.start_time_nanos,
            self.end_time_nanos,
            self.logical_clock,
            StatusCode::Ok,
            String::new(),
            attributes,
            resource,
            std::process::id(),
            0, // thread_id - would need platform-specific code
        )
    }
}

/// Equivalence score for golden trace comparison
///
/// Quantifies how similar two traces are across multiple dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct EquivalenceScore {
    /// Syscall sequence match score (0.0-1.0)
    ///
    /// Measures how well the syscall sequences align.
    /// 1.0 = identical syscalls in same order
    pub syscall_match: f64,

    /// Timing variance between traces (0.0-1.0)
    ///
    /// Measures how much timing differs between traces.
    /// 0.0 = identical timing, 1.0 = completely different timing
    pub timing_variance: f64,

    /// Semantic equivalence score (0.0-1.0)
    ///
    /// Measures whether observable behavior matches.
    /// 1.0 = semantically equivalent
    pub semantic_equiv: f64,
}

impl EquivalenceScore {
    /// Calculate overall equivalence score
    ///
    /// Weighted combination: 40% syscall, 20% timing, 40% semantic
    pub fn overall(&self) -> f64 {
        // Weight: syscall match and semantic equiv are most important
        // Timing variance is inverted (lower is better)
        let timing_score = 1.0 - self.timing_variance;
        0.4 * self.syscall_match + 0.2 * timing_score + 0.4 * self.semantic_equiv
    }

    /// Check if traces are considered equivalent (threshold: 0.85)
    pub fn is_equivalent(&self) -> bool {
        self.overall() >= 0.85
    }
}

/// Compare two traces and compute equivalence score
///
/// Analyzes syscall sequences, timing, and semantic behavior to
/// determine how similar two traces are.
///
/// # Arguments
///
/// * `baseline` - Reference trace (e.g., from original program)
/// * `candidate` - Candidate trace (e.g., from transpiled program)
///
/// # Returns
///
/// `EquivalenceScore` with detailed metrics on trace similarity.
///
/// # Example
///
/// ```
/// use renacer::experiment_span::compare_traces;
/// use renacer::unified_trace::UnifiedTrace;
///
/// let baseline = UnifiedTrace::new(1234, "test".to_string());
/// let candidate = UnifiedTrace::new(1234, "test".to_string());
///
/// let score = compare_traces(&baseline, &candidate);
/// assert!(score.is_equivalent());
/// ```
pub fn compare_traces(baseline: &UnifiedTrace, candidate: &UnifiedTrace) -> EquivalenceScore {
    contract_pre_error_handling!();
    let syscall_match = compute_syscall_match(baseline, candidate);
    let timing_variance = compute_timing_variance(baseline, candidate);
    let semantic_equiv = compute_semantic_equiv(baseline, candidate);

    EquivalenceScore { syscall_match, timing_variance, semantic_equiv }
}

/// Compute syscall sequence match score using LCS algorithm
fn compute_syscall_match(baseline: &UnifiedTrace, candidate: &UnifiedTrace) -> f64 {
    let baseline_syscalls: Vec<&str> =
        baseline.syscall_spans.iter().map(|s| s.name.as_ref()).collect();
    let candidate_syscalls: Vec<&str> =
        candidate.syscall_spans.iter().map(|s| s.name.as_ref()).collect();

    if baseline_syscalls.is_empty() && candidate_syscalls.is_empty() {
        return 1.0;
    }

    if baseline_syscalls.is_empty() || candidate_syscalls.is_empty() {
        return 0.0;
    }

    // Compute Longest Common Subsequence length
    let lcs_len = lcs_length(&baseline_syscalls, &candidate_syscalls);
    let max_len = baseline_syscalls.len().max(candidate_syscalls.len());

    lcs_len as f64 / max_len as f64
}

/// LCS length computation for syscall matching
fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    // Space-optimized LCS using two rows
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = curr[j - 1].max(prev[j]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[n]
}

/// Compute timing variance between traces
fn compute_timing_variance(baseline: &UnifiedTrace, candidate: &UnifiedTrace) -> f64 {
    let baseline_total: u64 = baseline.syscall_spans.iter().map(|s| s.duration_nanos).sum();
    let candidate_total: u64 = candidate.syscall_spans.iter().map(|s| s.duration_nanos).sum();

    if baseline_total == 0 && candidate_total == 0 {
        return 0.0;
    }

    if baseline_total == 0 || candidate_total == 0 {
        return 1.0;
    }

    // Compute relative difference
    let diff = (baseline_total as f64 - candidate_total as f64).abs();
    let max_total = baseline_total.max(candidate_total) as f64;

    (diff / max_total).min(1.0)
}

/// Compute semantic equivalence score
fn compute_semantic_equiv(baseline: &UnifiedTrace, candidate: &UnifiedTrace) -> f64 {
    // Filter to observable syscalls (I/O operations)
    let baseline_obs = filter_observable(baseline);
    let candidate_obs = filter_observable(candidate);

    if baseline_obs.is_empty() && candidate_obs.is_empty() {
        return 1.0;
    }

    if baseline_obs.is_empty() || candidate_obs.is_empty() {
        // One has observable syscalls, other doesn't
        return 0.0;
    }

    // Compare observable syscall sequences
    let matching = baseline_obs
        .iter()
        .zip(candidate_obs.iter())
        .filter(|(b, c)| b.name == c.name && b.return_value.signum() == c.return_value.signum())
        .count();

    let max_len = baseline_obs.len().max(candidate_obs.len());
    matching as f64 / max_len as f64
}

/// Filter to observable syscalls (I/O operations that affect external state)
fn filter_observable(trace: &UnifiedTrace) -> Vec<&crate::unified_trace::SyscallSpan> {
    const OBSERVABLE_SYSCALLS: &[&str] = &[
        "read",
        "write",
        "open",
        "openat",
        "close",
        "stat",
        "fstat",
        "lstat",
        "socket",
        "connect",
        "accept",
        "sendto",
        "recvfrom",
        "send",
        "recv",
        "sendmsg",
        "recvmsg",
        "pipe",
        "pipe2",
        "dup",
        "dup2",
        "dup3",
        "fcntl",
        "ioctl",
        "mkdir",
        "rmdir",
        "unlink",
        "rename",
        "link",
        "symlink",
        "chmod",
        "chown",
        "truncate",
        "ftruncate",
    ];

    trace.syscall_spans.iter().filter(|s| OBSERVABLE_SYSCALLS.contains(&s.name.as_ref())).collect()
}

/// Generate a random trace ID (128-bit)
fn generate_trace_id() -> [u8; 16] {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

    let nanos = now.as_nanos();
    let pid = std::process::id();

    // Simple pseudo-random generation based on time and pid
    let mut id = [0u8; 16];
    let bytes = nanos.to_le_bytes();
    id[0..8].copy_from_slice(&bytes[0..8]);
    id[8..12].copy_from_slice(&pid.to_le_bytes());

    // Add some entropy from thread-local counter
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    id[12..16].copy_from_slice(&counter.to_le_bytes());

    id
}

/// Generate a random span ID (64-bit)
fn generate_span_id() -> [u8; 8] {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

    let nanos = now.as_nanos() as u64;

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // XOR nanos with counter for uniqueness
    let id = nanos ^ (counter << 32);
    id.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_type_variants() {
        assert_eq!(SpanType::default(), SpanType::Syscall);
        let _experiment = SpanType::Experiment;
        let _gpu = SpanType::Gpu;
    }

    #[test]
    fn test_experiment_metadata_default() {
        let meta = ExperimentMetadata::default();
        assert!(meta.model_name.is_empty());
        assert!(meta.epoch.is_none());
        assert!(meta.step.is_none());
        assert!(meta.loss.is_none());
        assert!(meta.metrics.is_empty());
    }

    #[test]
    fn test_experiment_metadata_to_json() {
        let meta = ExperimentMetadata {
            model_name: "test".to_string(),
            epoch: Some(5),
            step: Some(100),
            loss: Some(0.1),
            metrics: HashMap::new(),
        };

        let json = meta.to_json();
        assert!(json.contains("test"));
        assert!(json.contains("5"));
    }

    #[test]
    fn test_experiment_metadata_from_json() {
        let json = r#"{"model_name":"test","epoch":5,"step":100,"loss":0.1,"metrics":{}}"#;
        let meta = ExperimentMetadata::from_json(json).expect("test");
        assert_eq!(meta.model_name, "test");
        assert_eq!(meta.epoch, Some(5));
    }

    #[test]
    fn test_new_experiment() {
        let meta = ExperimentMetadata {
            model_name: "gpt".to_string(),
            epoch: Some(1),
            ..Default::default()
        };

        let span = ExperimentSpan::new_experiment("training", meta);

        assert_eq!(span.name, "training");
        assert_eq!(span.span_type, SpanType::Experiment);
        assert_ne!(span.trace_id, [0u8; 16]);
        assert_ne!(span.span_id, [0u8; 8]);
        assert!(span.start_time_nanos > 0);
    }

    #[test]
    fn test_to_span_record() {
        let meta = ExperimentMetadata {
            model_name: "bert".to_string(),
            epoch: Some(10),
            step: Some(1000),
            loss: Some(0.05),
            metrics: HashMap::new(),
        };

        let span = ExperimentSpan::new_experiment("eval", meta);
        let record = span.to_span_record();

        assert_eq!(record.span_name, "eval");
        let attrs = record.parse_attributes();
        assert_eq!(attrs.get("experiment.model_name"), Some(&"bert".to_string()));
        assert_eq!(attrs.get("experiment.epoch"), Some(&"10".to_string()));
    }

    #[test]
    fn test_equivalence_score_overall() {
        let score =
            EquivalenceScore { syscall_match: 1.0, timing_variance: 0.0, semantic_equiv: 1.0 };

        assert_eq!(score.overall(), 1.0);
        assert!(score.is_equivalent());
    }

    #[test]
    fn test_equivalence_score_not_equivalent() {
        let score =
            EquivalenceScore { syscall_match: 0.3, timing_variance: 0.8, semantic_equiv: 0.3 };

        assert!(!score.is_equivalent());
    }

    #[test]
    fn test_compare_empty_traces() {
        let t1 = UnifiedTrace::new(1, "p1".to_string());
        let t2 = UnifiedTrace::new(1, "p2".to_string());

        let score = compare_traces(&t1, &t2);
        assert_eq!(score.syscall_match, 1.0);
        assert_eq!(score.timing_variance, 0.0);
    }

    #[test]
    fn test_lcs_length() {
        let a = vec!["read", "write", "close"];
        let b = vec!["read", "write", "close"];
        assert_eq!(lcs_length(&a, &b), 3);

        let c = vec!["read", "close"];
        assert_eq!(lcs_length(&a, &c), 2);
    }

    #[test]
    fn test_span_type_serialize_deserialize() {
        let syscall = SpanType::Syscall;
        let json = serde_json::to_string(&syscall).expect("test");
        let deser: SpanType = serde_json::from_str(&json).expect("test");
        assert_eq!(deser, SpanType::Syscall);

        let gpu = SpanType::Gpu;
        let json = serde_json::to_string(&gpu).expect("test");
        let deser: SpanType = serde_json::from_str(&json).expect("test");
        assert_eq!(deser, SpanType::Gpu);

        let experiment = SpanType::Experiment;
        let json = serde_json::to_string(&experiment).expect("test");
        let deser: SpanType = serde_json::from_str(&json).expect("test");
        assert_eq!(deser, SpanType::Experiment);
    }

    #[test]
    fn test_experiment_metadata_to_attributes() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let meta = ExperimentMetadata {
            model_name: "bert".to_string(),
            epoch: Some(10),
            step: Some(100),
            loss: Some(0.05),
            metrics,
        };

        let attrs = meta.to_attributes();
        assert_eq!(attrs.get("experiment.model_name"), Some(&"bert".to_string()));
        assert_eq!(attrs.get("experiment.epoch"), Some(&"10".to_string()));
        assert_eq!(attrs.get("experiment.step"), Some(&"100".to_string()));
        assert_eq!(attrs.get("experiment.loss"), Some(&"0.05".to_string()));
        assert_eq!(attrs.get("experiment.metrics.accuracy"), Some(&"0.95".to_string()));
    }

    #[test]
    fn test_experiment_metadata_to_attributes_partial() {
        let meta = ExperimentMetadata {
            model_name: "model".to_string(),
            epoch: None,
            step: None,
            loss: None,
            metrics: HashMap::new(),
        };

        let attrs = meta.to_attributes();
        assert_eq!(attrs.get("experiment.model_name"), Some(&"model".to_string()));
        assert!(attrs.get("experiment.epoch").is_none());
        assert!(attrs.get("experiment.step").is_none());
        assert!(attrs.get("experiment.loss").is_none());
    }

    #[test]
    fn test_experiment_span_with_parent() {
        let meta = ExperimentMetadata::default();
        let trace_id = [1u8; 16];
        let parent_id = [2u8; 8];

        let span = ExperimentSpan::new_experiment_with_parent("child", meta, trace_id, parent_id);

        assert_eq!(span.trace_id, trace_id);
        assert_eq!(span.parent_span_id, Some(parent_id));
        assert_eq!(span.name, "child");
        assert_eq!(span.span_type, SpanType::Experiment);
    }

    #[test]
    fn test_experiment_span_end() {
        let meta = ExperimentMetadata::default();
        let mut span = ExperimentSpan::new_experiment("test", meta);

        assert_eq!(span.end_time_nanos, 0);
        span.end();
        assert!(span.end_time_nanos > 0);
        assert!(span.end_time_nanos >= span.start_time_nanos);
    }

    #[test]
    fn test_lcs_length_edge_cases() {
        // Empty arrays
        let empty: Vec<&str> = vec![];
        let non_empty = vec!["a"];
        assert_eq!(lcs_length(&empty, &empty), 0);
        assert_eq!(lcs_length(&empty, &non_empty), 0);
        assert_eq!(lcs_length(&non_empty, &empty), 0);

        // No common elements
        let a = vec!["a", "b"];
        let b = vec!["c", "d"];
        assert_eq!(lcs_length(&a, &b), 0);

        // Partial match
        let c = vec!["a", "b", "c"];
        let d = vec!["b", "c", "d"];
        assert_eq!(lcs_length(&c, &d), 2);
    }

    #[test]
    fn test_equivalence_score_partial() {
        let score =
            EquivalenceScore { syscall_match: 0.9, timing_variance: 0.1, semantic_equiv: 0.9 };

        // 0.4*0.9 + 0.2*(1-0.1) + 0.4*0.9 = 0.36 + 0.18 + 0.36 = 0.90
        assert!((score.overall() - 0.90).abs() < f64::EPSILON);
        assert!(score.is_equivalent());
    }

    #[test]
    fn test_equivalence_threshold() {
        // Well above threshold (overall = 0.4*1.0 + 0.2*1.0 + 0.4*1.0 = 1.0)
        let score =
            EquivalenceScore { syscall_match: 1.0, timing_variance: 0.0, semantic_equiv: 1.0 };
        assert!(score.is_equivalent());

        // Well below threshold (overall = 0.4*0.5 + 0.2*0.5 + 0.4*0.5 = 0.5)
        let score2 = EquivalenceScore {
            syscall_match: 0.5,
            timing_variance: 0.5, // timing_score = 1.0 - 0.5 = 0.5
            semantic_equiv: 0.5,
        };
        assert!(!score2.is_equivalent());
    }

    #[test]
    fn test_span_type_clone_eq() {
        let a = SpanType::Syscall;
        let b = a;
        assert_eq!(a, b);

        let c = SpanType::Gpu;
        assert_ne!(a, c);
    }

    #[test]
    fn test_experiment_metadata_clone() {
        let mut metrics = HashMap::new();
        metrics.insert("acc".to_string(), 0.9);

        let meta = ExperimentMetadata {
            model_name: "test".to_string(),
            epoch: Some(5),
            step: Some(100),
            loss: Some(0.1),
            metrics,
        };

        let cloned = meta.clone();
        assert_eq!(cloned.model_name, meta.model_name);
        assert_eq!(cloned.epoch, meta.epoch);
        assert_eq!(cloned.metrics.get("acc"), meta.metrics.get("acc"));
    }

    #[test]
    fn test_experiment_span_clone() {
        let meta = ExperimentMetadata::default();
        let span = ExperimentSpan::new_experiment("test", meta);
        let cloned = span.clone();

        assert_eq!(cloned.name, span.name);
        assert_eq!(cloned.trace_id, span.trace_id);
        assert_eq!(cloned.span_id, span.span_id);
    }

    #[test]
    fn test_equivalence_score_clone() {
        let score =
            EquivalenceScore { syscall_match: 0.8, timing_variance: 0.2, semantic_equiv: 0.7 };
        let cloned = score.clone();
        assert_eq!(cloned.syscall_match, score.syscall_match);
    }

    #[test]
    fn test_generate_trace_id_unique() {
        let id1 = generate_trace_id();
        let id2 = generate_trace_id();
        // Should be different due to counter
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_generate_span_id_unique() {
        let id1 = generate_span_id();
        let id2 = generate_span_id();
        assert_ne!(id1, id2);
    }
}
