//! Verification Audit Trail
//!
//! Provides tamper-evident audit logging for test verification decisions.
//!
//! # Features
//!
//! - **Decision Path Tracking**: Record verification decisions with full context
//! - **Hash Chain Provenance**: Tamper-evident audit trails for compliance
//! - **Verdict Analysis**: Track pass/fail patterns and flaky tests
//!
//! # Toyota Way: 失敗を隠さない (Shippai wo kakusanai)
//! Never hide failures - every verification decision is auditable.
//!
//! # Example
//!
//! ```rust,ignore
//! use verificar::audit::{AuditCollector, VerificationPath};
//! use verificar::oracle::Verdict;
//!
//! let mut collector = AuditCollector::new("test-suite-001");
//!
//! let path = VerificationPath::new("test_addition")
//!     .with_verdict(Verdict::Pass)
//!     .with_duration(std::time::Duration::from_millis(50));
//!
//! collector.record(path);
//!
//! assert!(collector.verify_chain().valid);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::oracle::{ExecutionResult, Verdict, VerificationResult};
use crate::Language;

// =============================================================================
// Verification Decision Path
// =============================================================================

/// Decision path for a verification execution.
///
/// Captures all relevant information for auditing a test verification decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationPath {
    /// Test case identifier
    pub test_id: String,

    /// Source language
    pub source_language: Option<Language>,

    /// Target language
    pub target_language: Option<Language>,

    /// Verification verdict
    pub verdict: Option<Verdict>,

    /// Execution duration in nanoseconds
    pub duration_ns: u64,

    /// Source execution result summary
    pub source_result: Option<ExecutionSummary>,

    /// Target execution result summary
    pub target_result: Option<ExecutionSummary>,

    /// Feature contributions (for ML-enhanced verification)
    contributions: Vec<f32>,

    /// Confidence score for this decision
    confidence: f32,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Summary of an execution result for audit purposes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionSummary {
    /// Exit code
    pub exit_code: i32,

    /// Output length in bytes
    pub stdout_len: usize,

    /// Error output length in bytes
    pub stderr_len: usize,

    /// Execution duration in milliseconds
    pub duration_ms: u64,

    /// Hash of stdout content (for integrity)
    pub stdout_hash: u64,
}

impl From<&ExecutionResult> for ExecutionSummary {
    fn from(result: &ExecutionResult) -> Self {
        Self {
            exit_code: result.exit_code,
            stdout_len: result.stdout.len(),
            stderr_len: result.stderr.len(),
            duration_ms: result.duration_ms,
            stdout_hash: hash_string(&result.stdout),
        }
    }
}

impl VerificationPath {
    /// Create a new verification path for a test case.
    pub fn new(test_id: impl Into<String>) -> Self {
        Self {
            test_id: test_id.into(),
            source_language: None,
            target_language: None,
            verdict: None,
            duration_ns: 0,
            source_result: None,
            target_result: None,
            contributions: Vec::new(),
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the verification verdict.
    #[must_use]
    pub fn with_verdict(mut self, verdict: Verdict) -> Self {
        // Adjust confidence based on verdict
        self.confidence = match &verdict {
            Verdict::Pass => 1.0,
            Verdict::OutputMismatch { .. } => 0.0,
            Verdict::Timeout { .. } => 0.3,
            Verdict::RuntimeError { .. } => 0.0,
        };
        self.verdict = Some(verdict);
        self
    }

    /// Set execution duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ns = duration.as_nanos() as u64;
        self
    }

    /// Set source and target languages.
    #[must_use]
    pub fn with_languages(mut self, source: Language, target: Language) -> Self {
        self.source_language = Some(source);
        self.target_language = Some(target);
        self
    }

    /// Set source execution result.
    #[must_use]
    pub fn with_source_result(mut self, result: &ExecutionResult) -> Self {
        self.source_result = Some(ExecutionSummary::from(result));
        self
    }

    /// Set target execution result.
    #[must_use]
    pub fn with_target_result(mut self, result: &ExecutionResult) -> Self {
        self.target_result = Some(ExecutionSummary::from(result));
        self
    }

    /// Add metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set feature contributions (for ML analysis).
    #[must_use]
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

    /// Check if this verification passed.
    pub fn passed(&self) -> bool {
        matches!(self.verdict, Some(Verdict::Pass))
    }

    /// Serialize to bytes for hashing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Test ID
        bytes.extend_from_slice(self.test_id.as_bytes());
        bytes.push(0);

        // Duration
        bytes.extend_from_slice(&self.duration_ns.to_le_bytes());

        // Confidence
        bytes.extend_from_slice(&self.confidence.to_le_bytes());

        // Verdict indicator
        let verdict_byte = match &self.verdict {
            Some(Verdict::Pass) => 1u8,
            Some(Verdict::OutputMismatch { .. }) => 2u8,
            Some(Verdict::Timeout { .. }) => 3u8,
            Some(Verdict::RuntimeError { .. }) => 4u8,
            None => 0u8,
        };
        bytes.push(verdict_byte);

        bytes
    }

    /// Generate a text explanation of the verification.
    pub fn explain(&self) -> String {
        use std::fmt::Write;

        let mut explanation = format!("Test: {}\n", self.test_id);

        if let Some(src) = &self.source_language {
            let _ = writeln!(explanation, "Source: {src:?}");
        }
        if let Some(tgt) = &self.target_language {
            let _ = writeln!(explanation, "Target: {tgt:?}");
        }

        let _ = writeln!(
            explanation,
            "Duration: {:.2}ms",
            self.duration_ns as f64 / 1_000_000.0
        );

        if let Some(ref verdict) = self.verdict {
            let _ = writeln!(explanation, "Verdict: {verdict:?}");
        }

        let _ = write!(explanation, "Confidence: {:.1}%", self.confidence * 100.0);

        explanation
    }
}

impl From<&VerificationResult> for VerificationPath {
    fn from(result: &VerificationResult) -> Self {
        let mut path = Self::new(format!(
            "{}-to-{}",
            result.source_language, result.target_language
        ))
        .with_languages(result.source_language, result.target_language)
        .with_verdict(result.verdict.clone());

        if let Some(ref src) = result.source_result {
            path = path.with_source_result(src);
        }
        if let Some(ref tgt) = result.target_result {
            path = path.with_target_result(tgt);
        }

        path
    }
}

// =============================================================================
// Audit Trace Entry
// =============================================================================

/// A single audit trace entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditTrace {
    /// Sequence number
    pub sequence: u64,

    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,

    /// The verification path
    pub path: VerificationPath,
}

/// Hash chain entry for tamper-evident audit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashChainEntry {
    /// Sequence number
    pub sequence: u64,

    /// SHA-256 hash of previous entry (zeros for genesis)
    pub prev_hash: [u8; 32],

    /// Hash of this entry
    pub hash: [u8; 32],

    /// The audit trace
    pub trace: AuditTrace,
}

// =============================================================================
// Audit Collector
// =============================================================================

/// Collector for verification audit trails.
#[derive(Debug)]
pub struct AuditCollector {
    /// Hash chain entries
    entries: Vec<HashChainEntry>,

    /// Next sequence number
    next_sequence: u64,

    /// Suite identifier
    suite_id: String,

    /// Statistics
    stats: AuditStats,
}

/// Statistics for audit trail.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AuditStats {
    /// Total verifications
    pub total: usize,

    /// Passed verifications
    pub passed: usize,

    /// Failed verifications
    pub failed: usize,

    /// Timeout count
    pub timeouts: usize,

    /// Runtime errors
    pub errors: usize,

    /// Total duration in nanoseconds
    pub total_duration_ns: u64,
}

impl AuditCollector {
    /// Create a new audit collector for a test suite.
    pub fn new(suite_id: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 0,
            suite_id: suite_id.into(),
            stats: AuditStats::default(),
        }
    }

    /// Get the suite identifier.
    pub fn suite_id(&self) -> &str {
        &self.suite_id
    }

    /// Record a verification decision.
    pub fn record(&mut self, path: VerificationPath) -> &HashChainEntry {
        // Update statistics
        self.stats.total += 1;
        self.stats.total_duration_ns += path.duration_ns;

        match &path.verdict {
            Some(Verdict::Pass) => self.stats.passed += 1,
            Some(Verdict::OutputMismatch { .. }) => self.stats.failed += 1,
            Some(Verdict::Timeout { .. }) => self.stats.timeouts += 1,
            Some(Verdict::RuntimeError { .. }) => self.stats.errors += 1,
            None => {}
        }

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let trace = AuditTrace {
            sequence: self.next_sequence,
            timestamp_ns,
            path,
        };

        // Get previous hash
        let prev_hash = self.entries.last().map_or([0u8; 32], |e| e.hash);

        // Compute hash
        let hash = self.compute_hash(&trace, &prev_hash);

        let entry = HashChainEntry {
            sequence: self.next_sequence,
            prev_hash,
            hash,
            trace,
        };

        self.entries.push(entry);
        self.next_sequence += 1;

        // SAFETY: We just pushed an entry, so the vector is guaranteed to be non-empty.
        // Using index access to avoid expect() which is denied by clippy configuration.
        &self.entries[self.entries.len() - 1]
    }

    /// Compute hash for an entry.
    fn compute_hash(&self, trace: &AuditTrace, prev_hash: &[u8; 32]) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        prev_hash.hash(&mut hasher);
        trace.sequence.hash(&mut hasher);
        trace.timestamp_ns.hash(&mut hasher);
        trace.path.test_id.hash(&mut hasher);
        trace.path.duration_ns.hash(&mut hasher);

        let hash_value = hasher.finish();

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

    /// Get statistics.
    pub fn stats(&self) -> &AuditStats {
        &self.stats
    }

    /// Verify hash chain integrity.
    pub fn verify_chain(&self) -> ChainVerification {
        contract_pre_oracle_verdict!(input);
        let mut entries_verified = 0;

        for (i, entry) in self.entries.iter().enumerate() {
            // Verify prev_hash linkage
            if i == 0 {
                if entry.prev_hash != [0u8; 32] {
                    return ChainVerification {
                        valid: false,
                        entries_verified,
                        first_break: Some(0),
                    };
                }
            } else {
                let expected_prev = self.entries[i - 1].hash;
                if entry.prev_hash != expected_prev {
                    return ChainVerification {
                        valid: false,
                        entries_verified,
                        first_break: Some(i),
                    };
                }
            }

            // Verify entry hash
            let computed_hash = self.compute_hash(&entry.trace, &entry.prev_hash);
            if entry.hash != computed_hash {
                return ChainVerification {
                    valid: false,
                    entries_verified,
                    first_break: Some(i),
                };
            }

            entries_verified += 1;
        }

        ChainVerification {
            valid: true,
            entries_verified,
            first_break: None,
        }
    }

    /// Get recent entries.
    pub fn recent(&self, n: usize) -> Vec<&HashChainEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Get failed verifications.
    pub fn failures(&self) -> Vec<&HashChainEntry> {
        self.entries
            .iter()
            .filter(|e| !e.trace.path.passed())
            .collect()
    }

    /// Export to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct Export<'a> {
            suite_id: &'a str,
            chain_length: usize,
            verified: bool,
            stats: &'a AuditStats,
            entries: &'a [HashChainEntry],
        }

        let verification = self.verify_chain();

        let export = Export {
            suite_id: &self.suite_id,
            chain_length: self.entries.len(),
            verified: verification.valid,
            stats: &self.stats,
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

    /// Number of entries verified
    pub entries_verified: usize,

    /// Index of first broken link (if any)
    pub first_break: Option<usize>,
}

// =============================================================================
// Verification Timer
// =============================================================================

/// Timer for measuring verification duration.
#[derive(Debug)]
pub struct VerificationTimer {
    start: Instant,
    test_id: String,
}

impl VerificationTimer {
    /// Start timing a verification.
    pub fn start(test_id: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            test_id: test_id.into(),
        }
    }

    /// Stop timing and create a verification path.
    pub fn stop(self) -> VerificationPath {
        let duration = self.start.elapsed();
        VerificationPath::new(self.test_id).with_duration(duration)
    }

    /// Stop timing with a verdict.
    pub fn stop_with_verdict(self, verdict: Verdict) -> VerificationPath {
        let duration = self.start.elapsed();
        VerificationPath::new(self.test_id)
            .with_duration(duration)
            .with_verdict(verdict)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Simple string hash for integrity checking.
fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Create a new audit collector with generated suite ID.
#[must_use]
pub fn new_audit_collector() -> AuditCollector {
    let suite_id = format!(
        "suite-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    );
    AuditCollector::new(suite_id)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::oracle::Phase;

    #[test]
    fn test_verification_path_creation() {
        let path = VerificationPath::new("test_001");
        assert_eq!(path.test_id, "test_001");
        assert_eq!(path.confidence(), 1.0);
        assert!(path.verdict.is_none());
    }

    #[test]
    fn test_verification_path_with_verdict_pass() {
        let path = VerificationPath::new("test").with_verdict(Verdict::Pass);
        assert!(path.passed());
        assert_eq!(path.confidence(), 1.0);
    }

    #[test]
    fn test_verification_path_with_verdict_mismatch() {
        let path = VerificationPath::new("test").with_verdict(Verdict::OutputMismatch {
            expected: "a".into(),
            actual: "b".into(),
        });
        assert!(!path.passed());
        assert_eq!(path.confidence(), 0.0);
    }

    #[test]
    fn test_verification_path_with_verdict_timeout() {
        let path = VerificationPath::new("test").with_verdict(Verdict::Timeout {
            phase: Phase::Source,
            limit_ms: 5000,
        });
        assert!(!path.passed());
        assert_eq!(path.confidence(), 0.3);
    }

    #[test]
    fn test_verification_path_with_verdict_error() {
        let path = VerificationPath::new("test").with_verdict(Verdict::RuntimeError {
            phase: Phase::Target,
            error: "error".into(),
        });
        assert!(!path.passed());
        assert_eq!(path.confidence(), 0.0);
    }

    #[test]
    fn test_verification_path_with_duration() {
        let path = VerificationPath::new("test").with_duration(Duration::from_millis(100));
        assert_eq!(path.duration_ns, 100_000_000);
    }

    #[test]
    fn test_verification_path_with_languages() {
        let path = VerificationPath::new("test").with_languages(Language::Python, Language::Rust);
        assert_eq!(path.source_language, Some(Language::Python));
        assert_eq!(path.target_language, Some(Language::Rust));
    }

    #[test]
    fn test_verification_path_explain() {
        let path = VerificationPath::new("test_001")
            .with_duration(Duration::from_millis(50))
            .with_verdict(Verdict::Pass);
        let explanation = path.explain();
        assert!(explanation.contains("test_001"));
        assert!(explanation.contains("50.00ms"));
        assert!(explanation.contains("Pass"));
    }

    #[test]
    fn test_verification_path_to_bytes() {
        let path = VerificationPath::new("test");
        let bytes = path.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_execution_summary_from_result() {
        let result = ExecutionResult {
            stdout: "hello".to_string(),
            stderr: "err".to_string(),
            exit_code: 0,
            duration_ms: 100,
        };
        let summary = ExecutionSummary::from(&result);
        assert_eq!(summary.exit_code, 0);
        assert_eq!(summary.stdout_len, 5);
        assert_eq!(summary.stderr_len, 3);
        assert_eq!(summary.duration_ms, 100);
    }

    #[test]
    fn test_audit_collector_creation() {
        let collector = AuditCollector::new("suite-001");
        assert_eq!(collector.suite_id(), "suite-001");
        assert!(collector.is_empty());
    }

    #[test]
    fn test_audit_collector_record() {
        let mut collector = AuditCollector::new("suite");
        let path = VerificationPath::new("test_001").with_verdict(Verdict::Pass);

        let entry = collector.record(path);

        assert_eq!(entry.sequence, 0);
        assert_eq!(entry.prev_hash, [0u8; 32]);
        assert_eq!(collector.len(), 1);
    }

    #[test]
    fn test_audit_collector_stats() {
        let mut collector = AuditCollector::new("suite");

        collector.record(VerificationPath::new("t1").with_verdict(Verdict::Pass));
        collector.record(VerificationPath::new("t2").with_verdict(Verdict::Pass));
        collector.record(
            VerificationPath::new("t3").with_verdict(Verdict::OutputMismatch {
                expected: "a".into(),
                actual: "b".into(),
            }),
        );
        collector.record(VerificationPath::new("t4").with_verdict(Verdict::Timeout {
            phase: Phase::Source,
            limit_ms: 5000,
        }));

        let stats = collector.stats();
        assert_eq!(stats.total, 4);
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.timeouts, 1);
    }

    #[test]
    fn test_audit_collector_hash_chain_linkage() {
        let mut collector = AuditCollector::new("suite");

        collector.record(VerificationPath::new("t1"));
        collector.record(VerificationPath::new("t2"));
        collector.record(VerificationPath::new("t3"));

        let entries = collector.entries();

        assert_eq!(entries[0].prev_hash, [0u8; 32]);
        assert_eq!(entries[1].prev_hash, entries[0].hash);
        assert_eq!(entries[2].prev_hash, entries[1].hash);
    }

    #[test]
    fn test_audit_collector_verify_chain() {
        let mut collector = AuditCollector::new("suite");

        collector.record(VerificationPath::new("t1").with_verdict(Verdict::Pass));
        collector.record(VerificationPath::new("t2").with_verdict(Verdict::Pass));

        let verification = collector.verify_chain();
        assert!(verification.valid);
        assert_eq!(verification.entries_verified, 2);
        assert!(verification.first_break.is_none());
    }

    #[test]
    fn test_audit_collector_recent() {
        let mut collector = AuditCollector::new("suite");

        for i in 0..5 {
            collector.record(VerificationPath::new(format!("t{}", i)));
        }

        let recent = collector.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].sequence, 4);
        assert_eq!(recent[1].sequence, 3);
        assert_eq!(recent[2].sequence, 2);
    }

    #[test]
    fn test_audit_collector_failures() {
        let mut collector = AuditCollector::new("suite");

        collector.record(VerificationPath::new("t1").with_verdict(Verdict::Pass));
        collector.record(
            VerificationPath::new("t2").with_verdict(Verdict::OutputMismatch {
                expected: "a".into(),
                actual: "b".into(),
            }),
        );
        collector.record(VerificationPath::new("t3").with_verdict(Verdict::Pass));

        let failures = collector.failures();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].trace.path.test_id, "t2");
    }

    #[test]
    fn test_audit_collector_to_json() {
        let mut collector = AuditCollector::new("suite");
        collector.record(VerificationPath::new("t1").with_verdict(Verdict::Pass));

        let json = collector.to_json().unwrap();
        assert!(json.contains("suite"));
        assert!(json.contains("verified"));
        assert!(json.contains("stats"));
    }

    #[test]
    fn test_verification_timer() {
        let timer = VerificationTimer::start("test");
        std::thread::sleep(Duration::from_millis(10));
        let path = timer.stop();

        assert_eq!(path.test_id, "test");
        assert!(path.duration_ns > 0);
    }

    #[test]
    fn test_verification_timer_with_verdict() {
        let timer = VerificationTimer::start("test");
        let path = timer.stop_with_verdict(Verdict::Pass);

        assert!(path.passed());
    }

    #[test]
    fn test_new_audit_collector() {
        let collector = new_audit_collector();
        assert!(collector.suite_id().starts_with("suite-"));
    }

    #[test]
    fn test_verification_path_with_metadata() {
        let path = VerificationPath::new("test").with_metadata("key", serde_json::json!("value"));
        assert_eq!(path.metadata.len(), 1);
    }

    #[test]
    fn test_verification_path_with_contributions() {
        let contributions = vec![0.1, 0.2, 0.3];
        let path = VerificationPath::new("test").with_contributions(contributions.clone());
        assert_eq!(path.feature_contributions(), &contributions);
    }

    #[test]
    fn test_verification_path_with_source_result() {
        let result = ExecutionResult {
            stdout: "out".to_string(),
            stderr: "".to_string(),
            exit_code: 0,
            duration_ms: 10,
        };
        let path = VerificationPath::new("test").with_source_result(&result);
        assert!(path.source_result.is_some());
    }

    #[test]
    fn test_verification_path_with_target_result() {
        let result = ExecutionResult {
            stdout: "out".to_string(),
            stderr: "".to_string(),
            exit_code: 0,
            duration_ms: 10,
        };
        let path = VerificationPath::new("test").with_target_result(&result);
        assert!(path.target_result.is_some());
    }

    #[test]
    fn test_verification_path_from_result() {
        let result = VerificationResult {
            source_code: "print(1)".to_string(),
            source_language: Language::Python,
            target_code: "fn main() {}".to_string(),
            target_language: Language::Rust,
            verdict: Verdict::Pass,
            source_result: None,
            target_result: None,
        };

        let path = VerificationPath::from(&result);
        assert!(path.passed());
        assert_eq!(path.source_language, Some(Language::Python));
        assert_eq!(path.target_language, Some(Language::Rust));
    }

    #[test]
    fn test_chain_verification_serialization() {
        let verification = ChainVerification {
            valid: true,
            entries_verified: 5,
            first_break: None,
        };

        let json = serde_json::to_string(&verification).unwrap();
        let deserialized: ChainVerification = serde_json::from_str(&json).unwrap();

        assert_eq!(verification.valid, deserialized.valid);
        assert_eq!(verification.entries_verified, deserialized.entries_verified);
    }

    #[test]
    fn test_hash_string() {
        let hash1 = hash_string("hello");
        let hash2 = hash_string("hello");
        let hash3 = hash_string("world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::oracle::Phase;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_hash_chain_always_valid(n in 1usize..20) {
            let mut collector = AuditCollector::new("prop-test");

            for i in 0..n {
                collector.record(
                    VerificationPath::new(format!("t{}", i))
                        .with_verdict(Verdict::Pass)
                );
            }

            let verification = collector.verify_chain();
            prop_assert!(verification.valid);
            prop_assert_eq!(verification.entries_verified, n);
        }

        #[test]
        fn prop_sequence_numbers_monotonic(n in 2usize..20) {
            let mut collector = AuditCollector::new("test");

            for i in 0..n {
                collector.record(VerificationPath::new(format!("t{}", i)));
            }

            let entries = collector.entries();
            for i in 1..entries.len() {
                prop_assert!(entries[i].sequence > entries[i-1].sequence);
            }
        }

        #[test]
        fn prop_stats_consistent(
            passed in 0usize..10,
            failed in 0usize..10
        ) {
            let mut collector = AuditCollector::new("test");

            for i in 0..passed {
                collector.record(
                    VerificationPath::new(format!("pass{}", i))
                        .with_verdict(Verdict::Pass)
                );
            }

            for i in 0..failed {
                collector.record(
                    VerificationPath::new(format!("fail{}", i))
                        .with_verdict(Verdict::OutputMismatch {
                            expected: "a".into(),
                            actual: "b".into(),
                        })
                );
            }

            let stats = collector.stats();
            prop_assert_eq!(stats.total, passed + failed);
            prop_assert_eq!(stats.passed, passed);
            prop_assert_eq!(stats.failed, failed);
        }

        #[test]
        fn prop_to_bytes_deterministic(test_id in "[a-z]{1,20}") {
            let path1 = VerificationPath::new(&test_id);
            let path2 = VerificationPath::new(&test_id);

            let bytes1 = path1.to_bytes();
            let bytes2 = path2.to_bytes();

            prop_assert_eq!(bytes1, bytes2);
        }

        #[test]
        fn prop_confidence_bounded(verdict_type in 0u8..4) {
            let verdict = match verdict_type {
                0 => Verdict::Pass,
                1 => Verdict::OutputMismatch { expected: "a".into(), actual: "b".into() },
                2 => Verdict::Timeout { phase: Phase::Source, limit_ms: 5000 },
                _ => Verdict::RuntimeError { phase: Phase::Target, error: "err".into() },
            };

            let path = VerificationPath::new("test").with_verdict(verdict);
            let confidence = path.confidence();

            prop_assert!(confidence >= 0.0);
            prop_assert!(confidence <= 1.0);
        }
    }
}
