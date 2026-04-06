//! Trace Storage Policy (ENT-015)
//!
//! Provides configurable trace storage with compression and retention policies
//! for managing experiment traces efficiently.

use serde::{Deserialize, Serialize};

/// Number of days in a standard year, used for archival retention policy.
const DAYS_PER_YEAR: u32 = 365;

/// Compression algorithm for trace data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CompressionAlgorithm {
    /// No compression
    #[default]
    None,
    /// Run-length encoding (good for sparse data)
    Rle,
    /// Zstandard compression (good balance of speed/ratio)
    Zstd,
    /// LZ4 compression (fastest)
    Lz4,
}

impl CompressionAlgorithm {
    /// Returns the typical compression ratio for this algorithm
    ///
    /// Values are approximate and depend on data characteristics.
    pub fn typical_ratio(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Rle => 2.0,  // Varies widely based on data
            Self::Zstd => 4.0, // Good general compression
            Self::Lz4 => 2.5,  // Fast but lower ratio
        }
    }

    /// Returns the relative speed of this algorithm (higher = faster)
    ///
    /// Scale: 1.0 = baseline (Zstd), higher = faster
    pub fn relative_speed(&self) -> f64 {
        match self {
            Self::None => 10.0, // No processing
            Self::Rle => 5.0,   // Simple algorithm
            Self::Zstd => 1.0,  // Baseline
            Self::Lz4 => 3.0,   // Optimized for speed
        }
    }

    /// Parse compression algorithm from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "off" | "disabled" => Some(Self::None),
            "rle" | "runlength" | "run-length" => Some(Self::Rle),
            "zstd" | "zstandard" => Some(Self::Zstd),
            "lz4" => Some(Self::Lz4),
            _ => None,
        }
    }
}

impl std::fmt::Display for CompressionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Rle => write!(f, "rle"),
            Self::Zstd => write!(f, "zstd"),
            Self::Lz4 => write!(f, "lz4"),
        }
    }
}

/// Trace storage configuration policy
///
/// Controls how experiment traces are stored, compressed, and retained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceStoragePolicy {
    /// Compression algorithm to use for trace data
    pub compression: CompressionAlgorithm,

    /// Number of days to retain trace data (0 = indefinite)
    pub retention_days: u32,

    /// Maximum storage size in bytes (0 = unlimited)
    pub max_size_bytes: u64,

    /// Sampling rate for trace collection (0.0-1.0)
    /// 1.0 = collect all traces, 0.5 = collect 50%, etc.
    pub sample_rate: f64,
}

impl Default for TraceStoragePolicy {
    fn default() -> Self {
        Self {
            compression: CompressionAlgorithm::Zstd,
            retention_days: 30,
            max_size_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            sample_rate: 1.0,
        }
    }
}

impl TraceStoragePolicy {
    /// Create a new trace storage policy
    pub fn new(
        compression: CompressionAlgorithm,
        retention_days: u32,
        max_size_bytes: u64,
        sample_rate: f64,
    ) -> Self {
        Self {
            compression,
            retention_days,
            max_size_bytes,
            sample_rate: sample_rate.clamp(0.0, 1.0),
        }
    }

    /// Create a minimal policy (no compression, short retention)
    pub fn minimal() -> Self {
        Self {
            compression: CompressionAlgorithm::None,
            retention_days: 7,
            max_size_bytes: 1024 * 1024 * 1024, // 1 GB
            sample_rate: 0.1,
        }
    }

    /// Create a policy optimized for development (high detail, short retention)
    pub fn development() -> Self {
        Self {
            compression: CompressionAlgorithm::Lz4,
            retention_days: 7,
            max_size_bytes: 5 * 1024 * 1024 * 1024, // 5 GB
            sample_rate: 1.0,
        }
    }

    /// Create a policy optimized for production (balanced)
    pub fn production() -> Self {
        Self {
            compression: CompressionAlgorithm::Zstd,
            retention_days: 90,
            max_size_bytes: 50 * 1024 * 1024 * 1024, // 50 GB
            sample_rate: 0.5,
        }
    }

    /// Create a policy for archival (high compression, long retention)
    pub fn archival() -> Self {
        Self {
            compression: CompressionAlgorithm::Zstd,
            retention_days: DAYS_PER_YEAR,
            max_size_bytes: 100 * 1024 * 1024 * 1024, // 100 GB
            sample_rate: 0.25,
        }
    }

    /// Check if a trace should be sampled based on the sample rate
    ///
    /// Uses a deterministic hash of the trace ID for consistent sampling.
    pub fn should_sample(&self, trace_id: &str) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        if self.sample_rate <= 0.0 {
            return false;
        }

        // Simple deterministic hash for consistent sampling
        let hash =
            trace_id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(u64::from(b)));

        let normalized = (hash % 10000) as f64 / 10000.0;
        normalized < self.sample_rate
    }

    /// Estimate the compressed size of data given uncompressed size
    pub fn estimate_compressed_size(&self, uncompressed_bytes: u64) -> u64 {
        let ratio = self.compression.typical_ratio();
        (uncompressed_bytes as f64 / ratio).ceil() as u64
    }

    /// Check if adding data would exceed the storage limit
    pub fn would_exceed_limit(&self, current_bytes: u64, additional_bytes: u64) -> bool {
        if self.max_size_bytes == 0 {
            return false; // Unlimited
        }

        let estimated_additional = self.estimate_compressed_size(additional_bytes);
        current_bytes.saturating_add(estimated_additional) > self.max_size_bytes
    }

    /// Check if retention is indefinite
    pub fn is_indefinite_retention(&self) -> bool {
        self.retention_days == 0
    }

    /// Check if storage is unlimited
    pub fn is_unlimited_storage(&self) -> bool {
        self.max_size_bytes == 0
    }

    /// Get the effective sample percentage (0-100)
    pub fn sample_percentage(&self) -> f64 {
        self.sample_rate * 100.0
    }

    /// Validate the policy configuration
    pub fn validate(&self) -> Result<(), PolicyValidationError> {
        if self.sample_rate < 0.0 || self.sample_rate > 1.0 {
            return Err(PolicyValidationError::InvalidSampleRate(self.sample_rate));
        }

        // Warn about potentially problematic configurations
        if self.sample_rate < 0.01 && self.retention_days > 30 {
            // Very low sampling with long retention might indicate misconfiguration
        }

        Ok(())
    }
}

/// Errors from policy validation
#[derive(Debug, Clone, PartialEq)]
pub enum PolicyValidationError {
    /// Sample rate must be between 0.0 and 1.0
    InvalidSampleRate(f64),
}

impl std::fmt::Display for PolicyValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSampleRate(rate) => {
                write!(f, "Invalid sample rate {rate}: must be between 0.0 and 1.0")
            }
        }
    }
}

impl std::error::Error for PolicyValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_algorithm_default() {
        let algo = CompressionAlgorithm::default();
        assert_eq!(algo, CompressionAlgorithm::None);
    }

    #[test]
    fn test_compression_algorithm_typical_ratio() {
        assert!((CompressionAlgorithm::None.typical_ratio() - 1.0).abs() < f64::EPSILON);
        assert!(
            CompressionAlgorithm::Zstd.typical_ratio() > CompressionAlgorithm::Lz4.typical_ratio()
        );
    }

    #[test]
    fn test_compression_algorithm_relative_speed() {
        assert!(
            CompressionAlgorithm::None.relative_speed()
                > CompressionAlgorithm::Lz4.relative_speed()
        );
        assert!(
            CompressionAlgorithm::Lz4.relative_speed()
                > CompressionAlgorithm::Zstd.relative_speed()
        );
    }

    #[test]
    fn test_compression_algorithm_parse() {
        assert_eq!(CompressionAlgorithm::parse("none"), Some(CompressionAlgorithm::None));
        assert_eq!(CompressionAlgorithm::parse("off"), Some(CompressionAlgorithm::None));
        assert_eq!(CompressionAlgorithm::parse("rle"), Some(CompressionAlgorithm::Rle));
        assert_eq!(CompressionAlgorithm::parse("zstd"), Some(CompressionAlgorithm::Zstd));
        assert_eq!(CompressionAlgorithm::parse("ZSTANDARD"), Some(CompressionAlgorithm::Zstd));
        assert_eq!(CompressionAlgorithm::parse("lz4"), Some(CompressionAlgorithm::Lz4));
        assert_eq!(CompressionAlgorithm::parse("invalid"), None);
    }

    #[test]
    fn test_compression_algorithm_display() {
        assert_eq!(format!("{}", CompressionAlgorithm::None), "none");
        assert_eq!(format!("{}", CompressionAlgorithm::Rle), "rle");
        assert_eq!(format!("{}", CompressionAlgorithm::Zstd), "zstd");
        assert_eq!(format!("{}", CompressionAlgorithm::Lz4), "lz4");
    }

    #[test]
    fn test_trace_storage_policy_default() {
        let policy = TraceStoragePolicy::default();

        assert_eq!(policy.compression, CompressionAlgorithm::Zstd);
        assert_eq!(policy.retention_days, 30);
        assert_eq!(policy.max_size_bytes, 10 * 1024 * 1024 * 1024);
        assert!((policy.sample_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_storage_policy_new() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::Lz4, 14, 1024 * 1024, 0.75);

        assert_eq!(policy.compression, CompressionAlgorithm::Lz4);
        assert_eq!(policy.retention_days, 14);
        assert_eq!(policy.max_size_bytes, 1024 * 1024);
        assert!((policy.sample_rate - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_storage_policy_new_clamps_sample_rate() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 1.5);
        assert!((policy.sample_rate - 1.0).abs() < f64::EPSILON);

        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, -0.5);
        assert!((policy.sample_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trace_storage_policy_presets() {
        let minimal = TraceStoragePolicy::minimal();
        assert_eq!(minimal.compression, CompressionAlgorithm::None);
        assert!(minimal.sample_rate < 0.5);

        let dev = TraceStoragePolicy::development();
        assert_eq!(dev.compression, CompressionAlgorithm::Lz4);
        assert!((dev.sample_rate - 1.0).abs() < f64::EPSILON);

        let prod = TraceStoragePolicy::production();
        assert_eq!(prod.compression, CompressionAlgorithm::Zstd);
        assert!(prod.retention_days > dev.retention_days);

        let archive = TraceStoragePolicy::archival();
        assert!(archive.retention_days > prod.retention_days);
    }

    #[test]
    fn test_should_sample_always() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 1.0);

        // Should always sample at rate 1.0
        assert!(policy.should_sample("trace-001"));
        assert!(policy.should_sample("trace-002"));
        assert!(policy.should_sample("any-trace-id"));
    }

    #[test]
    fn test_should_sample_never() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 0.0);

        // Should never sample at rate 0.0
        assert!(!policy.should_sample("trace-001"));
        assert!(!policy.should_sample("trace-002"));
    }

    #[test]
    fn test_should_sample_deterministic() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 0.5);

        // Same trace ID should always get same result
        let result1 = policy.should_sample("trace-001");
        let result2 = policy.should_sample("trace-001");
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_should_sample_distribution() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 0.5);

        // With enough samples, should be roughly 50%
        let sampled: usize =
            (0..1000).filter(|i| policy.should_sample(&format!("trace-{i}"))).count();

        // Allow 10% tolerance
        assert!(sampled > 400 && sampled < 600, "Expected ~500 samples, got {sampled}");
    }

    #[test]
    fn test_estimate_compressed_size() {
        let policy_none = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 1.0);
        let policy_zstd = TraceStoragePolicy::new(CompressionAlgorithm::Zstd, 7, 1024, 1.0);

        let size = 1000u64;

        // No compression = same size
        assert_eq!(policy_none.estimate_compressed_size(size), 1000);

        // Zstd should estimate smaller
        assert!(policy_zstd.estimate_compressed_size(size) < size);
    }

    #[test]
    fn test_would_exceed_limit() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1000, 1.0);

        // Not exceeding
        assert!(!policy.would_exceed_limit(0, 500));
        assert!(!policy.would_exceed_limit(500, 500));

        // Exceeding
        assert!(policy.would_exceed_limit(500, 600));
        assert!(policy.would_exceed_limit(1000, 1));
    }

    #[test]
    fn test_would_exceed_limit_unlimited() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 0, 1.0);

        // Should never exceed with unlimited storage
        assert!(!policy.would_exceed_limit(u64::MAX - 1, 1));
    }

    #[test]
    fn test_is_indefinite_retention() {
        let indefinite = TraceStoragePolicy::new(CompressionAlgorithm::None, 0, 1024, 1.0);
        let limited = TraceStoragePolicy::new(CompressionAlgorithm::None, 30, 1024, 1.0);

        assert!(indefinite.is_indefinite_retention());
        assert!(!limited.is_indefinite_retention());
    }

    #[test]
    fn test_is_unlimited_storage() {
        let unlimited = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 0, 1.0);
        let limited = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 1.0);

        assert!(unlimited.is_unlimited_storage());
        assert!(!limited.is_unlimited_storage());
    }

    #[test]
    fn test_sample_percentage() {
        let policy = TraceStoragePolicy::new(CompressionAlgorithm::None, 7, 1024, 0.75);
        assert!((policy.sample_percentage() - 75.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_valid_policy() {
        let policy = TraceStoragePolicy::default();
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_sample_rate() {
        let mut policy = TraceStoragePolicy::default();
        policy.sample_rate = 1.5; // Invalid (bypassing new's clamp)
        assert!(matches!(policy.validate(), Err(PolicyValidationError::InvalidSampleRate(_))));
    }

    #[test]
    fn test_policy_validation_error_display() {
        let err = PolicyValidationError::InvalidSampleRate(1.5);
        let msg = format!("{err}");
        assert!(msg.contains("1.5"));
        assert!(msg.contains("0.0"));
        assert!(msg.contains("1.0"));
    }

    #[test]
    fn test_trace_storage_policy_serialization() {
        let policy = TraceStoragePolicy::production();
        let json = serde_json::to_string(&policy).expect("JSON serialization should succeed");
        let parsed: TraceStoragePolicy =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(parsed.compression, policy.compression);
        assert_eq!(parsed.retention_days, policy.retention_days);
        assert_eq!(parsed.max_size_bytes, policy.max_size_bytes);
        assert!((parsed.sample_rate - policy.sample_rate).abs() < f64::EPSILON);
    }
}
