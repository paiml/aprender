//! Core types for incremental snapshots: errors, retention tiers, and metric data.

use std::time::Duration;

/// Result type for snapshot operations
pub type SnapshotResult<T> = Result<T, SnapshotError>;

/// Errors in snapshot operations
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    /// IO error
    IoError { reason: String },
    /// Snapshot not found
    NotFound { index: usize },
    /// Corrupt snapshot
    Corrupt { reason: String },
    /// Checksum mismatch
    ChecksumMismatch { expected: u32, actual: u32 },
    /// Index out of bounds
    IndexOutOfBounds { index: usize, max: usize },
    /// Memory limit exceeded
    MemoryExceeded { limit_bytes: usize },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError { reason } => write!(f, "IO error: {}", reason),
            Self::NotFound { index } => write!(f, "Snapshot {} not found", index),
            Self::Corrupt { reason } => write!(f, "Corrupt snapshot: {}", reason),
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::IndexOutOfBounds { index, max } => {
                write!(f, "Index {} out of bounds (max {})", index, max)
            }
            Self::MemoryExceeded { limit_bytes } => {
                write!(f, "Memory limit {} exceeded", limit_bytes)
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Retention tier for snapshots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionTier {
    /// Raw snapshots (full data)
    Raw,
    /// Compressed snapshots (delta encoded)
    Compressed,
    /// Archived snapshots (highly compressed, read-only)
    Archive,
}

impl RetentionTier {
    /// Get maximum age for this tier
    pub fn max_age(&self) -> Duration {
        match self {
            Self::Raw => Duration::from_secs(24 * 60 * 60), // 1 day
            Self::Compressed => Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            Self::Archive => Duration::from_secs(365 * 24 * 60 * 60), // 1 year
        }
    }
}

/// Profile metric data
#[derive(Debug, Clone)]
pub struct MetricData {
    /// Metric name
    pub name: String,
    /// Metric values (time series)
    pub values: Vec<f64>,
    /// Timestamps (nanoseconds since epoch)
    pub timestamps: Vec<u64>,
}

impl MetricData {
    /// Create new metric data
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add a sample
    pub fn add(&mut self, value: f64, timestamp: u64) {
        self.values.push(value);
        self.timestamps.push(timestamp);
    }

    /// Get serialized size estimate
    pub fn size_bytes(&self) -> usize {
        self.name.len() + self.values.len() * 8 + self.timestamps.len() * 8
    }

    /// Compute delta from another metric
    pub fn delta_from(&self, other: &MetricData) -> DeltaMetric {
        let mut changed_indices = Vec::new();
        let mut changed_values = Vec::new();

        // Find changed values
        let max_len = self.values.len().max(other.values.len());
        for i in 0..max_len {
            let self_val = self.values.get(i).copied().unwrap_or(0.0);
            let other_val = other.values.get(i).copied().unwrap_or(0.0);

            if (self_val - other_val).abs() > 1e-10 {
                changed_indices.push(i);
                changed_values.push(self_val);
            }
        }

        DeltaMetric {
            name: self.name.clone(),
            base_len: other.values.len(),
            new_len: self.values.len(),
            changed_indices,
            changed_values,
        }
    }

    /// Apply delta to reconstruct
    pub fn apply_delta(&self, delta: &DeltaMetric) -> MetricData {
        let mut result = MetricData::new(&delta.name);

        // Start with base values
        result.values = self.values.clone();
        result.timestamps = self.timestamps.clone();

        // Resize if needed
        result.values.resize(delta.new_len, 0.0);
        result.timestamps.resize(delta.new_len, 0);

        // Apply changes
        for (i, &idx) in delta.changed_indices.iter().enumerate() {
            if idx < result.values.len() {
                result.values[idx] = delta.changed_values[i];
            }
        }

        result
    }
}

/// Delta-encoded metric
#[derive(Debug, Clone)]
pub struct DeltaMetric {
    /// Metric name
    pub name: String,
    /// Base array length
    pub base_len: usize,
    /// New array length
    pub new_len: usize,
    /// Indices of changed values
    pub changed_indices: Vec<usize>,
    /// Changed values
    pub changed_values: Vec<f64>,
}

impl DeltaMetric {
    /// Get compressed size estimate
    pub fn size_bytes(&self) -> usize {
        self.name.len() + 16 + self.changed_indices.len() * 8 + self.changed_values.len() * 8
    }

    /// Get compression ratio (compressed/original)
    pub fn compression_ratio(&self, original_size: usize) -> f64 {
        if original_size == 0 {
            1.0
        } else {
            self.size_bytes() as f64 / original_size as f64
        }
    }
}
