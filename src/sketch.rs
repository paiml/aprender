//! Sketch-based statistics for distributed/federated drift detection
//!
//! Provides privacy-preserving statistical summaries that can be computed
//! locally and merged without sharing raw data.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::sketch::{TDigest, DDSketch};
//!
//! // Create sketches on each node
//! let mut digest1 = TDigest::new(100);
//! digest1.add_batch(&node1_data);
//!
//! let mut digest2 = TDigest::new(100);
//! digest2.add_batch(&node2_data);
//!
//! // Merge sketches (no raw data shared)
//! let merged = TDigest::merge(&[digest1, digest2]);
//!
//! // Query quantiles from merged sketch
//! println!("Median: {}", merged.quantile(0.5));
//! ```

// Sketch algorithms require numeric casts and float comparisons
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::float_cmp)]
#![allow(clippy::suboptimal_flops)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    dataset::{ArrowDataset, Dataset},
    drift::DriftSeverity,
    error::{Error, Result},
};

/// A centroid in a T-Digest (mean and weight)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Centroid {
    /// Mean value of this centroid
    pub mean: f64,
    /// Weight (count) of this centroid
    pub weight: f64,
}

impl Centroid {
    /// Create a new centroid
    pub fn new(mean: f64, weight: f64) -> Self {
        Self { mean, weight }
    }

    /// Merge another centroid into this one
    pub fn merge(&mut self, other: &Self) {
        let total_weight = self.weight + other.weight;
        if total_weight > 0.0 {
            self.mean = (self.mean * self.weight + other.mean * other.weight) / total_weight;
            self.weight = total_weight;
        }
    }
}

/// T-Digest for streaming quantile estimation
///
/// Provides accurate quantile estimates with bounded memory usage.
/// Based on the algorithm by Ted Dunning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDigest {
    /// Centroids sorted by mean
    centroids: Vec<Centroid>,
    /// Compression parameter (higher = more accuracy, more memory)
    compression: f64,
    /// Total weight of all centroids
    total_weight: f64,
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
}

impl TDigest {
    /// Create a new T-Digest with given compression factor
    ///
    /// Compression of 100 is a good default. Higher values give more accuracy.
    pub fn new(compression: f64) -> Self {
        Self {
            centroids: Vec::new(),
            compression,
            total_weight: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add a single value
    pub fn add(&mut self, value: f64) {
        self.add_weighted(value, 1.0);
    }

    /// Add a value with weight
    pub fn add_weighted(&mut self, value: f64, weight: f64) {
        if !value.is_finite() || weight <= 0.0 {
            return;
        }

        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.total_weight += weight;

        // Find insertion point
        let idx = self.find_insertion_point(value);

        // Try to merge with existing centroid
        if !self.centroids.is_empty() {
            let max_weight = self.max_weight_at(idx);
            let nearest = if idx < self.centroids.len() {
                idx
            } else {
                self.centroids.len() - 1
            };

            if self.centroids[nearest].weight + weight <= max_weight {
                self.centroids[nearest].merge(&Centroid::new(value, weight));
                return;
            }
        }

        // Insert new centroid
        self.centroids.insert(idx, Centroid::new(value, weight));

        // Compress if needed
        if self.centroids.len() > self.compression as usize * 2 {
            self.compress();
        }
    }

    /// Add a batch of values
    pub fn add_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Get estimated quantile (0-1)
    pub fn quantile(&self, q: f64) -> f64 {
        if self.centroids.is_empty() {
            return f64::NAN;
        }

        let q = q.clamp(0.0, 1.0);

        if q == 0.0 {
            return self.min;
        }
        if q == 1.0 {
            return self.max;
        }

        let target_weight = q * self.total_weight;
        let mut cumulative = 0.0;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let next_cumulative = cumulative + centroid.weight;

            if next_cumulative >= target_weight {
                // Interpolate within this centroid
                let prev_mean = if i > 0 {
                    self.centroids[i - 1].mean
                } else {
                    self.min
                };
                let next_mean = if i < self.centroids.len() - 1 {
                    self.centroids[i + 1].mean
                } else {
                    self.max
                };

                let ratio = if centroid.weight > 0.0 {
                    (target_weight - cumulative) / centroid.weight
                } else {
                    0.5
                };

                // Linear interpolation
                let low = (prev_mean + centroid.mean) / 2.0;
                let high = (centroid.mean + next_mean) / 2.0;

                return low + ratio * (high - low);
            }

            cumulative = next_cumulative;
        }

        self.max
    }

    /// Get the CDF value for a given x (proportion of values <= x)
    pub fn cdf(&self, x: f64) -> f64 {
        if self.centroids.is_empty() || self.total_weight == 0.0 {
            return 0.0;
        }

        if x <= self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }

        let mut cumulative = 0.0;

        for centroid in &self.centroids {
            if x < centroid.mean {
                // Interpolate
                return cumulative / self.total_weight;
            }
            cumulative += centroid.weight;
        }

        cumulative / self.total_weight
    }

    /// Merge multiple T-Digests into one
    pub fn merge(digests: &[Self]) -> Self {
        if digests.is_empty() {
            return Self::new(100.0);
        }

        let compression = digests.iter().map(|d| d.compression).fold(0.0, f64::max);
        let mut result = Self::new(compression);

        // Collect all centroids
        let mut all_centroids: Vec<Centroid> = digests
            .iter()
            .flat_map(|d| d.centroids.iter().cloned())
            .collect();

        // Sort by mean
        all_centroids.sort_by(|a, b| {
            a.mean
                .partial_cmp(&b.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update min/max
        result.min = digests.iter().map(|d| d.min).fold(f64::INFINITY, f64::min);
        result.max = digests
            .iter()
            .map(|d| d.max)
            .fold(f64::NEG_INFINITY, f64::max);
        result.total_weight = digests.iter().map(|d| d.total_weight).sum();

        // Add centroids
        for centroid in all_centroids {
            result.centroids.push(centroid);
        }

        // Compress
        result.compress();

        result
    }

    /// Get total count of values
    pub fn count(&self) -> f64 {
        self.total_weight
    }

    /// Get minimum value
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get maximum value
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Get number of centroids
    pub fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rmp_serde::to_vec(self)
            .map_err(|e| Error::Format(format!("Failed to serialize TDigest: {e}")))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes)
            .map_err(|e| Error::Format(format!("Failed to deserialize TDigest: {e}")))
    }

    // Internal methods

    fn find_insertion_point(&self, value: f64) -> usize {
        self.centroids
            .binary_search_by(|c| {
                c.mean
                    .partial_cmp(&value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|i| i)
    }

    fn max_weight_at(&self, index: usize) -> f64 {
        // Scale function for T-Digest
        let q = if self.total_weight > 0.0 {
            let cumulative: f64 = self.centroids.iter().take(index).map(|c| c.weight).sum();
            cumulative / self.total_weight
        } else {
            0.5
        };

        // k1 scale function
        let k = 4.0 * self.compression * q * (1.0 - q);
        k.max(1.0)
    }

    fn compress(&mut self) {
        if self.centroids.len() <= 1 {
            return;
        }

        // Sort by mean
        self.centroids.sort_by(|a, b| {
            a.mean
                .partial_cmp(&b.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_centroids = Vec::with_capacity(self.compression as usize);
        let mut current = self.centroids[0].clone();
        let mut cumulative = 0.0;

        for centroid in self.centroids.iter().skip(1) {
            let q = cumulative / self.total_weight;
            let max_weight = 4.0 * self.compression * q * (1.0 - q);

            if current.weight + centroid.weight <= max_weight.max(1.0) {
                current.merge(centroid);
            } else {
                cumulative += current.weight;
                new_centroids.push(current);
                current = centroid.clone();
            }
        }

        new_centroids.push(current);
        self.centroids = new_centroids;
    }
}

/// DDSketch for distribution estimation
///
/// Provides relative-error guarantees for quantile estimation.
/// Based on the algorithm by DataDog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDSketch {
    /// Relative accuracy parameter (e.g., 0.01 for 1% error)
    alpha: f64,
    /// Gamma = (1 + alpha) / (1 - alpha)
    gamma: f64,
    /// Log of gamma for bucket mapping
    ln_gamma: f64,
    /// Positive value buckets
    positive_buckets: HashMap<i32, u64>,
    /// Negative value buckets
    negative_buckets: HashMap<i32, u64>,
    /// Count of zero values
    zero_count: u64,
    /// Total count
    total_count: u64,
    /// Minimum value
    min: f64,
    /// Maximum value
    max: f64,
}

impl DDSketch {
    /// Create a new DDSketch with given relative accuracy
    ///
    /// Alpha of 0.01 gives 1% relative error on quantiles.
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(0.0001, 0.5);
        let gamma = (1.0 + alpha) / (1.0 - alpha);

        Self {
            alpha,
            gamma,
            ln_gamma: gamma.ln(),
            positive_buckets: HashMap::new(),
            negative_buckets: HashMap::new(),
            zero_count: 0,
            total_count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add a value
    pub fn add(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }

        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.total_count += 1;

        if value > 0.0 {
            let bucket = self.bucket_index(value);
            *self.positive_buckets.entry(bucket).or_insert(0) += 1;
        } else if value < 0.0 {
            let bucket = self.bucket_index(-value);
            *self.negative_buckets.entry(bucket).or_insert(0) += 1;
        } else {
            self.zero_count += 1;
        }
    }

    /// Add a batch of values
    pub fn add_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.add(v);
        }
    }

    /// Get estimated quantile (0-1)
    pub fn quantile(&self, q: f64) -> f64 {
        if self.total_count == 0 {
            return f64::NAN;
        }

        let q = q.clamp(0.0, 1.0);

        if q == 0.0 {
            return self.min;
        }
        if q == 1.0 {
            return self.max;
        }

        let target_rank = (q * self.total_count as f64).ceil() as u64;
        let mut cumulative: u64 = 0;

        // Check negative buckets (sorted descending by bucket index = ascending by
        // value)
        let mut neg_buckets: Vec<_> = self.negative_buckets.iter().collect();
        neg_buckets.sort_by(|a, b| b.0.cmp(a.0));

        for (&bucket, &count) in &neg_buckets {
            cumulative += count;
            if cumulative >= target_rank {
                return -self.bucket_to_value(bucket);
            }
        }

        // Check zero
        cumulative += self.zero_count;
        if cumulative >= target_rank {
            return 0.0;
        }

        // Check positive buckets (sorted ascending)
        let mut pos_buckets: Vec<_> = self.positive_buckets.iter().collect();
        pos_buckets.sort_by_key(|&(k, _)| *k);

        for (&bucket, &count) in &pos_buckets {
            cumulative += count;
            if cumulative >= target_rank {
                return self.bucket_to_value(bucket);
            }
        }

        self.max
    }

    /// Merge multiple DDSketches
    pub fn merge(sketches: &[Self]) -> Self {
        if sketches.is_empty() {
            return Self::new(0.01);
        }

        // Use minimum alpha for best accuracy
        let alpha = sketches
            .iter()
            .map(|s| s.alpha)
            .fold(f64::INFINITY, f64::min);
        let mut result = Self::new(alpha);

        result.min = sketches.iter().map(|s| s.min).fold(f64::INFINITY, f64::min);
        result.max = sketches
            .iter()
            .map(|s| s.max)
            .fold(f64::NEG_INFINITY, f64::max);
        result.total_count = sketches.iter().map(|s| s.total_count).sum();
        result.zero_count = sketches.iter().map(|s| s.zero_count).sum();

        for sketch in sketches {
            for (&bucket, &count) in &sketch.positive_buckets {
                *result.positive_buckets.entry(bucket).or_insert(0) += count;
            }
            for (&bucket, &count) in &sketch.negative_buckets {
                *result.negative_buckets.entry(bucket).or_insert(0) += count;
            }
        }

        result
    }

    /// Get total count
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Get minimum value
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get maximum value
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rmp_serde::to_vec(self)
            .map_err(|e| Error::Format(format!("Failed to serialize DDSketch: {e}")))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes)
            .map_err(|e| Error::Format(format!("Failed to deserialize DDSketch: {e}")))
    }

    // Internal methods

    fn bucket_index(&self, value: f64) -> i32 {
        if value <= 0.0 {
            return i32::MIN;
        }
        (value.ln() / self.ln_gamma).ceil() as i32
    }

    fn bucket_to_value(&self, bucket: i32) -> f64 {
        (2.0 * self.gamma.powi(bucket)) / (1.0 + self.gamma)
    }
}

/// Type of sketch algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SketchType {
    /// T-Digest for quantile estimation
    TDigest,
    /// DDSketch for relative-error quantiles
    DDSketch,
}

impl SketchType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::TDigest => "T-Digest",
            Self::DDSketch => "DDSketch",
        }
    }
}

/// Serializable data sketch containing distribution summaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSketch {
    /// Sketch type used
    pub sketch_type: SketchType,
    /// Per-column T-Digest sketches
    pub tdigests: HashMap<String, TDigest>,
    /// Per-column DDSketch sketches
    pub ddsketches: HashMap<String, DDSketch>,
    /// Total row count
    pub row_count: u64,
    /// Source identifier (e.g., node name)
    pub source: Option<String>,
}

impl DataSketch {
    /// Create a new empty data sketch
    pub fn new(sketch_type: SketchType) -> Self {
        Self {
            sketch_type,
            tdigests: HashMap::new(),
            ddsketches: HashMap::new(),
            row_count: 0,
            source: None,
        }
    }

    /// Set source identifier
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Create sketch from dataset
    pub fn from_dataset(dataset: &ArrowDataset, sketch_type: SketchType) -> Result<Self> {
        let mut sketch = Self::new(sketch_type);
        sketch.add_dataset(dataset)?;
        Ok(sketch)
    }

    /// Add data from dataset to sketch
    pub fn add_dataset(&mut self, dataset: &ArrowDataset) -> Result<()> {
        use arrow::{
            array::{Array, Float64Array, Int32Array, Int64Array},
            datatypes::DataType,
        };

        let schema = dataset.schema();

        for batch in dataset.iter() {
            self.row_count += batch.num_rows() as u64;

            for (col_idx, field) in schema.fields().iter().enumerate() {
                // Only sketch numeric columns
                let is_numeric = matches!(
                    field.data_type(),
                    DataType::Float64 | DataType::Float32 | DataType::Int32 | DataType::Int64
                );

                if !is_numeric {
                    continue;
                }

                let col_name = field.name();
                let array = batch.column(col_idx);

                // Collect values
                let values: Vec<f64> = match field.data_type() {
                    DataType::Float64 => {
                        if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
                            (0..arr.len())
                                .filter(|&i| !arr.is_null(i))
                                .map(|i| arr.value(i))
                                .collect()
                        } else {
                            continue;
                        }
                    }
                    DataType::Float32 => {
                        if let Some(arr) =
                            array.as_any().downcast_ref::<arrow::array::Float32Array>()
                        {
                            (0..arr.len())
                                .filter(|&i| !arr.is_null(i))
                                .map(|i| f64::from(arr.value(i)))
                                .collect()
                        } else {
                            continue;
                        }
                    }
                    DataType::Int32 => {
                        if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                            (0..arr.len())
                                .filter(|&i| !arr.is_null(i))
                                .map(|i| f64::from(arr.value(i)))
                                .collect()
                        } else {
                            continue;
                        }
                    }
                    DataType::Int64 => {
                        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                            (0..arr.len())
                                .filter(|&i| !arr.is_null(i))
                                .map(|i| arr.value(i) as f64)
                                .collect()
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                // Add to appropriate sketch
                match self.sketch_type {
                    SketchType::TDigest => {
                        let digest = self
                            .tdigests
                            .entry(col_name.clone())
                            .or_insert_with(|| TDigest::new(100.0));
                        digest.add_batch(&values);
                    }
                    SketchType::DDSketch => {
                        let sketch = self
                            .ddsketches
                            .entry(col_name.clone())
                            .or_insert_with(|| DDSketch::new(0.01));
                        sketch.add_batch(&values);
                    }
                }
            }
        }

        Ok(())
    }

    /// Merge multiple data sketches
    pub fn merge(sketches: &[Self]) -> Result<Self> {
        if sketches.is_empty() {
            return Err(Error::invalid_config("Cannot merge empty sketch list"));
        }

        let sketch_type = sketches[0].sketch_type;

        // Verify all sketches use same type
        for s in sketches {
            if s.sketch_type != sketch_type {
                return Err(Error::invalid_config(
                    "Cannot merge sketches of different types",
                ));
            }
        }

        let mut result = Self::new(sketch_type);
        result.row_count = sketches.iter().map(|s| s.row_count).sum();

        // Collect column names based on sketch type
        let columns: std::collections::HashSet<String> = match sketch_type {
            SketchType::TDigest => sketches
                .iter()
                .flat_map(|s| s.tdigests.keys().cloned())
                .collect(),
            SketchType::DDSketch => sketches
                .iter()
                .flat_map(|s| s.ddsketches.keys().cloned())
                .collect(),
        };

        // Merge each column
        for col in columns {
            match sketch_type {
                SketchType::TDigest => {
                    let digests: Vec<TDigest> = sketches
                        .iter()
                        .filter_map(|s| s.tdigests.get(&col).cloned())
                        .collect();
                    if !digests.is_empty() {
                        result.tdigests.insert(col, TDigest::merge(&digests));
                    }
                }
                SketchType::DDSketch => {
                    let dd_sketches: Vec<DDSketch> = sketches
                        .iter()
                        .filter_map(|s| s.ddsketches.get(&col).cloned())
                        .collect();
                    if !dd_sketches.is_empty() {
                        result.ddsketches.insert(col, DDSketch::merge(&dd_sketches));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Get quantile for a column
    pub fn quantile(&self, column: &str, q: f64) -> Option<f64> {
        match self.sketch_type {
            SketchType::TDigest => self.tdigests.get(column).map(|d| d.quantile(q)),
            SketchType::DDSketch => self.ddsketches.get(column).map(|d| d.quantile(q)),
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rmp_serde::to_vec(self)
            .map_err(|e| Error::Format(format!("Failed to serialize DataSketch: {e}")))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes)
            .map_err(|e| Error::Format(format!("Failed to deserialize DataSketch: {e}")))
    }
}

/// Result of distributed drift comparison
#[derive(Debug, Clone)]
pub struct SketchDriftResult {
    /// Column name
    pub column: String,
    /// KS-like statistic based on quantile differences
    pub statistic: f64,
    /// Estimated severity
    pub severity: DriftSeverity,
    /// Quantile differences at key points
    pub quantile_diffs: Vec<(f64, f64)>,
}

/// Distributed drift detector using sketches
pub struct DistributedDriftDetector {
    /// Sketch type to use
    sketch_type: SketchType,
    /// Number of quantile points to compare
    num_quantiles: usize,
    /// Threshold for drift detection
    threshold: f64,
}

impl Default for DistributedDriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedDriftDetector {
    /// Create a new distributed drift detector
    pub fn new() -> Self {
        Self {
            sketch_type: SketchType::TDigest,
            num_quantiles: 20,
            threshold: 0.1,
        }
    }

    /// Set sketch type
    #[must_use]
    pub fn with_sketch_type(mut self, sketch_type: SketchType) -> Self {
        self.sketch_type = sketch_type;
        self
    }

    /// Set number of quantile points to compare
    #[must_use]
    pub fn with_num_quantiles(mut self, n: usize) -> Self {
        self.num_quantiles = n.max(5);
        self
    }

    /// Set drift detection threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Create sketch from dataset
    pub fn create_sketch(&self, dataset: &ArrowDataset) -> Result<DataSketch> {
        DataSketch::from_dataset(dataset, self.sketch_type)
    }

    /// Compare two sketches for drift
    pub fn compare(
        &self,
        reference: &DataSketch,
        current: &DataSketch,
    ) -> Result<Vec<SketchDriftResult>> {
        if reference.sketch_type != current.sketch_type {
            return Err(Error::invalid_config("Sketch types must match"));
        }

        let mut results = Vec::new();

        // Get all columns
        let columns: std::collections::HashSet<&String> = match self.sketch_type {
            SketchType::TDigest => reference
                .tdigests
                .keys()
                .chain(current.tdigests.keys())
                .collect(),
            SketchType::DDSketch => reference
                .ddsketches
                .keys()
                .chain(current.ddsketches.keys())
                .collect(),
        };

        for col in columns {
            let result = self.compare_column(reference, current, col);
            results.push(result);
        }

        Ok(results)
    }

    /// Compare a single column between sketches
    fn compare_column(
        &self,
        reference: &DataSketch,
        current: &DataSketch,
        column: &str,
    ) -> SketchDriftResult {
        let mut max_diff = 0.0_f64;
        let mut quantile_diffs = Vec::new();

        // Compare at multiple quantile points
        for i in 1..self.num_quantiles {
            let q = i as f64 / self.num_quantiles as f64;

            let ref_val = reference.quantile(column, q);
            let cur_val = current.quantile(column, q);

            if let (Some(r), Some(c)) = (ref_val, cur_val) {
                // Relative difference
                let diff = if r.abs() > f64::EPSILON {
                    ((c - r) / r).abs()
                } else if c.abs() > f64::EPSILON {
                    1.0
                } else {
                    0.0
                };

                max_diff = max_diff.max(diff);
                quantile_diffs.push((q, diff));
            }
        }

        // Map max difference to severity
        let severity = if max_diff < self.threshold {
            DriftSeverity::None
        } else if max_diff < self.threshold * 2.0 {
            DriftSeverity::Low
        } else if max_diff < self.threshold * 5.0 {
            DriftSeverity::Medium
        } else if max_diff < self.threshold * 10.0 {
            DriftSeverity::High
        } else {
            DriftSeverity::Critical
        };

        SketchDriftResult {
            column: column.to_string(),
            statistic: max_diff,
            severity,
            quantile_diffs,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::Float64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;

    // ========== TDigest tests ==========

    #[test]
    fn test_tdigest_new() {
        let digest = TDigest::new(100.0);
        assert!(digest.is_empty());
        assert_eq!(digest.count(), 0.0);
    }

    #[test]
    fn test_tdigest_add_single() {
        let mut digest = TDigest::new(100.0);
        digest.add(5.0);

        assert!(!digest.is_empty());
        assert_eq!(digest.count(), 1.0);
        assert_eq!(digest.min(), 5.0);
        assert_eq!(digest.max(), 5.0);
    }

    #[test]
    fn test_tdigest_add_batch() {
        let mut digest = TDigest::new(100.0);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        digest.add_batch(&values);

        assert_eq!(digest.count(), 100.0);
        assert_eq!(digest.min(), 0.0);
        assert_eq!(digest.max(), 99.0);
    }

    #[test]
    fn test_tdigest_quantile_median() {
        let mut digest = TDigest::new(100.0);
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        digest.add_batch(&values);

        let median = digest.quantile(0.5);
        // Should be close to 500
        assert!((median - 500.0).abs() < 50.0, "Median was {}", median);
    }

    #[test]
    fn test_tdigest_quantile_extremes() {
        let mut digest = TDigest::new(100.0);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        digest.add_batch(&values);

        assert_eq!(digest.quantile(0.0), 0.0);
        assert_eq!(digest.quantile(1.0), 99.0);
    }

    #[test]
    fn test_tdigest_quantile_quartiles() {
        let mut digest = TDigest::new(100.0);
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        digest.add_batch(&values);

        let q1 = digest.quantile(0.25);
        let q3 = digest.quantile(0.75);

        // Should be approximately 250 and 750
        assert!((q1 - 250.0).abs() < 50.0, "Q1 was {}", q1);
        assert!((q3 - 750.0).abs() < 50.0, "Q3 was {}", q3);
    }

    #[test]
    fn test_tdigest_merge() {
        let mut digest1 = TDigest::new(100.0);
        let mut digest2 = TDigest::new(100.0);

        let values1: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let values2: Vec<f64> = (500..1000).map(|i| i as f64).collect();

        digest1.add_batch(&values1);
        digest2.add_batch(&values2);

        let merged = TDigest::merge(&[digest1, digest2]);

        assert_eq!(merged.count(), 1000.0);
        assert_eq!(merged.min(), 0.0);
        assert_eq!(merged.max(), 999.0);

        let median = merged.quantile(0.5);
        assert!(
            (median - 500.0).abs() < 50.0,
            "Merged median was {}",
            median
        );
    }

    #[test]
    fn test_tdigest_serialization() {
        let mut digest = TDigest::new(100.0);
        digest.add_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let bytes = digest.to_bytes().expect("serialize");
        let restored = TDigest::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.count(), digest.count());
        assert_eq!(restored.min(), digest.min());
        assert_eq!(restored.max(), digest.max());
    }

    #[test]
    fn test_tdigest_cdf() {
        let mut digest = TDigest::new(100.0);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        digest.add_batch(&values);

        assert_eq!(digest.cdf(-1.0), 0.0);
        assert_eq!(digest.cdf(100.0), 1.0);

        let cdf_50 = digest.cdf(50.0);
        assert!(cdf_50 > 0.4 && cdf_50 < 0.6, "CDF at 50 was {}", cdf_50);
    }

    #[test]
    fn test_tdigest_empty_quantile() {
        let digest = TDigest::new(100.0);
        assert!(digest.quantile(0.5).is_nan());
    }

    // ========== DDSketch tests ==========

    #[test]
    fn test_ddsketch_new() {
        let sketch = DDSketch::new(0.01);
        assert!(sketch.is_empty());
        assert_eq!(sketch.count(), 0);
    }

    #[test]
    fn test_ddsketch_add_single() {
        let mut sketch = DDSketch::new(0.01);
        sketch.add(5.0);

        assert!(!sketch.is_empty());
        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.min(), 5.0);
        assert_eq!(sketch.max(), 5.0);
    }

    #[test]
    fn test_ddsketch_add_batch() {
        let mut sketch = DDSketch::new(0.01);
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        sketch.add_batch(&values);

        assert_eq!(sketch.count(), 100);
        assert_eq!(sketch.min(), 1.0);
        assert_eq!(sketch.max(), 100.0);
    }

    #[test]
    fn test_ddsketch_quantile_median() {
        let mut sketch = DDSketch::new(0.01);
        let values: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        sketch.add_batch(&values);

        let median = sketch.quantile(0.5);
        // Should be close to 500, within relative error
        assert!((median - 500.0).abs() < 100.0, "Median was {}", median);
    }

    #[test]
    fn test_ddsketch_quantile_extremes() {
        let mut sketch = DDSketch::new(0.01);
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        sketch.add_batch(&values);

        assert_eq!(sketch.quantile(0.0), 1.0);
        assert_eq!(sketch.quantile(1.0), 100.0);
    }

    #[test]
    fn test_ddsketch_negative_values() {
        let mut sketch = DDSketch::new(0.01);
        let values: Vec<f64> = (-50..=50).map(|i| i as f64).collect();
        sketch.add_batch(&values);

        assert_eq!(sketch.min(), -50.0);
        assert_eq!(sketch.max(), 50.0);

        let median = sketch.quantile(0.5);
        assert!((median).abs() < 20.0, "Median was {}", median);
    }

    #[test]
    fn test_ddsketch_merge() {
        let mut sketch1 = DDSketch::new(0.01);
        let mut sketch2 = DDSketch::new(0.01);

        let values1: Vec<f64> = (1..=500).map(|i| i as f64).collect();
        let values2: Vec<f64> = (501..=1000).map(|i| i as f64).collect();

        sketch1.add_batch(&values1);
        sketch2.add_batch(&values2);

        let merged = DDSketch::merge(&[sketch1, sketch2]);

        assert_eq!(merged.count(), 1000);
        assert_eq!(merged.min(), 1.0);
        assert_eq!(merged.max(), 1000.0);
    }

    #[test]
    fn test_ddsketch_serialization() {
        let mut sketch = DDSketch::new(0.01);
        sketch.add_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let bytes = sketch.to_bytes().expect("serialize");
        let restored = DDSketch::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.count(), sketch.count());
        assert_eq!(restored.min(), sketch.min());
        assert_eq!(restored.max(), sketch.max());
    }

    #[test]
    fn test_ddsketch_empty_quantile() {
        let sketch = DDSketch::new(0.01);
        assert!(sketch.quantile(0.5).is_nan());
    }

    // ========== SketchType tests ==========

    #[test]
    fn test_sketch_type_name() {
        assert_eq!(SketchType::TDigest.name(), "T-Digest");
        assert_eq!(SketchType::DDSketch.name(), "DDSketch");
    }

    // ========== DataSketch tests ==========

    fn make_float_dataset(values: Vec<f64>) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(values))],
        )
        .expect("batch");

        ArrowDataset::from_batch(batch).expect("dataset")
    }

    #[test]
    fn test_data_sketch_from_dataset_tdigest() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let dataset = make_float_dataset(values);

        let sketch = DataSketch::from_dataset(&dataset, SketchType::TDigest).expect("sketch");

        assert_eq!(sketch.row_count, 100);
        assert!(sketch.tdigests.contains_key("value"));

        let median = sketch.quantile("value", 0.5);
        assert!(median.is_some());
    }

    #[test]
    fn test_data_sketch_from_dataset_ddsketch() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let dataset = make_float_dataset(values);

        let sketch = DataSketch::from_dataset(&dataset, SketchType::DDSketch).expect("sketch");

        assert_eq!(sketch.row_count, 100);
        assert!(sketch.ddsketches.contains_key("value"));
    }

    #[test]
    fn test_data_sketch_merge() {
        let values1: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let values2: Vec<f64> = (50..100).map(|i| i as f64).collect();

        let dataset1 = make_float_dataset(values1);
        let dataset2 = make_float_dataset(values2);

        let sketch1 = DataSketch::from_dataset(&dataset1, SketchType::TDigest).expect("sketch1");
        let sketch2 = DataSketch::from_dataset(&dataset2, SketchType::TDigest).expect("sketch2");

        let merged = DataSketch::merge(&[sketch1, sketch2]).expect("merge");

        assert_eq!(merged.row_count, 100);
    }

    #[test]
    fn test_data_sketch_serialization() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let dataset = make_float_dataset(values);

        let sketch = DataSketch::from_dataset(&dataset, SketchType::TDigest).expect("sketch");
        let bytes = sketch.to_bytes().expect("serialize");
        let restored = DataSketch::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.row_count, sketch.row_count);
        assert_eq!(restored.sketch_type, sketch.sketch_type);
    }

    // ========== DistributedDriftDetector tests ==========

    #[test]
    fn test_distributed_detector_new() {
        let detector = DistributedDriftDetector::new();
        assert_eq!(detector.sketch_type, SketchType::TDigest);
    }

    #[test]
    fn test_distributed_detector_builder() {
        let detector = DistributedDriftDetector::new()
            .with_sketch_type(SketchType::DDSketch)
            .with_num_quantiles(50)
            .with_threshold(0.2);

        assert_eq!(detector.sketch_type, SketchType::DDSketch);
        assert_eq!(detector.num_quantiles, 50);
        assert!((detector.threshold - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distributed_detector_no_drift() {
        let values: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let dataset1 = make_float_dataset(values.clone());
        let dataset2 = make_float_dataset(values);

        let detector = DistributedDriftDetector::new();
        let sketch1 = detector.create_sketch(&dataset1).expect("sketch1");
        let sketch2 = detector.create_sketch(&dataset2).expect("sketch2");

        let results = detector.compare(&sketch1, &sketch2).expect("compare");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].severity, DriftSeverity::None);
    }

    #[test]
    fn test_distributed_detector_with_drift() {
        let values1: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let values2: Vec<f64> = (500..1000).map(|i| i as f64).collect();

        let dataset1 = make_float_dataset(values1);
        let dataset2 = make_float_dataset(values2);

        let detector = DistributedDriftDetector::new().with_threshold(0.1);
        let sketch1 = detector.create_sketch(&dataset1).expect("sketch1");
        let sketch2 = detector.create_sketch(&dataset2).expect("sketch2");

        let results = detector.compare(&sketch1, &sketch2).expect("compare");

        assert_eq!(results.len(), 1);
        assert!(results[0].severity.is_drift(), "Should detect drift");
        assert!(results[0].statistic > 0.0);
    }

    #[test]
    fn test_distributed_detector_ddsketch() {
        let values1: Vec<f64> = (1..=500).map(|i| i as f64).collect();
        let values2: Vec<f64> = (1..=500).map(|i| i as f64).collect();

        let dataset1 = make_float_dataset(values1);
        let dataset2 = make_float_dataset(values2);

        let detector = DistributedDriftDetector::new().with_sketch_type(SketchType::DDSketch);
        let sketch1 = detector.create_sketch(&dataset1).expect("sketch1");
        let sketch2 = detector.create_sketch(&dataset2).expect("sketch2");

        let results = detector.compare(&sketch1, &sketch2).expect("compare");

        assert!(!results.is_empty());
        assert_eq!(results[0].severity, DriftSeverity::None);
    }

    #[test]
    fn test_distributed_detector_quantile_diffs() {
        let values1: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let values2: Vec<f64> = (100..600).map(|i| i as f64).collect();

        let dataset1 = make_float_dataset(values1);
        let dataset2 = make_float_dataset(values2);

        let detector = DistributedDriftDetector::new().with_num_quantiles(10);
        let sketch1 = detector.create_sketch(&dataset1).expect("sketch1");
        let sketch2 = detector.create_sketch(&dataset2).expect("sketch2");

        let results = detector.compare(&sketch1, &sketch2).expect("compare");

        assert!(!results[0].quantile_diffs.is_empty());
    }

    // ========== Additional edge case tests ==========

    #[test]
    fn test_centroid_new_and_merge() {
        let mut c1 = Centroid::new(10.0, 2.0);
        let c2 = Centroid::new(20.0, 3.0);
        c1.merge(&c2);
        // Weighted average: (10*2 + 20*3) / 5 = 80/5 = 16
        assert!((c1.mean - 16.0).abs() < f64::EPSILON);
        assert_eq!(c1.weight, 5.0);
    }

    #[test]
    fn test_centroid_merge_zero_weights() {
        let mut c1 = Centroid::new(10.0, 0.0);
        let c2 = Centroid::new(20.0, 0.0);
        c1.merge(&c2);
        // total_weight = 0, so merge shouldn't change mean
        assert_eq!(c1.mean, 10.0);
        assert_eq!(c1.weight, 0.0);
    }

    #[test]
    fn test_tdigest_add_weighted_non_finite() {
        let mut digest = TDigest::new(100.0);
        digest.add_weighted(f64::NAN, 1.0);
        digest.add_weighted(f64::INFINITY, 1.0);
        digest.add_weighted(f64::NEG_INFINITY, 1.0);
        assert!(digest.is_empty());
    }

    #[test]
    fn test_tdigest_add_weighted_zero_weight() {
        let mut digest = TDigest::new(100.0);
        digest.add_weighted(5.0, 0.0);
        digest.add_weighted(10.0, -1.0);
        assert!(digest.is_empty());
    }

    #[test]
    fn test_tdigest_num_centroids() {
        let mut digest = TDigest::new(100.0);
        assert_eq!(digest.num_centroids(), 0);

        digest.add(5.0);
        assert!(digest.num_centroids() > 0);
    }

    #[test]
    fn test_tdigest_quantile_clamp() {
        let mut digest = TDigest::new(100.0);
        digest.add_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test clamping of out-of-range quantiles
        let q_neg = digest.quantile(-0.5);
        let q_over = digest.quantile(1.5);
        assert_eq!(q_neg, digest.min());
        assert_eq!(q_over, digest.max());
    }

    #[test]
    fn test_tdigest_merge_empty() {
        let merged = TDigest::merge(&[]);
        assert!(merged.is_empty());
        assert_eq!(merged.compression, 100.0);
    }

    #[test]
    fn test_tdigest_cdf_empty() {
        let digest = TDigest::new(100.0);
        assert_eq!(digest.cdf(5.0), 0.0);
    }

    #[test]
    fn test_tdigest_clone() {
        let mut digest = TDigest::new(100.0);
        digest.add_batch(&[1.0, 2.0, 3.0]);

        let cloned = digest.clone();
        assert_eq!(cloned.count(), digest.count());
        assert_eq!(cloned.min(), digest.min());
        assert_eq!(cloned.max(), digest.max());
    }

    #[test]
    fn test_tdigest_debug() {
        let digest = TDigest::new(100.0);
        let debug = format!("{:?}", digest);
        assert!(debug.contains("TDigest"));
    }

    #[test]
    fn test_ddsketch_add_non_finite() {
        let mut sketch = DDSketch::new(0.01);
        sketch.add(f64::NAN);
        sketch.add(f64::INFINITY);
        sketch.add(f64::NEG_INFINITY);
        assert!(sketch.is_empty());
    }

    #[test]
    fn test_ddsketch_add_zero() {
        let mut sketch = DDSketch::new(0.01);
        sketch.add(0.0);
        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.quantile(0.5), 0.0);
    }

    #[test]
    fn test_ddsketch_quantile_clamp() {
        let mut sketch = DDSketch::new(0.01);
        sketch.add_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let q_neg = sketch.quantile(-0.5);
        let q_over = sketch.quantile(1.5);
        assert_eq!(q_neg, sketch.min());
        assert_eq!(q_over, sketch.max());
    }

    #[test]
    fn test_ddsketch_merge_empty() {
        let merged = DDSketch::merge(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_ddsketch_alpha_clamp() {
        // Alpha too small
        let sketch1 = DDSketch::new(0.00001);
        assert!(sketch1.alpha >= 0.0001);

        // Alpha too large
        let sketch2 = DDSketch::new(0.9);
        assert!(sketch2.alpha <= 0.5);
    }

    #[test]
    fn test_ddsketch_clone() {
        let mut sketch = DDSketch::new(0.01);
        sketch.add_batch(&[1.0, 2.0, 3.0]);

        let cloned = sketch.clone();
        assert_eq!(cloned.count(), sketch.count());
        assert_eq!(cloned.min(), sketch.min());
    }

    #[test]
    fn test_ddsketch_debug() {
        let sketch = DDSketch::new(0.01);
        let debug = format!("{:?}", sketch);
        assert!(debug.contains("DDSketch"));
    }

    #[test]
    fn test_sketch_type_equality() {
        assert_eq!(SketchType::TDigest, SketchType::TDigest);
        assert_ne!(SketchType::TDigest, SketchType::DDSketch);
    }

    #[test]
    fn test_sketch_type_clone() {
        let st = SketchType::DDSketch;
        let cloned = st;
        assert_eq!(st, cloned);
    }

    #[test]
    fn test_sketch_type_debug() {
        let st = SketchType::TDigest;
        let debug = format!("{:?}", st);
        assert!(debug.contains("TDigest"));
    }

    #[test]
    fn test_data_sketch_new() {
        let sketch = DataSketch::new(SketchType::TDigest);
        assert_eq!(sketch.sketch_type, SketchType::TDigest);
        assert_eq!(sketch.row_count, 0);
        assert!(sketch.source.is_none());
    }

    #[test]
    fn test_data_sketch_with_source() {
        let sketch = DataSketch::new(SketchType::TDigest).with_source("node1");
        assert_eq!(sketch.source, Some("node1".to_string()));
    }

    #[test]
    fn test_data_sketch_merge_empty_error() {
        let result = DataSketch::merge(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_sketch_merge_different_types_error() {
        let sketch1 = DataSketch::new(SketchType::TDigest);
        let sketch2 = DataSketch::new(SketchType::DDSketch);

        let result = DataSketch::merge(&[sketch1, sketch2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_sketch_quantile_not_found() {
        let sketch = DataSketch::new(SketchType::TDigest);
        assert!(sketch.quantile("nonexistent", 0.5).is_none());
    }

    #[test]
    fn test_data_sketch_clone() {
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let dataset = make_float_dataset(values);
        let sketch = DataSketch::from_dataset(&dataset, SketchType::TDigest).expect("sketch");

        let cloned = sketch.clone();
        assert_eq!(cloned.row_count, sketch.row_count);
        assert_eq!(cloned.sketch_type, sketch.sketch_type);
    }

    #[test]
    fn test_data_sketch_debug() {
        let sketch = DataSketch::new(SketchType::DDSketch);
        let debug = format!("{:?}", sketch);
        assert!(debug.contains("DataSketch"));
    }

    #[test]
    fn test_sketch_drift_result_clone() {
        let result = SketchDriftResult {
            column: "test".to_string(),
            statistic: 0.5,
            severity: DriftSeverity::Medium,
            quantile_diffs: vec![(0.5, 0.1)],
        };

        let cloned = result.clone();
        assert_eq!(cloned.column, result.column);
        assert_eq!(cloned.statistic, result.statistic);
    }

    #[test]
    fn test_sketch_drift_result_debug() {
        let result = SketchDriftResult {
            column: "test".to_string(),
            statistic: 0.5,
            severity: DriftSeverity::None,
            quantile_diffs: vec![],
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("SketchDriftResult"));
    }

    #[test]
    fn test_distributed_detector_default() {
        let detector = DistributedDriftDetector::default();
        assert_eq!(detector.sketch_type, SketchType::TDigest);
    }

    #[test]
    fn test_distributed_detector_compare_type_mismatch() {
        let sketch1 = DataSketch::new(SketchType::TDigest);
        let sketch2 = DataSketch::new(SketchType::DDSketch);

        let detector = DistributedDriftDetector::new();
        let result = detector.compare(&sketch1, &sketch2);
        assert!(result.is_err());
    }

    #[test]
    fn test_distributed_detector_num_quantiles_min() {
        let detector = DistributedDriftDetector::new().with_num_quantiles(1);
        assert!(detector.num_quantiles >= 5);
    }

    #[test]
    fn test_tdigest_compression_triggers() {
        // Add enough values to trigger compression
        let mut digest = TDigest::new(10.0); // Low compression to trigger often
        for i in 0..1000 {
            digest.add(i as f64);
        }
        // Should have compressed
        assert!(digest.num_centroids() < 1000);
    }

    #[test]
    fn test_tdigest_serialization_invalid() {
        let result = TDigest::from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ddsketch_serialization_invalid() {
        let result = DDSketch::from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_sketch_serialization_invalid() {
        let result = DataSketch::from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_centroid_clone() {
        let c = Centroid::new(5.0, 2.0);
        let cloned = c.clone();
        assert_eq!(cloned.mean, c.mean);
        assert_eq!(cloned.weight, c.weight);
    }

    #[test]
    fn test_centroid_debug() {
        let c = Centroid::new(5.0, 2.0);
        let debug = format!("{:?}", c);
        assert!(debug.contains("Centroid"));
    }

    #[test]
    fn test_data_sketch_merge_ddsketch() {
        let values1: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let values2: Vec<f64> = (51..=100).map(|i| i as f64).collect();

        let dataset1 = make_float_dataset(values1);
        let dataset2 = make_float_dataset(values2);

        let sketch1 = DataSketch::from_dataset(&dataset1, SketchType::DDSketch).expect("sketch1");
        let sketch2 = DataSketch::from_dataset(&dataset2, SketchType::DDSketch).expect("sketch2");

        let merged = DataSketch::merge(&[sketch1, sketch2]).expect("merge");

        assert_eq!(merged.row_count, 100);
        assert_eq!(merged.sketch_type, SketchType::DDSketch);
    }

    #[test]
    fn test_distributed_detector_severity_levels() {
        // Create significantly different distributions to trigger different severity
        // levels
        let values1: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let values2: Vec<f64> = (0..100).map(|i| (i * 50) as f64).collect(); // 50x different

        let dataset1 = make_float_dataset(values1);
        let dataset2 = make_float_dataset(values2);

        let detector = DistributedDriftDetector::new().with_threshold(0.01);
        let sketch1 = detector.create_sketch(&dataset1).expect("sketch1");
        let sketch2 = detector.create_sketch(&dataset2).expect("sketch2");

        let results = detector.compare(&sketch1, &sketch2).expect("compare");
        assert!(!results.is_empty());
        // Should detect some level of drift
        assert!(results[0].statistic > 0.0);
    }

    #[test]
    fn test_data_sketch_add_dataset_int_types() {
        // Test with Int32 and Int64 columns
        use arrow::array::{Int32Array, Int64Array};

        let schema = Arc::new(Schema::new(vec![
            Field::new("int32_col", DataType::Int32, false),
            Field::new("int64_col", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50])),
            ],
        )
        .expect("batch");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset");
        let sketch = DataSketch::from_dataset(&dataset, SketchType::TDigest).expect("sketch");

        assert_eq!(sketch.row_count, 5);
        assert!(sketch.tdigests.contains_key("int32_col"));
        assert!(sketch.tdigests.contains_key("int64_col"));
    }

    #[test]
    fn test_data_sketch_add_dataset_float32() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "float32_col",
            DataType::Float32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float32Array::from(vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0,
            ]))],
        )
        .expect("batch");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset");
        let sketch = DataSketch::from_dataset(&dataset, SketchType::TDigest).expect("sketch");

        assert_eq!(sketch.row_count, 5);
        assert!(sketch.tdigests.contains_key("float32_col"));
    }

    #[test]
    fn test_data_sketch_non_numeric_columns_skipped() {
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .expect("batch");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset");
        let sketch = DataSketch::from_dataset(&dataset, SketchType::TDigest).expect("sketch");

        // Only numeric column should be sketched
        assert!(sketch.tdigests.contains_key("value"));
        assert!(!sketch.tdigests.contains_key("name"));
    }

    #[test]
    fn test_distributed_detector_compare_missing_column() {
        // Test when one sketch has a column the other doesn't
        let values1: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let dataset1 = make_float_dataset(values1);

        let detector = DistributedDriftDetector::new();
        let sketch1 = detector.create_sketch(&dataset1).expect("sketch1");

        // Create empty sketch
        let sketch2 = DataSketch::new(SketchType::TDigest);

        let results = detector.compare(&sketch1, &sketch2).expect("compare");
        // Should still produce results, even if quantiles are None
        assert!(!results.is_empty());
    }
}
