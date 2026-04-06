//! Workload Characterization System (PMAT-035)
//!
//! Automatic workload classification based on runtime metrics.
//!
//! # Features
//!
//! - Feature extraction from workload metrics
//! - Workload classification (GEMM, Bandwidth, Attention, etc.)
//! - Similarity computation between workloads
//! - Backend recommendation based on classification
//!
//! # Falsification Criteria (F1271-F1280)
//!
//! See `tests/workload_characterization_f1271.rs` for falsification tests.

/// Known workload categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadCategory {
    /// Matrix multiplication (compute-bound)
    Gemm,
    /// Memory bandwidth test (memory-bound)
    Bandwidth,
    /// Attention mechanism (mixed)
    Attention,
    /// Convolution (compute-bound)
    Conv2d,
    /// Elementwise operations (memory-bound)
    Elementwise,
    /// Reduction operations (mixed)
    Reduction,
    /// Unknown workload type
    Unknown,
}

impl WorkloadCategory {
    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemm => "gemm",
            Self::Bandwidth => "bandwidth",
            Self::Attention => "attention",
            Self::Conv2d => "conv2d",
            Self::Elementwise => "elementwise",
            Self::Reduction => "reduction",
            Self::Unknown => "unknown",
        }
    }

    /// Check if compute-bound
    pub fn is_compute_bound(&self) -> bool {
        matches!(self, Self::Gemm | Self::Conv2d)
    }

    /// Check if memory-bound
    pub fn is_memory_bound(&self) -> bool {
        matches!(self, Self::Bandwidth | Self::Elementwise)
    }

    /// Get typical arithmetic intensity range
    pub fn typical_intensity_range(&self) -> (f64, f64) {
        match self {
            Self::Gemm => (10.0, 100.0),
            Self::Bandwidth => (0.1, 1.0),
            Self::Attention => (1.0, 20.0),
            Self::Conv2d => (5.0, 50.0),
            Self::Elementwise => (0.1, 0.5),
            Self::Reduction => (0.5, 5.0),
            Self::Unknown => (0.0, 100.0),
        }
    }
}

/// Recommended compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedBackend {
    /// CPU with SIMD (small sizes)
    CpuSimd,
    /// GPU (large sizes, parallel workloads)
    Gpu,
    /// Either CPU or GPU acceptable
    Either,
}

impl RecommendedBackend {
    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuSimd => "cpu_simd",
            Self::Gpu => "gpu",
            Self::Either => "either",
        }
    }
}

/// Workload feature vector
#[derive(Debug, Clone)]
pub struct WorkloadFeatures {
    /// Arithmetic intensity (FLOPs / Bytes)
    pub arithmetic_intensity: f64,
    /// Memory footprint in bytes
    pub memory_footprint: usize,
    /// Working set size in bytes
    pub working_set: usize,
    /// Access pattern score (0 = random, 1 = sequential)
    pub access_pattern: f64,
    /// Compute density (ops per cycle)
    pub compute_density: f64,
    /// Branch rate (branches per operation)
    pub branch_rate: f64,
    /// Data reuse factor
    pub data_reuse: f64,
}

impl Default for WorkloadFeatures {
    fn default() -> Self {
        Self {
            arithmetic_intensity: 1.0,
            memory_footprint: 0,
            working_set: 0,
            access_pattern: 0.5,
            compute_density: 1.0,
            branch_rate: 0.0,
            data_reuse: 1.0,
        }
    }
}

impl WorkloadFeatures {
    /// Create new feature vector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set arithmetic intensity
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.arithmetic_intensity = intensity.max(0.0);
        self
    }

    /// Set memory footprint
    pub fn with_memory(mut self, footprint: usize, working_set: usize) -> Self {
        self.memory_footprint = footprint;
        self.working_set = working_set;
        self
    }

    /// Set access pattern (0 = random, 1 = sequential)
    pub fn with_access_pattern(mut self, pattern: f64) -> Self {
        self.access_pattern = pattern.clamp(0.0, 1.0);
        self
    }

    /// Set compute density
    pub fn with_compute_density(mut self, density: f64) -> Self {
        self.compute_density = density.max(0.0);
        self
    }

    /// Set branch rate
    pub fn with_branch_rate(mut self, rate: f64) -> Self {
        self.branch_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set data reuse factor
    pub fn with_data_reuse(mut self, reuse: f64) -> Self {
        self.data_reuse = reuse.max(1.0);
        self
    }

    /// Normalize features to Z-scores
    pub fn normalize(&self, means: &[f64], stds: &[f64]) -> Vec<f64> {
        let features = self.to_vec();
        features
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if stds[i] > 1e-10 {
                    (v - means[i]) / stds[i]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Convert to feature vector
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.arithmetic_intensity,
            self.memory_footprint as f64,
            self.working_set as f64,
            self.access_pattern,
            self.compute_density,
            self.branch_rate,
            self.data_reuse,
        ]
    }

    /// Compute Euclidean distance to another feature vector
    pub fn distance(&self, other: &Self) -> f64 {
        let a = self.to_vec();
        let b = other.to_vec();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute cosine similarity
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let a = self.to_vec();
        let b = other.to_vec();

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Classification result
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted category
    pub category: WorkloadCategory,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Distance to nearest prototype
    pub distance: f64,
    /// Recommended backend
    pub recommended_backend: RecommendedBackend,
    /// Size threshold for GPU crossover
    pub gpu_crossover_size: Option<usize>,
}

impl ClassificationResult {
    /// Check if classification is confident
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.7
    }
}

/// Workload characterization system
#[derive(Debug)]
pub struct WorkloadCharacterizer {
    /// Prototype features for known workloads
    prototypes: Vec<(WorkloadCategory, WorkloadFeatures)>,
    /// GPU crossover thresholds by category
    gpu_thresholds: Vec<(WorkloadCategory, usize)>,
}

impl Default for WorkloadCharacterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadCharacterizer {
    /// Create new characterizer with default prototypes
    pub fn new() -> Self {
        let prototypes = vec![
            // GEMM: High intensity, high compute density, high reuse
            (
                WorkloadCategory::Gemm,
                WorkloadFeatures::new()
                    .with_intensity(50.0)
                    .with_compute_density(8.0)
                    .with_access_pattern(0.8)
                    .with_data_reuse(32.0)
                    .with_branch_rate(0.01),
            ),
            // Bandwidth: Low intensity, sequential access
            (
                WorkloadCategory::Bandwidth,
                WorkloadFeatures::new()
                    .with_intensity(0.25)
                    .with_compute_density(0.5)
                    .with_access_pattern(1.0)
                    .with_data_reuse(1.0)
                    .with_branch_rate(0.0),
            ),
            // Attention: Medium intensity, mixed access
            (
                WorkloadCategory::Attention,
                WorkloadFeatures::new()
                    .with_intensity(5.0)
                    .with_compute_density(4.0)
                    .with_access_pattern(0.6)
                    .with_data_reuse(4.0)
                    .with_branch_rate(0.05),
            ),
            // Conv2D: High intensity, sliding window access
            (
                WorkloadCategory::Conv2d,
                WorkloadFeatures::new()
                    .with_intensity(20.0)
                    .with_compute_density(6.0)
                    .with_access_pattern(0.7)
                    .with_data_reuse(9.0)
                    .with_branch_rate(0.02),
            ),
            // Elementwise: Very low intensity, sequential
            (
                WorkloadCategory::Elementwise,
                WorkloadFeatures::new()
                    .with_intensity(0.125)
                    .with_compute_density(1.0)
                    .with_access_pattern(1.0)
                    .with_data_reuse(1.0)
                    .with_branch_rate(0.0),
            ),
            // Reduction: Low intensity, tree pattern
            (
                WorkloadCategory::Reduction,
                WorkloadFeatures::new()
                    .with_intensity(1.0)
                    .with_compute_density(2.0)
                    .with_access_pattern(0.5)
                    .with_data_reuse(2.0)
                    .with_branch_rate(0.1),
            ),
        ];

        let gpu_thresholds = vec![
            (WorkloadCategory::Gemm, 10_000),         // GPU wins at ~100x100
            (WorkloadCategory::Bandwidth, 1_000_000), // GPU wins at 1M elements
            (WorkloadCategory::Attention, 50_000),    // GPU wins at ~224 seq len
            (WorkloadCategory::Conv2d, 100_000),      // GPU wins at moderate sizes
            (WorkloadCategory::Elementwise, 500_000), // GPU wins at 500K elements
            (WorkloadCategory::Reduction, 100_000),   // GPU wins at 100K elements
        ];

        Self {
            prototypes,
            gpu_thresholds,
        }
    }

    /// Extract features from workload metrics
    pub fn extract_features(
        &self,
        flops: f64,
        bytes_accessed: f64,
        memory_footprint: usize,
        working_set: usize,
    ) -> WorkloadFeatures {
        let intensity = if bytes_accessed > 0.0 {
            flops / bytes_accessed
        } else {
            0.0
        };

        WorkloadFeatures::new()
            .with_intensity(intensity)
            .with_memory(memory_footprint, working_set)
    }

    /// Classify workload based on features
    pub fn classify(&self, features: &WorkloadFeatures) -> ClassificationResult {
        let mut best_category = WorkloadCategory::Unknown;
        let mut best_distance = f64::MAX;
        let mut second_best_distance = f64::MAX;

        for (category, prototype) in &self.prototypes {
            let distance = features.distance(prototype);
            if distance < best_distance {
                second_best_distance = best_distance;
                best_distance = distance;
                best_category = *category;
            } else if distance < second_best_distance {
                second_best_distance = distance;
            }
        }

        // Confidence based on distance ratio
        let confidence = if second_best_distance > 1e-10 {
            (1.0 - best_distance / second_best_distance).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Recommend backend
        let recommended_backend = self.recommend_backend(best_category, features.memory_footprint);

        // Get GPU crossover threshold
        let gpu_crossover_size = self
            .gpu_thresholds
            .iter()
            .find(|(c, _)| *c == best_category)
            .map(|(_, t)| *t);

        ClassificationResult {
            category: best_category,
            confidence,
            distance: best_distance,
            recommended_backend,
            gpu_crossover_size,
        }
    }

    /// Compute similarity between two workloads
    pub fn workload_similarity(&self, a: &WorkloadFeatures, b: &WorkloadFeatures) -> f64 {
        // Normalize cosine similarity to 0-1 range
        (a.cosine_similarity(b) + 1.0) / 2.0
    }

    /// Recommend backend for workload
    pub fn recommend_backend(&self, category: WorkloadCategory, size: usize) -> RecommendedBackend {
        let threshold = self
            .gpu_thresholds
            .iter()
            .find(|(c, _)| *c == category)
            .map(|(_, t)| *t)
            .unwrap_or(100_000);

        if size < threshold / 2 {
            RecommendedBackend::CpuSimd
        } else if size > threshold * 2 {
            RecommendedBackend::Gpu
        } else {
            RecommendedBackend::Either
        }
    }

    /// Predict GPU crossover size
    pub fn predict_crossover(&self, category: WorkloadCategory) -> Option<usize> {
        self.gpu_thresholds
            .iter()
            .find(|(c, _)| *c == category)
            .map(|(_, t)| *t)
    }

    /// Add custom prototype
    pub fn add_prototype(&mut self, category: WorkloadCategory, features: WorkloadFeatures) {
        self.prototypes.push((category, features));
    }

    /// Get all prototypes
    pub fn get_prototypes(&self) -> &[(WorkloadCategory, WorkloadFeatures)] {
        &self.prototypes
    }
}

#[cfg(test)]
mod tests;
