//! Cache Efficiency Analysis Module (PMAT-025)
//!
//! Implements cache efficiency analysis for L1/L2/L3 cache behavior prediction
//! and optimization recommendations based on working set size.
//!
//! # Motivation
//!
//! §31.2 identifies memory bandwidth cliff at 4M elements (32MB) due to L3 overflow.
//! This module predicts and recommends optimal problem sizes.
//!
//! # Components
//!
//! | Component | Description | Use Case |
//! |-----------|-------------|----------|
//! | Working Set Estimator | Bytes = elements × sizeof(T) × factor | Predict cache fit |
//! | Cache Level Classifier | L1/L2/L3/RAM based on size | Identify bottleneck |
//! | Tiling Recommender | Optimal tile size for cache | Loop blocking advice |
//! | Bandwidth Estimator | Theoretical vs achieved BW | Efficiency score |

/// Cache level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// L1 data cache (typically 32KB-64KB per core)
    L1,
    /// L2 cache (typically 256KB-1MB per core)
    L2,
    /// L3 cache (typically 4MB-64MB shared)
    L3,
    /// Main memory (RAM)
    Ram,
}

impl CacheLevel {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            CacheLevel::L1 => "L1 cache",
            CacheLevel::L2 => "L2 cache",
            CacheLevel::L3 => "L3 cache",
            CacheLevel::Ram => "main memory",
        }
    }

    /// Get typical latency in CPU cycles
    pub fn typical_latency_cycles(&self) -> u32 {
        match self {
            CacheLevel::L1 => 4,
            CacheLevel::L2 => 12,
            CacheLevel::L3 => 40,
            CacheLevel::Ram => 200,
        }
    }

    /// Get typical bandwidth relative to L1 (as fraction)
    pub fn relative_bandwidth(&self) -> f64 {
        match self {
            CacheLevel::L1 => 1.0,
            CacheLevel::L2 => 0.8,
            CacheLevel::L3 => 0.5,
            CacheLevel::Ram => 0.1,
        }
    }
}

/// Cache hierarchy configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// L1 data cache size in bytes per core
    pub l1_size: usize,
    /// L2 cache size in bytes per core
    pub l2_size: usize,
    /// L3 cache size in bytes (shared)
    pub l3_size: usize,
    /// Number of cores sharing L3
    pub l3_sharing: usize,
    /// Cache line size in bytes
    pub line_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 32 * 1024,        // 32KB
            l2_size: 512 * 1024,       // 512KB
            l3_size: 32 * 1024 * 1024, // 32MB
            l3_sharing: 8,             // 8 cores share L3
            line_size: 64,             // 64 bytes
        }
    }
}

impl CacheConfig {
    /// Create config for AMD Zen4 (Ryzen 7000 series)
    pub fn zen4() -> Self {
        Self {
            l1_size: 32 * 1024,        // 32KB per core
            l2_size: 1024 * 1024,      // 1MB per core
            l3_size: 32 * 1024 * 1024, // 32MB per CCD
            l3_sharing: 8,
            line_size: 64,
        }
    }

    /// Create config for Intel Sapphire Rapids
    pub fn sapphire_rapids() -> Self {
        Self {
            l1_size: 48 * 1024,        // 48KB per core
            l2_size: 2 * 1024 * 1024,  // 2MB per core
            l3_size: 60 * 1024 * 1024, // 60MB shared
            l3_sharing: 16,
            line_size: 64,
        }
    }

    /// Create config for Apple M2
    pub fn apple_m2() -> Self {
        Self {
            l1_size: 128 * 1024,       // 128KB per P-core
            l2_size: 16 * 1024 * 1024, // 16MB shared L2
            l3_size: 0,                // No L3
            l3_sharing: 1,
            line_size: 128, // 128-byte cache lines
        }
    }

    /// Classify working set size to cache level
    pub fn classify(&self, working_set_bytes: usize) -> CacheLevel {
        if working_set_bytes <= self.l1_size * 3 / 4 {
            CacheLevel::L1
        } else if working_set_bytes <= self.l2_size * 3 / 4 {
            CacheLevel::L2
        } else if self.l3_size > 0 && working_set_bytes <= self.l3_size * 3 / 4 {
            CacheLevel::L3
        } else {
            CacheLevel::Ram
        }
    }

    /// Get effective L3 size per core (accounting for sharing)
    pub fn l3_per_core(&self) -> usize {
        if self.l3_sharing > 0 {
            self.l3_size / self.l3_sharing
        } else {
            self.l3_size
        }
    }
}

/// Working set analysis result
#[derive(Debug, Clone)]
pub struct WorkingSetAnalysis {
    /// Total working set in bytes
    pub working_set_bytes: usize,
    /// Cache level where working set fits
    pub cache_level: CacheLevel,
    /// Cache utilization percentage (working_set / cache_size * 100)
    pub utilization_percent: f64,
    /// Expected bandwidth efficiency (relative to peak)
    pub expected_efficiency: f64,
    /// Whether tiling is recommended
    pub tiling_recommended: bool,
    /// Recommended tile size if tiling is recommended
    pub recommended_tile_bytes: Option<usize>,
}

impl WorkingSetAnalysis {
    /// Analyze working set against cache hierarchy
    pub fn analyze(
        elements: usize,
        element_size: usize,
        access_factor: f64,
        config: &CacheConfig,
    ) -> Self {
        let working_set_bytes = (elements as f64 * element_size as f64 * access_factor) as usize;
        let cache_level = config.classify(working_set_bytes);

        let (utilization_percent, _cache_size) = match cache_level {
            CacheLevel::L1 => (
                (working_set_bytes as f64 / config.l1_size as f64) * 100.0,
                config.l1_size,
            ),
            CacheLevel::L2 => (
                (working_set_bytes as f64 / config.l2_size as f64) * 100.0,
                config.l2_size,
            ),
            CacheLevel::L3 => (
                (working_set_bytes as f64 / config.l3_size as f64) * 100.0,
                config.l3_size,
            ),
            CacheLevel::Ram => (100.0, working_set_bytes),
        };

        let expected_efficiency = cache_level.relative_bandwidth();

        // Recommend tiling if working set exceeds L2
        let tiling_recommended = working_set_bytes > config.l2_size;

        let recommended_tile_bytes = if tiling_recommended {
            // Recommend tile size that fits 75% of L2
            Some(config.l2_size * 3 / 4)
        } else {
            None
        };

        Self {
            working_set_bytes,
            cache_level,
            utilization_percent,
            expected_efficiency,
            tiling_recommended,
            recommended_tile_bytes,
        }
    }

    /// Get recommendation string
    pub fn recommendation(&self) -> String {
        if self.tiling_recommended {
            format!(
                "Working set ({} bytes) exceeds L2. Recommend tiling with {} byte tiles for {} cache.",
                self.working_set_bytes,
                self.recommended_tile_bytes.unwrap_or(0),
                CacheLevel::L2.name()
            )
        } else {
            format!(
                "Working set ({} bytes) fits in {}. No tiling needed.",
                self.working_set_bytes,
                self.cache_level.name()
            )
        }
    }
}

/// Calculate working set size for matrix operations
pub fn matrix_working_set(m: usize, n: usize, k: usize, element_size: usize) -> usize {
    // For C = A × B: A is m×k, B is k×n, C is m×n
    let a_size = m * k * element_size;
    let b_size = k * n * element_size;
    let c_size = m * n * element_size;
    a_size + b_size + c_size
}

/// Calculate optimal tile size for matrix multiply
pub fn optimal_matmul_tile(config: &CacheConfig, element_size: usize) -> usize {
    // For tiled matmul, need 3 tiles: A_tile, B_tile, C_tile
    // Each tile is tile_size × tile_size
    // Total: 3 × tile_size² × element_size ≤ L2 × 0.75
    let target_bytes = config.l2_size * 3 / 4;
    let max_tile_elements = target_bytes / (3 * element_size);
    let tile_size = (max_tile_elements as f64).sqrt() as usize;

    // Round down to multiple of cache line for alignment
    let elements_per_line = config.line_size / element_size;
    (tile_size / elements_per_line) * elements_per_line
}

/// Calculate working set for elementwise operations
pub fn elementwise_working_set(
    elements: usize,
    inputs: usize,
    outputs: usize,
    element_size: usize,
) -> usize {
    elements * (inputs + outputs) * element_size
}

/// Streaming vs reuse pattern detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Data accessed once and discarded (streaming)
    Streaming,
    /// Data reused multiple times (cache-friendly)
    Reuse,
    /// Random access pattern (cache-unfriendly)
    Random,
}

impl AccessPattern {
    /// Estimate based on working set vs cache size
    pub fn estimate(working_set: usize, iterations: usize, config: &CacheConfig) -> Self {
        if iterations == 1 {
            AccessPattern::Streaming
        } else if working_set <= config.l2_size {
            AccessPattern::Reuse
        } else {
            AccessPattern::Random
        }
    }

    /// Get name
    pub fn name(&self) -> &'static str {
        match self {
            AccessPattern::Streaming => "streaming",
            AccessPattern::Reuse => "reuse",
            AccessPattern::Random => "random",
        }
    }

    /// Get expected efficiency multiplier
    pub fn efficiency_factor(&self) -> f64 {
        match self {
            AccessPattern::Streaming => 0.5, // 50% - prefetching helps
            AccessPattern::Reuse => 1.0,     // 100% - cache hits
            AccessPattern::Random => 0.1,    // 10% - cache misses
        }
    }
}

/// Bandwidth prediction result
#[derive(Debug, Clone)]
pub struct BandwidthPrediction {
    /// Peak theoretical bandwidth (GB/s)
    pub peak_bandwidth_gbps: f64,
    /// Predicted achievable bandwidth (GB/s)
    pub predicted_bandwidth_gbps: f64,
    /// Efficiency percentage
    pub efficiency_percent: f64,
    /// Limiting factor description
    pub limiting_factor: String,
}

impl BandwidthPrediction {
    /// Predict bandwidth for given access pattern
    pub fn predict(
        peak_bandwidth_gbps: f64,
        working_set: usize,
        access_pattern: AccessPattern,
        config: &CacheConfig,
    ) -> Self {
        let cache_level = config.classify(working_set);
        let cache_efficiency = cache_level.relative_bandwidth();
        let pattern_efficiency = access_pattern.efficiency_factor();

        let overall_efficiency = cache_efficiency * pattern_efficiency;
        let predicted_bandwidth_gbps = peak_bandwidth_gbps * overall_efficiency;

        let limiting_factor = if pattern_efficiency < cache_efficiency {
            format!("{} access pattern", access_pattern.name())
        } else {
            format!("{} bandwidth", cache_level.name())
        };

        Self {
            peak_bandwidth_gbps,
            predicted_bandwidth_gbps,
            efficiency_percent: overall_efficiency * 100.0,
            limiting_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_level_classification() {
        let config = CacheConfig::default();

        assert_eq!(config.classify(1024), CacheLevel::L1);
        assert_eq!(config.classify(100 * 1024), CacheLevel::L2);
        assert_eq!(config.classify(10 * 1024 * 1024), CacheLevel::L3);
        assert_eq!(config.classify(100 * 1024 * 1024), CacheLevel::Ram);
    }

    #[test]
    fn test_working_set_analysis() {
        let config = CacheConfig::default();
        let analysis = WorkingSetAnalysis::analyze(1000, 4, 2.0, &config);

        // 1000 elements × 4 bytes × 2.0 factor = 8KB → fits in L1
        assert_eq!(analysis.cache_level, CacheLevel::L1);
        assert!(!analysis.tiling_recommended);
    }

    #[test]
    fn test_matrix_working_set() {
        let ws = matrix_working_set(1024, 1024, 1024, 4);
        // 3 matrices × 1024² × 4 = 12MB
        assert_eq!(ws, 3 * 1024 * 1024 * 4);
    }

    #[test]
    fn test_optimal_tile_size() {
        let config = CacheConfig::default();
        let tile = optimal_matmul_tile(&config, 4);

        // Tile should be reasonable size
        assert!(tile > 0);
        assert!(tile <= 512); // Should be ≤ 512 for typical cache
    }

    #[test]
    fn test_access_pattern() {
        let config = CacheConfig::default();

        assert_eq!(
            AccessPattern::estimate(1024, 1, &config),
            AccessPattern::Streaming
        );
        assert_eq!(
            AccessPattern::estimate(1024, 10, &config),
            AccessPattern::Reuse
        );
        assert_eq!(
            AccessPattern::estimate(100 * 1024 * 1024, 10, &config),
            AccessPattern::Random
        );
    }

    #[test]
    fn test_bandwidth_prediction() {
        let config = CacheConfig::default();
        let prediction = BandwidthPrediction::predict(
            100.0, // 100 GB/s peak
            1024,  // 1KB working set
            AccessPattern::Reuse,
            &config,
        );

        // L1 with reuse should be near peak
        assert!(prediction.efficiency_percent > 90.0);
    }
}
