//! Benchmark result types
//!
//! Data structures for quantization benchmark results.

use super::super::granularity::{QuantGranularity, QuantMode};
use serde::{Deserialize, Serialize};

/// Benchmark results for quantization accuracy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantBenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of elements tested
    pub num_elements: usize,
    /// Bits used for quantization
    pub bits: u8,
    /// Granularity used
    pub granularity: QuantGranularity,
    /// Mode used (symmetric/asymmetric)
    pub mode: QuantMode,
    /// MSE error
    pub mse: f32,
    /// Max error
    pub max_error: f32,
    /// SQNR in dB
    pub sqnr_db: f32,
    /// Compression ratio
    pub compression_ratio: f32,
}

impl QuantBenchmarkResult {
    /// Quality score (higher is better): SQNR / compression overhead
    pub fn quality_score(&self) -> f32 {
        if self.compression_ratio > 0.0 {
            self.sqnr_db / self.compression_ratio.max(1.0)
        } else {
            0.0
        }
    }
}

/// Suite of benchmark results
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct BenchmarkSuite {
    pub results: Vec<QuantBenchmarkResult>,
}

impl BenchmarkSuite {
    /// Add a benchmark result
    pub fn add(&mut self, result: QuantBenchmarkResult) {
        self.results.push(result);
    }

    /// Get best result by SQNR
    pub fn best_by_sqnr(&self) -> Option<&QuantBenchmarkResult> {
        self.results.iter().max_by(|a, b| a.sqnr_db.total_cmp(&b.sqnr_db))
    }

    /// Get best result by MSE (lowest)
    pub fn best_by_mse(&self) -> Option<&QuantBenchmarkResult> {
        self.results.iter().min_by(|a, b| a.mse.total_cmp(&b.mse))
    }

    /// Get results sorted by quality score
    pub fn sorted_by_quality(&self) -> Vec<&QuantBenchmarkResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| b.quality_score().total_cmp(&a.quality_score()));
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, sqnr: f32, mse: f32, compression: f32) -> QuantBenchmarkResult {
        QuantBenchmarkResult {
            name: name.to_string(),
            num_elements: 1000,
            bits: 8,
            granularity: QuantGranularity::PerTensor,
            mode: QuantMode::Symmetric,
            mse,
            max_error: mse * 2.0,
            sqnr_db: sqnr,
            compression_ratio: compression,
        }
    }

    #[test]
    fn test_quality_score_normal() {
        let result = make_result("test", 40.0, 0.01, 4.0);
        assert!((result.quality_score() - 10.0).abs() < 1e-6); // 40 / 4 = 10
    }

    #[test]
    fn test_quality_score_zero_compression() {
        let result = make_result("test", 40.0, 0.01, 0.0);
        assert_eq!(result.quality_score(), 0.0);
    }

    #[test]
    fn test_quality_score_low_compression() {
        let result = make_result("test", 40.0, 0.01, 0.5);
        // compression_ratio.max(1.0) = 1.0, so 40 / 1 = 40
        assert!((result.quality_score() - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_benchmark_suite_default() {
        let suite = BenchmarkSuite::default();
        assert!(suite.results.is_empty());
    }

    #[test]
    fn test_benchmark_suite_add() {
        let mut suite = BenchmarkSuite::default();
        suite.add(make_result("test1", 40.0, 0.01, 4.0));
        assert_eq!(suite.results.len(), 1);
        suite.add(make_result("test2", 50.0, 0.005, 4.0));
        assert_eq!(suite.results.len(), 2);
    }

    #[test]
    fn test_best_by_sqnr() {
        let mut suite = BenchmarkSuite::default();
        suite.add(make_result("low", 30.0, 0.02, 4.0));
        suite.add(make_result("high", 50.0, 0.01, 4.0));
        suite.add(make_result("mid", 40.0, 0.015, 4.0));

        let best = suite.best_by_sqnr().expect("operation should succeed");
        assert_eq!(best.name, "high");
        assert!((best.sqnr_db - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_best_by_sqnr_empty() {
        let suite = BenchmarkSuite::default();
        assert!(suite.best_by_sqnr().is_none());
    }

    #[test]
    fn test_best_by_mse() {
        let mut suite = BenchmarkSuite::default();
        suite.add(make_result("high_error", 30.0, 0.02, 4.0));
        suite.add(make_result("low_error", 50.0, 0.005, 4.0));
        suite.add(make_result("mid_error", 40.0, 0.01, 4.0));

        let best = suite.best_by_mse().expect("operation should succeed");
        assert_eq!(best.name, "low_error");
        assert!((best.mse - 0.005).abs() < 1e-6);
    }

    #[test]
    fn test_best_by_mse_empty() {
        let suite = BenchmarkSuite::default();
        assert!(suite.best_by_mse().is_none());
    }

    #[test]
    fn test_sorted_by_quality() {
        let mut suite = BenchmarkSuite::default();
        suite.add(make_result("low_quality", 20.0, 0.02, 4.0)); // 20/4 = 5
        suite.add(make_result("high_quality", 60.0, 0.01, 4.0)); // 60/4 = 15
        suite.add(make_result("mid_quality", 40.0, 0.015, 4.0)); // 40/4 = 10

        let sorted = suite.sorted_by_quality();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].name, "high_quality");
        assert_eq!(sorted[1].name, "mid_quality");
        assert_eq!(sorted[2].name, "low_quality");
    }

    #[test]
    fn test_sorted_by_quality_empty() {
        let suite = BenchmarkSuite::default();
        let sorted = suite.sorted_by_quality();
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_quant_benchmark_result_serde() {
        let result = make_result("test", 40.0, 0.01, 4.0);
        let json = serde_json::to_string(&result).expect("JSON serialization should succeed");
        let deserialized: QuantBenchmarkResult =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(result.name, deserialized.name);
        assert!((result.sqnr_db - deserialized.sqnr_db).abs() < 1e-6);
    }

    #[test]
    fn test_benchmark_suite_serde() {
        let mut suite = BenchmarkSuite::default();
        suite.add(make_result("test1", 40.0, 0.01, 4.0));
        suite.add(make_result("test2", 50.0, 0.005, 4.0));

        let json = serde_json::to_string(&suite).expect("JSON serialization should succeed");
        let deserialized: BenchmarkSuite =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(suite.results.len(), deserialized.results.len());
    }
}
