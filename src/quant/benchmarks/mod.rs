//! Accuracy Degradation Benchmarks
//!
//! Provides benchmarks for measuring quantization accuracy degradation:
//! - Synthetic workload benchmarks
//! - Bit-width comparison (4-bit vs 8-bit)
//! - Granularity comparison (per-tensor vs per-channel vs per-group)
//! - Model-like weight pattern tests
//! - Numerical precision edge cases

mod generators;
mod runners;
mod types;

#[cfg(test)]
mod tests;

// Re-export types
pub use types::{BenchmarkSuite, QuantBenchmarkResult};

// Re-export generators
pub use generators::{
    generate_gaussian_weights, generate_multi_channel_weights, generate_uniform_weights,
    generate_weights_with_outliers,
};

// Re-export runners
pub use runners::{
    accuracy_retention, compare_bit_width_degradation, run_benchmark, run_full_benchmark_suite,
};
