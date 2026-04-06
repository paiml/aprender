//! Cost-Performance Benchmarking (ENT-011)
//!
//! Provides benchmarking infrastructure with Pareto frontier analysis
//! for cost-performance optimization.

mod collection;
mod entry;
mod statistics;

pub use collection::CostPerformanceBenchmark;
pub use entry::BenchmarkEntry;
pub use statistics::BenchmarkStatistics;

// Re-export types needed by this module's public API

#[cfg(test)]
mod tests;
