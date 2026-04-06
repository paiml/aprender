//! HTTP Client Tests
//!
//! Split into parts for pmat compliance (<2000 lines per file).
//!
//! Part organization:
//! - part_01: Unit tests, Integration tests, Benchmark runner tests, IMP-144-152
//! - part_02: IMP-153 through IMP-158 (Performance tracking, Gate validation)
//! - part_03: IMP-159 through IMP-161 (Statistical analysis)
//! - part_04: IMP-162 through IMP-164 (More statistical tests)
//! - part_05: IMP-165 through IMP-168 (Benchmark variants)
//! - part_06: IMP-169 through IMP-173 (Coverage tests)
//! - part_07: IMP-174 through IMP-179 (More coverage tests)
//! - part_08: IMP-180 through IMP-189 (Benchmark infrastructure)
//! - part_09: IMP-190 through IMP-198 (Benchmark variants)
//! - part_10: IMP-199 through IMP-207 (APR GPU, QA tests)
//! - part_11: IMP-208 through IMP-305 (Activation, KV cache, Trueno SIMD)
//! - part_12: IMP-306 through IMP-400 (wgpu GPU, E2E comparison)

mod client_creation_custom;
mod performance_trend_entry;
mod standard_benchmark;
mod mad_outlier;
mod cold_start;
mod quantized_quality;
mod timeout_gpu;
mod success_thread_safety;
mod benchmark_version;
mod model_format_comparison;
mod softmax_verification;
mod check_imp306_gpu;
