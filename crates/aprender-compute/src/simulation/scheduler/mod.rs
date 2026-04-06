//! Heijunka Scheduler (Leveled Testing)
//!
//! Implements Toyota Production System's Heijunka principle:
//! level the workload to reduce waste and variability.

use crate::Backend;
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Backend-specific tolerance configuration
///
/// Implements Poka-Yoke (mistake-proofing) by providing compile-time
/// guarantees for correct tolerance values per backend type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackendTolerance {
    /// Scalar vs SIMD tolerance (should be exact: 0.0)
    pub scalar_vs_simd: f32,
    /// SIMD vs GPU tolerance (IEEE 754: 1e-5)
    pub simd_vs_gpu: f32,
    /// GPU vs GPU tolerance (same precision: 1e-6)
    pub gpu_vs_gpu: f32,
}

impl Default for BackendTolerance {
    fn default() -> Self {
        Self { scalar_vs_simd: 0.0, simd_vs_gpu: 1e-5, gpu_vs_gpu: 1e-6 }
    }
}

impl BackendTolerance {
    /// Strict tolerance for exact comparisons
    #[must_use]
    pub const fn strict() -> Self {
        Self { scalar_vs_simd: 0.0, simd_vs_gpu: 0.0, gpu_vs_gpu: 0.0 }
    }

    /// Relaxed tolerance for approximate comparisons
    #[must_use]
    pub const fn relaxed() -> Self {
        Self { scalar_vs_simd: 1e-6, simd_vs_gpu: 1e-4, gpu_vs_gpu: 1e-5 }
    }

    /// Get tolerance for comparing two backends
    #[must_use]
    pub fn for_backends(&self, a: Backend, b: Backend) -> f32 {
        match (a, b) {
            (Backend::Scalar, Backend::Scalar) => 0.0,
            (
                Backend::Scalar,
                Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
                | Backend::Auto,
            )
            | (
                Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
                | Backend::Auto,
                Backend::Scalar,
            ) => self.scalar_vs_simd,
            (Backend::GPU, Backend::GPU) => self.gpu_vs_gpu,
            (
                Backend::GPU,
                Backend::Scalar
                | Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
                | Backend::Auto,
            )
            | (
                Backend::Scalar
                | Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
                | Backend::Auto,
                Backend::GPU,
            ) => self.simd_vs_gpu,
            // SIMD vs SIMD (all remaining non-Scalar, non-GPU combinations)
            (
                Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
                | Backend::Auto,
                Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
                | Backend::Auto,
            ) => self.scalar_vs_simd,
        }
    }
}

/// Poka-Yoke: Type-safe backend selection
///
/// Provides compile-time and runtime guarantees for correct backend selection
/// based on input size and operation type.
#[derive(Debug, Clone)]
pub struct BackendSelector {
    /// Minimum size for GPU offload (default: 100,000)
    gpu_threshold: usize,
    /// Minimum size for parallel execution (default: 1,000)
    parallel_threshold: usize,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self { gpu_threshold: 100_000, parallel_threshold: 1_000 }
    }
}

impl BackendSelector {
    /// Create a new backend selector with custom thresholds
    #[must_use]
    pub const fn new(gpu_threshold: usize, parallel_threshold: usize) -> Self {
        Self { gpu_threshold, parallel_threshold }
    }

    /// Get the GPU threshold
    #[must_use]
    pub const fn gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    /// Get the parallel threshold
    #[must_use]
    pub const fn parallel_threshold(&self) -> usize {
        self.parallel_threshold
    }

    /// Select backend based on input size
    ///
    /// # Decision Logic (TRUENO-SPEC-012)
    ///
    /// - N < 1,000: Pure SIMD (no parallelization overhead)
    /// - 1,000 <= N < 100,000: SIMD + Parallel (Rayon)
    /// - N >= 100,000: GPU (if available), else SIMD + Parallel
    #[must_use]
    pub fn select_for_size(&self, size: usize, gpu_available: bool) -> BackendCategory {
        if size < self.parallel_threshold {
            BackendCategory::SimdOnly
        } else if size < self.gpu_threshold {
            BackendCategory::SimdParallel
        } else if gpu_available {
            BackendCategory::Gpu
        } else {
            BackendCategory::SimdParallel // Graceful fallback
        }
    }

    /// Check if size is at GPU threshold boundary (for testing)
    #[must_use]
    pub fn is_at_gpu_boundary(&self, size: usize) -> bool {
        size == self.gpu_threshold || size == self.gpu_threshold - 1
    }

    /// Check if size is at parallel threshold boundary (for testing)
    #[must_use]
    pub fn is_at_parallel_boundary(&self, size: usize) -> bool {
        size == self.parallel_threshold || size == self.parallel_threshold - 1
    }
}

/// Backend category for selection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCategory {
    /// Pure SIMD (N < 1,000)
    SimdOnly,
    /// SIMD with parallel execution (1,000 <= N < 100,000)
    SimdParallel,
    /// GPU compute (N >= 100,000)
    Gpu,
}

/// Simulation test configuration
#[derive(Debug, Clone)]
pub struct SimulationTest {
    /// Backend to test
    pub backend: Backend,
    /// Input size
    pub input_size: usize,
    /// Test cycle number
    pub cycle: u32,
    /// Seed for deterministic RNG
    pub seed: u64,
}

/// Heijunka: Balanced test distribution across backends and sizes
///
/// Implements Toyota Production System's Heijunka principle:
/// level the workload to reduce waste and variability.
#[derive(Debug)]
pub struct HeijunkaScheduler {
    /// Test queue balanced across backends
    queue: VecDeque<SimulationTest>,
    /// Backends to cycle through
    backends: Vec<Backend>,
}

impl HeijunkaScheduler {
    /// Create a leveled test schedule
    #[must_use]
    pub fn new(
        backends: Vec<Backend>,
        input_sizes: Vec<usize>,
        cycles_per_backend: u32,
        master_seed: u64,
    ) -> Self {
        let mut queue = VecDeque::new();

        // Interleave tests across backends (leveling)
        for size in &input_sizes {
            for backend in &backends {
                for cycle in 0..cycles_per_backend {
                    let seed = compute_seed(*backend, *size, cycle, master_seed);
                    queue.push_back(SimulationTest {
                        backend: *backend,
                        input_size: *size,
                        cycle,
                        seed,
                    });
                }
            }
        }

        Self { queue, backends: backends.clone() }
    }

    /// Get the next test from the queue
    pub fn next_test(&mut self) -> Option<SimulationTest> {
        self.queue.pop_front()
    }

    /// Get remaining test count
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }

    /// Get backends being tested
    #[must_use]
    pub fn backends(&self) -> &[Backend] {
        &self.backends
    }

    /// Check if schedule is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Compute deterministic seed for a test configuration
pub(crate) fn compute_seed(backend: Backend, size: usize, cycle: u32, master_seed: u64) -> u64 {
    let backend_bits = backend as u64;
    let size_bits = size as u64;
    let cycle_bits = u64::from(cycle);

    master_seed
        .wrapping_add(backend_bits.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(size_bits.wrapping_mul(0x6A09_E667_BB67_AE85))
        .wrapping_add(cycle_bits.wrapping_mul(0x3C6E_F372_FE94_F82B))
}

/// Simulation test configuration builder
#[derive(Debug, Clone)]
pub struct SimTestConfigBuilder<S> {
    seed: u64,
    tolerance: BackendTolerance,
    backends: Vec<Backend>,
    input_sizes: Vec<usize>,
    cycles: u32,
    _state: PhantomData<S>,
}

/// Builder state: seed not set
pub struct NeedsSeed;
/// Builder state: ready to build
pub struct Ready;

impl Default for SimTestConfigBuilder<NeedsSeed> {
    fn default() -> Self {
        Self::new()
    }
}

impl SimTestConfigBuilder<NeedsSeed> {
    /// Create a new config builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: 0,
            tolerance: BackendTolerance::default(),
            backends: vec![Backend::Scalar, Backend::AVX2],
            input_sizes: vec![100, 1_000, 10_000, 100_000],
            cycles: 10,
            _state: PhantomData,
        }
    }

    /// Set the master seed (required)
    #[must_use]
    pub fn seed(self, seed: u64) -> SimTestConfigBuilder<Ready> {
        SimTestConfigBuilder {
            seed,
            tolerance: self.tolerance,
            backends: self.backends,
            input_sizes: self.input_sizes,
            cycles: self.cycles,
            _state: PhantomData,
        }
    }
}

impl SimTestConfigBuilder<Ready> {
    /// Set tolerance configuration
    #[must_use]
    pub fn tolerance(mut self, tolerance: BackendTolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set backends to test
    #[must_use]
    pub fn backends(mut self, backends: Vec<Backend>) -> Self {
        self.backends = backends;
        self
    }

    /// Set input sizes to test
    #[must_use]
    pub fn input_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.input_sizes = sizes;
        self
    }

    /// Set number of test cycles
    #[must_use]
    pub fn cycles(mut self, cycles: u32) -> Self {
        self.cycles = cycles;
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> SimTestConfig {
        SimTestConfig {
            seed: self.seed,
            tolerance: self.tolerance,
            backends: self.backends,
            input_sizes: self.input_sizes,
            cycles: self.cycles,
        }
    }
}

/// Simulation test configuration
#[derive(Debug, Clone)]
pub struct SimTestConfig {
    /// Master seed for deterministic RNG
    pub seed: u64,
    /// Backend tolerance configuration
    pub tolerance: BackendTolerance,
    /// Backends to test
    pub backends: Vec<Backend>,
    /// Input sizes to test
    pub input_sizes: Vec<usize>,
    /// Number of test cycles
    pub cycles: u32,
}

impl SimTestConfig {
    /// Create a config builder
    #[must_use]
    pub fn builder() -> SimTestConfigBuilder<NeedsSeed> {
        SimTestConfigBuilder::new()
    }

    /// Create a Heijunka scheduler from this config
    #[must_use]
    pub fn create_scheduler(&self) -> HeijunkaScheduler {
        HeijunkaScheduler::new(
            self.backends.clone(),
            self.input_sizes.clone(),
            self.cycles,
            self.seed,
        )
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;
