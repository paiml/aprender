//! GPU parity benchmarks for comparing Realizar vs Ollama/llama.cpp
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - IMP-800: TRUE GPU Parity Benchmark (M2 Milestone)
//! - IMP-900: Closing the 18x Gap (M3/M4 Milestones)

use serde::{Deserialize, Serialize};

// ============================================================================
// IMP-800: TRUE GPU Parity Benchmark (M2 Milestone)
// ============================================================================

/// GPU parity benchmark configuration (IMP-800b)
///
/// Configures apples-to-apples throughput comparison on same GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuParityBenchmark {
    /// Model to benchmark (phi-2 Q4_K_M)
    pub model_path: String,
    /// Prompt for generation
    pub prompt: String,
    /// Number of tokens to generate
    pub max_tokens: usize,
    /// Ollama endpoint for comparison
    pub ollama_endpoint: String,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Target CV for stable measurements
    pub target_cv: f64,
}

impl Default for GpuParityBenchmark {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "The quick brown fox".to_string(),
            max_tokens: 32,
            ollama_endpoint: "http://localhost:11434".to_string(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            target_cv: 0.05,
        }
    }
}

impl GpuParityBenchmark {
    /// Create a new GPU parity benchmark with model path
    #[must_use]
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the prompt for generation
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Set the number of tokens to generate
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the Ollama endpoint
    #[must_use]
    pub fn with_ollama_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.ollama_endpoint = endpoint.into();
        self
    }

    /// Set the number of warmup iterations
    #[must_use]
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup_iterations = warmup;
        self
    }

    /// Set the number of measurement iterations
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }
}

/// Benchmark result with statistical analysis (IMP-800b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuParityResult {
    /// Realizar GPU throughput (tok/s)
    pub realizar_gpu_tps: f64,
    /// Ollama throughput (tok/s)
    pub ollama_tps: f64,
    /// Performance gap ratio (Ollama / Realizar)
    pub gap_ratio: f64,
    /// Coefficient of variation (measurement stability)
    pub cv: f64,
    /// GPU device name
    pub gpu_device: String,
    /// VRAM usage (MB)
    pub vram_mb: u64,
    /// Realizar latency p50 (ms)
    pub realizar_p50_ms: f64,
    /// Ollama latency p50 (ms)
    pub ollama_p50_ms: f64,
}

impl GpuParityResult {
    /// Create a new GPU parity result
    #[must_use]
    pub fn new(
        realizar_gpu_tps: f64,
        ollama_tps: f64,
        cv: f64,
        gpu_device: impl Into<String>,
        vram_mb: u64,
    ) -> Self {
        let gap_ratio = if realizar_gpu_tps > 0.0 {
            ollama_tps / realizar_gpu_tps
        } else {
            f64::INFINITY
        };

        Self {
            realizar_gpu_tps,
            ollama_tps,
            gap_ratio,
            cv,
            gpu_device: gpu_device.into(),
            vram_mb,
            realizar_p50_ms: 0.0,
            ollama_p50_ms: 0.0,
        }
    }

    /// Returns true if within 2x of Ollama (M2 target)
    #[must_use]
    pub fn achieves_m2_parity(&self) -> bool {
        self.gap_ratio <= 2.0
    }

    /// Returns true if within 1.25x of Ollama (M4 target)
    #[must_use]
    pub fn achieves_m4_parity(&self) -> bool {
        self.gap_ratio <= 1.25
    }

    /// Returns true if GPU is faster than CPU SIMD baseline (5 tok/s)
    #[must_use]
    pub fn gpu_faster_than_cpu(&self) -> bool {
        self.realizar_gpu_tps > 5.0
    }

    /// Returns true if measurements are stable (CV < 0.05)
    #[must_use]
    pub fn measurements_stable(&self) -> bool {
        self.cv < 0.05
    }

    /// Get speedup over CPU SIMD baseline
    #[must_use]
    pub fn cpu_speedup(&self) -> f64 {
        self.realizar_gpu_tps / 5.0 // CPU baseline ~5 tok/s
    }
}

/// Gap analysis with falsifiable claims (IMP-800c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysis {
    /// Claimed gap reduction
    pub claimed_gap: f64,
    /// Measured gap
    pub measured_gap: f64,
    /// Statistical significance (p-value)
    pub p_value: f64,
    /// Confidence interval lower bound (95%)
    pub ci_95_lower: f64,
    /// Confidence interval upper bound (95%)
    pub ci_95_upper: f64,
    /// Popper score (falsifiability, 0-100)
    pub popper_score: f64,
    /// Claim descriptions
    pub claims: Vec<FalsifiableClaim>,
}

/// A falsifiable claim for Popperian testing (IMP-800c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsifiableClaim {
    /// Claim identifier
    pub id: String,
    /// Claim description
    pub description: String,
    /// Expected value
    pub expected: f64,
    /// Threshold for verification
    pub threshold: f64,
    /// Measured value
    pub measured: f64,
    /// Whether claim is verified
    pub verified: bool,
}

impl FalsifiableClaim {
    /// Create a new falsifiable claim
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        expected: f64,
        threshold: f64,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            expected,
            threshold,
            measured: 0.0,
            verified: false,
        }
    }

    /// Evaluate the claim against a measured value
    #[must_use]
    pub fn evaluate(mut self, measured: f64) -> Self {
        self.measured = measured;
        self.verified = measured >= self.threshold;
        self
    }
}

impl GapAnalysis {
    /// Create a new gap analysis
    #[must_use]
    pub fn new(claimed_gap: f64, measured_gap: f64) -> Self {
        Self {
            claimed_gap,
            measured_gap,
            p_value: 0.0,
            ci_95_lower: 0.0,
            ci_95_upper: 0.0,
            popper_score: 0.0,
            claims: Vec::new(),
        }
    }

    /// Add statistical bounds
    #[must_use]
    pub fn with_statistics(mut self, p_value: f64, ci_lower: f64, ci_upper: f64) -> Self {
        self.p_value = p_value;
        self.ci_95_lower = ci_lower;
        self.ci_95_upper = ci_upper;
        self
    }

    /// Calculate and set Popper score based on claims
    pub fn calculate_popper_score(&mut self) {
        if self.claims.is_empty() {
            self.popper_score = 0.0;
            return;
        }

        let verified_count = self.claims.iter().filter(|c| c.verified).count();
        self.popper_score = (verified_count as f64 / self.claims.len() as f64) * 100.0;
    }

    /// Add a falsifiable claim
    pub fn add_claim(&mut self, claim: FalsifiableClaim) {
        self.claims.push(claim);
    }

    /// Claim is verified if measured within CI
    #[must_use]
    pub fn claim_verified(&self) -> bool {
        self.measured_gap >= self.ci_95_lower && self.measured_gap <= self.ci_95_upper
    }

    /// Create default IMP-800c claims
    #[must_use]
    pub fn with_default_claims(mut self, realizar_gpu_tps: f64) -> Self {
        // IMP-800c-1: GPU faster than CPU SIMD (>5x, threshold 25 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-1", "GPU faster than CPU SIMD (>5x)", 5.0, 25.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-2: GPU within 10x of Ollama (threshold 24 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-2", "GPU within 10x of Ollama", 10.0, 24.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-3: GPU within 2x of Ollama - M2 (threshold 120 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-3", "GPU within 2x of Ollama (M2)", 2.0, 120.0)
                .evaluate(realizar_gpu_tps),
        );

        // IMP-800c-4: GPU at parity with Ollama - M4 (threshold 192 tok/s)
        self.claims.push(
            FalsifiableClaim::new("IMP-800c-4", "GPU at parity with Ollama (M4)", 1.25, 192.0)
                .evaluate(realizar_gpu_tps),
        );

        self.calculate_popper_score();
        self
    }
}

// ============================================================================
// IMP-900: Closing the 18x Gap (M3/M4 Milestones)
// ============================================================================

/// Optimized GEMM configuration (IMP-900a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedGemmConfig {
    /// Tile size for shared memory (typically 32 or 64)
    pub tile_size: u32,
    /// Register blocking factor (typically 4 or 8)
    pub reg_block: u32,
    /// Use tensor cores if available (SM 7.0+)
    pub use_tensor_cores: bool,
    /// Vectorized loads (float4 = 4)
    pub vector_width: u32,
    /// Unroll factor for K-loop
    pub k_unroll: u32,
    /// Use double buffering for tile prefetch
    pub double_buffer: bool,
}

impl Default for OptimizedGemmConfig {
    fn default() -> Self {
        Self {
            tile_size: 32,
            reg_block: 4,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 4,
            double_buffer: true,
        }
    }
}

impl OptimizedGemmConfig {
    /// Create configuration for small matrices (256x256)
    #[must_use]
    pub fn small() -> Self {
        Self {
            tile_size: 16,
            reg_block: 2,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 4,
            double_buffer: false,
        }
    }

    /// Create configuration for large matrices (1024+)
    #[must_use]
    pub fn large() -> Self {
        Self {
            tile_size: 64,
            reg_block: 8,
            use_tensor_cores: false,
            vector_width: 4,
            k_unroll: 8,
            double_buffer: true,
        }
    }

    /// Calculate shared memory requirement (bytes)
    #[must_use]
    pub fn shared_memory_bytes(&self) -> u32 {
        // Two tiles (A and B) in shared memory
        // Each tile is tile_size × tile_size × sizeof(f32)
        let tile_bytes = self.tile_size * self.tile_size * 4;
        if self.double_buffer {
            tile_bytes * 4 // 2 tiles × 2 buffers
        } else {
            tile_bytes * 2 // 2 tiles
        }
    }

    /// Calculate threads per block
    #[must_use]
    pub fn threads_per_block(&self) -> u32 {
        // Each thread computes reg_block × reg_block elements
        let threads_per_dim = self.tile_size / self.reg_block;
        threads_per_dim * threads_per_dim
    }

    /// Calculate registers per thread (for accumulators)
    #[must_use]
    pub fn registers_per_thread(&self) -> u32 {
        // reg_block × reg_block accumulator values
        self.reg_block * self.reg_block
    }
}

/// GEMM performance result (IMP-900a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmPerformanceResult {
    /// Matrix M dimension (rows of A, rows of C)
    pub m: u32,
    /// Matrix N dimension (cols of B, cols of C)
    pub n: u32,
    /// Matrix K dimension (cols of A, rows of B)
    pub k: u32,
    /// Time in milliseconds
    pub time_ms: f64,
    /// GFLOP/s achieved
    pub gflops: f64,
    /// Memory bandwidth achieved (GB/s)
    pub bandwidth_gbs: f64,
    /// Percentage of peak performance
    pub efficiency: f64,
}

include!("gemm_performance.rs");
include!("gpu_parity_gpu_parity.rs");
