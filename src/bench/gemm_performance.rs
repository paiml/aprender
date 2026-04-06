
impl GemmPerformanceResult {
    /// Create a new GEMM performance result
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32, time_ms: f64) -> Self {
        // GEMM operations: 2 * M * N * K (multiply-add)
        let ops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        let gflops = ops / (time_ms * 1e6);

        // Memory: read A (M*K), read B (K*N), write C (M*N)
        let bytes = (f64::from(m) * f64::from(k)
            + f64::from(k) * f64::from(n)
            + f64::from(m) * f64::from(n))
            * 4.0;
        let bandwidth_gbs = bytes / (time_ms * 1e6);

        Self {
            m,
            n,
            k,
            time_ms,
            gflops,
            bandwidth_gbs,
            efficiency: 0.0, // Set by caller based on peak
        }
    }

    /// Set efficiency based on peak GFLOP/s
    #[must_use]
    pub fn with_peak(mut self, peak_gflops: f64) -> Self {
        self.efficiency = (self.gflops / peak_gflops) * 100.0;
        self
    }

    /// Check if performance improved by at least the given factor
    #[must_use]
    pub fn improved_by(&self, baseline_gflops: f64, factor: f64) -> bool {
        self.gflops >= baseline_gflops * factor
    }
}

/// Optimized GEMM benchmark runner (IMP-900a)
#[derive(Debug)]
pub struct OptimizedGemmBenchmark {
    /// Configuration
    pub config: OptimizedGemmConfig,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Measurement iterations
    pub measurement_iterations: usize,
    /// Target coefficient of variation
    pub target_cv: f64,
}

impl Default for OptimizedGemmBenchmark {
    fn default() -> Self {
        Self {
            config: OptimizedGemmConfig::default(),
            warmup_iterations: 5,
            measurement_iterations: 20,
            target_cv: 0.05,
        }
    }
}

impl OptimizedGemmBenchmark {
    /// Create benchmark with custom config
    #[must_use]
    pub fn with_config(config: OptimizedGemmConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Calculate expected improvement over naive GEMM
    #[must_use]
    pub fn expected_improvement(&self) -> f64 {
        let mut improvement = 1.0;

        // Shared memory tiling: ~2x for cache efficiency
        improvement *= 2.0;

        // Register blocking: ~1.5x for reduced memory traffic
        if self.config.reg_block >= 4 {
            improvement *= 1.5;
        }

        // Vectorized loads: ~1.3x for coalesced access
        if self.config.vector_width >= 4 {
            improvement *= 1.3;
        }

        // Double buffering: ~1.2x for latency hiding
        if self.config.double_buffer {
            improvement *= 1.2;
        }

        improvement
    }
}

/// Kernel fusion configuration (IMP-900b)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusedOpType {
    /// GEMM + bias + activation
    GemmBiasActivation,
    /// Layer normalization + linear projection
    LayerNormLinear,
    /// Fused attention (FlashAttention-style)
    FusedAttention,
    /// FFN: up projection + gate + down projection
    FusedFfn,
}

/// Fused operation specification (IMP-900b)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOpSpec {
    /// Type of fused operation
    pub op_type: FusedOpType,
    /// Input dimensions
    pub input_dims: Vec<u32>,
    /// Output dimensions
    pub output_dims: Vec<u32>,
    /// Activation function (if applicable)
    pub activation: Option<String>,
    /// Number of kernel launches when fused
    pub fused_launches: u32,
    /// Number of kernel launches when unfused
    pub unfused_launches: u32,
}

impl FusedOpSpec {
    /// Calculate launch reduction factor
    #[must_use]
    pub fn launch_reduction(&self) -> f64 {
        f64::from(self.unfused_launches) / f64::from(self.fused_launches)
    }

    /// Check if fusion reduces launches by at least 50%
    #[must_use]
    pub fn achieves_target_reduction(&self) -> bool {
        self.launch_reduction() >= 2.0
    }
}

/// FlashAttention configuration (IMP-900c)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    /// Block size for Q tiling (Br)
    pub block_size_q: u32,
    /// Block size for K/V tiling (Bc)
    pub block_size_kv: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Use causal masking
    pub causal: bool,
    /// Softmax scale (default: 1/sqrt(head_dim))
    pub scale: f32,
}

impl FlashAttentionConfig {
    /// Create configuration for phi-2 model
    #[must_use]
    pub fn phi2() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            head_dim: 80, // phi-2: 2560 / 32 heads
            num_heads: 32,
            causal: true,
            scale: 1.0 / (80.0_f32).sqrt(),
        }
    }

    /// Calculate memory required for attention (naive vs flash)
    #[must_use]
    pub fn memory_comparison(&self, seq_len: u32) -> (u64, u64) {
        // Naive: O(N²) attention matrix
        let naive_bytes = u64::from(seq_len) * u64::from(seq_len) * 4;

        // FlashAttention: O(N) working memory
        let flash_bytes = u64::from(self.block_size_q) * u64::from(self.block_size_kv) * 4 * 2; // S and P blocks

        (naive_bytes, flash_bytes)
    }

    /// Calculate memory savings factor
    #[must_use]
    pub fn memory_savings(&self, seq_len: u32) -> f64 {
        let (naive, flash) = self.memory_comparison(seq_len);
        naive as f64 / flash as f64
    }
}

/// Memory pool configuration (IMP-900d)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size (bytes)
    pub initial_size: usize,
    /// Maximum pool size (bytes)
    pub max_size: usize,
    /// Size classes for allocation (powers of 2)
    pub size_classes: Vec<usize>,
    /// Use pinned memory for host staging
    pub use_pinned_memory: bool,
    /// Enable async transfers
    pub async_transfers: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024,  // 256 MB
            max_size: 2 * 1024 * 1024 * 1024, // 2 GB
            size_classes: vec![
                4096,        // 4 KB
                16384,       // 16 KB
                65536,       // 64 KB
                262_144,     // 256 KB
                1_048_576,   // 1 MB
                4_194_304,   // 4 MB
                16_777_216,  // 16 MB
                67_108_864,  // 64 MB
                268_435_456, // 256 MB
            ],
            use_pinned_memory: true,
            async_transfers: true,
        }
    }
}

impl MemoryPoolConfig {
    /// Find the smallest size class that fits the requested size
    #[must_use]
    pub fn find_size_class(&self, requested: usize) -> Option<usize> {
        self.size_classes
            .iter()
            .copied()
            .find(|&size| size >= requested)
    }

    /// Calculate expected bandwidth improvement from pinned memory
    #[must_use]
    pub fn expected_bandwidth_improvement(&self) -> f64 {
        if self.use_pinned_memory {
            2.4 // Pinned memory typically 2-3x faster
        } else {
            1.0
        }
    }
}

/// IMP-900 combined result (M3/M4 targets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Imp900Result {
    /// Baseline throughput (13.1 tok/s from IMP-800)
    pub baseline_tps: f64,
    /// Throughput after optimizations
    pub optimized_tps: f64,
    /// GEMM optimization improvement factor
    pub gemm_improvement: f64,
    /// Kernel fusion improvement factor
    pub fusion_improvement: f64,
    /// FlashAttention improvement factor
    pub flash_attention_improvement: f64,
    /// Memory optimization improvement factor
    pub memory_improvement: f64,
    /// Gap to Ollama
    pub gap_ratio: f64,
    /// Target milestone achieved
    pub milestone: Option<String>,
}

impl Imp900Result {
    /// Create result from baseline
    #[must_use]
    pub fn from_baseline(baseline_tps: f64) -> Self {
        Self {
            baseline_tps,
            optimized_tps: baseline_tps,
            gemm_improvement: 1.0,
            fusion_improvement: 1.0,
            flash_attention_improvement: 1.0,
            memory_improvement: 1.0,
            gap_ratio: 240.0 / baseline_tps,
            milestone: None,
        }
    }

    /// Apply GEMM optimization
    #[must_use]
    pub fn with_gemm_improvement(mut self, factor: f64) -> Self {
        self.gemm_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply fusion optimization
    #[must_use]
    pub fn with_fusion_improvement(mut self, factor: f64) -> Self {
        self.fusion_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply FlashAttention optimization
    #[must_use]
    pub fn with_flash_attention_improvement(mut self, factor: f64) -> Self {
        self.flash_attention_improvement = factor;
        self.recalculate();
        self
    }

    /// Apply memory optimization
    #[must_use]
    pub fn with_memory_improvement(mut self, factor: f64) -> Self {
        self.memory_improvement = factor;
        self.recalculate();
        self
    }

    /// Recalculate throughput and milestone
    fn recalculate(&mut self) {
        let total_improvement = self.gemm_improvement
            * self.fusion_improvement
            * self.flash_attention_improvement
            * self.memory_improvement;

        self.optimized_tps = self.baseline_tps * total_improvement;
        self.gap_ratio = 240.0 / self.optimized_tps;

        self.milestone = if self.gap_ratio <= 1.25 {
            Some("M4".to_string()) // Full parity
        } else if self.gap_ratio <= 2.0 {
            Some("M3".to_string()) // Near parity
        } else if self.gap_ratio <= 5.0 {
            Some("M2".to_string()) // Within 5x
        } else {
            None
        };
    }

    /// Check if M3 target achieved (>48 tok/s, <5x gap)
    #[must_use]
    pub fn achieves_m3(&self) -> bool {
        self.optimized_tps >= 48.0 && self.gap_ratio <= 5.0
    }

    /// Check if M4 target achieved (>192 tok/s, <1.25x gap)
    #[must_use]
    pub fn achieves_m4(&self) -> bool {
        self.optimized_tps >= 192.0 && self.gap_ratio <= 1.25
    }

    /// Get combined improvement factor
    #[must_use]
    pub fn total_improvement(&self) -> f64 {
        self.optimized_tps / self.baseline_tps
    }
}
