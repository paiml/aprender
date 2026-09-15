// ============================================================================
// Development-phase lint allows - to be addressed incrementally
// ============================================================================
// Allow manual_div_ceil - clearer for block calculations
#![allow(clippy::manual_div_ceil)]
// Allow manual_is_multiple_of - clearer alignment checks
#![allow(clippy::manual_is_multiple_of)]
// Allow needless_range_loop - index access is clearer in some SIMD algorithms
#![allow(clippy::needless_range_loop)]
// Allow empty line after doc comments - formatting preference
#![allow(clippy::empty_line_after_doc_comments)]
// Allow similar names - semantic distinction is clear
#![allow(clippy::similar_names)]
// Allow many single char names - standard math/matrix notation
#![allow(clippy::many_single_char_names)]
// Allow too many arguments - SIMD/compute APIs require many parameters
#![allow(clippy::too_many_arguments)]
// Allow type complexity - complex SIMD types
#![allow(clippy::type_complexity)]
// Allow macro metavars in unsafe - necessary for SIMD dispatch macros
#![allow(clippy::macro_metavars_in_unsafe)]
// Allow missing panics doc - will be added incrementally
#![allow(clippy::missing_panics_doc)]
// Allow uninit_vec - intentional pattern for perf-critical paths where
// every element is SET (not accumulated) before any read. Each use has
// a SAFETY comment documenting the write-before-read invariant.
#![allow(clippy::uninit_vec)]
// Allow missing errors doc - will be added incrementally
#![allow(clippy::missing_errors_doc)]
// Allow missing safety doc - will be added incrementally
#![allow(clippy::missing_safety_doc)]
// Allow excessive precision - SIMD math constants need specific precision
#![allow(clippy::excessive_precision)]
// Allow unnecessary cast - clearer type annotations in some cases
#![allow(clippy::unnecessary_cast)]
// Allow cast_possible_truncation - handled in SIMD code
#![allow(clippy::cast_possible_truncation)]
// Allow cast_sign_loss - handled in SIMD code
#![allow(clippy::cast_sign_loss)]
// Allow cast_precision_loss - handled in SIMD code
#![allow(clippy::cast_precision_loss)]
// Allow large stack arrays - SIMD/GPU test data and proptest expansions
#![allow(clippy::large_stack_arrays)]
// Allow unwrap/float_cmp in test code — safe in assertions, banned in production
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::float_cmp))]

//! Trueno: Multi-Target High-Performance Compute Library
//!
//! **Trueno** (Spanish: "thunder") provides unified, high-performance compute primitives
//! across three execution targets:
//!
//! 1. **CPU SIMD** - x86 (SSE2/AVX/AVX2/AVX-512), ARM (NEON), WASM (SIMD128)
//! 2. **GPU** - Vulkan/Metal/DX12/WebGPU via `wgpu`
//! 3. **WebAssembly** - Portable SIMD128 for browser/edge deployment
//!
//! # Design Principles
//!
//! - **Write once, optimize everywhere**: Single algorithm, multiple backends
//! - **Runtime dispatch**: Auto-select best implementation based on CPU features
//! - **Zero unsafe in public API**: Safety via type system, `unsafe` isolated in backends
//! - **Benchmarked performance**: Every optimization must prove ≥10% speedup
//! - **Extreme TDD**: >90% test coverage, mutation testing, property-based tests
//!
//! # Quick Start
//!
//! ```rust
//! use trueno::Vector;
//!
//! let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
//! let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
//!
//! // Auto-selects best backend (AVX2/GPU/WASM)
//! let result = a.add(&b).unwrap();
//! assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
//! ```

// Contract assertions from YAML (pv codegen)
#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

// Fallback macros for contracts not yet in codegen
macro_rules! contract_pre_add { () => {{}}; ($($x:expr),+ $(,)?) => {{ $(let _ = &$x;)+ }}; }
macro_rules! contract_pre_gemv { () => {{}}; ($($x:expr),+ $(,)?) => {{ $(let _ = &$x;)+ }}; }

pub mod activations;
pub mod backends;
pub mod blis;
pub mod brick;
pub mod chaos;
pub mod contracts;
pub mod eigen;
pub mod error;
pub mod hardware;
pub mod hash;
pub mod inference;
pub mod matrix;
pub mod monitor;
/// Backend discovery: probe → enumerate → print (PP-066 R-0a).
pub mod registry;
pub mod simulation;
pub mod tiling;
pub mod tuner;
pub mod vector;

// Canonical scalar activation functions (UCBD §4, trueno #103)
pub use activations::{
    f16_to_f32, f32_to_f16, gelu_scalar, relu_scalar, sigmoid_scalar, silu_scalar, tanh_scalar,
};
pub use eigen::SymmetricEigen;
pub use error::{Result, TruenoError};
pub use hash::{hash_bytes, hash_key, hash_keys_batch, hash_keys_batch_with_backend};
pub use matrix::Matrix;
pub use monitor::{
    cuda_monitor_available, GpuBackend, GpuClockMetrics, GpuDeviceInfo, GpuMemoryMetrics,
    GpuMetrics, GpuMonitor, GpuPcieMetrics, GpuPowerMetrics, GpuThermalMetrics, GpuUtilization,
    GpuVendor, MonitorConfig, MonitorError,
};
#[cfg(feature = "cuda-monitor")]
pub use monitor::{enumerate_cuda_devices, query_cuda_device_info, query_cuda_memory};
pub use vector::Vector;

// ComputeBrick exports
pub use brick::{
    fnv1a_f32_checksum,
    AddOp,
    AssertionResult,
    AttentionOp,
    // QUANT-Q5K: Q5_K and Q6_K quantization formats (llama.cpp compatible)
    BlockQ5K,
    BlockQ6K,
    BrickBottleneck,
    BrickCategory,
    BrickError,
    // PAR-200: BrickProfiler v2 types
    BrickId,
    BrickIdTimer,
    BrickLayer,
    BrickProfiler,
    BrickSample,
    BrickStats,
    BrickTimer,
    BrickVerification,
    ByteBudget,
    CategoryStats,
    ComputeAssertion,
    ComputeBackend,
    ComputeBrick,
    ComputeOp,
    DivergenceInfo,
    DotOp,
    DotQ5KOp,
    DotQ6KOp,
    EdgeType,
    ExecutionEdge,
    ExecutionGraph,
    ExecutionNode,
    // PAR-201: Execution path graph types
    ExecutionNodeId,
    FusedGateUpOp,
    FusedGateUpWeights,
    FusedQKVOp,
    FusedQKVWeights,
    // CORRECTNESS-011: Divergence detection types
    KernelChecksum,
    MatmulOp,
    PtxRegistry,
    SoftmaxOp,
    SyncMode,
    // TILING-SPEC-001: Tile-level profiling types
    TileLevel,
    TileStats,
    TileTimer,
    TokenBudget,
    TokenResult,
};

// Hardware capability exports (PMAT-447)
pub use hardware::{
    default_hardware_path, Bottleneck, CpuCapability, GpuBackend as HardwareGpuBackend,
    GpuCapability, HardwareCapability, RooflineParams, SimdWidth,
};

// ML Tuner exports (T-TUNER-003 through T-TUNER-007, GH#80-84)
pub use tuner::{
    BottleneckClass, BottleneckPrediction, BrickTuner, ConceptDriftStatus, ExperimentSuggestion,
    FeatureExtractor, KernelClassifier, KernelRecommendation, KernelType, QuantType, RunConfig,
    ThroughputPrediction, ThroughputRegressor, TrainingSample, TrainingStats, TunerDataCollector,
    TunerError, TunerFeatures, TunerRecommendation, UserFeedback,
};

// Tiling Compute Blocks exports (TILING-SPEC-001)
pub use tiling::{
    optimal_prefetch_distance, pack_a_index, pack_b_index, swizzle_index, PackingLayout,
    PrefetchLocality, TcbGeometry, TcbIndexCalculator, TcbLevel, TiledQ4KMatvec, TilingBackend,
    TilingConfig, TilingError, TilingStats, Q4K_SUPERBLOCK_BYTES, Q4K_SUPERBLOCK_SIZE,
};

/// Backend execution target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE2 (x86_64 baseline)
    SSE2,
    /// AVX (256-bit)
    AVX,
    /// AVX2 (256-bit with FMA)
    AVX2,
    /// AVX-512 (512-bit)
    AVX512,
    /// ARM NEON
    NEON,
    /// WebAssembly SIMD128
    WasmSIMD,
    /// GPU compute (wgpu)
    GPU,
    /// Auto-select best available
    Auto,
}

impl Backend {
    /// Select the best available backend for the current platform
    ///
    /// This is a convenience wrapper around `select_best_available_backend()`
    pub fn select_best() -> Self {
        select_best_available_backend()
    }
}

/// Operation complexity for GPU dispatch eligibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpComplexity {
    /// Simple operations (add, mul) - prefer SIMD unless very large
    Low = 0,
    /// Moderate operations (dot, reduce) - GPU beneficial at 100K+
    Medium = 1,
    /// Complex operations (matmul, convolution) - GPU beneficial at 10K+
    High = 2,
}

/// Operation type for SIMD backend selection
///
/// Based on AVX-512 performance analysis (see AVX512_ANALYSIS.md), operations are
/// categorized by their memory vs compute characteristics to guide optimal backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Memory-bound operations (add, sub, mul, scale, div)
    ///
    /// These operations perform minimal computation per memory access (arithmetic intensity < 1 op/byte).
    /// Prefer AVX2 over AVX-512 due to memory bandwidth bottleneck.
    ///
    /// AVX-512 performance: 0.67-1.20x scalar (often slower!)
    /// AVX2 performance: 1.0-1.2x scalar
    MemoryBound,

    /// Compute-bound operations (dot, max, min, argmax, argmin)
    ///
    /// These operations perform significant computation per memory access (arithmetic intensity > 1 op/byte).
    /// AVX-512 excels due to wider SIMD parallelism.
    ///
    /// AVX-512 performance: 7-14x scalar (validated)
    /// AVX2 performance: 4-12x scalar (validated)
    ComputeBound,

    /// Mixed operations (fma, sqrt, exp, sigmoid, activations)
    ///
    /// Performance depends on data size and hardware.
    /// Use size-based heuristics or default to AVX2 for safety.
    Mixed,
}

/// Detect best SIMD backend for x86/x86_64 platforms
///
/// **IMPORTANT**: Prefers AVX2 over AVX-512 by default based on performance analysis.
///
/// AVX-512 is **NOT** universally faster - it causes 10-33% slowdown for memory-bound
/// operations (add, mul, sub) due to memory bandwidth bottleneck and thermal throttling.
/// See AVX512_ANALYSIS.md for detailed benchmarking results.
///
/// For operation-specific backend selection, use `select_backend_for_operation()`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn detect_x86_backend() -> Backend {
    // Prefer AVX2 over AVX-512 for safety (AVX-512 causes regressions for memory-bound ops)
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Backend::AVX2;
    }
    // Note: AVX-512 is intentionally NOT checked here
    // Use select_backend_for_operation(OperationType::ComputeBound) for AVX-512
    if is_x86_feature_detected!("avx") {
        return Backend::AVX;
    }
    if is_x86_feature_detected!("sse2") {
        return Backend::SSE2;
    }
    Backend::Scalar
}

/// Detect best SIMD backend for ARM platforms
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
fn detect_arm_backend() -> Backend {
    #[cfg(target_feature = "neon")]
    {
        Backend::NEON
    }
    #[cfg(not(target_feature = "neon"))]
    {
        Backend::Scalar
    }
}

/// Detect best SIMD backend for WebAssembly
#[cfg(target_arch = "wasm32")]
fn detect_wasm_backend() -> Backend {
    #[cfg(target_feature = "simd128")]
    {
        Backend::WasmSIMD
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        Backend::Scalar
    }
}

/// Select the best available backend for the current platform
///
/// This function performs runtime CPU feature detection and selects the most
/// optimized backend available. The selection follows this priority:
///
/// **x86/x86_64**:
/// 1. AVX-512 (if `avx512f` feature detected)
/// 2. AVX2 (if `avx2` and `fma` features detected)
/// 3. AVX (if `avx` feature detected)
/// 4. SSE2 (baseline for x86_64)
/// 5. Scalar (fallback)
///
/// **ARM**:
/// 1. NEON (if available)
/// 2. Scalar (fallback)
///
/// **WASM**: SIMD128 (if available), else Scalar
///
/// **Other platforms**: Scalar
///
/// # Returns
///
/// The most optimized backend available on this CPU/platform
///
/// # Examples
///
/// ```
/// use trueno::select_best_available_backend;
///
/// let backend = select_best_available_backend();
/// println!("Using backend: {:?}", backend);
/// ```
pub fn select_best_available_backend() -> Backend {
    // Cache backend selection using OnceLock to avoid repeated CPU feature detection
    // This eliminates 3-5% overhead from calling is_x86_feature_detected!() repeatedly
    static BEST_BACKEND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();

    *BEST_BACKEND.get_or_init(|| {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            detect_x86_backend()
        }

        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            detect_arm_backend()
        }

        #[cfg(target_arch = "wasm32")]
        {
            detect_wasm_backend()
        }

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "x86",
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "wasm32"
        )))]
        {
            Backend::Scalar
        }
    })
}

/// Select the optimal backend for a specific operation type
///
/// This function considers the memory vs compute characteristics of operations
/// to select the backend that will provide the best performance. Based on
/// comprehensive benchmarking (see AVX512_ANALYSIS.md), AVX-512 is avoided
/// for memory-bound operations where it causes 10-33% performance degradation.
///
/// # Operation Classification
///
/// - **MemoryBound**: add, sub, mul, div, scale, abs, clamp, lerp, relu
///   - Prefer AVX2 (1.0-1.2x scalar) over AVX-512 (0.67-1.20x scalar)
///   - Memory bandwidth bottleneck limits wider SIMD benefit
///
/// - **ComputeBound**: dot, max, min, argmax, argmin, norm_l1, norm_l2, norm_linf
///   - Prefer AVX-512 (7-14x scalar) over AVX2 (4-12x scalar)
///   - High arithmetic intensity benefits from wider SIMD
///
/// - **Mixed**: fma, sqrt, exp, ln, sigmoid, tanh, gelu, swish
///   - Default to AVX2 for safety (avoids AVX-512 thermal throttling)
///   - Size-based heuristics could improve this in future
///
/// # Backend Selection Priority
///
/// **For MemoryBound operations**:
/// 1. AVX2 (if available) - BEST for memory-bound
/// 2. SSE2 (x86_64 baseline)
/// 3. AVX-512 (AVOIDED - causes slowdown)
/// 4. NEON (ARM)
/// 5. WASM SIMD128
/// 6. Scalar (fallback)
///
/// **For ComputeBound operations**:
/// 1. AVX-512 (if available) - BEST for compute-bound
/// 2. AVX2
/// 3. SSE2
/// 4. NEON (ARM)
/// 5. WASM SIMD128
/// 6. Scalar (fallback)
///
/// # Arguments
///
/// * `op_type` - The type of operation being performed
///
/// # Returns
///
/// The optimal backend for the given operation type
///
/// # Examples
///
/// ```
/// use trueno::{select_backend_for_operation, OperationType};
///
/// // Memory-bound operation - prefers AVX2 over AVX-512
/// let backend = select_backend_for_operation(OperationType::MemoryBound);
///
/// // Compute-bound operation - uses AVX-512 if available
/// let backend = select_backend_for_operation(OperationType::ComputeBound);
/// ```
///
/// # Performance Impact
///
/// Using operation-aware backend selection fixes performance regressions:
/// - mul with AVX-512: 0.67x → 1.0x (use AVX2 instead)
/// - sub with AVX-512: 0.87x → 1.0x (use AVX2 instead)
/// - dot with AVX-512: 7.89x (keep AVX-512)
pub fn select_backend_for_operation(op_type: OperationType) -> Backend {
    // Allow unused on non-x86 architectures
    let _ = &op_type;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        select_x86_backend_for_operation(op_type)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        detect_arm_backend()
    }

    #[cfg(target_arch = "wasm32")]
    {
        detect_wasm_backend()
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "wasm32"
    )))]
    {
        Backend::Scalar
    }
}

/// Select the best x86 backend based on operation type and available features.
///
/// Separated from `select_backend_for_operation` to reduce cyclomatic complexity.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn select_x86_backend_for_operation(op_type: OperationType) -> Backend {
    use std::arch::is_x86_feature_detected;

    // Check for AVX-512 (only for compute-bound operations)
    let use_avx512 = op_type == OperationType::ComputeBound && is_x86_feature_detected!("avx512f");
    if use_avx512 {
        return Backend::AVX512;
    }

    // AVX2 with FMA is preferred for most operations
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Backend::AVX2;
    }

    // Fallback chain: AVX -> SSE2 -> Scalar
    if is_x86_feature_detected!("avx") {
        return Backend::AVX;
    }
    if is_x86_feature_detected!("sse2") {
        return Backend::SSE2;
    }

    Backend::Scalar
}

#[cfg(test)]
mod contract_tests;

#[cfg(test)]
mod contract_tests_image;

#[cfg(test)]
mod contract_tests_linalg;

#[cfg(test)]
mod tests;
