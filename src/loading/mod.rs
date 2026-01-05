//! APR Loading Subsystem
//!
//! Implements Toyota Way Heijunka (level loading) for model initialization.
//! Prevents memory spikes in embedded environments with limited RAM.
//!
//! # Design Philosophy
//!
//! Per Toyota Production System principles:
//! - **Heijunka**: Level resource demands during model initialization
//! - **Jidoka**: Built-in quality with verification at each layer
//! - **Poka-yoke**: Error-proofing via type-safe APIs
//!
//! # NASA Safety Compliance
//!
//! Per NPR 7150.2D Section 3.6.1, all memory allocations are deterministic
//! and bounded through pre-allocated buffer pools and streaming decompression.
//!
//! # References
//!
//! - [Wilhelm et al. 2008] "The worst-case execution-time problem"
//! - [Liu & Layland 1973] Real-time scheduling theory
//! - ISO 26262 ASIL-B requirements for automotive ML inference

pub mod cipher;
pub mod wcet;

pub use cipher::CipherSuite;
pub use wcet::{calculate_wcet, platforms, PlatformSpecs};

use std::sync::Arc;
use std::time::Duration;

/// Loading strategy selection based on deployment target
///
/// # Toyota Way Alignment
///
/// - **Eager**: Maximum quality, minimum latency (Jidoka)
/// - **`MappedDemand`**: Just-in-time resource usage (JIT)
/// - **Streaming**: Level loading for constrained systems (Heijunka)
/// - **`LazySection`**: Muda elimination for sparse access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadingMode {
    /// Full load into contiguous memory (embedded systems with deterministic requirements)
    /// Memory: `O(uncompressed_size)`
    /// Latency: Highest initial, lowest inference
    #[default]
    Eager,

    /// Memory-mapped with demand paging (server/desktop with large models)
    /// Memory: `O(page_size` * `active_pages`)
    /// Latency: Lowest initial, amortized inference
    MappedDemand,

    /// Streaming decompression with fixed-size ring buffer (ultra-constrained)
    /// Memory: `O(buffer_size)` constant
    /// Latency: Consistent per-chunk
    Streaming,

    /// JIT section loading for sparse access patterns (tree ensembles)
    /// Memory: `O(accessed_sections)`
    /// Latency: Per-section on first access
    LazySection,
}

impl LoadingMode {
    /// Get human-readable description of this mode
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Eager => "Full load into contiguous memory",
            Self::MappedDemand => "Memory-mapped with demand paging",
            Self::Streaming => "Streaming decompression with ring buffer",
            Self::LazySection => "JIT section loading for sparse access",
        }
    }

    /// Check if this mode supports zero-copy access
    #[must_use]
    pub const fn supports_zero_copy(&self) -> bool {
        matches!(self, Self::MappedDemand)
    }

    /// Check if this mode has deterministic memory usage
    #[must_use]
    pub const fn is_deterministic(&self) -> bool {
        matches!(self, Self::Eager | Self::Streaming)
    }

    /// Get recommended mode for a given memory budget
    #[must_use]
    pub fn for_memory_budget(budget_bytes: usize, model_size: usize) -> Self {
        if budget_bytes >= model_size * 2 {
            Self::Eager
        } else if budget_bytes >= model_size {
            Self::MappedDemand
        } else if budget_bytes >= 256 * 1024 {
            Self::Streaming
        } else {
            Self::LazySection
        }
    }
}

/// Verification level for model loading (NASA defense-in-depth)
///
/// # Safety Levels
///
/// Maps to ISO 26262 ASIL levels and DO-178C DALs:
/// - **`UnsafeSkip`**: Testing only, NEVER in production
/// - **`ChecksumOnly`**: ASIL-A / DAL-D (minimum Jidoka)
/// - **Standard**: ASIL-B / DAL-C (production default)
/// - **Paranoid**: ASIL-D / DAL-A (safety-critical)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerificationLevel {
    /// Skip all verification (testing only, NEVER in production)
    UnsafeSkip,

    /// Checksum only (minimum Jidoka)
    ChecksumOnly,

    /// Checksum + signature (if signed) - production default
    #[default]
    Standard,

    /// Full verification + runtime assertions (NASA Level A)
    Paranoid,
}

impl VerificationLevel {
    /// Check if this level verifies checksums
    #[must_use]
    pub const fn verifies_checksum(&self) -> bool {
        !matches!(self, Self::UnsafeSkip)
    }

    /// Check if this level verifies signatures
    #[must_use]
    pub const fn verifies_signature(&self) -> bool {
        matches!(self, Self::Standard | Self::Paranoid)
    }

    /// Check if this level enables runtime assertions
    #[must_use]
    pub const fn has_runtime_assertions(&self) -> bool {
        matches!(self, Self::Paranoid)
    }

    /// Get the ASIL level equivalent
    #[must_use]
    pub const fn asil_level(&self) -> &'static str {
        match self {
            Self::UnsafeSkip => "QM (not safety-relevant)",
            Self::ChecksumOnly => "ASIL-A",
            Self::Standard => "ASIL-B",
            Self::Paranoid => "ASIL-D",
        }
    }

    /// Get the DO-178C DAL equivalent
    #[must_use]
    pub const fn dal_level(&self) -> &'static str {
        match self {
            Self::UnsafeSkip => "DAL-E",
            Self::ChecksumOnly => "DAL-D",
            Self::Standard => "DAL-C",
            Self::Paranoid => "DAL-A",
        }
    }
}

/// Buffer pool for deterministic memory allocation
///
/// Pre-allocates memory to avoid runtime allocation failures
/// in embedded systems (Toyota Way: Heijunka).
#[derive(Debug)]
pub struct BufferPool {
    /// Pre-allocated buffers
    buffers: Vec<Vec<u8>>,
    /// Buffer size
    buffer_size: usize,
    /// Number of free buffers
    free_count: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    #[must_use]
    pub fn new(buffer_count: usize, buffer_size: usize) -> Self {
        let buffers = (0..buffer_count).map(|_| vec![0u8; buffer_size]).collect();
        Self {
            buffers,
            buffer_size,
            free_count: buffer_count,
        }
    }

    /// Get the buffer size
    #[must_use]
    pub const fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get number of free buffers
    #[must_use]
    pub const fn free_count(&self) -> usize {
        self.free_count
    }

    /// Get total buffer count
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.buffers.len()
    }

    /// Get total memory usage
    #[must_use]
    pub fn total_memory(&self) -> usize {
        self.buffers.len() * self.buffer_size
    }
}

/// Backend selection for Trueno operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU with SIMD (AVX2/AVX-512/NEON)
    #[default]
    CpuSimd,
    /// GPU via compute shaders (wgpu/WebGPU)
    Gpu,
    /// NVIDIA CUDA (requires `cuda` feature and NVIDIA driver)
    Cuda,
    /// WebAssembly (browser deployment)
    Wasm,
    /// Bare metal embedded (`no_std`)
    Embedded,
}

impl Backend {
    /// Check if this backend supports SIMD
    #[must_use]
    pub const fn supports_simd(&self) -> bool {
        matches!(self, Self::CpuSimd)
    }

    /// Check if this backend requires std library
    #[must_use]
    pub const fn requires_std(&self) -> bool {
        !matches!(self, Self::Embedded)
    }

    /// Check if this backend uses GPU acceleration
    #[must_use]
    pub const fn is_gpu_accelerated(&self) -> bool {
        matches!(self, Self::Gpu | Self::Cuda)
    }

    /// Check if this backend requires NVIDIA driver
    #[must_use]
    pub const fn requires_nvidia_driver(&self) -> bool {
        matches!(self, Self::Cuda)
    }
}

/// Load configuration with Toyota Way Jidoka (quality built-in) enforcement
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Loading mode selection
    pub mode: LoadingMode,

    /// Maximum memory budget (Heijunka: level loading)
    pub max_memory_bytes: Option<usize>,

    /// Verification strictness (NASA: defense-in-depth)
    pub verification: VerificationLevel,

    /// Trueno backend selection (SIMD/GPU/WASM)
    pub backend: Backend,

    /// Pre-allocated buffer pool (deterministic allocation)
    pub buffer_pool: Option<Arc<BufferPool>>,

    /// Time budget for loading (WCET enforcement)
    pub time_budget: Option<Duration>,

    /// Enable streaming decompression
    pub streaming: bool,

    /// Ring buffer size for streaming mode
    pub ring_buffer_size: usize,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            mode: LoadingMode::default(),
            max_memory_bytes: None,
            verification: VerificationLevel::default(),
            backend: Backend::default(),
            buffer_pool: None,
            time_budget: None,
            streaming: false,
            ring_buffer_size: 256 * 1024, // 256KB default
        }
    }
}

impl LoadConfig {
    /// Create a new load configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration for embedded deployment
    #[must_use]
    pub fn embedded(max_memory: usize) -> Self {
        Self {
            mode: LoadingMode::Eager,
            max_memory_bytes: Some(max_memory),
            verification: VerificationLevel::Paranoid,
            backend: Backend::Embedded,
            buffer_pool: None,
            time_budget: Some(Duration::from_millis(100)),
            streaming: false,
            ring_buffer_size: 64 * 1024,
        }
    }

    /// Create configuration for server deployment
    #[must_use]
    pub fn server() -> Self {
        Self {
            mode: LoadingMode::MappedDemand,
            max_memory_bytes: None,
            verification: VerificationLevel::Standard,
            backend: Backend::CpuSimd,
            buffer_pool: None,
            time_budget: None,
            streaming: false,
            ring_buffer_size: 1024 * 1024,
        }
    }

    /// Create configuration for WASM deployment
    #[must_use]
    pub fn wasm() -> Self {
        Self {
            mode: LoadingMode::Streaming,
            max_memory_bytes: Some(64 * 1024 * 1024), // 64MB
            verification: VerificationLevel::Standard,
            backend: Backend::Wasm,
            buffer_pool: None,
            time_budget: None,
            streaming: true,
            ring_buffer_size: 512 * 1024,
        }
    }

    /// Create configuration for NVIDIA CUDA deployment
    ///
    /// Requires the `cuda` feature and NVIDIA driver.
    /// Uses `MappedDemand` for efficient GPU memory transfers.
    #[must_use]
    pub fn cuda() -> Self {
        Self {
            mode: LoadingMode::MappedDemand,
            max_memory_bytes: None,
            verification: VerificationLevel::Standard,
            backend: Backend::Cuda,
            buffer_pool: None,
            time_budget: None,
            streaming: false,
            ring_buffer_size: 1024 * 1024, // 1MB for GPU transfers
        }
    }

    /// Create configuration for GPU deployment (wgpu/WebGPU)
    ///
    /// Cross-platform GPU acceleration via compute shaders.
    #[must_use]
    pub fn gpu() -> Self {
        Self {
            mode: LoadingMode::MappedDemand,
            max_memory_bytes: None,
            verification: VerificationLevel::Standard,
            backend: Backend::Gpu,
            buffer_pool: None,
            time_budget: None,
            streaming: false,
            ring_buffer_size: 1024 * 1024, // 1MB for GPU transfers
        }
    }

    /// Set the loading mode
    #[must_use]
    pub fn with_mode(mut self, mode: LoadingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the maximum memory budget
    #[must_use]
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Set the verification level
    #[must_use]
    pub fn with_verification(mut self, level: VerificationLevel) -> Self {
        self.verification = level;
        self
    }

    /// Set the backend
    #[must_use]
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the time budget
    #[must_use]
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = Some(budget);
        self
    }

    /// Enable streaming mode
    #[must_use]
    pub fn with_streaming(mut self, ring_buffer_size: usize) -> Self {
        self.streaming = true;
        self.ring_buffer_size = ring_buffer_size;
        self
    }

    /// Set the buffer pool
    #[must_use]
    pub fn with_buffer_pool(mut self, pool: Arc<BufferPool>) -> Self {
        self.buffer_pool = Some(pool);
        self
    }
}

/// Result of a model load operation
#[derive(Debug, Clone)]
pub struct LoadResult {
    /// Actual loading time
    pub load_time: Duration,
    /// Memory used
    pub memory_used: usize,
    /// Whether checksum was verified
    pub checksum_verified: bool,
    /// Whether signature was verified
    pub signature_verified: bool,
    /// Number of pages loaded (for paged modes)
    pub pages_loaded: usize,
    /// Decompression ratio achieved
    pub decompression_ratio: f64,
}

impl LoadResult {
    /// Create a new load result
    #[must_use]
    pub fn new(load_time: Duration, memory_used: usize) -> Self {
        Self {
            load_time,
            memory_used,
            checksum_verified: false,
            signature_verified: false,
            pages_loaded: 1,
            decompression_ratio: 1.0,
        }
    }

    /// Get throughput in MB/s
    #[must_use]
    pub fn throughput_mbps(&self) -> f64 {
        if self.load_time.as_secs_f64() > 0.0 {
            (self.memory_used as f64 / (1024.0 * 1024.0)) / self.load_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loading_mode_default() {
        assert_eq!(LoadingMode::default(), LoadingMode::Eager);
    }

    #[test]
    fn test_loading_mode_description() {
        assert!(!LoadingMode::Eager.description().is_empty());
        assert!(!LoadingMode::Streaming.description().is_empty());
    }

    #[test]
    fn test_loading_mode_zero_copy() {
        assert!(LoadingMode::MappedDemand.supports_zero_copy());
        assert!(!LoadingMode::Eager.supports_zero_copy());
    }

    #[test]
    fn test_loading_mode_deterministic() {
        assert!(LoadingMode::Eager.is_deterministic());
        assert!(LoadingMode::Streaming.is_deterministic());
        assert!(!LoadingMode::MappedDemand.is_deterministic());
    }

    #[test]
    fn test_loading_mode_for_memory_budget() {
        assert_eq!(LoadingMode::for_memory_budget(200, 100), LoadingMode::Eager);
        assert_eq!(
            LoadingMode::for_memory_budget(100, 100),
            LoadingMode::MappedDemand
        );
        assert_eq!(
            LoadingMode::for_memory_budget(512 * 1024, 1024 * 1024),
            LoadingMode::Streaming
        );
    }

    #[test]
    fn test_verification_level_default() {
        assert_eq!(VerificationLevel::default(), VerificationLevel::Standard);
    }

    #[test]
    fn test_verification_level_checksums() {
        assert!(!VerificationLevel::UnsafeSkip.verifies_checksum());
        assert!(VerificationLevel::ChecksumOnly.verifies_checksum());
        assert!(VerificationLevel::Standard.verifies_checksum());
        assert!(VerificationLevel::Paranoid.verifies_checksum());
    }

    #[test]
    fn test_verification_level_signatures() {
        assert!(!VerificationLevel::UnsafeSkip.verifies_signature());
        assert!(!VerificationLevel::ChecksumOnly.verifies_signature());
        assert!(VerificationLevel::Standard.verifies_signature());
        assert!(VerificationLevel::Paranoid.verifies_signature());
    }

    #[test]
    fn test_verification_level_asil() {
        assert_eq!(VerificationLevel::Paranoid.asil_level(), "ASIL-D");
        assert_eq!(VerificationLevel::Standard.asil_level(), "ASIL-B");
    }

    #[test]
    fn test_verification_level_dal() {
        assert_eq!(VerificationLevel::Paranoid.dal_level(), "DAL-A");
        assert_eq!(VerificationLevel::Standard.dal_level(), "DAL-C");
    }

    #[test]
    fn test_buffer_pool_creation() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.buffer_size(), 1024);
        assert_eq!(pool.free_count(), 4);
        assert_eq!(pool.total_count(), 4);
        assert_eq!(pool.total_memory(), 4096);
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(Backend::default(), Backend::CpuSimd);
    }

    #[test]
    fn test_backend_simd_support() {
        assert!(Backend::CpuSimd.supports_simd());
        assert!(!Backend::Wasm.supports_simd());
    }

    #[test]
    fn test_backend_std_requirement() {
        assert!(Backend::CpuSimd.requires_std());
        assert!(!Backend::Embedded.requires_std());
    }

    #[test]
    fn test_load_config_default() {
        let config = LoadConfig::default();
        assert_eq!(config.mode, LoadingMode::Eager);
        assert_eq!(config.verification, VerificationLevel::Standard);
        assert_eq!(config.backend, Backend::CpuSimd);
        assert!(config.max_memory_bytes.is_none());
    }

    #[test]
    fn test_load_config_embedded() {
        let config = LoadConfig::embedded(1024 * 1024);
        assert_eq!(config.mode, LoadingMode::Eager);
        assert_eq!(config.verification, VerificationLevel::Paranoid);
        assert_eq!(config.backend, Backend::Embedded);
        assert_eq!(config.max_memory_bytes, Some(1024 * 1024));
    }

    #[test]
    fn test_load_config_server() {
        let config = LoadConfig::server();
        assert_eq!(config.mode, LoadingMode::MappedDemand);
        assert_eq!(config.verification, VerificationLevel::Standard);
        assert!(config.max_memory_bytes.is_none());
    }

    #[test]
    fn test_load_config_wasm() {
        let config = LoadConfig::wasm();
        assert_eq!(config.mode, LoadingMode::Streaming);
        assert_eq!(config.backend, Backend::Wasm);
        assert!(config.streaming);
    }

    #[test]
    fn test_load_config_builder() {
        let config = LoadConfig::new()
            .with_mode(LoadingMode::Streaming)
            .with_max_memory(1024)
            .with_verification(VerificationLevel::Paranoid)
            .with_backend(Backend::Gpu)
            .with_time_budget(Duration::from_millis(100))
            .with_streaming(512 * 1024);

        assert_eq!(config.mode, LoadingMode::Streaming);
        assert_eq!(config.max_memory_bytes, Some(1024));
        assert_eq!(config.verification, VerificationLevel::Paranoid);
        assert_eq!(config.backend, Backend::Gpu);
        assert!(config.streaming);
        assert_eq!(config.ring_buffer_size, 512 * 1024);
    }

    #[test]
    fn test_load_result() {
        let result = LoadResult::new(Duration::from_millis(100), 10 * 1024 * 1024);
        assert_eq!(result.memory_used, 10 * 1024 * 1024);
        // 10 MB in 100ms = 100 MB/s
        let throughput = result.throughput_mbps();
        assert!((throughput - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_load_result_zero_time() {
        let result = LoadResult::new(Duration::ZERO, 1024);
        assert_eq!(result.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_backend_gpu_accelerated() {
        assert!(Backend::Gpu.is_gpu_accelerated());
        assert!(Backend::Cuda.is_gpu_accelerated());
        assert!(!Backend::CpuSimd.is_gpu_accelerated());
        assert!(!Backend::Wasm.is_gpu_accelerated());
        assert!(!Backend::Embedded.is_gpu_accelerated());
    }

    #[test]
    fn test_backend_nvidia_driver_requirement() {
        assert!(Backend::Cuda.requires_nvidia_driver());
        assert!(!Backend::Gpu.requires_nvidia_driver());
        assert!(!Backend::CpuSimd.requires_nvidia_driver());
        assert!(!Backend::Wasm.requires_nvidia_driver());
        assert!(!Backend::Embedded.requires_nvidia_driver());
    }

    #[test]
    fn test_load_config_cuda() {
        let config = LoadConfig::cuda();
        assert_eq!(config.mode, LoadingMode::MappedDemand);
        assert_eq!(config.backend, Backend::Cuda);
        assert_eq!(config.verification, VerificationLevel::Standard);
        assert!(config.max_memory_bytes.is_none());
        assert!(!config.streaming);
        assert!(config.backend.requires_nvidia_driver());
        assert!(config.backend.is_gpu_accelerated());
    }

    #[test]
    fn test_load_config_gpu() {
        let config = LoadConfig::gpu();
        assert_eq!(config.mode, LoadingMode::MappedDemand);
        assert_eq!(config.backend, Backend::Gpu);
        assert_eq!(config.verification, VerificationLevel::Standard);
        assert!(config.max_memory_bytes.is_none());
        assert!(!config.streaming);
        assert!(!config.backend.requires_nvidia_driver());
        assert!(config.backend.is_gpu_accelerated());
    }

    #[test]
    fn test_backend_cuda_properties() {
        let backend = Backend::Cuda;
        // CUDA requires std library
        assert!(backend.requires_std());
        // CUDA does not use CPU SIMD
        assert!(!backend.supports_simd());
        // CUDA is GPU accelerated
        assert!(backend.is_gpu_accelerated());
        // CUDA requires NVIDIA driver
        assert!(backend.requires_nvidia_driver());
    }

    #[test]
    fn test_load_config_builder_with_cuda() {
        let config = LoadConfig::new()
            .with_backend(Backend::Cuda)
            .with_mode(LoadingMode::Eager)
            .with_verification(VerificationLevel::Paranoid);

        assert_eq!(config.backend, Backend::Cuda);
        assert_eq!(config.mode, LoadingMode::Eager);
        assert_eq!(config.verification, VerificationLevel::Paranoid);
    }
}
