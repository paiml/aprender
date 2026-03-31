# APR File Tooling Specification

**Document**: APR Loading, Inspection, Diff, Debugging, Bundling, and Data Tooling Improvements
**Version**: 1.0.0
**Status**: Draft Specification
**Author**: Noah Gift / Pragmatic AI Labs
**Date**: 2025-12-08
**Target Platforms**: Embedded (Automotive/Aerospace), WASM (Browser), Native (Server/Desktop)
**TDG Target**: A+ (95.0+/100)
**Safety Standard Compliance**: Toyota Way, NASA Software Safety (NPR 7150.2D)

---

## Executive Summary

This specification defines comprehensive tooling improvements for `.apr` (Aprender Model Format) files, targeting safety-critical embedded systems (automotive ECUs, aerospace flight computers), resource-constrained devices, and interactive WASM playgrounds. The design philosophy integrates Toyota Production System (TPS) principles with NASA software safety standards to achieve fault-tolerant, deterministic model loading suitable for mission-critical applications.

**Core Objectives**:
1. Sub-millisecond model loading for real-time inference
2. Deterministic memory allocation for embedded deployment
3. Zero-copy data access with SIMD acceleration via Trueno
4. 100-point model quality scoring using ML best practices
5. Interactive model inspection for debugging and validation
6. WASM-native playground support for model zoo distribution
7. Seamless integration with Sovereign AI Stack (alimentar, pacha, realizar, presentar, batuta)

---

## Table of Contents

1. [Loading Subsystem](#1-loading-subsystem)
2. [Memory Paging Architecture](#2-memory-paging-architecture)
3. [Binary Caching and Bundling](#3-binary-caching-and-bundling)
4. [Data Embedding with Trueno Acceleration](#4-data-embedding-with-trueno-acceleration)
5. [World's Fastest Model Type](#5-worlds-fastest-model-type)
6. [Model Inspection Tooling](#6-model-inspection-tooling)
7. [100-Point Model Quality Scoring](#7-100-point-model-quality-scoring)
8. [WASM Playground Support](#8-wasm-playground-support)
9. [Sovereign AI Stack Integration](#9-sovereign-ai-stack-integration)
10. [References](#10-references)
11. [Toyota Way Compliance Matrix](#11-toyota-way-compliance-matrix)

---

## 1. Loading Subsystem

### 1.1 Design Philosophy (Toyota Way: Heijunka - Level Loading)

The loading subsystem implements Toyota's Heijunka principle by leveling resource demands during model initialization. This prevents memory spikes that could cause system instability in embedded environments with limited RAM.

**NASA Safety Compliance**: Per NPR 7150.2D Section 3.6.1, all memory allocations must be deterministic and bounded. The loading subsystem achieves this through pre-allocated buffer pools and streaming decompression.

### 1.2 Loading Modes

```rust
/// Loading strategy selection based on deployment target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadingMode {
    /// Full load into contiguous memory (embedded systems with deterministic requirements)
    /// Memory: O(uncompressed_size)
    /// Latency: Highest initial, lowest inference
    Eager,

    /// Memory-mapped with demand paging (server/desktop with large models)
    /// Memory: O(page_size * active_pages)
    /// Latency: Lowest initial, amortized inference
    MappedDemand,

    /// Streaming decompression with fixed-size ring buffer (ultra-constrained)
    /// Memory: O(buffer_size) constant
    /// Latency: Consistent per-chunk
    Streaming,

    /// JIT section loading for sparse access patterns (tree ensembles)
    /// Memory: O(accessed_sections)
    /// Latency: Per-section on first access
    LazySection,
}
```

### 1.3 Hierarchical Loading Pipeline

The loading pipeline follows NASA's defense-in-depth principle (NPR 7150.2D Section 3.7) with multiple validation layers:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Header Validation (32 bytes, constant time)                   │
│   - Magic number verification (APRN)                                    │
│   - Version compatibility check                                         │
│   - Feature flag parsing                                                │
│   - Size bounds validation (prevent billion-laugh attacks)              │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Metadata Extraction (MessagePack, variable)                   │
│   - Model type verification                                             │
│   - Hyperparameter restoration                                          │
│   - Training provenance chain                                           │
│   - License enforcement (if LICENSED flag set)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Security Layer (optional, per flags)                          │
│   - Ed25519 signature verification (SIGNED flag)                        │
│   - AES-256-GCM decryption (ENCRYPTED flag)                            │
│   - Key derivation via Argon2id                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Payload Decompression (Zstd/LZ4, streaming capable)           │
│   - Compression bomb protection (MAX_UNCOMPRESSED_SIZE = 1GB)          │
│   - Streaming decompression for memory efficiency                       │
│   - Zero-copy path for uncompressed payloads                           │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Checksum Verification (CRC32, constant time)                  │
│   - Full payload integrity verification                                 │
│   - Jidoka: Stop-the-line on corruption detection                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Loading API

```rust
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
}

#[derive(Debug, Clone, Copy, Default)]
pub enum VerificationLevel {
    /// Skip all verification (testing only, NEVER in production)
    UnsafeSkip,

    /// Checksum only (minimum Jidoka)
    ChecksumOnly,

    /// Checksum + signature (if signed)
    #[default]
    Standard,

    /// Full verification + runtime assertions (NASA Level A)
    Paranoid,
}

/// Primary loading entrypoint with comprehensive error handling
pub fn load_apr<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    config: LoadConfig,
) -> Result<LoadedModel<M>, AprenderError> {
    // Implementation follows 5-layer pipeline
}

/// Zero-copy loading from bytes (WASM/embedded)
pub fn load_apr_bytes<M: DeserializeOwned>(
    bytes: &[u8],
    expected_type: ModelType,
    config: LoadConfig,
) -> Result<LoadedModel<M>, AprenderError> {
    // Direct buffer processing without filesystem
}
```

### 1.5 Performance Targets (Automotive-Grade)

Per ISO 26262 ASIL-B requirements for automotive ML inference:

| Operation | Target Latency | Memory Bound | Verification |
|-----------|---------------|--------------|--------------|
| Header validation | < 1 μs | 32 bytes | Always |
| Metadata parsing | < 100 μs | 64 KB | Always |
| Signature verify | < 1 ms | O(1) | If signed |
| Decompression | < 10 ms/MB | 256 KB buffer | Always |
| Full load (10 MB) | < 50 ms | 10 MB + overhead | All layers |

### 1.6 Deterministic Timing Analysis (WCET)

For safety-critical systems (ISO 26262 ASIL-D, DO-178C Level A), loading time must be **mathematically bounded**. The Worst-Case Execution Time (WCET) provides certification evidence that the system will always complete within the allocated time budget.

**Reference**: [Wilhelm et al. 2008] "The worst-case execution-time problem"

#### 1.6.1 WCET Formula

The theoretical worst-case loading time is:

```
T_load = T_header + (S_comp / R_read) + (S_comp / R_decomp) + T_verify + T_deserialize
```

Where:
- `T_header`: Header validation time (constant, ~1μs)
- `S_comp`: Compressed payload size in bytes
- `R_read`: Minimum storage read speed (worst-case fragmentation)
- `R_decomp`: Minimum decompression throughput (Zstd is strictly bounded)
- `T_verify`: CRC32/signature verification time
- `T_deserialize`: Bincode deserialization time

#### 1.6.2 WCET Calculator Implementation

```rust
/// Platform-specific timing parameters for WCET calculation
/// Must be characterized via hardware profiling (Genchi Genbutsu)
#[derive(Debug, Clone)]
pub struct PlatformSpecs {
    /// Minimum guaranteed read speed from storage (MB/s)
    /// Account for worst-case fragmentation and bus contention
    pub min_read_speed_mbps: f64,

    /// Minimum decompression throughput (MB/s uncompressed output)
    /// Zstd has bounded worst-case: ~200 MB/s on Cortex-A53
    pub min_decomp_speed_mbps: f64,

    /// CRC32 throughput using hardware acceleration (MB/s)
    /// Most ARM cores: ~2000 MB/s with CRC32 instruction
    pub crc32_throughput_mbps: f64,

    /// Ed25519 signature verification time (microseconds)
    /// Constant time regardless of message size
    pub ed25519_verify_us: f64,

    /// Bincode deserialization overhead (bytes/μs)
    pub deserialize_throughput_bps: f64,
}

/// Pre-characterized platform specifications
pub mod platforms {
    use super::PlatformSpecs;

    /// Automotive-grade ECU (NXP S32G, Cortex-A53)
    pub const AUTOMOTIVE_S32G: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 50.0,    // eMMC worst-case
        min_decomp_speed_mbps: 200.0,  // Zstd level 3
        crc32_throughput_mbps: 2000.0, // Hardware CRC
        ed25519_verify_us: 800.0,      // Software implementation
        deserialize_throughput_bps: 500_000_000.0,
    };

    /// Aerospace flight computer (RAD750-class)
    pub const AEROSPACE_RAD750: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 10.0,     // Radiation-hardened flash
        min_decomp_speed_mbps: 50.0,   // Conservative estimate
        crc32_throughput_mbps: 100.0,  // Software CRC
        ed25519_verify_us: 5000.0,     // No crypto acceleration
        deserialize_throughput_bps: 50_000_000.0,
    };

    /// Edge device (Raspberry Pi 4, Cortex-A72)
    pub const EDGE_RPI4: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 100.0,
        min_decomp_speed_mbps: 400.0,
        crc32_throughput_mbps: 3000.0,
        ed25519_verify_us: 200.0,
        deserialize_throughput_bps: 1_000_000_000.0,
    };
}

/// Calculates the theoretical Worst-Case Execution Time (WCET) for model loading.
/// Used for ISO 26262 ASIL-D and DO-178C Level A certification evidence.
///
/// # Safety Rationale (Jidoka)
///
/// This function provides a **conservative upper bound**. Actual execution
/// will typically be faster. The WCET is used to guarantee the system
/// meets real-time deadlines under all operating conditions.
///
/// # References
///
/// - [Wilhelm et al. 2008] WCET analysis overview
/// - [Liu & Layland 1973] Real-time scheduling theory
/// - [Zstd Format Spec] Bounded decompression complexity
pub fn calculate_wcet(header: &HeaderInfo, platform: &PlatformSpecs) -> Duration {
    // 1. Header validation (constant time)
    let header_time_us = 1.0;

    // 2. Storage read latency (worst-case sequential read)
    let compressed_mb = header.compressed_size_bytes as f64 / (1024.0 * 1024.0);
    let read_time_us = (compressed_mb / platform.min_read_speed_mbps) * 1_000_000.0;

    // 3. Decompression latency (Zstd worst-case is strictly bounded)
    // Reference: [Collet 2016] Zstd compression algorithm
    let uncompressed_mb = header.uncompressed_size_bytes as f64 / (1024.0 * 1024.0);
    let decomp_time_us = (uncompressed_mb / platform.min_decomp_speed_mbps) * 1_000_000.0;

    // 4. Integrity verification
    let verify_time_us = if header.flags.is_signed() {
        // Ed25519 verification is constant-time
        platform.ed25519_verify_us
    } else {
        // CRC32 only
        let payload_mb = header.payload_size_bytes as f64 / (1024.0 * 1024.0);
        (payload_mb / platform.crc32_throughput_mbps) * 1_000_000.0
    };

    // 5. Deserialization overhead
    let deserialize_time_us = header.uncompressed_size_bytes as f64
        / platform.deserialize_throughput_bps
        * 1_000_000.0;

    // Total WCET with 10% safety margin (Toyota Way: conservative design)
    let total_us = (header_time_us + read_time_us + decomp_time_us
        + verify_time_us + deserialize_time_us) * 1.1;

    Duration::from_micros(total_us.ceil() as u64)
}

/// Runtime assertion for time budget compliance
///
/// # Jidoka Principle
///
/// If the model cannot be loaded within the time budget, this function
/// returns an error immediately (stop-the-line). This prevents:
/// - Missed real-time deadlines
/// - System instability during boot
/// - Undefined behavior in safety-critical contexts
pub fn assert_time_budget(
    model: &HeaderInfo,
    platform: &PlatformSpecs,
    budget: Duration,
) -> Result<(), SafetyError> {
    let worst_case = calculate_wcet(model, platform);

    if worst_case > budget {
        return Err(SafetyError::TimeBudgetExceeded {
            budget,
            worst_case,
            model_type: model.model_type,
            compressed_size: model.compressed_size_bytes,
            recommendation: format!(
                "Reduce model size below {} bytes or use faster storage",
                estimate_max_size_for_budget(platform, budget)
            ),
        });
    }

    Ok(())
}

/// Estimate maximum model size that fits within time budget
fn estimate_max_size_for_budget(platform: &PlatformSpecs, budget: Duration) -> usize {
    let budget_us = budget.as_micros() as f64;
    let effective_throughput = platform.min_read_speed_mbps
        .min(platform.min_decomp_speed_mbps);
    let max_mb = (budget_us / 1_000_000.0) * effective_throughput * 0.8; // 80% utilization
    (max_mb * 1024.0 * 1024.0) as usize
}

#[derive(Debug, Clone)]
pub enum SafetyError {
    TimeBudgetExceeded {
        budget: Duration,
        worst_case: Duration,
        model_type: ModelType,
        compressed_size: u64,
        recommendation: String,
    },
    MemoryBudgetExceeded {
        budget: usize,
        required: usize,
    },
    IntegrityCheckFailed {
        expected: u32,
        computed: u32,
    },
}
```

#### 1.6.3 Streaming Mode Jitter Analysis

For `LoadingMode::Streaming`, the ring buffer must absorb burst rates. The buffer size `B` must satisfy:

```
B ≥ (R_decomp_max - R_consume_min) × T_window
```

Where:
- `R_decomp_max`: Maximum decompression burst rate
- `R_consume_min`: Minimum consumption rate by inference engine
- `T_window`: Scheduling window (typically 1-10ms for real-time systems)

```rust
/// Calculate minimum ring buffer size for jitter-free streaming
pub fn min_ring_buffer_size(
    decomp_max_mbps: f64,
    consume_min_mbps: f64,
    window_ms: f64,
) -> usize {
    let rate_diff_mbps = decomp_max_mbps - consume_min_mbps;
    if rate_diff_mbps <= 0.0 {
        // Consumer is always faster; minimal buffer needed
        return 64 * 1024; // 64KB minimum
    }

    let buffer_mb = rate_diff_mbps * (window_ms / 1000.0);
    let buffer_bytes = (buffer_mb * 1024.0 * 1024.0).ceil() as usize;

    // Round up to page boundary (4KB alignment)
    ((buffer_bytes + 4095) / 4096) * 4096
}
```

### 1.7 Cryptographic Agility (Post-Quantum Ready)

To support algorithm rotation without breaking file format compatibility, the header includes a **cipher suite identifier**. This enables migration to post-quantum cryptography (NIST PQC standards) without format version bumps.

**Reference**: [Barker 2020] NIST SP 800-57 Key Management Recommendations

```rust
/// Supported cipher suites for cryptographic agility.
/// Allows rotation from classical to post-quantum algorithms.
///
/// # Design Rationale (Poka-yoke)
///
/// Hardcoding Ed25519 creates long-term risk. Post-quantum computers
/// could break classical signatures by 2030-2040. This enum allows
/// graceful migration without breaking existing files.
///
/// # References
///
/// - [NIST PQC] Post-Quantum Cryptography Standardization
/// - [Bernstein et al. 2012] Ed25519 specification
/// - [Cryptographic Agility RFC 7696]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CipherSuite {
    /// Current standard (2025): Fast, small keys
    /// - Sign: Ed25519 (ed25519-dalek)
    /// - Encrypt: XChaCha20-Poly1305
    /// - Hash: BLAKE3
    /// - KDF: Argon2id
    Standard2025 = 0x01,

    /// NIST PQC Standard (2030+): Post-Quantum resistant
    /// - Sign: ML-DSA-65 (Dilithium3)
    /// - KEM: ML-KEM-768 (Kyber768)
    /// - Hash: SHA3-256
    /// - KDF: HKDF-SHA3
    PostQuantum2030 = 0x02,

    /// Hybrid mode: Classical + Post-Quantum (transition period)
    /// - Sign: Ed25519 + ML-DSA-65 (both required)
    /// - KEM: X25519 + ML-KEM-768
    /// Provides security against both classical and quantum adversaries
    Hybrid2028 = 0x03,

    /// Government/High-Assurance (NSA Suite B compatible)
    /// - Sign: ECDSA P-384
    /// - Encrypt: AES-256-GCM
    /// - Hash: SHA-384
    /// Required for some government/defense applications
    GovHighAssurance = 0x04,

    /// Legacy support (deprecated, load-only)
    /// - Sign: RSA-2048
    /// - Encrypt: AES-128-CBC
    /// WARNING: Do not create new files with this suite
    LegacyRSA = 0xFF,
}

impl CipherSuite {
    /// Convert from u8 value with validation
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::Standard2025),
            0x02 => Some(Self::PostQuantum2030),
            0x03 => Some(Self::Hybrid2028),
            0x04 => Some(Self::GovHighAssurance),
            0xFF => Some(Self::LegacyRSA),
            _ => None,
        }
    }

    /// Check if this suite is considered secure (not deprecated)
    pub fn is_secure(&self) -> bool {
        !matches!(self, Self::LegacyRSA)
    }

    /// Check if this suite provides post-quantum resistance
    pub fn is_post_quantum(&self) -> bool {
        matches!(self, Self::PostQuantum2030 | Self::Hybrid2028)
    }

    /// Get signature size in bytes
    pub fn signature_size(&self) -> usize {
        match self {
            Self::Standard2025 => 64,         // Ed25519
            Self::PostQuantum2030 => 3293,    // ML-DSA-65
            Self::Hybrid2028 => 64 + 3293,    // Ed25519 + ML-DSA-65
            Self::GovHighAssurance => 96,     // ECDSA P-384
            Self::LegacyRSA => 256,           // RSA-2048
        }
    }

    /// Get public key size in bytes
    pub fn public_key_size(&self) -> usize {
        match self {
            Self::Standard2025 => 32,         // Ed25519
            Self::PostQuantum2030 => 1952,    // ML-DSA-65
            Self::Hybrid2028 => 32 + 1952,    // Ed25519 + ML-DSA-65
            Self::GovHighAssurance => 97,     // P-384 uncompressed
            Self::LegacyRSA => 256,           // RSA-2048
        }
    }
}

/// Runtime-supported cipher suites (compile-time configuration)
#[cfg(feature = "crypto-standard")]
pub const RUNTIME_SUPPORTED_SUITES: &[CipherSuite] = &[
    CipherSuite::Standard2025,
];

#[cfg(feature = "crypto-pqc")]
pub const RUNTIME_SUPPORTED_SUITES: &[CipherSuite] = &[
    CipherSuite::Standard2025,
    CipherSuite::PostQuantum2030,
    CipherSuite::Hybrid2028,
];

#[cfg(feature = "crypto-gov")]
pub const RUNTIME_SUPPORTED_SUITES: &[CipherSuite] = &[
    CipherSuite::Standard2025,
    CipherSuite::GovHighAssurance,
];

impl Header {
    /// Validates that the runtime supports the file's cipher suite
    ///
    /// # Poka-yoke (Mistake Proofing)
    ///
    /// This check prevents:
    /// 1. Loading files signed with deprecated algorithms
    /// 2. Loading files requiring crypto features not compiled in
    /// 3. Silently accepting unverifiable signatures
    pub fn validate_crypto(&self) -> Result<(), SecurityError> {
        // Check if cipher suite is recognized
        let suite = CipherSuite::from_u8(self.cipher_suite_id)
            .ok_or(SecurityError::UnknownCipherSuite {
                suite_id: self.cipher_suite_id,
            })?;

        // Check if cipher suite is deprecated
        if !suite.is_secure() {
            return Err(SecurityError::DeprecatedCrypto {
                suite,
                recommendation: "Re-sign model with Standard2025 or newer".to_string(),
            });
        }

        // Check if runtime supports this suite
        if !RUNTIME_SUPPORTED_SUITES.contains(&suite) {
            return Err(SecurityError::UnsupportedCrypto {
                suite,
                supported: RUNTIME_SUPPORTED_SUITES.to_vec(),
                recommendation: format!(
                    "Rebuild with feature '{}' or re-sign with supported suite",
                    feature_for_suite(suite)
                ),
            });
        }

        Ok(())
    }
}

fn feature_for_suite(suite: CipherSuite) -> &'static str {
    match suite {
        CipherSuite::Standard2025 => "crypto-standard",
        CipherSuite::PostQuantum2030 | CipherSuite::Hybrid2028 => "crypto-pqc",
        CipherSuite::GovHighAssurance => "crypto-gov",
        CipherSuite::LegacyRSA => "crypto-legacy",
    }
}

#[derive(Debug, Clone)]
pub enum SecurityError {
    UnknownCipherSuite { suite_id: u8 },
    DeprecatedCrypto { suite: CipherSuite, recommendation: String },
    UnsupportedCrypto { suite: CipherSuite, supported: Vec<CipherSuite>, recommendation: String },
    SignatureVerificationFailed,
    DecryptionFailed,
}
```

### 1.8 Execution Modalities (Universal Deployment)

A single `.apr` file deploys across **all** execution environments without modification. The format is runner-agnostic; the Trueno backend handles hardware abstraction.

| Runner | Environment | Backend | Use Case |
|--------|-------------|---------|----------|
| **Server** | Linux/macOS/Windows | Trueno SIMD (AVX2/AVX-512) | Batch inference, training |
| **GPU** | CUDA/Vulkan/Metal | Trueno wgpu | Large models, parallel inference |
| **Browser** | WASM playground | Trueno WASM SIMD | Interactive demos, model zoo |
| **Automotive ECU** | Bare metal `no_std` | Cortex-M4/R5 fixed-point | ADAS, brake control |
| **Aerospace** | RTOS (VxWorks/RTEMS) | Deterministic WCET path | Flight computers, guidance |
| **Edge** | Raspberry Pi/Jetson | ARM NEON | IoT, on-device inference |
| **Mobile** | iOS/Android | NEON/GPU | On-device ML apps |
| **realizár** | Inference engine | Any Trueno backend | Production serving |
| **batuta** | Orchestrator/Oracle | Distributed multi-node | Fleet coordination |

**Format Independence Guarantee**: The `.apr` file contains no runner-specific code. Backend selection occurs at load time via `LoadConfig::backend`.

```rust
/// Backend selection for model execution
#[derive(Debug, Clone, Copy)]
pub enum Backend {
    /// Auto-detect best available (default)
    Auto,
    /// CPU with SIMD (AVX2/AVX-512/NEON)
    CpuSimd,
    /// GPU via wgpu (Vulkan/Metal/DX12)
    Gpu,
    /// WebAssembly with SIMD128
    Wasm,
    /// Bare metal no_std (fixed-point arithmetic)
    BareMetal,
    /// Deterministic WCET mode for safety-critical
    SafetyCritical,
}
```

---

## 2. Memory Paging Architecture

### 2.1 Design Rationale (Toyota Way: Muda Elimination)

Traditional model loading suffers from Muda (waste) through:
1. **Overproduction**: Loading entire model when only subset needed
2. **Waiting**: Blocking on full decompression before first use
3. **Transportation**: Unnecessary data copies between buffers

The paging architecture eliminates these wastes through demand-driven loading with Trueno's SIMD-accelerated page operations.

### 2.2 Page Table Design

```rust
/// Page size aligned to SIMD vector width (AVX-512: 64 bytes)
/// Also aligned to common cache line size for optimal prefetch
pub const PAGE_SIZE: usize = 64 * 1024;  // 64 KB pages

/// Page states following virtual memory semantics [Vahalia 1996]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// Page not yet loaded (demand paging)
    NotPresent,

    /// Page loading in progress (async I/O)
    Loading,

    /// Page in memory, clean (can evict without writeback)
    Clean,

    /// Page in memory, accessed (LRU tracking)
    Accessed,

    /// Page locked in memory (inference in progress)
    Pinned,
}

/// Page table entry with metadata for eviction policy
#[derive(Debug, Clone)]
pub struct PageTableEntry {
    /// Offset in the .apr file
    pub file_offset: u64,

    /// Current state
    pub state: PageState,

    /// Access count (LFU metric)
    pub access_count: u64,

    /// Last access timestamp (LRU metric)
    pub last_access: u64,

    /// Page content (None if not present)
    pub data: Option<Arc<AlignedBuffer>>,

    /// Trueno tensor reference (for zero-copy SIMD access)
    pub tensor_ref: Option<trueno::TensorRef>,
}

/// SIMD-aligned buffer for Trueno zero-copy operations
#[repr(align(64))]  // AVX-512 alignment
pub struct AlignedBuffer {
    data: Vec<u8>,
    capacity: usize,
}
```

### 2.3 Eviction Policies

The paging system implements adaptive eviction based on access patterns:

```rust
/// Eviction policy selection (configurable per deployment)
#[derive(Debug, Clone, Copy, Default)]
pub enum EvictionPolicy {
    /// Least Recently Used (temporal locality)
    /// Best for: Sequential inference, time-series models
    #[default]
    LRU,

    /// Least Frequently Used (frequency locality)
    /// Best for: Random access, sparse models
    LFU,

    /// Adaptive Replacement Cache [Megiddo & Modha 2003]
    /// Best for: Mixed workloads, production deployments
    ARC,

    /// Clock algorithm (efficient approximation of LRU)
    /// Best for: Embedded systems with limited CPU
    Clock,

    /// No eviction (fixed memory pool)
    /// Best for: Deterministic embedded, NASA Level A
    Fixed,
}

/// Memory budget enforcement (Heijunka principle)
pub struct MemoryBudget {
    /// Maximum pages in memory
    pub max_pages: usize,

    /// High watermark (start eviction)
    pub high_watermark: usize,

    /// Low watermark (stop eviction)
    pub low_watermark: usize,

    /// Reserved pages (never evict)
    pub reserved_pages: HashSet<u64>,
}
```

### 2.4 Prefetch Heuristics

Intelligent prefetching reduces page fault latency using access pattern prediction:

```rust
/// Prefetch strategy based on model type
pub trait PrefetchStrategy {
    /// Predict next pages based on current access
    fn predict_next(&self, current_page: u64, history: &AccessHistory) -> Vec<u64>;

    /// Confidence score for prediction (0.0 - 1.0)
    fn confidence(&self) -> f32;
}

/// Sequential prefetch for dense models (Linear Regression, Neural Networks)
pub struct SequentialPrefetch {
    pub lookahead: usize,  // Default: 4 pages
}

/// Tree-based prefetch for ensemble models (Random Forest, XGBoost)
pub struct TreePrefetch {
    pub depth_lookahead: usize,  // Prefetch child nodes
}

/// Stride-based prefetch for convolutional models
pub struct StridePrefetch {
    pub detected_stride: Option<i64>,
    pub stride_confidence: f32,
}
```

### 2.5 SIGBUS Handling (Safety-Critical)

Per NASA NPR 7150.2D Section 3.8 (Fault Tolerance), memory-mapped regions must handle SIGBUS gracefully:

```rust
/// SIGBUS recovery strategy for safety-critical deployments
/// References: [Vahalia 1996], [McKusick & Karels 1988]
pub struct SigbusRecovery {
    /// Enable mmap checksumming before access (defensive)
    pub pre_access_checksum: bool,

    /// Fallback to read() on SIGBUS (recovery mode)
    pub fallback_to_read: bool,

    /// Maximum retry attempts before failure
    pub max_retries: u32,

    /// Alert callback for monitoring systems
    pub alert_callback: Option<Box<dyn Fn(SigbusEvent)>>,
}
```

---

## 3. Binary Caching and Bundling

### 3.1 Design Goals (Toyota Way: Just-In-Time)

The caching system implements Just-In-Time principles:
- **Right amount**: Cache only what's needed for current inference
- **Right time**: Prefetch before access, evict after use
- **Right place**: Hierarchical caching (L1 = hot, L2 = warm, L3 = cold storage)

### 3.2 Cache Hierarchy

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L0: Trueno Tensor Cache (SIMD-aligned, in-register)                    │
│     Size: CPU vector registers (256-512 bits per lane)                  │
│     Latency: 1 cycle                                                    │
│     Policy: Implicit (hardware managed)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ L1: Hot Model Cache (heap-allocated, aligned buffers)                  │
│     Size: 64 MB default (configurable)                                 │
│     Latency: < 100 ns                                                   │
│     Policy: LRU with pinning                                           │
│     Content: Decompressed model weights, bias vectors                   │
├─────────────────────────────────────────────────────────────────────────┤
│ L2: Warm Disk Cache (memory-mapped files)                              │
│     Size: 1 GB default (configurable)                                  │
│     Latency: < 1 ms (page fault + decompression)                       │
│     Policy: ARC (Adaptive Replacement Cache)                           │
│     Content: Decompressed .apr segments                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ L3: Cold Storage (filesystem or network)                               │
│     Size: Unbounded                                                     │
│     Latency: 10-100 ms (SSD), 50-500 ms (network)                      │
│     Policy: TTL-based expiration                                       │
│     Content: Original .apr files, remote model zoo                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Cache Entry Structure

```rust
/// Cache entry with comprehensive metadata for debugging and eviction
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Unique model identifier (SHA-256 of header + metadata)
    pub model_hash: [u8; 32],

    /// Model type for type-safe deserialization
    pub model_type: ModelType,

    /// Cached data (compressed or decompressed based on tier)
    pub data: CacheData,

    /// Entry metadata
    pub metadata: CacheMetadata,

    /// Access statistics for eviction
    pub stats: AccessStats,
}

#[derive(Debug, Clone)]
pub struct CacheMetadata {
    /// Original .apr file path (for invalidation)
    pub source_path: Option<PathBuf>,

    /// Source file modification time (staleness check)
    pub source_mtime: Option<SystemTime>,

    /// Cache entry creation time
    pub cached_at: SystemTime,

    /// Time-to-live (None = infinite)
    pub ttl: Option<Duration>,

    /// Entry size in bytes
    pub size_bytes: usize,

    /// Compression ratio achieved
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    pub hit_count: u64,
    pub miss_count: u64,
    pub last_access: u64,
    pub total_access_time_ns: u64,
    pub prefetch_hits: u64,
}
```

### 3.4 Binary Bundling for Distribution

For embedded deployment, models can be bundled into the binary using Rust's `include_bytes!` with a structured wrapper:

```rust
/// Compile-time model embedding for embedded/WASM deployment
///
/// # Example
///
/// ```rust,ignore
/// // Build script generates this from models/*.apr
/// static BUNDLED_MODELS: &[BundledModel] = &[
///     BundledModel::new(
///         "iris_classifier",
///         ModelType::RandomForest,
///         include_bytes!("../models/iris_rf.apr"),
///     ),
///     BundledModel::new(
///         "sentiment_analyzer",
///         ModelType::NaiveBayes,
///         include_bytes!("../models/sentiment_nb.apr"),
///     ),
/// ];
/// ```
#[derive(Debug)]
pub struct BundledModel {
    /// Human-readable model name
    pub name: &'static str,

    /// Model type for type safety
    pub model_type: ModelType,

    /// Embedded .apr bytes (compressed at compile time)
    pub data: &'static [u8],

    /// Original uncompressed size (for allocation)
    pub uncompressed_size: usize,

    /// Checksum for integrity verification
    pub checksum: u32,
}

/// Bundle configuration for build script
#[derive(Debug, Clone)]
pub struct BundleConfig {
    /// Source directory containing .apr files
    pub source_dir: PathBuf,

    /// Output Rust file with bundled models
    pub output_file: PathBuf,

    /// Compression level (0 = none, 1-3 = zstd levels)
    pub compression: CompressionLevel,

    /// Maximum bundle size (fail if exceeded)
    pub max_size_bytes: Option<usize>,

    /// Include test data for tiny models
    pub include_test_data: bool,
}
```

### 3.5 Bundled Model Registry

```rust
/// Runtime registry for bundled and dynamically loaded models
pub struct ModelRegistry {
    /// Bundled models (available at compile time)
    bundled: HashMap<String, &'static BundledModel>,

    /// Runtime-loaded models (from filesystem or network)
    dynamic: HashMap<String, Arc<LoadedModel<dyn ModelTrait>>>,

    /// Cache configuration
    cache_config: CacheConfig,

    /// Trueno backend for tensor operations
    backend: Backend,
}

impl ModelRegistry {
    /// Load model by name (checks bundled first, then cache, then filesystem)
    pub fn get<M: DeserializeOwned + 'static>(
        &mut self,
        name: &str,
        expected_type: ModelType,
    ) -> Result<Arc<M>, AprenderError> {
        // 1. Check bundled models (O(1), no allocation)
        if let Some(bundled) = self.bundled.get(name) {
            return self.load_bundled(bundled);
        }

        // 2. Check dynamic cache (O(1), may allocate)
        if let Some(cached) = self.dynamic.get(name) {
            return Ok(cached.clone().downcast()?);
        }

        // 3. Load from filesystem (O(n), allocates)
        Err(AprenderError::ModelNotFound(name.to_string()))
    }

    /// List all available models
    pub fn list(&self) -> Vec<ModelInfo> {
        let mut models: Vec<_> = self.bundled.values()
            .map(|b| ModelInfo::from_bundled(b))
            .collect();
        models.extend(self.dynamic.values().map(|d| d.info().clone()));
        models
    }
}
```

---

## 4. Data Embedding with Trueno Acceleration

### 4.1 Tiny Problem Data Embedding

For educational and testing purposes, models can bundle sample datasets:

```rust
/// Embedded test data for model validation and demos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedTestData {
    /// Feature matrix (flattened, row-major)
    pub x_data: Vec<f32>,

    /// Feature matrix shape (n_samples, n_features)
    pub x_shape: (usize, usize),

    /// Target vector (for supervised models)
    pub y_data: Option<Vec<f32>>,

    /// Feature names (for inspection/debugging)
    pub feature_names: Option<Vec<String>>,

    /// Sample identifiers (for traceability)
    pub sample_ids: Option<Vec<String>>,

    /// Data provenance (Toyota Way: traceability)
    pub provenance: DataProvenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProvenance {
    /// Original dataset name (e.g., "UCI Iris")
    pub source: String,

    /// Subset selection criteria
    pub subset_criteria: Option<String>,

    /// Preprocessing steps applied
    pub preprocessing: Vec<String>,

    /// Creation timestamp
    pub created_at: String,

    /// License/attribution
    pub license: Option<String>,
}
```

### 4.2 Compression Strategies

Data compression leverages Trueno's SIMD operations for both compression and decompression:

```rust
/// Data-aware compression strategy selection
#[derive(Debug, Clone, Copy)]
pub enum DataCompression {
    /// No compression (raw f32 values)
    None,

    /// Zstd compression (general purpose)
    /// Ratio: 2-10x, Speed: 500 MB/s decompress
    Zstd { level: u8 },

    /// Delta encoding + Zstd (time series, sorted data)
    /// Ratio: 5-20x for sequential data
    DeltaZstd { level: u8 },

    /// Quantization + entropy coding (ML-specific)
    /// Ratio: 4-8x with minimal accuracy loss
    QuantizedEntropy { bits: u8 },

    /// Sparse representation (for sparse features)
    /// Ratio: proportional to sparsity
    Sparse { threshold: f32 },
}

/// Trueno-accelerated decompression
pub fn decompress_with_trueno(
    compressed: &[u8],
    strategy: DataCompression,
    backend: &Backend,
) -> Result<trueno::Vector<f32>, AprenderError> {
    match strategy {
        DataCompression::None => {
            // Zero-copy conversion to Trueno vector
            let floats = bytemuck::cast_slice(compressed);
            Ok(trueno::Vector::from_slice(floats, backend.clone())?)
        }
        DataCompression::Zstd { .. } => {
            // Streaming decompression to Trueno buffer
            let decompressed = zstd::decode_all(compressed)?;
            let floats = bytemuck::cast_slice(&decompressed);
            Ok(trueno::Vector::from_slice(floats, backend.clone())?)
        }
        DataCompression::DeltaZstd { .. } => {
            // Delta decode using SIMD prefix sum
            let decompressed = zstd::decode_all(compressed)?;
            let deltas: &[f32] = bytemuck::cast_slice(&decompressed);
            let mut values = trueno::Vector::zeros(deltas.len(), backend.clone())?;

            // SIMD-accelerated prefix sum (cumulative sum of deltas)
            trueno::ops::prefix_sum(deltas, values.as_mut_slice())?;
            Ok(values)
        }
        DataCompression::Sparse { threshold } => {
            // Sparse representation: indices + values
            let (indices, values) = deserialize_sparse(compressed)?;
            let mut dense = trueno::Vector::zeros(indices.len(), backend.clone())?;

            // SIMD scatter operation
            trueno::ops::scatter(&indices, &values, dense.as_mut_slice())?;
            Ok(dense)
        }
        _ => todo!("Quantized entropy compression"),
    }
}
```

### 4.3 Efficient Small Model Representation

For tiny models (< 1 MB), specialized representations minimize overhead:

```rust
/// Compact representation for tiny models (educational/edge deployment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TinyModelRepr {
    /// Linear model: coefficients + intercept (< 1 KB typical)
    Linear {
        coefficients: Vec<f32>,
        intercept: f32,
    },

    /// Decision stump: single split (< 100 bytes)
    Stump {
        feature_idx: u16,
        threshold: f32,
        left_value: f32,
        right_value: f32,
    },

    /// Naive Bayes: means + variances per class (< 10 KB typical)
    NaiveBayes {
        class_priors: Vec<f32>,
        means: Vec<Vec<f32>>,     // [n_classes][n_features]
        variances: Vec<Vec<f32>>, // [n_classes][n_features]
    },

    /// K-Means: cluster centroids (< 100 KB typical)
    KMeans {
        centroids: Vec<Vec<f32>>,  // [n_clusters][n_features]
    },

    /// Compressed representation for larger tiny models
    Compressed {
        compression: DataCompression,
        data: Vec<u8>,
    },
}

impl TinyModelRepr {
    /// Total size in bytes (for bundling decisions)
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Linear { coefficients, .. } => coefficients.len() * 4 + 4,
            Self::Stump { .. } => 14,
            Self::NaiveBayes { class_priors, means, .. } => {
                let n_classes = class_priors.len();
                let n_features = means.first().map_or(0, Vec::len);
                n_classes * 4 + n_classes * n_features * 8
            }
            Self::KMeans { centroids } => {
                centroids.iter().map(|c| c.len() * 4).sum()
            }
            Self::Compressed { data, .. } => data.len(),
        }
    }
}
```

---

## 5. World's Fastest Model Type

### 5.1 Design Principles

Achieving world-class inference speed requires:

1. **Zero-copy data paths**: Model weights stay in aligned buffers accessed directly by SIMD operations
2. **Kernel fusion**: Combine sequential operations into single passes
3. **Memory locality**: Data layout optimized for cache line utilization
4. **Branch elimination**: Branchless algorithms where possible

### 5.2 Trueno-Native Model Format

```rust
/// Model format optimized for Trueno SIMD operations
///
/// Memory layout guarantees:
/// - 64-byte alignment (AVX-512 compatible)
/// - Contiguous storage (no pointer chasing)
/// - Row-major ordering (matches Trueno convention)
/// - Padding to SIMD width boundaries
#[repr(C, align(64))]
pub struct TruenoNativeModel {
    /// Model type identifier
    pub model_type: ModelType,

    /// Number of parameters
    pub n_params: u32,

    /// Number of features expected in input
    pub n_features: u32,

    /// Number of outputs (classes for classification, 1 for regression)
    pub n_outputs: u32,

    /// Reserved for future use (alignment padding)
    _reserved: [u8; 48],

    /// Model parameters (64-byte aligned)
    pub params: AlignedVec<f32>,

    /// Bias terms (64-byte aligned)
    pub bias: Option<AlignedVec<f32>>,

    /// Additional model-specific data
    pub extra: Option<ModelExtra>,
}

/// 64-byte aligned vector for SIMD operations
#[repr(C, align(64))]
pub struct AlignedVec<T> {
    data: Vec<T>,
    len: usize,
    capacity: usize,
}

impl<T: Copy + Default> AlignedVec<T> {
    /// Create with capacity rounded up to 64-byte boundary
    pub fn with_capacity(capacity: usize) -> Self {
        let aligned_cap = (capacity * size_of::<T>() + 63) / 64 * 64 / size_of::<T>();
        let mut data = Vec::with_capacity(aligned_cap);
        data.resize(capacity, T::default());
        Self {
            data,
            len: capacity,
            capacity: aligned_cap,
        }
    }

    /// Get raw pointer for SIMD operations (guaranteed 64-byte aligned)
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        debug_assert!(self.data.as_ptr() as usize % 64 == 0);
        self.data.as_ptr()
    }
}
```

### 5.3 SIMD-Optimized Inference Kernels

```rust
/// Linear model inference with Trueno SIMD acceleration
///
/// Performance: ~10 GFLOPS on AVX2, ~20 GFLOPS on AVX-512
/// Reference: [Intel Intrinsics Guide], [Fog 2023]
pub fn linear_predict_simd(
    model: &TruenoNativeModel,
    x: &trueno::Matrix<f32>,
    out: &mut trueno::Vector<f32>,
) -> Result<(), AprenderError> {
    // Validate dimensions
    if x.ncols() != model.n_features as usize {
        return Err(AprenderError::DimensionMismatch {
            expected: model.n_features as usize,
            got: x.ncols(),
        });
    }

    // Matrix-vector multiplication: y = X @ w + b
    // Uses Trueno's SIMD-accelerated matmul
    let weights = trueno::Vector::from_aligned_ptr(
        model.params.as_ptr(),
        model.n_params as usize,
        x.backend().clone(),
    )?;

    // Fused multiply-add: out = X @ w
    trueno::ops::matvec(x, &weights, out)?;

    // Add bias (fused with SIMD broadcast)
    if let Some(ref bias) = model.bias {
        let bias_vec = trueno::Vector::from_aligned_ptr(
            bias.as_ptr(),
            bias.len,
            x.backend().clone(),
        )?;
        trueno::ops::add_inplace(out, &bias_vec)?;
    }

    Ok(())
}

/// K-Means prediction with SIMD distance computation
///
/// Performance: O(n * k * d) with SIMD parallelism over d
/// Uses squared Euclidean distance (avoids sqrt in inner loop)
pub fn kmeans_predict_simd(
    model: &TruenoNativeModel,
    x: &trueno::Matrix<f32>,
    labels: &mut trueno::Vector<u32>,
) -> Result<(), AprenderError> {
    let n_clusters = model.n_outputs as usize;
    let n_features = model.n_features as usize;
    let n_samples = x.nrows();

    // Extract centroids from model params
    let centroids = trueno::Matrix::from_aligned_ptr(
        model.params.as_ptr(),
        n_clusters,
        n_features,
        x.backend().clone(),
    )?;

    // Compute all pairwise distances using SIMD
    // dist[i, j] = ||x[i] - centroids[j]||^2
    let mut distances = trueno::Matrix::zeros(n_samples, n_clusters, x.backend().clone())?;
    trueno::ops::pairwise_l2_squared(x, &centroids, &mut distances)?;

    // Find argmin for each sample (SIMD parallel over samples)
    trueno::ops::argmin_axis1(&distances, labels)?;

    Ok(())
}
```

### 5.4 Benchmark Targets

| Model Type | Dataset Size | Target Latency | Throughput |
|------------|--------------|----------------|------------|
| Linear (100 features) | 1K samples | < 10 μs | 100M predictions/s |
| Linear (10K features) | 1K samples | < 100 μs | 10M predictions/s |
| K-Means (10 clusters, 100d) | 1K samples | < 50 μs | 20M predictions/s |
| Random Forest (100 trees) | 1K samples | < 1 ms | 1M predictions/s |
| Neural Net (3 layers, 256 units) | 1K samples | < 500 μs | 2M predictions/s |

### 5.5 Bare Metal (`no_std`) Implementation

To prove the "Target Platforms: Embedded" claim, this section demonstrates how `.apr` models run on bare-metal microcontrollers **without an Operating System** (no heap, no `std` library).

**Target**: Automotive ECU (NXP S32G), Aerospace flight computer, Industrial PLC

**Toyota Way Principle**: Muda Elimination - Remove the waste of OS overhead entirely

#### 5.5.1 Memory Layout (Deterministic)

```rust
#![no_std]
#![no_main]

use aprender::embedded::{FixedBuffer, ModelView, PredictNoStd};
use aprender::format::ModelType;
use core::panic::PanicInfo;

/// Global static buffer in dedicated SRAM section (Heijunka: pre-allocated)
/// 128KB Tightly Coupled Memory (TCM) for deterministic access
///
/// # Safety Justification
///
/// Static mut is safe here because:
/// 1. Only accessed from single ISR (no data races)
/// 2. Initialized before interrupts enabled in main()
/// 3. TCM has deterministic latency (no cache misses)
#[link_section = ".ram_tcm"]
static mut MODEL_BUFFER: [u8; 128 * 1024] = [0u8; 128 * 1024];

/// Pre-computed model hash for integrity verification
/// Generated at compile time via build.rs
const MODEL_HASH: u32 = include!(concat!(env!("OUT_DIR"), "/model_hash.rs"));

/// Panic handler for safety-critical systems (Jidoka: stop immediately)
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // Trigger hardware watchdog reset
    unsafe { trigger_failsafe() };
    loop {}
}
```

#### 5.5.2 Initialization (Main Function)

```rust
#[no_mangle]
pub extern "C" fn main() -> ! {
    // 1. Initialize hardware
    init_clocks();
    init_gpio();
    init_watchdog();

    // 2. Load model from flash to TCM (one-time copy)
    // Model is embedded in .rodata section at compile time
    let model_bytes: &'static [u8] = include_bytes!("../models/brake_classifier.apr");

    // 3. Verify integrity before use (Jidoka)
    let computed_hash = crc32_compute(model_bytes);
    if computed_hash != MODEL_HASH {
        // Model corrupted - do not proceed
        unsafe { trigger_failsafe() };
    }

    // 4. Copy to TCM for deterministic access
    unsafe {
        MODEL_BUFFER[..model_bytes.len()].copy_from_slice(model_bytes);
    }

    // 5. Validate model can be parsed
    let model_view = unsafe { ModelView::from_bytes(&MODEL_BUFFER) };
    if model_view.is_err() {
        unsafe { trigger_failsafe() };
    }

    // 6. Enable interrupts (model now ready for inference)
    enable_sensor_interrupt();

    // 7. Enter low-power loop (inference happens in ISR)
    loop {
        wait_for_interrupt();
        kick_watchdog();
    }
}
```

#### 5.5.3 Real-Time Inference (ISR)

```rust
/// Interrupt Service Routine for sensor-triggered inference
///
/// # Timing Requirements (ISO 26262 ASIL-D)
///
/// - Maximum execution time: 100μs
/// - Jitter: < 5μs
/// - Stack usage: < 1KB
///
/// # Safety Properties
///
/// - Zero heap allocations (all stack-based)
/// - Zero syscalls (bare metal)
/// - Constant-time execution (no branches on secret data)
/// - Bounded recursion (none)
#[no_mangle]
#[link_section = ".fast_code"]  // Execute from ITCM for speed
pub extern "C" fn SENSOR_IRQ_Handler() {
    // 1. Read sensor data into stack-allocated array (No heap!)
    let mut inputs: [f32; 8] = [0.0; 8];
    read_wheel_speed_sensors(&mut inputs[0..4]);
    read_brake_pressure_sensors(&mut inputs[4..8]);

    // 2. Get zero-copy view into the pre-loaded model
    // SAFETY: MODEL_BUFFER initialized in main() before interrupts enabled
    let buffer_ref: &[u8] = unsafe { &MODEL_BUFFER };
    let model = match ModelView::from_bytes(buffer_ref) {
        Ok(m) => m,
        Err(_) => {
            // Model parsing failed - engage failsafe
            unsafe { trigger_failsafe() };
            return;
        }
    };

    // 3. Run inference (Zero allocations, SIMD-accelerated)
    // Uses Trueno-Micro intrinsics for Cortex-M (NEON/DSP)
    let mut output: [f32; 1] = [0.0];
    match model.predict_no_std(&inputs, &mut output) {
        Ok(()) => {
            // Successful inference
            let brake_probability = output[0];

            // 4. Apply control action
            if brake_probability > 0.9 {
                engage_emergency_brake();
            } else if brake_probability > 0.5 {
                apply_regenerative_braking(brake_probability);
            }
        }
        Err(_) => {
            // Inference failed - Jidoka: stop immediately
            unsafe { trigger_failsafe() };
        }
    }

    // 5. Clear interrupt flag
    clear_sensor_irq();
}
```

#### 5.5.4 Stack-Only Inference Trait

```rust
/// Inference trait for `no_std` environments
///
/// # Contract
///
/// Implementations MUST:
/// 1. Use only stack allocation
/// 2. Complete in bounded time (no unbounded loops)
/// 3. Never panic (return Result instead)
/// 4. Support constant-time execution for crypto
pub trait PredictNoStd {
    /// Maximum stack usage in bytes
    const MAX_STACK_BYTES: usize;

    /// Worst-case execution cycles
    const MAX_CYCLES: u32;

    /// Perform inference without heap allocation
    ///
    /// # Arguments
    /// * `input` - Feature vector (must match model's expected dimensions)
    /// * `output` - Output buffer (caller-allocated)
    ///
    /// # Errors
    /// * `InferenceError::DimensionMismatch` - Input size wrong
    /// * `InferenceError::OutputBufferTooSmall` - Output buffer insufficient
    /// * `InferenceError::NumericalInstability` - NaN/Inf detected
    fn predict_no_std(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError>;
}

/// Zero-copy model view for embedded systems
pub struct ModelView<'a> {
    /// Raw model bytes (points into static buffer)
    bytes: &'a [u8],

    /// Parsed header (stack-allocated)
    header: HeaderInfo,

    /// Coefficient pointer (into bytes)
    coefficients_offset: usize,

    /// Bias pointer (into bytes)
    bias_offset: usize,
}

impl<'a> ModelView<'a> {
    /// Create model view from bytes without allocation
    ///
    /// # Safety
    ///
    /// The `bytes` slice must remain valid for the lifetime of the ModelView.
    /// Caller must ensure bytes are not modified during use.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, ParseError> {
        // Validate header
        if bytes.len() < HEADER_SIZE {
            return Err(ParseError::TooSmall);
        }

        // Parse header (stack allocation only)
        let header = parse_header_no_alloc(&bytes[..HEADER_SIZE])?;

        // Validate magic number (Poka-yoke)
        if header.magic != MAGIC {
            return Err(ParseError::InvalidMagic);
        }

        // Verify checksum (Jidoka)
        let expected_crc = u32::from_le_bytes([
            bytes[bytes.len() - 4],
            bytes[bytes.len() - 3],
            bytes[bytes.len() - 2],
            bytes[bytes.len() - 1],
        ]);
        let computed_crc = crc32_no_alloc(&bytes[..bytes.len() - 4]);
        if expected_crc != computed_crc {
            return Err(ParseError::ChecksumMismatch);
        }

        Ok(Self {
            bytes,
            header,
            coefficients_offset: HEADER_SIZE + header.metadata_size as usize,
            bias_offset: HEADER_SIZE + header.metadata_size as usize
                + (header.n_features as usize * 4),
        })
    }
}

impl<'a> PredictNoStd for ModelView<'a> {
    const MAX_STACK_BYTES: usize = 256;  // Only loop variables
    const MAX_CYCLES: u32 = 10_000;      // ~50μs on 200MHz Cortex-M

    fn predict_no_std(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), InferenceError> {
        // Dimension check
        if input.len() != self.header.n_features as usize {
            return Err(InferenceError::DimensionMismatch {
                expected: self.header.n_features as usize,
                got: input.len(),
            });
        }

        if output.is_empty() {
            return Err(InferenceError::OutputBufferTooSmall);
        }

        // Linear model inference: y = x · w + b
        // Zero allocations, uses stack-only dot product
        let coefficients = unsafe {
            core::slice::from_raw_parts(
                self.bytes[self.coefficients_offset..].as_ptr() as *const f32,
                self.header.n_features as usize,
            )
        };

        let bias = unsafe {
            *(self.bytes[self.bias_offset..].as_ptr() as *const f32)
        };

        // SIMD dot product (Trueno-Micro for Cortex-M)
        let mut sum = 0.0f32;
        for (x, w) in input.iter().zip(coefficients.iter()) {
            sum += x * w;
        }
        output[0] = sum + bias;

        // Numerical stability check (Jidoka)
        if !output[0].is_finite() {
            return Err(InferenceError::NumericalInstability);
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InferenceError {
    DimensionMismatch { expected: usize, got: usize },
    OutputBufferTooSmall,
    NumericalInstability,
    ModelCorrupted,
}

#[derive(Debug, Clone, Copy)]
pub enum ParseError {
    TooSmall,
    InvalidMagic,
    ChecksumMismatch,
    UnsupportedVersion,
    UnsupportedModelType,
}
```

#### 5.5.5 Build Configuration for Embedded

```toml
# Cargo.toml for embedded target
[package]
name = "brake-controller"
version = "0.1.0"
edition = "2021"

[dependencies]
aprender = { version = "0.15", default-features = false, features = ["embedded"] }

[profile.release]
opt-level = "z"        # Optimize for size
lto = true             # Link-time optimization
codegen-units = 1      # Single codegen unit for better optimization
panic = "abort"        # No unwinding
strip = true           # Strip symbols

[profile.release.package."*"]
opt-level = "z"

# Target-specific configuration
[target.thumbv7em-none-eabihf]
runner = "probe-run --chip STM32F407VGT6"
rustflags = [
    "-C", "link-arg=-Tlink.x",
    "-C", "link-arg=-Map=output.map",
]
```

```bash
# Build for Cortex-M4F (automotive ECU)
cargo build --release --target thumbv7em-none-eabihf

# Check binary size (target: <64KB)
arm-none-eabi-size target/thumbv7em-none-eabihf/release/brake-controller
# Output:
#    text    data     bss     dec     hex filename
#   32768    2048   131072  165888   28800 brake-controller
```

### 5.6 GPU Acceleration Path

For large-scale inference, Trueno's GPU backend provides additional acceleration:

```rust
/// GPU inference configuration
#[derive(Debug, Clone)]
pub struct GpuInferenceConfig {
    /// Minimum batch size for GPU dispatch
    /// Below this threshold, SIMD CPU is faster due to transfer overhead
    pub min_batch_size: usize,  // Default: 1000

    /// Maximum GPU memory usage
    pub max_gpu_memory_bytes: usize,  // Default: 1 GB

    /// Enable async transfer (overlap compute and data movement)
    pub async_transfer: bool,

    /// Preferred GPU device (if multiple available)
    pub device_id: usize,
}

/// Automatic dispatch based on workload size
pub fn predict_auto_dispatch<M: TruenoNativeModelTrait>(
    model: &M,
    x: &trueno::Matrix<f32>,
    config: &GpuInferenceConfig,
) -> Result<trueno::Vector<f32>, AprenderError> {
    if x.nrows() >= config.min_batch_size && trueno::gpu_available() {
        model.predict_gpu(x, config)
    } else {
        model.predict_simd(x)
    }
}
```

---

## 6. Model Inspection Tooling

### 6.1 CLI Tool Design

```bash
# Inspect .apr file structure and metadata
apr-inspect model.apr

# Output (example):
# ═══════════════════════════════════════════════════════════════════════════════
# APR Model Inspector v1.0.0
# ═══════════════════════════════════════════════════════════════════════════════
#
# File: model.apr
# Size: 2.45 MB (compressed), 8.32 MB (uncompressed)
# Compression: Zstd (level 3), ratio: 3.4x
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ HEADER                                                                      │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Magic:           APRN (0x4150524E)                                          │
# │ Format Version:  1.0                                                        │
# │ Model Type:      RandomForest (0x0004)                                      │
# │ Flags:           SIGNED | TRUENO_NATIVE | HAS_MODEL_CARD                    │
# │ Checksum:        0xA3B2C1D0 (VALID)                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ METADATA                                                                    │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Model Name:      iris_random_forest_v2                                      │
# │ Description:     Iris species classifier (100 trees, max_depth=10)         │
# │ Created:         2025-12-08T14:32:00Z                                       │
# │ Aprender:        0.15.0                                                     │
# │ Training:        150 samples, 2.3 seconds                                   │
# │ Hyperparameters:                                                            │
# │   - n_estimators: 100                                                       │
# │   - max_depth: 10                                                           │
# │   - min_samples_split: 2                                                    │
# │ Metrics:                                                                    │
# │   - accuracy: 0.9667                                                        │
# │   - f1_macro: 0.9662                                                        │
# │   - cv_score_mean: 0.9533                                                   │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SIGNATURE (Ed25519)                                                         │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Signer:          noah@paiml.com                                             │
# │ Public Key:      3gG7...Kx2F (fingerprint)                                  │
# │ Signature:       VALID                                                      │
# │ Signed At:       2025-12-08T14:35:00Z                                       │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MODEL STRUCTURE                                                             │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Trees: 100                                                                  │
# │ Features: 4 (sepal_length, sepal_width, petal_length, petal_width)         │
# │ Classes: 3 (setosa, versicolor, virginica)                                 │
# │ Total Nodes: 2,847                                                          │
# │ Max Depth: 10                                                               │
# │ Parameters: 34,164 (133.5 KB uncompressed)                                  │
# └─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Programmatic Inspection API

```rust
/// Model inspection result with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionResult {
    /// File path (if loaded from file)
    pub source: Option<PathBuf>,

    /// File size information
    pub size: SizeInfo,

    /// Header information (always available)
    pub header: HeaderInfo,

    /// Metadata (if present)
    pub metadata: Option<MetadataInfo>,

    /// Signature verification result (if signed)
    pub signature: Option<SignatureInfo>,

    /// Model structure analysis
    pub structure: ModelStructure,

    /// Quality score (0-100)
    pub quality_score: Option<QualityScore>,

    /// Embedded test data summary (if present)
    pub test_data: Option<TestDataSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeInfo {
    pub compressed_bytes: u64,
    pub uncompressed_bytes: u64,
    pub compression_ratio: f32,
    pub header_bytes: u64,
    pub metadata_bytes: u64,
    pub payload_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStructure {
    /// Model type
    pub model_type: ModelType,

    /// Number of input features
    pub n_features: usize,

    /// Number of outputs
    pub n_outputs: usize,

    /// Total parameters
    pub n_parameters: usize,

    /// Model-specific structure info
    pub details: ModelDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelDetails {
    Linear {
        has_intercept: bool,
        regularization: Option<String>,
    },
    Tree {
        n_nodes: usize,
        max_depth: usize,
        feature_importances: Vec<(String, f32)>,
    },
    Ensemble {
        n_estimators: usize,
        base_model: Box<ModelDetails>,
        aggregation: String,
    },
    Neural {
        layers: Vec<LayerInfo>,
        total_params: usize,
        activation: String,
    },
    Clustering {
        n_clusters: usize,
        algorithm: String,
    },
}

/// Inspect an .apr file without loading the full model
pub fn inspect(path: impl AsRef<Path>) -> Result<InspectionResult, AprenderError> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    // Read header (32 bytes)
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = parse_header(&header_bytes)?;

    // Read metadata
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = rmp_serde::from_slice(&metadata_bytes)?;

    // Analyze structure without decompressing payload
    let structure = analyze_structure(&header, &metadata)?;

    Ok(InspectionResult {
        source: Some(path.as_ref().to_path_buf()),
        size: compute_size_info(&header, path.as_ref())?,
        header: HeaderInfo::from(&header),
        metadata: Some(MetadataInfo::from(&metadata)),
        signature: verify_signature_if_present(&reader, &header)?,
        structure,
        quality_score: None,  // Computed separately via score() function
        test_data: None,
    })
}
```

### 6.3 Diff Tool for Model Comparison

```bash
# Compare two model versions
apr-diff model_v1.apr model_v2.apr

# Output (example):
# ═══════════════════════════════════════════════════════════════════════════════
# APR Model Diff
# ═══════════════════════════════════════════════════════════════════════════════
#
# Left:  model_v1.apr (v1, 2025-12-01)
# Right: model_v2.apr (v2, 2025-12-08)
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ METADATA CHANGES                                                            │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ - Hyperparameters:                                                          │
# │     n_estimators: 50 → 100 (+100%)                                          │
# │     max_depth: 5 → 10 (+100%)                                               │
# │ - Metrics:                                                                  │
# │     accuracy: 0.9333 → 0.9667 (+3.57%)                                      │
# │     f1_macro: 0.9298 → 0.9662 (+3.91%)                                      │
# │ - Size: 1.2 MB → 2.45 MB (+104%)                                            │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ STRUCTURAL CHANGES                                                          │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ Trees: 50 → 100 (+100%)                                                     │
# │ Total Nodes: 1,423 → 2,847 (+100%)                                          │
# │ Parameters: 17,082 → 34,164 (+100%)                                         │
# │ Feature Importance Changes:                                                 │
# │   petal_length: 0.52 → 0.48 (-7.7%)                                         │
# │   petal_width: 0.34 → 0.38 (+11.8%)                                         │
# │   sepal_length: 0.09 → 0.10 (+11.1%)                                        │
# │   sepal_width: 0.05 → 0.04 (-20.0%)                                         │
# └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 100-Point Model Quality Scoring

### 7.1 Scoring Framework Overview

The quality scoring system evaluates models across **seven dimensions** based on data science and ML best practices, aligned with Toyota Way principles:

| Dimension | Max Points | Toyota Way Principle |
|-----------|-----------|---------------------|
| **Accuracy & Performance** | 25 | Kaizen (continuous improvement) |
| **Generalization & Robustness** | 20 | Jidoka (quality built-in) |
| **Model Complexity** | 15 | Muda elimination (waste reduction) |
| **Documentation & Provenance** | 15 | Genchi Genbutsu (go and see) |
| **Reproducibility** | 15 | Standardization |
| **Fairness & Bias** | 10 | Respect for People |
| **Security & Safety** | 10 | Poka-yoke (error-proofing) |
| **TOTAL** | **110** | (Normalized to 100) |

**Note**: Scores are normalized to 100 points for final grading. A model achieving 88/110 would receive a normalized score of 80/100 (Grade B+).

### 7.2 Scoring Algorithm

```rust
/// 100-point model quality score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScore {
    /// Total score (0-100)
    pub total: f32,

    /// Grade letter (A+, A, A-, B+, ...)
    pub grade: Grade,

    /// Individual dimension scores
    pub dimensions: DimensionScores,

    /// Detailed findings and recommendations
    pub findings: Vec<Finding>,

    /// Critical issues that must be addressed
    pub critical_issues: Vec<CriticalIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScores {
    pub accuracy_performance: DimensionScore,
    pub generalization_robustness: DimensionScore,
    pub model_complexity: DimensionScore,
    pub documentation_provenance: DimensionScore,
    pub reproducibility: DimensionScore,
    pub security_safety: DimensionScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    pub score: f32,
    pub max_score: f32,
    pub percentage: f32,
    pub breakdown: Vec<(String, f32, f32)>,  // (criterion, score, max)
}

/// Compute quality score for a model
pub fn compute_quality_score(
    model: &InspectionResult,
    test_data: Option<&TestData>,
    config: &ScoringConfig,
) -> Result<QualityScore, AprenderError> {
    let mut total = 0.0;
    let mut findings = Vec::new();
    let mut critical = Vec::new();

    // Dimension 1: Accuracy & Performance (25 points)
    let accuracy = score_accuracy_performance(model, test_data, &mut findings)?;
    total += accuracy.score;

    // Dimension 2: Generalization & Robustness (20 points)
    let generalization = score_generalization_robustness(model, test_data, &mut findings)?;
    total += generalization.score;

    // Dimension 3: Model Complexity (15 points)
    let complexity = score_model_complexity(model, &mut findings)?;
    total += complexity.score;

    // Dimension 4: Documentation & Provenance (15 points)
    let documentation = score_documentation_provenance(model, &mut findings, &mut critical)?;
    total += documentation.score;

    // Dimension 5: Reproducibility (15 points)
    let reproducibility = score_reproducibility(model, &mut findings)?;
    total += reproducibility.score;

    // Dimension 6: Security & Safety (10 points)
    let security = score_security_safety(model, &mut findings, &mut critical)?;
    total += security.score;

    Ok(QualityScore {
        total,
        grade: Grade::from_score(total),
        dimensions: DimensionScores {
            accuracy_performance: accuracy,
            generalization_robustness: generalization,
            model_complexity: complexity,
            documentation_provenance: documentation,
            reproducibility: reproducibility,
            security_safety: security,
        },
        findings,
        critical_issues: critical,
    })
}
```

### 7.3 Dimension 1: Accuracy & Performance (25 points)

```rust
fn score_accuracy_performance(
    model: &InspectionResult,
    test_data: Option<&TestData>,
    findings: &mut Vec<Finding>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 1.1 Primary metric meets threshold (10 points)
    // References: [Raschka 2018], [Hastie et al. 2009]
    if let Some(metrics) = &model.metadata.as_ref().and_then(|m| m.metrics.as_ref()) {
        let primary_metric = match model.structure.model_type {
            ModelType::LinearRegression => metrics.get("r2_score"),
            ModelType::LogisticRegression | ModelType::RandomForest => metrics.get("accuracy"),
            ModelType::KMeans => metrics.get("silhouette_score"),
            _ => metrics.get("primary_score"),
        };

        if let Some(value) = primary_metric {
            let threshold = get_acceptable_threshold(model.structure.model_type);
            let metric_score = (value / threshold).min(1.0) * 10.0;
            score += metric_score;
            breakdown.push(("primary_metric".to_string(), metric_score, 10.0));
        } else {
            findings.push(Finding::Warning {
                message: "No primary metric recorded in model metadata".to_string(),
                recommendation: "Include primary evaluation metric during training".to_string(),
            });
        }
    }

    // 1.2 Cross-validation performed (8 points)
    // References: [Kohavi 1995], [Varma & Simon 2006]
    if let Some(cv_score) = model.metadata.as_ref()
        .and_then(|m| m.metrics.as_ref())
        .and_then(|metrics| metrics.get("cv_score_mean"))
    {
        let cv_std = model.metadata.as_ref()
            .and_then(|m| m.metrics.as_ref())
            .and_then(|metrics| metrics.get("cv_score_std"))
            .unwrap_or(&0.0);

        // Penalize high variance (indicates overfitting risk)
        let cv_quality = if *cv_std < 0.05 { 8.0 }
                        else if *cv_std < 0.1 { 6.0 }
                        else { 4.0 };
        score += cv_quality;
        breakdown.push(("cross_validation".to_string(), cv_quality, 8.0));
    } else {
        findings.push(Finding::Info {
            message: "No cross-validation results found".to_string(),
            recommendation: "Use k-fold cross-validation to estimate generalization".to_string(),
        });
    }

    // 1.3 Inference latency documented (4 points)
    if model.metadata.as_ref()
        .and_then(|m| m.metrics.as_ref())
        .and_then(|metrics| metrics.get("inference_latency_ms"))
        .is_some()
    {
        score += 4.0;
        breakdown.push(("latency_documented".to_string(), 4.0, 4.0));
    }

    // 1.4 Live evaluation on test data (3 points, bonus)
    if let Some(_test_data) = test_data {
        // Perform live evaluation
        score += 3.0;
        breakdown.push(("live_evaluation".to_string(), 3.0, 3.0));
    }

    Ok(DimensionScore {
        score,
        max_score: 25.0,
        percentage: score / 25.0 * 100.0,
        breakdown,
    })
}
```

### 7.4 Dimension 2: Generalization & Robustness (20 points)

```rust
fn score_generalization_robustness(
    model: &InspectionResult,
    test_data: Option<&TestData>,
    findings: &mut Vec<Finding>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 2.1 Train/test split used (5 points)
    // References: [Bishop 2006], [Murphy 2022]
    if model.metadata.as_ref()
        .and_then(|m| m.training.as_ref())
        .map(|t| t.test_size.is_some())
        .unwrap_or(false)
    {
        score += 5.0;
        breakdown.push(("train_test_split".to_string(), 5.0, 5.0));
    }

    // 2.2 Regularization applied (5 points, if applicable)
    // References: [Tibshirani 1996], [Hastie et al. 2009]
    let needs_regularization = matches!(
        model.structure.model_type,
        ModelType::LinearRegression | ModelType::LogisticRegression | ModelType::NeuralSequential
    );

    if needs_regularization {
        if model.metadata.as_ref()
            .and_then(|m| m.hyperparameters.as_ref())
            .map(|h| h.contains_key("alpha") || h.contains_key("lambda") || h.contains_key("l2_penalty"))
            .unwrap_or(false)
        {
            score += 5.0;
            breakdown.push(("regularization".to_string(), 5.0, 5.0));
        } else {
            findings.push(Finding::Warning {
                message: "No regularization detected for linear/neural model".to_string(),
                recommendation: "Consider adding L2 regularization to prevent overfitting".to_string(),
            });
        }
    } else {
        score += 5.0;  // N/A, give full points
        breakdown.push(("regularization".to_string(), 5.0, 5.0));
    }

    // 2.3 Training/test performance gap (5 points)
    // References: [Domingos 2012], [Ng & Jordan 2002]
    if let (Some(train_score), Some(test_score)) = (
        get_metric(&model.metadata, "train_score"),
        get_metric(&model.metadata, "test_score"),
    ) {
        let gap = train_score - test_score;
        let gap_score = if gap < 0.05 { 5.0 }
                       else if gap < 0.1 { 3.0 }
                       else if gap < 0.2 { 1.0 }
                       else { 0.0 };
        score += gap_score;
        breakdown.push(("generalization_gap".to_string(), gap_score, 5.0));

        if gap >= 0.1 {
            findings.push(Finding::Warning {
                message: format!("High train/test gap detected: {:.1}%", gap * 100.0),
                recommendation: "Model may be overfitting. Consider regularization or simpler model.".to_string(),
            });
        }
    }

    // 2.4 Handles edge cases (5 points)
    // Evidence of handling: empty input, NaN, extreme values
    let edge_case_handling = model.metadata.as_ref()
        .and_then(|m| m.custom.as_ref())
        .and_then(|c| c.get("edge_case_tests"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if edge_case_handling {
        score += 5.0;
        breakdown.push(("edge_cases".to_string(), 5.0, 5.0));
    }

    Ok(DimensionScore {
        score,
        max_score: 20.0,
        percentage: score / 20.0 * 100.0,
        breakdown,
    })
}
```

### 7.5 Dimension 3: Model Complexity (15 points)

```rust
fn score_model_complexity(
    model: &InspectionResult,
    findings: &mut Vec<Finding>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 3.1 Parameter efficiency (5 points)
    // References: [Akaike 1974], [Schwarz 1978]
    // Metric: parameters per training sample
    if let Some(n_samples) = model.metadata.as_ref()
        .and_then(|m| m.training.as_ref())
        .and_then(|t| t.n_samples)
    {
        let params_per_sample = model.structure.n_parameters as f64 / n_samples as f64;

        // Rule of thumb: < 0.1 params/sample is efficient
        let efficiency_score = if params_per_sample < 0.1 { 5.0 }
                              else if params_per_sample < 0.5 { 4.0 }
                              else if params_per_sample < 1.0 { 3.0 }
                              else if params_per_sample < 5.0 { 2.0 }
                              else { 1.0 };
        score += efficiency_score;
        breakdown.push(("parameter_efficiency".to_string(), efficiency_score, 5.0));

        if params_per_sample > 1.0 {
            findings.push(Finding::Info {
                message: format!("High parameter count relative to data: {:.2} params/sample", params_per_sample),
                recommendation: "Consider feature selection or simpler model architecture".to_string(),
            });
        }
    }

    // 3.2 Model interpretability (5 points)
    // References: [Ribeiro et al. 2016], [Lundberg & Lee 2017]
    let interpretability_score = match model.structure.model_type {
        ModelType::LinearRegression | ModelType::LogisticRegression => 5.0,  // Highly interpretable
        ModelType::DecisionTree | ModelType::NaiveBayes => 4.0,  // Interpretable
        ModelType::RandomForest | ModelType::GradientBoosting => 3.0,  // Partially interpretable
        ModelType::Knn => 2.0,  // Instance-based
        ModelType::NeuralSequential | ModelType::NeuralCustom => 1.0,  // Black box
        _ => 2.5,
    };
    score += interpretability_score;
    breakdown.push(("interpretability".to_string(), interpretability_score, 5.0));

    // 3.3 Feature importance available (5 points)
    let has_feature_importance = model.metadata.as_ref()
        .and_then(|m| m.custom.as_ref())
        .map(|c| c.contains_key("feature_importance"))
        .unwrap_or(false);

    if has_feature_importance || matches!(model.structure.details, ModelDetails::Tree { .. }) {
        score += 5.0;
        breakdown.push(("feature_importance".to_string(), 5.0, 5.0));
    } else {
        findings.push(Finding::Info {
            message: "No feature importance information available".to_string(),
            recommendation: "Include feature importance for model interpretability".to_string(),
        });
    }

    Ok(DimensionScore {
        score,
        max_score: 15.0,
        percentage: score / 15.0 * 100.0,
        breakdown,
    })
}
```

### 7.6 Dimension 4: Documentation & Provenance (15 points)

```rust
fn score_documentation_provenance(
    model: &InspectionResult,
    findings: &mut Vec<Finding>,
    critical: &mut Vec<CriticalIssue>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 4.1 Model name and description (3 points)
    if model.metadata.as_ref().and_then(|m| m.model_name.as_ref()).is_some() {
        score += 1.5;
    }
    if model.metadata.as_ref().and_then(|m| m.description.as_ref()).is_some() {
        score += 1.5;
    }
    breakdown.push(("name_description".to_string(), score.min(3.0), 3.0));

    // 4.2 Training provenance (4 points)
    // References: [Gebru et al. 2021] (Datasheets for Datasets)
    let mut provenance_score = 0.0;
    if let Some(training) = model.metadata.as_ref().and_then(|m| m.training.as_ref()) {
        if training.source.is_some() { provenance_score += 1.0; }
        if training.n_samples.is_some() { provenance_score += 1.0; }
        if training.duration_ms.is_some() { provenance_score += 1.0; }
        if training.random_seed.is_some() { provenance_score += 1.0; }
    }
    score += provenance_score;
    breakdown.push(("training_provenance".to_string(), provenance_score, 4.0));

    if provenance_score < 2.0 {
        findings.push(Finding::Warning {
            message: "Incomplete training provenance".to_string(),
            recommendation: "Record data source, sample count, training duration, and random seed".to_string(),
        });
    }

    // 4.3 Hyperparameters documented (4 points)
    if let Some(hyperparams) = model.metadata.as_ref().and_then(|m| m.hyperparameters.as_ref()) {
        let hp_score = (hyperparams.len() as f32 / 5.0).min(1.0) * 4.0;
        score += hp_score;
        breakdown.push(("hyperparameters".to_string(), hp_score, 4.0));
    }

    // 4.4 Model card present (4 points)
    // References: [Mitchell et al. 2019] (Model Cards)
    if model.header.flags.contains(Flags::HAS_MODEL_CARD) {
        score += 4.0;
        breakdown.push(("model_card".to_string(), 4.0, 4.0));
    } else {
        findings.push(Finding::Info {
            message: "No model card attached".to_string(),
            recommendation: "Add model card for comprehensive documentation (see Mitchell et al. 2019)".to_string(),
        });
    }

    Ok(DimensionScore {
        score,
        max_score: 15.0,
        percentage: score / 15.0 * 100.0,
        breakdown,
    })
}
```

### 7.7 Dimension 5: Reproducibility (15 points)

```rust
fn score_reproducibility(
    model: &InspectionResult,
    findings: &mut Vec<Finding>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 5.1 Random seed recorded (5 points)
    // References: [Pineau et al. 2021] (ML Reproducibility Checklist)
    if model.metadata.as_ref()
        .and_then(|m| m.training.as_ref())
        .and_then(|t| t.random_seed)
        .is_some()
    {
        score += 5.0;
        breakdown.push(("random_seed".to_string(), 5.0, 5.0));
    } else {
        findings.push(Finding::Warning {
            message: "No random seed recorded".to_string(),
            recommendation: "Set and record random seed for reproducibility".to_string(),
        });
    }

    // 5.2 Framework version recorded (3 points)
    if model.metadata.as_ref().and_then(|m| m.aprender_version.as_ref()).is_some() {
        score += 3.0;
        breakdown.push(("framework_version".to_string(), 3.0, 3.0));
    }

    // 5.3 Data preprocessing documented (4 points)
    let preprocessing_documented = model.metadata.as_ref()
        .and_then(|m| m.custom.as_ref())
        .map(|c| c.contains_key("preprocessing_steps"))
        .unwrap_or(false);

    if preprocessing_documented {
        score += 4.0;
        breakdown.push(("preprocessing".to_string(), 4.0, 4.0));
    }

    // 5.4 Checksum/hash for integrity (3 points)
    // Always present in valid .apr files
    score += 3.0;
    breakdown.push(("checksum".to_string(), 3.0, 3.0));

    Ok(DimensionScore {
        score,
        max_score: 15.0,
        percentage: score / 15.0 * 100.0,
        breakdown,
    })
}
```

### 7.8 Dimension 6: Fairness & Bias (10 points)

**Toyota Way Principle**: Respect for People

A model can achieve high accuracy while exhibiting discriminatory behavior against protected groups. This dimension measures algorithmic fairness, ensuring ML systems respect human dignity.

**References**: [Mehrabi et al. 2021], [Barocas et al. 2019], [Feldman et al. 2015]

```rust
/// Fairness metrics following the "Four-Fifths Rule" (EEOC)
/// and academic fairness definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessMetrics {
    /// Disparate Impact Ratio (DIR)
    /// DIR = P(positive | unprivileged) / P(positive | privileged)
    /// Target: 0.8 <= DIR <= 1.25 (Four-Fifths Rule)
    pub disparate_impact_ratio: f32,

    /// Equal Opportunity Difference (EOD)
    /// EOD = |TPR_privileged - TPR_unprivileged|
    /// Target: EOD < 0.1 (10% tolerance)
    pub equal_opportunity_diff: f32,

    /// Demographic Parity Difference (DPD)
    /// DPD = |P(Y=1|privileged) - P(Y=1|unprivileged)|
    /// Target: DPD < 0.1
    pub demographic_parity_diff: f32,

    /// Calibration error per group
    /// Measures if predicted probabilities match actual outcomes per group
    pub calibration_error: HashMap<String, f32>,
}

/// Protected attributes that should be checked for bias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectedAttribute {
    /// Attribute name (e.g., "gender", "race", "age_group")
    pub name: String,

    /// Privileged group value(s)
    pub privileged: Vec<String>,

    /// Unprivileged group value(s)
    pub unprivileged: Vec<String>,
}

fn score_fairness_bias(
    model: &InspectionResult,
    test_data: Option<&TestData>,
    protected_attrs: &[ProtectedAttribute],
    findings: &mut Vec<Finding>,
    critical: &mut Vec<CriticalIssue>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 7.8.1 Disparate Impact Ratio (4 points)
    // Reference: [Feldman et al. 2015], EEOC Four-Fifths Rule
    if let Some(data) = test_data {
        for attr in protected_attrs {
            let (privileged, unprivileged) = split_by_group(data, attr)?;
            let rate_priv = positive_rate(&privileged)?;
            let rate_unpriv = positive_rate(&unprivileged)?;

            let dir = if rate_priv > 0.0 { rate_unpriv / rate_priv } else { 1.0 };

            // Scoring based on Four-Fifths Rule
            let dir_score = if dir >= 0.8 && dir <= 1.25 {
                4.0  // Perfect: no disparate impact
            } else if dir >= 0.6 && dir <= 1.5 {
                2.0  // Warning: potential bias
            } else {
                0.0  // Critical: significant bias detected
            };

            if dir_score == 0.0 {
                critical.push(CriticalIssue {
                    severity: Severity::High,
                    message: format!(
                        "Disparate impact detected for '{}': DIR = {:.3}",
                        attr.name, dir
                    ),
                    action: format!(
                        "Review training data for bias. Consider resampling or fairness constraints."
                    ),
                });
            } else if dir_score == 2.0 {
                findings.push(Finding::Warning {
                    message: format!(
                        "Potential disparate impact for '{}': DIR = {:.3} (target: 0.8-1.25)",
                        attr.name, dir
                    ),
                    recommendation: "Monitor this metric and consider mitigation strategies".to_string(),
                });
            }

            score += dir_score / protected_attrs.len() as f32;
        }
        breakdown.push(("disparate_impact".to_string(), score.min(4.0), 4.0));
    } else {
        // No test data available - cannot compute fairness metrics
        findings.push(Finding::Info {
            message: "Fairness metrics not computed (no test data provided)".to_string(),
            recommendation: "Provide test data with protected attributes for fairness evaluation".to_string(),
        });
    }

    // 7.8.2 Equal Opportunity (3 points)
    // Reference: [Hardt et al. 2016] "Equality of Opportunity in Supervised Learning"
    if let Some(data) = test_data {
        for attr in protected_attrs {
            let eod = compute_equal_opportunity_diff(data, attr)?;

            let eod_score = if eod < 0.05 {
                3.0  // Excellent
            } else if eod < 0.1 {
                2.0  // Good
            } else if eod < 0.2 {
                1.0  // Marginal
            } else {
                0.0  // Failing
            };

            if eod_score <= 1.0 {
                findings.push(Finding::Warning {
                    message: format!(
                        "Equal opportunity difference for '{}': {:.1}% (target: <10%)",
                        attr.name, eod * 100.0
                    ),
                    recommendation: "True positive rates differ significantly between groups".to_string(),
                });
            }

            score += eod_score / protected_attrs.len() as f32;
        }
        breakdown.push(("equal_opportunity".to_string(), (score - 4.0).max(0.0).min(3.0), 3.0));
    }

    // 7.8.3 Fairness Documentation (3 points)
    // Did the model document fairness considerations?
    let fairness_documented = model.metadata.as_ref()
        .and_then(|m| m.custom.as_ref())
        .map(|c| {
            c.contains_key("fairness_evaluation") ||
            c.contains_key("protected_attributes") ||
            c.contains_key("bias_mitigation")
        })
        .unwrap_or(false);

    if fairness_documented {
        score += 3.0;
        breakdown.push(("fairness_documentation".to_string(), 3.0, 3.0));
    } else {
        findings.push(Finding::Info {
            message: "No fairness documentation in model metadata".to_string(),
            recommendation: "Document fairness evaluation methodology and results".to_string(),
        });
    }

    Ok(DimensionScore {
        score,
        max_score: 10.0,
        percentage: score / 10.0 * 100.0,
        breakdown,
    })
}

/// Split test data by protected attribute
fn split_by_group(
    data: &TestData,
    attr: &ProtectedAttribute,
) -> Result<(TestData, TestData), AprenderError> {
    let attr_idx = data.feature_names.as_ref()
        .and_then(|names| names.iter().position(|n| n == &attr.name))
        .ok_or(AprenderError::MissingField(attr.name.clone()))?;

    let mut privileged = TestData::empty_like(data);
    let mut unprivileged = TestData::empty_like(data);

    for i in 0..data.n_samples() {
        let value = data.get_feature(i, attr_idx)?;
        let value_str = format!("{}", value);

        if attr.privileged.contains(&value_str) {
            privileged.push_sample(data.get_sample(i)?);
        } else if attr.unprivileged.contains(&value_str) {
            unprivileged.push_sample(data.get_sample(i)?);
        }
    }

    Ok((privileged, unprivileged))
}

/// Compute positive prediction rate for a group
fn positive_rate(data: &TestData) -> Result<f32, AprenderError> {
    if data.n_samples() == 0 {
        return Ok(0.0);
    }

    let positives = data.predictions()
        .iter()
        .filter(|&&p| p > 0.5)  // Binary classification threshold
        .count();

    Ok(positives as f32 / data.n_samples() as f32)
}

/// Compute Equal Opportunity Difference (TPR gap)
fn compute_equal_opportunity_diff(
    data: &TestData,
    attr: &ProtectedAttribute,
) -> Result<f32, AprenderError> {
    let (privileged, unprivileged) = split_by_group(data, attr)?;

    let tpr_priv = true_positive_rate(&privileged)?;
    let tpr_unpriv = true_positive_rate(&unprivileged)?;

    Ok((tpr_priv - tpr_unpriv).abs())
}

/// Compute True Positive Rate for a group
fn true_positive_rate(data: &TestData) -> Result<f32, AprenderError> {
    let actual_positives: Vec<_> = data.labels()
        .iter()
        .enumerate()
        .filter(|(_, &l)| l > 0.5)
        .map(|(i, _)| i)
        .collect();

    if actual_positives.is_empty() {
        return Ok(0.0);
    }

    let true_positives = actual_positives.iter()
        .filter(|&&i| data.predictions()[i] > 0.5)
        .count();

    Ok(true_positives as f32 / actual_positives.len() as f32)
}
```

#### 7.8.4 Fairness Thresholds by Domain

Different domains have different fairness requirements:

| Domain | Disparate Impact | Equal Opportunity | Regulatory Basis |
|--------|------------------|-------------------|------------------|
| **Employment** | 0.8 - 1.25 | < 10% | EEOC Four-Fifths Rule |
| **Credit/Lending** | 0.9 - 1.11 | < 5% | ECOA, Fair Credit |
| **Healthcare** | 0.9 - 1.11 | < 5% | HIPAA, ACA §1557 |
| **Criminal Justice** | 0.95 - 1.05 | < 3% | COMPAS litigation |
| **General Purpose** | 0.8 - 1.25 | < 10% | Default threshold |

### 7.9 Dimension 7: Security & Safety (10 points)

```rust
fn score_security_safety(
    model: &InspectionResult,
    findings: &mut Vec<Finding>,
    critical: &mut Vec<CriticalIssue>,
) -> Result<DimensionScore, AprenderError> {
    let mut score = 0.0;
    let mut breakdown = Vec::new();

    // 6.1 Digital signature (4 points)
    // References: [Goldreich 2001] (Foundations of Cryptography)
    if model.header.flags.contains(Flags::SIGNED) {
        if model.signature.as_ref().map(|s| s.valid).unwrap_or(false) {
            score += 4.0;
            breakdown.push(("signature".to_string(), 4.0, 4.0));
        } else {
            critical.push(CriticalIssue {
                severity: Severity::High,
                message: "Model signature is INVALID".to_string(),
                action: "Do not use this model - signature verification failed".to_string(),
            });
        }
    } else {
        findings.push(Finding::Info {
            message: "Model is not signed".to_string(),
            recommendation: "Sign models for production deployment".to_string(),
        });
    }

    // 6.2 Trueno-native alignment (3 points)
    // Ensures safe SIMD operations without buffer overflows
    if model.header.flags.contains(Flags::TRUENO_NATIVE) {
        score += 3.0;
        breakdown.push(("simd_safe".to_string(), 3.0, 3.0));
    }

    // 6.3 Bounded inference (3 points)
    // References: [NASA NPR 7150.2D], [ISO 26262]
    // Model declares maximum memory/time bounds for inference
    let has_bounds = model.metadata.as_ref()
        .and_then(|m| m.custom.as_ref())
        .map(|c| c.contains_key("max_inference_memory_bytes") && c.contains_key("max_inference_time_ms"))
        .unwrap_or(false);

    if has_bounds {
        score += 3.0;
        breakdown.push(("bounded_inference".to_string(), 3.0, 3.0));
    } else {
        findings.push(Finding::Info {
            message: "No inference bounds declared".to_string(),
            recommendation: "Declare max memory and time bounds for safety-critical deployments".to_string(),
        });
    }

    Ok(DimensionScore {
        score,
        max_score: 10.0,
        percentage: score / 10.0 * 100.0,
        breakdown,
    })
}
```

### 7.9 Grade Scale

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    APlus,   // 95-100: Exceptional, production-ready
    A,       // 90-94: Excellent, minor improvements possible
    AMinus,  // 85-89: Very good, some recommendations
    BPlus,   // 80-84: Good, address recommendations
    B,       // 75-79: Acceptable, needs improvement
    BMinus,  // 70-74: Below standard, significant issues
    C,       // 60-69: Poor, major issues
    D,       // 50-59: Failing, not recommended for use
    F,       // 0-49: Critical failures, do not use
}

impl Grade {
    pub fn from_score(score: f32) -> Self {
        match score as u8 {
            95..=100 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::AMinus,
            80..=84 => Self::BPlus,
            75..=79 => Self::B,
            70..=74 => Self::BMinus,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    pub fn is_production_ready(&self) -> bool {
        matches!(self, Self::APlus | Self::A | Self::AMinus)
    }

    pub fn is_acceptable(&self) -> bool {
        matches!(self, Self::APlus | Self::A | Self::AMinus | Self::BPlus | Self::B)
    }
}
```

### 7.9 QA Module Architecture (`aprender::qa`)

The `aprender::qa` module provides a **100-point adversarial QA checklist** for production model validation. This separates *model quality* (aprender) from *code quality* (certeza).

#### 7.9.1 Module Structure

```
src/
├── qa/
│   ├── mod.rs           # QaChecklist, QaReport, orchestration
│   ├── adversarial.rs   # FGSM, PGD, perturbation attacks
│   ├── fairness.rs      # Disparate impact, EOD, subgroup parity
│   ├── robustness.rs    # Edge cases, OOD detection, input validation
│   ├── privacy.rs       # Membership inference, model inversion
│   ├── performance.rs   # Latency P99, memory, WCET compliance
│   └── reproducibility.rs # Determinism, seed stability
```

#### 7.9.2 100-Point Adversarial QA Checklist

| Category | Points | Tests | Pass Criteria |
|----------|--------|-------|---------------|
| **Robustness** | 20 | FGSM ε=0.1, PGD 10-step, Gaussian noise σ=0.05 | Accuracy drop < 5% |
| **Edge Cases** | 15 | NaN, Inf, empty input, max-size tensor, zero vector | No panic, graceful error |
| **Distribution Shift** | 15 | OOD detection AUC, covariate shift, label shift | OOD AUC > 0.85 |
| **Fairness** | 15 | Disparate impact ≥ 0.8, EOD ≤ 0.1, subgroup accuracy | EEOC compliant |
| **Privacy** | 10 | Membership inference AUC < 0.6, no PII leakage | Attack AUC near random |
| **Latency** | 10 | P50, P95, P99, WCET budget compliance | P99 < SLA, WCET pass |
| **Memory** | 10 | Peak allocation, leak detection, fragmentation | No leaks, < budget |
| **Reproducibility** | 5 | Same seed → same output, cross-platform determinism | Bit-exact match |

#### 7.9.3 Core Types

```rust
/// 100-point QA checklist for model validation
#[derive(Debug, Clone)]
pub struct QaChecklist {
    /// Model under test
    pub model_path: PathBuf,
    /// Test dataset (required for most checks)
    pub test_data: Option<PathBuf>,
    /// Protected attributes for fairness testing
    pub protected_attrs: Vec<String>,
    /// Latency SLA for performance testing
    pub latency_sla: Duration,
    /// Memory budget for resource testing
    pub memory_budget: usize,
    /// WCET platform specs (if safety-critical)
    pub wcet_platform: Option<PlatformSpecs>,
}

/// QA report with 100-point scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaReport {
    /// Model identifier
    pub model_id: String,
    /// Timestamp of QA run
    pub timestamp: DateTime<Utc>,
    /// Individual category scores
    pub categories: HashMap<QaCategory, CategoryScore>,
    /// Total score (0-100)
    pub total_score: u8,
    /// Pass/fail determination
    pub passed: bool,
    /// Blocking issues (must fix)
    pub blockers: Vec<QaIssue>,
    /// Warnings (should fix)
    pub warnings: Vec<QaIssue>,
    /// Evidence artifacts (plots, logs)
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QaCategory {
    Robustness,
    EdgeCases,
    DistributionShift,
    Fairness,
    Privacy,
    Latency,
    Memory,
    Reproducibility,
}

#[derive(Debug, Clone)]
pub struct CategoryScore {
    pub points_earned: u8,
    pub points_possible: u8,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub details: Vec<TestResult>,
}

#[derive(Debug, Clone)]
pub struct QaIssue {
    pub category: QaCategory,
    pub severity: Severity,
    pub message: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Copy)]
pub enum Severity {
    /// Blocks production deployment
    Blocker,
    /// Should fix before production
    Critical,
    /// Recommended improvement
    Warning,
    /// Informational only
    Info,
}
```

#### 7.9.4 CLI Integration (`aprender-shell`)

```bash
# Run full 100-point QA checklist
apr qa run model.apr --test-data test.ald --checklist full

# Run specific category
apr qa run model.apr --checklist robustness,fairness

# Run adversarial attacks only
apr qa attack model.apr --method fgsm --epsilon 0.1

# Generate QA report (JSON/HTML)
apr qa report model.apr --format html --output qa-report.html

# Check if model passes QA gate
apr qa gate model.apr --min-score 80 --no-blockers
```

#### 7.9.5 Ecosystem Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                     QA Validation Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ aprender │───▶│  batuta  │───▶│  pacha   │───▶│ realizár │  │
│  │   ::qa   │    │ (oracle) │    │(registry)│    │(serving) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  Define tests    Orchestrate     Store results   Gate deploy    │
│  Score model     Run campaign    Enforce policy  Serve if pass  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 7.9.6 Configuration (`.apr-qa.toml`)

```toml
# .apr-qa.toml - QA checklist configuration (git-tracked)

[checklist]
version = "1.0"
min_score = 80
block_on_blocker = true

[robustness]
enabled = true
fgsm_epsilon = 0.1
pgd_steps = 10
pgd_epsilon = 0.03
noise_sigma = 0.05
max_accuracy_drop = 0.05

[fairness]
enabled = true
protected_attributes = ["gender", "race", "age"]
disparate_impact_threshold = 0.8
equal_opportunity_threshold = 0.1

[privacy]
enabled = true
membership_inference_max_auc = 0.6
differential_privacy_epsilon = 1.0

[performance]
enabled = true
latency_p99_ms = 50
memory_budget_mb = 512
wcet_platform = "automotive"  # or "aerospace", "edge"

[reproducibility]
enabled = true
seeds = [42, 123, 456]
require_deterministic = true
```

#### 7.9.7 Pacha Pre-Publish Hook

```rust
/// Pacha registry enforces QA gate before model publication
impl PachaRegistry {
    pub fn publish(&self, model: &AprModel, report: &QaReport) -> Result<(), PublishError> {
        // Gate 1: Minimum score
        if report.total_score < self.config.min_qa_score {
            return Err(PublishError::QaScoreTooLow {
                actual: report.total_score,
                required: self.config.min_qa_score,
            });
        }

        // Gate 2: No blockers
        if !report.blockers.is_empty() {
            return Err(PublishError::HasBlockers {
                count: report.blockers.len(),
                issues: report.blockers.clone(),
            });
        }

        // Gate 3: Required categories passed
        for required in &self.config.required_categories {
            let score = report.categories.get(required)
                .ok_or(PublishError::MissingCategory(*required))?;
            if score.points_earned < score.points_possible / 2 {
                return Err(PublishError::CategoryFailed {
                    category: *required,
                    score: score.points_earned,
                    required: score.points_possible / 2,
                });
            }
        }

        // All gates passed - publish
        self.store_model(model, report)
    }
}
```

### 7.10 Model Evaluation Framework (`aprender::bench`)

The `aprender::bench` module provides a **multi-model comparison framework** for evaluating `.apr` models on custom tasks. Unlike QA (single-model validation), this module compares multiple models to find the **smallest model that meets a performance threshold**.

#### 7.10.1 Use Case: Code Translation Evaluation

**Scenario**: Compare 3 coding models on "single-shot compile" Python→Rust translation:
1. Measure success rate by turn (turn 1: 60%, turn 2: 85%, turn 3: 95%)
2. Track tokens and latency per turn
3. Find smallest model achieving ≥90% success
4. Extend to future tasks (Bash→Rust, SQL→Rust, etc.)

#### 7.10.2 Module Structure

```
src/
├── bench/
│   ├── mod.rs          # EvalSuite, EvalResult, ModelComparison
│   ├── task.rs         # EvalTask trait, built-in task types
│   ├── metrics.rs      # TurnMetrics, CompileSuccess, SizeEfficiency
│   ├── harness.rs      # Run evaluation, timeout, sandboxing
│   ├── pareto.rs       # Pareto frontier, smallest-model selection
│   └── report.rs       # Comparison tables, charts
```

#### 7.10.3 Core Abstractions

```rust
/// Custom evaluation task (extensible for any domain)
pub trait EvalTask: Send + Sync {
    /// Unique task identifier (e.g., "python-to-rust-v1")
    fn id(&self) -> &str;

    /// Human-readable description
    fn description(&self) -> &str;

    /// Input examples to evaluate
    fn examples(&self) -> &[Example];

    /// Evaluate model output for a given turn
    fn evaluate(&self, input: &Example, output: &str, turn: u32) -> TurnResult;

    /// Maximum turns before declaring failure
    fn max_turns(&self) -> u32 { 5 }

    /// Timeout per turn
    fn turn_timeout(&self) -> Duration { Duration::from_secs(60) }
}

/// Example input for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Unique example ID
    pub id: String,
    /// Input prompt/code
    pub input: String,
    /// Expected behavior description
    pub expected: ExpectedBehavior,
    /// Difficulty tier (for stratified analysis)
    pub difficulty: Difficulty,
    /// Tags for filtering
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ExpectedBehavior {
    /// Must compile successfully
    Compiles,
    /// Must compile and pass test cases
    PassesTests(Vec<TestCase>),
    /// Must produce exact output
    ExactOutput(String),
    /// Must match semantic equivalence (custom validator)
    SemanticMatch(Box<dyn Fn(&str) -> bool + Send + Sync>),
}

#[derive(Debug, Clone, Copy)]
pub enum Difficulty {
    Trivial,    // 1-liner, obvious translation
    Easy,       // Simple logic, standard patterns
    Medium,     // Multiple functions, error handling
    Hard,       // Complex algorithms, unsafe/FFI
    Expert,     // Requires deep language knowledge
}
```

#### 7.10.4 Code Translation Task

```rust
/// Built-in task for code translation evaluation
pub struct CodeTranslationTask {
    pub id: String,
    pub source_lang: Language,
    pub target_lang: Language,
    pub examples: Vec<Example>,
    pub compiler: CompilerConfig,
    pub sandbox: SandboxConfig,
}

#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub command: String,           // e.g., "rustc", "cargo build"
    pub args: Vec<String>,
    pub timeout: Duration,
    pub capture_diagnostics: bool,
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub enabled: bool,
    pub memory_limit: usize,
    pub cpu_time_limit: Duration,
    pub network_disabled: bool,
    pub temp_dir: PathBuf,
}

impl EvalTask for CodeTranslationTask {
    fn id(&self) -> &str { &self.id }

    fn description(&self) -> &str {
        // e.g., "Python to Rust single-shot compilation"
    }

    fn examples(&self) -> &[Example] { &self.examples }

    fn evaluate(&self, input: &Example, output: &str, turn: u32) -> TurnResult {
        // 1. Extract code block from model output
        let code = extract_code_block(output, &self.target_lang)?;

        // 2. Write to temp file in sandbox
        let temp_path = self.sandbox.temp_dir.join(format!("eval_{}.rs", input.id));
        std::fs::write(&temp_path, &code)?;

        // 3. Attempt compilation
        let compile_result = Command::new(&self.compiler.command)
            .args(&self.compiler.args)
            .arg(&temp_path)
            .timeout(self.compiler.timeout)
            .output()?;

        if !compile_result.status.success() {
            return TurnResult::Failed {
                turn,
                reason: FailureReason::CompileError {
                    stderr: String::from_utf8_lossy(&compile_result.stderr).to_string(),
                    diagnostics: parse_diagnostics(&compile_result.stderr),
                },
                recoverable: true,  // Model can retry with error feedback
            };
        }

        // 4. Run tests if specified
        match &input.expected {
            ExpectedBehavior::Compiles => TurnResult::Success { turn },
            ExpectedBehavior::PassesTests(tests) => {
                run_tests(&temp_path, tests, turn)
            }
            // ...
        }
    }
}
```

#### 7.10.5 Turn Tracking & Metrics

```rust
/// Result of evaluating a single model on a single task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Model identifier
    pub model_id: String,
    /// Model size in bytes (for Pareto analysis)
    pub model_size_bytes: u64,
    /// Model parameter count (if known)
    pub model_params: Option<u64>,
    /// Task evaluated
    pub task_id: String,
    /// Per-example results
    pub example_results: Vec<ExampleResult>,
    /// Aggregate: success rate by turn
    /// e.g., [0.60, 0.85, 0.95] = 60% turn 1, 85% turn 2, 95% turn 3
    pub success_by_turn: Vec<f64>,
    /// Aggregate: average turns to success (for successful examples)
    pub avg_turns_to_success: f64,
    /// Aggregate: overall success rate (any turn)
    pub overall_success_rate: f64,
    /// Total tokens consumed
    pub total_tokens: u64,
    /// Total latency
    pub total_latency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    pub example_id: String,
    pub difficulty: Difficulty,
    /// Which turn solved it (None = failed all turns)
    pub solved_at_turn: Option<u32>,
    /// Tokens per turn
    pub tokens_per_turn: Vec<u64>,
    /// Latency per turn
    pub latency_per_turn: Vec<Duration>,
    /// Final status
    pub status: ExampleStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExampleStatus {
    /// Solved within max_turns
    Solved { turn: u32 },
    /// Failed all turns
    Failed { attempts: u32, last_error: String },
    /// Timed out
    Timeout { turn: u32 },
    /// Skipped (e.g., dependency missing)
    Skipped { reason: String },
}
```

#### 7.10.6 Multi-Model Comparison

```rust
/// Compare multiple models on the same task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// Task being evaluated
    pub task_id: String,
    /// Results per model
    pub results: Vec<EvalResult>,
    /// Pareto-optimal models (size vs success rate)
    pub pareto_frontier: Vec<ParetoPoint>,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub model_id: String,
    pub size_bytes: u64,
    pub success_rate: f64,
    pub avg_turns: f64,
    /// True if no other model dominates (smaller AND more accurate)
    pub is_pareto_optimal: bool,
}

impl ModelComparison {
    /// Find smallest model meeting a success threshold
    pub fn smallest_meeting_threshold(&self, min_success: f64) -> Option<&EvalResult> {
        self.results.iter()
            .filter(|r| r.overall_success_rate >= min_success)
            .min_by_key(|r| r.model_size_bytes)
    }

    /// Find fastest model meeting a success threshold
    pub fn fastest_meeting_threshold(&self, min_success: f64) -> Option<&EvalResult> {
        self.results.iter()
            .filter(|r| r.overall_success_rate >= min_success)
            .min_by_key(|r| r.avg_turns_to_success as u64)
    }

    /// Compute Pareto frontier (size vs accuracy)
    pub fn compute_pareto_frontier(&self) -> Vec<ParetoPoint> {
        let mut points: Vec<_> = self.results.iter()
            .map(|r| ParetoPoint {
                model_id: r.model_id.clone(),
                size_bytes: r.model_size_bytes,
                success_rate: r.overall_success_rate,
                avg_turns: r.avg_turns_to_success,
                is_pareto_optimal: false,
            })
            .collect();

        // Mark Pareto-optimal points
        for i in 0..points.len() {
            let dominated = points.iter().enumerate().any(|(j, other)| {
                j != i &&
                other.size_bytes <= points[i].size_bytes &&
                other.success_rate >= points[i].success_rate &&
                (other.size_bytes < points[i].size_bytes ||
                 other.success_rate > points[i].success_rate)
            });
            points[i].is_pareto_optimal = !dominated;
        }

        points.into_iter().filter(|p| p.is_pareto_optimal).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub scenario: String,
    pub model_id: String,
    pub rationale: String,
}
```

#### 7.10.7 CLI Integration

```bash
# Create new evaluation suite
apr bench new python-to-rust \
    --task code-translation \
    --source python \
    --target rust \
    --examples ./examples/py2rs/

# Add examples to existing suite
apr bench add-examples python-to-rust ./more-examples/

# Run evaluation on multiple models
apr bench run python-to-rust \
    --models codegen-16b.apr,codegen-6b.apr,codegen-2b.apr \
    --max-turns 5 \
    --parallel 4

# Find smallest model meeting threshold
apr bench pareto python-to-rust --min-success 0.9

# Generate comparison report
apr bench compare python-to-rust --format table
apr bench compare python-to-rust --format html --output report.html
apr bench compare python-to-rust --format json > results.json

# Stratified analysis by difficulty
apr bench analyze python-to-rust --group-by difficulty

# Export for presentar visualization
apr bench export python-to-rust --format presentar
```

#### 7.10.8 Output Formats

**Table Output**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Task: python-to-rust (50 examples)                                      │
├──────────────┬──────────┬────────┬────────┬────────┬────────┬──────────┤
│ Model        │ Size     │ Turn 1 │ Turn 2 │ Turn 3 │ Turn 4 │ Avg Turn │
├──────────────┼──────────┼────────┼────────┼────────┼────────┼──────────┤
│ codegen-16b  │ 32 GB    │ 78%    │ 92%    │ 98%    │ 100%   │ 1.24     │
│ codegen-6b   │ 12 GB    │ 65%    │ 85%    │ 94%    │ 96%    │ 1.42     │
│ codegen-2b   │ 4 GB     │ 45%    │ 70%    │ 82%    │ 88%    │ 1.78     │
├──────────────┴──────────┴────────┴────────┴────────┴────────┴──────────┤
│ Pareto frontier: codegen-2b (smallest), codegen-16b (most accurate)    │
│ Smallest @ 90% success: codegen-6b (12 GB)                             │
│ Smallest @ 80% success: codegen-2b (4 GB) ← RECOMMENDED                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Stratified by Difficulty**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Task: python-to-rust - Stratified by Difficulty                         │
├──────────────┬──────────┬─────────┬────────┬────────┬────────┬─────────┤
│ Model        │ Trivial  │ Easy    │ Medium │ Hard   │ Expert │ Overall │
├──────────────┼──────────┼─────────┼────────┼────────┼────────┼─────────┤
│ codegen-16b  │ 100%     │ 98%     │ 95%    │ 85%    │ 60%    │ 88%     │
│ codegen-6b   │ 100%     │ 95%     │ 88%    │ 70%    │ 40%    │ 79%     │
│ codegen-2b   │ 98%      │ 90%     │ 75%    │ 50%    │ 20%    │ 67%     │
└──────────────┴──────────┴─────────┴────────┴────────┴────────┴─────────┘
```

#### 7.10.9 Ecosystem Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Model Evaluation Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ aprender │    │  batuta  │    │  pacha   │    │ presentar│          │
│  │  ::bench │───▶│(harness) │───▶│ (store)  │───▶│  (viz)   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│       │               │               │               │                 │
│       ▼               ▼               ▼               ▼                 │
│  Task schema     Execute evals   Store results   Pareto charts          │
│  Metrics def     Sandbox runs    Leaderboards    Turn heatmaps          │
│  Pareto calc     Parallel exec   Version track   Model cards            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 7.10.10 Configuration (`.apr-bench.toml`)

```toml
# .apr-bench.toml - Evaluation suite configuration

[suite]
id = "python-to-rust"
description = "Single-shot Python to Rust code translation"
version = "1.0"

[task]
type = "code-translation"
source_lang = "python"
target_lang = "rust"
max_turns = 5
turn_timeout_secs = 60

[compiler]
command = "rustc"
args = ["--edition", "2021", "-o", "/dev/null"]
timeout_secs = 30

[sandbox]
enabled = true
memory_limit_mb = 512
cpu_time_limit_secs = 30
network_disabled = true

[examples]
path = "./examples/"
pattern = "*.py"
metadata = "./examples/metadata.json"

[thresholds]
# For "smallest model" recommendations
min_success_rates = [0.80, 0.90, 0.95]

[output]
formats = ["table", "json", "html"]
export_to_pacha = true
export_to_presentar = true
```

#### 7.10.11 Day 1 Benchmark: Python→Rust Single-Shot Compile (10 Levels)

**Status**: MANDATORY - Ship with v1.0

This canonical benchmark measures model ability to translate Python to compilable Rust in a single shot (no iterative refinement). Models are scored on **highest level passed** and **turns to pass**.

| Level | Name | Python Input | Rust Must... | Example |
|-------|------|--------------|--------------|---------|
| **1** | Hello | `print("hello")` | Compile, print exact | `println!("hello");` |
| **2** | Variables | Assignment, arithmetic | Type inference | `let x = 5; let y = x * 2;` |
| **3** | Functions | `def` with args/return | Signature, ownership | `fn add(a: i32, b: i32) -> i32` |
| **4** | Collections | List/dict comprehension | Vec/HashMap | `vec![1,2,3].iter().map()` |
| **5** | Control Flow | Loops, match, recursion | Borrowing correct | Fibonacci, binary search |
| **6** | Error Handling | try/except | `Result<T,E>`, `?` operator | File read with error |
| **7** | OOP→Traits | Class with methods | `struct` + `impl` + trait | Shape hierarchy |
| **8** | Concurrency | Threading, async | `tokio`/`rayon`, no races | Parallel map-reduce |
| **9** | FFI/Unsafe | ctypes, numpy interop | Safe wrapper over unsafe | C library binding |
| **10** | Metaprogramming | Decorators, metaclass | Proc macro or generics | `@dataclass` → `#[derive]` |

**Scoring Formula**

```rust
/// Model score on Python→Rust benchmark
pub struct Py2RsScore {
    /// Highest level passed (1-10)
    pub max_level: u8,
    /// Levels passed on turn 1 (single-shot)
    pub single_shot_levels: Vec<u8>,
    /// Average turns per level (lower = better)
    pub avg_turns_by_level: [f32; 10],
    /// Composite score (0-100)
    pub composite: f32,
}

impl Py2RsScore {
    pub fn composite_score(&self) -> f32 {
        // Weight: level difficulty * single-shot bonus
        let level_weights = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0];
        let max_possible: f32 = level_weights.iter().sum(); // 68.5

        let earned: f32 = self.single_shot_levels.iter()
            .map(|&l| level_weights[(l - 1) as usize])
            .sum();

        (earned / max_possible) * 100.0
    }
}
```

**Example Inputs (10 canonical problems)**

```python
# Level 1: hello.py
print("hello world")

# Level 2: variables.py
x = 42
y = x * 2 + 1
print(f"Result: {y}")

# Level 3: functions.py
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Level 4: collections.py
squares = [x**2 for x in range(10) if x % 2 == 0]
counts = {word: len(word) for word in ["hello", "world"]}

# Level 5: control_flow.py
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Level 6: error_handling.py
def read_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

# Level 7: oop_traits.py
class Shape:
    def area(self): raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius): self.radius = radius
    def area(self): return 3.14159 * self.radius ** 2

# Level 8: concurrency.py
import asyncio
async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[fetch(session, u) for u in urls])

# Level 9: ffi_unsafe.py
import ctypes
lib = ctypes.CDLL("libcrypto.so")
lib.SHA256_Init.argtypes = [ctypes.POINTER(SHA256_CTX)]

# Level 10: metaprogramming.py
@dataclass
class Point:
    x: float
    y: float
    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
```

**CLI Usage**

```bash
# Run canonical Python→Rust benchmark
apr bench run py2rs-canonical --models model1.apr,model2.apr,model3.apr

# Output
┌────────────────────────────────────────────────────────────────┐
│ Benchmark: py2rs-canonical (10 levels)                         │
├──────────────┬───────┬────────────────────────────────┬────────┤
│ Model        │ Score │ Levels (● = T1, ◐ = T2+, ○ = fail) │ Max  │
├──────────────┼───────┼────────────────────────────────┼────────┤
│ codegen-16b  │ 92.3  │ ● ● ● ● ● ● ● ◐ ◐ ○            │ L9    │
│ codegen-6b   │ 71.5  │ ● ● ● ● ● ◐ ◐ ○ ○ ○            │ L7    │
│ codegen-2b   │ 43.8  │ ● ● ● ◐ ◐ ○ ○ ○ ○ ○            │ L5    │
├──────────────┴───────┴────────────────────────────────┴────────┤
│ Legend: ● Pass Turn 1 | ◐ Pass Turn 2+ | ○ Failed              │
└────────────────────────────────────────────────────────────────┘
```

#### 7.10.12 Extending to New Tasks

```rust
// Future task: Bash to Rust
let bash_to_rust = CodeTranslationTask {
    id: "bash-to-rust-v1".into(),
    source_lang: Language::Bash,
    target_lang: Language::Rust,
    examples: load_examples("./examples/bash2rs/"),
    compiler: CompilerConfig {
        command: "cargo".into(),
        args: vec!["build".into(), "--release".into()],
        ..Default::default()
    },
    ..Default::default()
};

// Future task: SQL to Rust (using sqlx)
let sql_to_rust = CodeTranslationTask {
    id: "sql-to-rust-v1".into(),
    source_lang: Language::SQL,
    target_lang: Language::Rust,
    examples: load_examples("./examples/sql2rs/"),
    // Custom compiler that checks sqlx macros
    compiler: CompilerConfig {
        command: "cargo".into(),
        args: vec!["sqlx".into(), "prepare".into(), "--check".into()],
        ..Default::default()
    },
    ..Default::default()
};

// Register tasks
suite.register_task(bash_to_rust);
suite.register_task(sql_to_rust);
```

---

## 8. WASM Playground Support

### 8.1 Design Goals

Enable interactive model exploration in browser-based environments:
- **interactive.paiml.com**: Educational ML demonstrations
- **presentar**: Presentation slides with live model inference
- **prs-cookbook**: Hands-on ML cookbook with editable examples
- **model-zoo**: Shareable model repository with instant preview

### 8.2 WASM Module Architecture

```rust
/// WASM-specific entry points for playground integration
/// Compiled with: cargo build --target wasm32-unknown-unknown --features wasm
#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;
    use super::*;

    /// Load model from JavaScript ArrayBuffer
    #[wasm_bindgen]
    pub fn load_model_bytes(
        bytes: &[u8],
        model_type: u16,
    ) -> Result<JsValue, JsValue> {
        let model_type = ModelType::from_u16(model_type)
            .ok_or_else(|| JsValue::from_str("Invalid model type"))?;

        let config = LoadConfig {
            mode: LoadingMode::Eager,  // No mmap in WASM
            max_memory_bytes: Some(50 * 1024 * 1024),  // 50 MB limit
            verification: VerificationLevel::Standard,
            backend: Backend::Wasm,
            buffer_pool: None,
        };

        let loaded = load_apr_bytes::<Box<dyn ModelTrait>>(bytes, model_type, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(serde_wasm_bindgen::to_value(&loaded.info())?)
    }

    /// Inspect model without full loading
    #[wasm_bindgen]
    pub fn inspect_model_bytes(bytes: &[u8]) -> Result<JsValue, JsValue> {
        let result = inspect_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(serde_wasm_bindgen::to_value(&result)?)
    }

    /// Compute quality score from inspection
    #[wasm_bindgen]
    pub fn score_model_bytes(bytes: &[u8]) -> Result<JsValue, JsValue> {
        let inspection = inspect_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let score = compute_quality_score(&inspection, None, &ScoringConfig::default())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(serde_wasm_bindgen::to_value(&score)?)
    }

    /// Run inference on loaded model
    #[wasm_bindgen]
    pub fn predict(
        model_handle: u32,
        x_data: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let model = get_model_handle(model_handle)
            .ok_or_else(|| JsValue::from_str("Invalid model handle"))?;

        // Construct Trueno matrix from flat array
        let x = trueno::Matrix::from_slice(x_data, n_samples, n_features, Backend::Wasm)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let predictions = model.predict(&x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(predictions.to_vec())
    }
}
```

### 8.3 Model Zoo Protocol

```rust
/// Model zoo entry for sharing and discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelZooEntry {
    /// Unique model identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Model description
    pub description: String,

    /// Version string
    pub version: String,

    /// Author information
    pub author: AuthorInfo,

    /// Model type
    pub model_type: ModelType,

    /// Quality score (cached)
    pub quality_score: f32,

    /// Tags for discovery
    pub tags: Vec<String>,

    /// Download URL
    pub download_url: String,

    /// File size in bytes
    pub size_bytes: u64,

    /// SHA-256 hash for verification
    pub sha256: String,

    /// License
    pub license: String,

    /// Creation timestamp
    pub created_at: String,

    /// Download count
    pub downloads: u64,
}

/// Model zoo index (JSON format for easy WASM consumption)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelZooIndex {
    /// Index version
    pub version: String,

    /// Last updated timestamp
    pub updated_at: String,

    /// Models in the zoo
    pub models: Vec<ModelZooEntry>,

    /// Total model count
    pub total_models: usize,

    /// Featured models (for homepage display)
    pub featured: Vec<String>,  // IDs
}
```

### 8.4 Interactive Playground Components

```typescript
// TypeScript interface for playground integration
interface AprenderPlayground {
  // Model loading
  loadModel(url: string): Promise<ModelHandle>;
  loadModelBytes(bytes: ArrayBuffer): Promise<ModelHandle>;

  // Inspection
  inspect(handle: ModelHandle): InspectionResult;
  getQualityScore(handle: ModelHandle): QualityScore;

  // Inference
  predict(handle: ModelHandle, data: Float32Array, shape: [number, number]): Float32Array;

  // Model zoo
  listModels(): Promise<ModelZooEntry[]>;
  searchModels(query: string): Promise<ModelZooEntry[]>;
  downloadModel(id: string): Promise<ArrayBuffer>;

  // Bundled models (compiled into WASM)
  listBundled(): ModelInfo[];
  loadBundled(name: string): ModelHandle;
}

// React component example
const ModelExplorer: React.FC<{ modelUrl: string }> = ({ modelUrl }) => {
  const [model, setModel] = useState<ModelHandle | null>(null);
  const [inspection, setInspection] = useState<InspectionResult | null>(null);
  const [score, setScore] = useState<QualityScore | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      const handle = await aprender.loadModel(modelUrl);
      setModel(handle);
      setInspection(aprender.inspect(handle));
      setScore(aprender.getQualityScore(handle));
    };
    loadModel();
  }, [modelUrl]);

  return (
    <div className="model-explorer">
      <InspectionPanel inspection={inspection} />
      <QualityScoreGauge score={score} />
      <InferencePanel model={model} />
    </div>
  );
};
```

### 8.5 WASM Size Optimization

Target WASM bundle size: < 500 KB (gzipped) for playground use:

```rust
/// Feature flags for minimal WASM builds
// Cargo.toml
[features]
wasm-minimal = []  // Core inference only, no signing/encryption
wasm-full = ["format-signing"]  // Include signature verification
wasm-playground = ["wasm-full"]  // Full playground support

/// Size optimization strategies
// 1. LTO (Link-Time Optimization)
// 2. panic = "abort" (no unwinding)
// 3. opt-level = "z" (size optimization)
// 4. strip = true (remove symbols)

// Cargo.toml profile
[profile.release-wasm]
inherits = "release"
lto = true
opt-level = "z"
panic = "abort"
strip = true
codegen-units = 1
```

---

## 9. Sovereign AI Stack Integration

The `.apr` format is a **standalone binary format** that integrates with the broader Pragmatic AI Labs Sovereign AI Stack. This section documents how `.apr` files interact with sibling tools while maintaining format independence.

### 9.1 Ecosystem Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SOVEREIGN AI STACK                                    │
│                    (Pure Rust, Zero Cloud Dependencies)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  alimentar  │───▶│  aprender   │───▶│   pacha     │───▶│  realizar   │      │
│  │  (Data)     │    │  (ML Algo)  │    │  (Registry) │    │  (Inference)│      │
│  │  .ald files │    │  .apr files │    │  Versioning │    │  REST API   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                  │                  │                  │              │
│         │                  │                  │                  │              │
│         ▼                  ▼                  ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          presentar                                       │   │
│  │              (WASM Visualization & Playgrounds)                          │   │
│  │     ModelCard Widget │ DataCard Widget │ Interactive Inference           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           batuta                                         │   │
│  │              (Orchestration & Oracle Mode)                               │   │
│  │   Component Recommendations │ Pipeline Orchestration │ Stack Health      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Alimentar Integration (Data Loading)

**Project**: `alimentar` ("to feed") - Pure Rust data loading and transformation library
**Format**: `.ald` (Alimentar Dataset Format) - Sister format to `.apr`

#### 9.2.1 Format Compatibility

| Aspect | `.apr` (Aprender) | `.ald` (Alimentar) |
|--------|-------------------|-------------------|
| Magic | `APRN` (0x4150524E) | `ALDF` (0x414C4446) |
| Version | 1.0 | 1.2 |
| Header Size | 32 bytes | 32 bytes |
| Compression | Zstd/LZ4/None | Zstd/LZ4/None |
| Encryption | AES-256-GCM | AES-256-GCM |
| Signing | Ed25519 | Ed25519 |
| Checksum | CRC32 | CRC32 |

**Key Difference**: `.apr` stores model weights; `.ald` stores Arrow RecordBatches for datasets.

#### 9.2.2 Data Flow: Alimentar → Aprender

```rust
/// Alimentar provides tensor extraction for aprender training
use alimentar::{ArrowDataset, TensorExtractor};
use aprender::linear_model::LinearRegression;

// Load dataset from .ald file
let dataset = ArrowDataset::from_ald("training_data.ald")?;

// Extract features as Trueno-compatible tensors
let extractor = TensorExtractor::new(&["feature_1", "feature_2", "target"]);
let tensor_data = extractor.extract_f32(&dataset.to_batch()?)?;

// Convert to aprender Matrix/Vector
let x = trueno::Matrix::from_slice(
    tensor_data.column_slice("feature_1"),
    n_samples,
    n_features,
    Backend::Auto,
)?;
let y = trueno::Vector::from_slice(tensor_data.column_slice("target"), Backend::Auto)?;

// Train model and save as .apr
let mut model = LinearRegression::new();
model.fit(&x, &y)?;
aprender::format::save(&model, ModelType::LinearRegression, "model.apr", SaveOptions::default())?;
```

#### 9.2.3 Shared Quality Infrastructure

Both formats share:
- **100-point quality scoring** (see Section 7 for `.apr`; alimentar has `quality.rs`)
- **Data drift detection** (alimentar: KS test, PSI; aprender: model drift via metadata)
- **Toyota Way principles** (Jidoka, Muda elimination, Kaizen)

### 9.3 Pacha Integration (Model Registry)

**Project**: `pacha` - Model, Data, and Recipe Registry with lineage tracking
**Role**: Central repository for `.apr` model versioning, signatures, and provenance

#### 9.3.1 Registry Operations

```bash
# Register a trained model
pacha model register iris_classifier model.apr \
    --version 1.0.0 \
    --description "Iris species classifier (Random Forest)" \
    --tags ml,classification,iris

# List model versions
pacha model list iris_classifier
# Output:
# iris_classifier
#   v1.0.0 (2025-12-08) - Development
#   v0.9.0 (2025-12-01) - Archived

# Transition to production
pacha model stage iris_classifier --version 1.0.0 --target production

# Download for inference
pacha model download iris_classifier --version 1.0.0 --output ./models/

# View lineage (fine-tuning, distillation, etc.)
pacha model lineage iris_classifier --version 1.0.0
```

#### 9.3.2 Lineage Tracking

Pacha tracks model derivation via a DAG (Directed Acyclic Graph):

```rust
/// Model derivation types tracked by Pacha
pub enum DerivationType {
    /// Original training run
    Original,

    /// Fine-tuning from parent model
    FineTune { parent_hash: [u8; 32], epochs: u32 },

    /// Knowledge distillation
    Distillation { teacher_hash: [u8; 32], temperature: f32 },

    /// Model merging (e.g., TIES, DARE)
    Merge { parent_hashes: Vec<[u8; 32]>, method: String },

    /// Quantization (precision reduction)
    Quantize { parent_hash: [u8; 32], quant_type: QuantType },

    /// Pruning (weight removal)
    Prune { parent_hash: [u8; 32], sparsity: f32 },
}
```

#### 9.3.3 Security Integration

Pacha provides cryptographic operations compatible with `.apr` format flags:

| `.apr` Flag | Pacha Feature | Implementation |
|-------------|--------------|----------------|
| `SIGNED` | `pacha model sign` | Ed25519 (ed25519-dalek) |
| `ENCRYPTED` | `pacha model encrypt` | ChaCha20-Poly1305 |
| `LICENSED` | `pacha model license` | License block with terms |

```bash
# Sign a model before distribution
pacha model sign model.apr --identity alice@example.com --output model_signed.apr

# Verify signature on load
pacha model verify model_signed.apr --trusted-keys ./keys/
```

### 9.4 Realizar Integration (Inference Engine)

**Project**: `realizar` ("to accomplish") - Pure Rust ML inference engine
**Performance**: 9.6x faster than PyTorch for CPU-only deployments

#### 9.4.1 REST API for .apr Models

```rust
// realizar serves .apr models via axum HTTP server
use realizar::serve::{AprServer, AprConfig};

let config = AprConfig {
    model_path: "models/iris_classifier.apr".into(),
    port: 8080,
    max_batch_size: 32,
    timeout_ms: 100,
};

let server = AprServer::new(config)?;
server.run().await?;
```

**Endpoints**:
```
POST /predict          - Single prediction
POST /batch_predict    - Batched predictions
GET  /health           - Health check
GET  /metrics          - Prometheus metrics
GET  /model/info       - Model inspection (metadata, quality score)
```

#### 9.4.2 Lambda Deployment

Realizar provides 53,000x faster cold starts than Python for AWS Lambda:

```rust
// realizar lambda handler for .apr models
use realizar::lambda::{handler, LambdaConfig};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let config = LambdaConfig {
        model_bytes: include_bytes!("../models/classifier.apr"),
        model_type: ModelType::RandomForest,
    };
    lambda_runtime::run(handler(config)).await
}
```

**Performance Comparison**:
| Metric | Realizar (Rust) | PyTorch (Python) |
|--------|-----------------|------------------|
| Cold Start | 15 μs | 800 ms |
| Inference | 0.52 μs | 5.0 μs |
| Binary Size | 3.2 KB | 500+ MB |
| Lambda RAM | 128 MB | 512 MB |

### 9.5 Presentar Integration (WASM Visualization)

**Project**: `presentar` - WASM-first visualization framework
**Role**: Browser-based model exploration, ModelCard display, interactive inference

#### 9.5.1 ModelCard Widget

Presentar renders `.apr` metadata as interactive ModelCard widgets:

```yaml
# presentar YAML manifest
models:
  classifier:
    source: "./models/iris_classifier.apr"
    format: "apr"

layout:
  - type: "model_card"
    data: "{{ models.classifier }}"
    show_metrics: true
    show_hyperparameters: true
```

**Rendered ModelCard Fields** (from `.apr` metadata):
- Model name, version, description
- Author, creation timestamp, framework version
- Training metrics (accuracy, F1, loss)
- Hyperparameters table
- Feature importance visualization
- Lifecycle status (Draft, Review, Published, Deprecated)

#### 9.5.2 Browser-Based Inference

Presentar's shell autocomplete demo runs `.apr` models entirely in-browser:

```rust
// WASM module for client-side .apr inference
#[wasm_bindgen]
pub struct ShellAutocomplete {
    model: NgramMarkovModel,  // Loaded from .apr
    trie: Trie,               // For O(k) prefix lookup
}

#[wasm_bindgen]
impl ShellAutocomplete {
    pub fn predict(&self, prefix: &str) -> Vec<JsValue> {
        // <1ms inference latency in browser
        self.model.predict_top_k(prefix, 5)
            .into_iter()
            .map(|s| JsValue::from_str(&s))
            .collect()
    }
}
```

**Performance**: <1ms inference, <100ms cold start, ~100KB WASM bundle

#### 9.5.3 Pacha URI Protocol

Presentar uses `pacha://` URIs to load models from registry:

```yaml
# Load model from Pacha registry
models:
  sentiment:
    source: "pacha://models/sentiment_analyzer@1.0.0"
    format: "apr"
```

### 9.6 Batuta Integration (Orchestration & Oracle)

**Project**: `batuta` - Orchestration framework with Oracle mode
**Role**: Component recommendation, pipeline orchestration, stack health

#### 9.6.1 Oracle Mode Queries

Batuta's Oracle mode recommends `.apr`-compatible workflows:

```bash
# Ask batuta for recommendations
batuta oracle "Train random forest on 1M samples"
# → aprender::tree::RandomForestClassifier + SIMD backend
# → Save as .apr with Zstd compression
# → Register in Pacha for versioning

batuta oracle "Serve model with <10ms latency"
# → realizar (Lambda) + .apr format
# → Enable TRUENO_NATIVE flag for SIMD

batuta oracle "Convert sklearn pipeline to Rust"
# → depyler → aprender (sklearn_converter)
# → Output: .apr model file
```

#### 9.6.2 sklearn → Aprender Conversion

Batuta's sklearn converter generates `.apr`-compatible models:

```python
# Input: Python sklearn
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

```rust
// Output: Rust aprender (via batuta transpilation)
use aprender::tree::RandomForestClassifier;

let mut rf = RandomForestClassifier::builder()
    .n_estimators(100)
    .build();
rf.fit(&x_train, &y_train)?;

// Save as .apr for deployment
aprender::format::save(&rf, ModelType::RandomForest, "model.apr", SaveOptions::default())?;
```

#### 9.6.3 Stack Health Monitoring

Batuta monitors `.apr` ecosystem health:

```bash
batuta stack check
# Output:
# ┌─────────────────────────────────────────────────────────┐
# │ Sovereign AI Stack Health                               │
# ├─────────────────────────────────────────────────────────┤
# │ aprender      0.15.0  ✓ healthy   96.94% coverage       │
# │ alimentar     0.2.2   ✓ healthy   85%+ coverage         │
# │ pacha         0.1.1   ✓ healthy   131 tests             │
# │ realizar      0.2.2   ✓ healthy   94.61% coverage       │
# │ presentar     0.1.0   ✓ healthy   91% coverage          │
# │ trueno        0.8.0   ✓ healthy   SIMD/GPU backends     │
# └─────────────────────────────────────────────────────────┘
```

### 9.7 Playground Targets

The `.apr` format supports multiple WASM playground deployments:

| Playground | URL | Use Case |
|------------|-----|----------|
| **interactive.paiml.com** | Production | Educational ML demonstrations |
| **presentar** | Local/Hosted | Slide decks with live inference |
| **apr-cookbook** | GitHub Pages | Hands-on aprender tutorials |
| **prs-cookbook** | GitHub Pages | Presentar visualization cookbook |
| **alm-cookbook** | GitHub Pages | Alimentar data loading tutorials |

#### 9.7.1 Model Zoo Distribution

```rust
/// Model zoo manifest for playground discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelZoo {
    /// Zoo identifier
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Models available in this zoo
    pub models: Vec<ModelZooEntry>,

    /// Update frequency
    pub update_schedule: Option<String>,

    /// Maintainer contact
    pub maintainer: String,
}

/// Integration with Pacha registry
impl ModelZoo {
    /// Sync from Pacha registry
    pub async fn sync_from_pacha(registry_url: &str) -> Result<Self, Error> {
        let client = pacha::RemoteClient::new(registry_url)?;
        let models = client.list_models().await?;

        Ok(Self {
            id: "pacha-sync".to_string(),
            name: "Pacha Registry Models".to_string(),
            models: models.into_iter()
                .map(ModelZooEntry::from_pacha)
                .collect(),
            update_schedule: Some("hourly".to_string()),
            maintainer: "registry@paiml.com".to_string(),
        })
    }
}
```

### 9.8 Format Independence Guarantee

**Critical Design Principle**: The `.apr` format is **standalone and self-contained**.

- **No runtime dependencies**: `.apr` files load without alimentar, pacha, realizar, or batuta
- **Self-describing**: All metadata embedded in file (no external manifest required)
- **Cross-platform**: Same binary format on Linux, macOS, Windows, WASM
- **Version-stable**: Format version 1.0 guaranteed backward-compatible

```rust
// .apr files load with aprender alone (no ecosystem dependencies)
use aprender::format::{load, ModelType};
use aprender::tree::RandomForestClassifier;

// Direct loading - no pacha, no alimentar, no realizar
let model: RandomForestClassifier = load("model.apr", ModelType::RandomForest)?;
let predictions = model.predict(&x_test);
```

**Ecosystem tools enhance but don't require**: Pacha adds versioning, Realizar adds serving, Presentar adds visualization - but the core `.apr` format works independently.

---

## 10. References

### 10.1 Academic Literature

#### A. Safety-Critical Systems & Determinism

1. **Wilhelm, R., et al.** (2008). The worst-case execution-time problem—overview of methods and survey of tools. *ACM Transactions on Embedded Computing Systems*, 7(3), 1-53. https://doi.org/10.1145/1347375.1347389
   - *Validates*: Section 1.6 WCET hybrid measurement approach

2. **Liu, C. L., & Layland, J. W.** (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. *Journal of the ACM*, 20(1), 46-61. https://doi.org/10.1145/321738.321743
   - *Validates*: `LoadingMode::Eager` for deterministic scheduling

3. **Koopman, P.** (2014). *Better Embedded System Software*. Drumnadrochit Press. ISBN: 978-0-9845920-0-4
   - *Validates*: Section 5.5 `PredictNoStd` stack-only allocation

4. **NASA.** (2019). *NASA Software Engineering Requirements (NPR 7150.2D)*. NASA OCIO.
   - *Validates*: Section 2.5 `SigbusRecovery` deterministic failure handling

5. **ISO 26262-6** (2018). Road vehicles – Functional safety – Part 6: Product development at the software level.
   - *Validates*: `assert_time_budget` FFI requirements

#### B. Storage, Compression, and I/O

6. **Collet, Y.** (2016). Zstandard: Real-time compression algorithm. *Facebook Engineering*.
   - *Validates*: Section 4.2 Zstd bounded decompression for WCET

7. **Vahalia, U.** (1996). *UNIX Internals: The New Frontiers*. Prentice Hall. ISBN: 978-0-13-101908-1
   - *Validates*: `MappedDemand` vs `Eager` bifurcation

8. **Didona, D., et al.** (2022). Understanding Modern Storage APIs: A systematic study. *USENIX ATC*.
   - *Validates*: `LoadingMode::Streaming` async I/O approach

9. **McKusick, M. K.** (1984). A Fast File System for UNIX. *ACM TOCS*, 2(3), 181-197.
   - *Validates*: `TruenoNativeModel` contiguous layout

10. **Stonebraker, M., et al.** (2005). C-Store: A column-oriented DBMS. *VLDB*, 553-564.
    - *Validates*: `DataCompression::DeltaZstd` SIMD prefix-sum decoding

#### C. Machine Learning Systems (MLSys)

11. **Abadi, M., et al.** (2016). TensorFlow: A system for large-scale machine learning. *OSDI*, 265-283.
    - *Validates*: Section 1.8 Trueno backend abstraction

12. **Sculley, D., et al.** (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
    - *Validates*: `BundledModel` freezing provenance

13. **Zaharia, M., et al.** (2018). Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Eng. Bull.*, 41(4), 39-45.
    - *Validates*: Section 9 Pacha registry integration

14. **Crankshaw, D., et al.** (2017). Clipper: A Low-Latency Online Prediction Serving System. *NSDI*, 613-627.
    - *Validates*: Section 3.2 Cache Hierarchy multi-tier strategy

15. **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *KDD*, 785-794.
    - *Validates*: `LazySection` out-of-core loading

#### D. Fairness, Ethics, and Governance

16. **Mitchell, M., et al.** (2019). Model Cards for Model Reporting. *FAT\**, 220-229.
    - *Validates*: Section 7.6 `HAS_MODEL_CARD` flag

17. **Gebru, T., et al.** (2021). Datasheets for Datasets. *CACM*, 64(12), 86-92.
    - *Validates*: Section 4.1 `DataProvenance` struct

18. **Barocas, S., Hardt, M., & Narayanan, A.** (2019). *Fairness and Machine Learning*. MIT Press.
    - *Validates*: Section 7.8 multi-metric fairness (DIR, EOD, DPD)

19. **Feldman, M., et al.** (2015). Certifying and Removing Disparate Impact. *KDD*, 259-268.
    - *Validates*: Four-Fifths Rule (0.8-1.25 range)

20. **Ribeiro, M. T., et al.** (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*, 1135-1144.
    - *Validates*: Section 7.5 `score_model_complexity` interpretability

#### E. Cryptography and Security

21. **Bernstein, D. J., et al.** (2012). High-speed high-security signatures. *Journal of Cryptographic Engineering*, 2(2), 77-89.
    - *Validates*: Section 1.7 Ed25519 selection

22. **Barker, E.** (2020). NIST SP 800-57 Part 1 Rev. 5: Recommendation for Key Management.
    - *Validates*: `CipherSuite` cryptographic agility

23. **Bernstein, D. J.** (2008). ChaCha, a variant of Salsa20. *State of the Art of Stream Ciphers Workshop*.
    - *Validates*: `Standard2025` XChaCha20-Poly1305

24. **Alagic, G., et al.** (2022). Status Report on the Third Round of the NIST Post-Quantum Cryptography Standardization Process. *NIST IR 8413*.
    - *Validates*: `PostQuantum2030` ML-DSA-65 + ML-KEM-768

25. **Matsakis, N. D., & Klock, F. S.** (2014). The Rust Language. *ACM SIGAda Ada Letters*, 34(3), 103-104.
    - *Validates*: Memory safety via ownership (entire spec)

#### F. Classical ML & Statistics

26. **Akaike, H.** (1974). A new look at the statistical model identification. *IEEE TAC*, 19(6), 716-723.

27. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer.

28. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32.

29. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

30. **Liker, J. K.** (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill.

#### G. Adversarial ML & Privacy

31. **Goodfellow, I. J., et al.** (2015). Explaining and Harnessing Adversarial Examples. *ICLR*.
    - *Validates*: Section 7.9 FGSM attacks in QA checklist

32. **Carlini, N., & Wagner, D.** (2017). Towards Evaluating the Robustness of Neural Networks. *IEEE S&P*.
    - *Validates*: Section 7.9 PGD iterative attacks

33. **Shokri, R., et al.** (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE S&P*.
    - *Validates*: Section 7.9 Privacy checks (AUC < 0.6)

#### H. Benchmarking & Optimization

34. **Deb, K., et al.** (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE TEVC*, 6(2), 182-197.
    - *Validates*: Section 7.10 Pareto frontier computation

35. **Chen, T., et al.** (2018). TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. *OSDI*.
    - *Validates*: Section 5.2 `TruenoNativeModel` kernel fusion

#### I. Embedded & WebAssembly

36. **Haas, A., et al.** (2017). Bringing the Web up to Speed with WebAssembly. *PLDI*, 185-200.
    - *Validates*: Section 8 WASM playground sandboxing

37. **Mazumdar, S., et al.** (2020). Concentric: A compression-aware AI accelerator. *IEEE Micro*.
    - *Validates*: `DataCompression::DeltaZstd` streaming decompression

#### J. Software Architecture

38. **Parnas, D. L.** (1972). On the Criteria To Be Used in Decomposing Systems into Modules. *CACM*, 15(12), 1053-1058.
    - *Validates*: Section 9 modular ecosystem (aprender/alimentar/pacha)

39. **Knight, J. C.** (2002). Safety Critical Systems: Challenges and Directions. *ICSE*, 547-550.
    - *Validates*: Section 1.6 WCET formal resource specification

### 10.2 Standards and Specifications

- **ISO 26262**: Road vehicles - Functional safety
- **NASA NPR 7150.2D**: NASA Software Engineering Requirements
- **DO-178C**: Software Considerations in Airborne Systems and Equipment Certification
- **Toyota Production System (TPS)**: Lean manufacturing principles

### 10.3 Implementation References

- **Trueno**: SIMD-accelerated tensor operations (github.com/paiml/trueno)
- **memmap2**: Memory-mapped file I/O for Rust (crates.io/crates/memmap2)
- **zstd**: Zstandard compression library (facebook.github.io/zstd/)
- **ed25519-dalek**: Ed25519 signatures for Rust (crates.io/crates/ed25519-dalek)

---

## 11. Toyota Way Compliance Matrix

*Reviewed by: Gemini (AI Assistant), 2025-12-08*
*Full review: `docs/specifications/reviews/apr-tooling-review-toyota-way.md`*

This section maps the 14 Toyota Way principles (Liker, 2004) to specific spec implementations.

### 11.1 Philosophy (Long-Term Thinking)

| Principle | Implementation | Section |
|-----------|----------------|---------|
| **P1**: Base decisions on long-term philosophy | Format designed for 10+ year stability; PQC-ready | 1.7 |

### 11.2 Process (Eliminate Waste)

| Principle | Implementation | Section |
|-----------|----------------|---------|
| **P2**: Create continuous flow | 5-layer hierarchical loading pipeline | 1.3 |
| **P3**: Use pull systems | Demand paging (`MappedDemand`, `LazySection`) | 1.2, 2.1 |
| **P4**: Level workload (Heijunka) | Ring buffer streaming, jitter absorption | 1.2, 1.6.3 |
| **P5**: Stop to fix (Jidoka) | 100-point scoring, `CriticalIssue` blocks deploy | 7.1 |
| **P6**: Standardize tasks | Single `.apr` format across all backends | 1.8 |
| **P7**: Visual control | `apr-inspect`, `apr-diff` CLI tools | 6.1, 6.2 |
| **P8**: Reliable technology | Zstd, Ed25519, Rust - proven, audited | 10.1 |

### 11.3 People (Respect & Challenge)

| Principle | Implementation | Section |
|-----------|----------------|---------|
| **P9**: Grow leaders | Comprehensive spec enables team scaling | All |
| **P10**: Develop teams | Modular architecture (qa, bench, format) | 7.9, 7.10 |
| **P11**: Respect partners | Ecosystem integration (alimentar, pacha, etc.) | 9 |

### 11.4 Problem Solving (Continuous Improvement)

| Principle | Implementation | Section |
|-----------|----------------|---------|
| **P12**: Go and see (Genchi Genbutsu) | `PlatformSpecs` requires HW measurement | 1.6.2 |
| **P13**: Decide slowly, implement fast | Spec-first design, rapid Trueno execution | All |
| **P14**: Become learning org (Kaizen) | Versioned format, benchmark tracking | 7.10 |

### 11.5 Muda (Waste) Elimination Checklist

| Waste Type | Eliminated By | Evidence |
|------------|---------------|----------|
| **Overproduction** | Demand paging, lazy loading | `LazySection` |
| **Waiting** | Async streaming, prefetch | `LoadingMode::Streaming` |
| **Transport** | Zero-copy mmap | `MappedView` |
| **Overprocessing** | Compression levels, format tiers | `CompressionLevel::Fast` |
| **Inventory** | LRU/ARC cache eviction | Section 3.2 |
| **Motion** | Single-file deployment | `.apr` bundling |
| **Defects** | 100-point scoring, QA gates | Section 7 |
| **Unused talent** | Self-documenting format, model cards | Section 7.6 |

### 11.6 Jidoka (Built-in Quality) Controls

```rust
/// Jidoka enforcement points in the loading pipeline
pub enum JidokaStop {
    /// Header magic/version mismatch - stop immediately
    InvalidHeader,
    /// Signature verification failed - stop, alert security
    SignatureFailed,
    /// Checksum mismatch - stop, data corrupted
    ChecksumFailed,
    /// WCET budget exceeded - stop, unsafe for deployment
    WcetViolation,
    /// Fairness threshold breached - stop, ethical concern
    FairnessViolation,
    /// Model score below threshold - stop, quality gate
    QualityGateFailed { score: u8, threshold: u8 },
}

impl JidokaStop {
    /// All stops are non-recoverable without human intervention
    pub fn requires_human_review(&self) -> bool {
        true  // Andon cord - always escalate
    }
}
```

### 11.7 Poka-yoke (Mistake-Proofing) Mechanisms

| Mechanism | Prevents | Implementation |
|-----------|----------|----------------|
| Magic number `APRN` | Wrong file type | Header validation |
| Version check | Incompatible format | `version_major` field |
| CRC32 checksum | Silent corruption | Trailing checksum |
| `CipherSuite` enum | Wrong crypto selection | Type-safe crypto |
| `PlatformSpecs` struct | Invalid timing assumptions | Required HW params |
| `Difficulty` enum | Unbounded eval complexity | Stratified benchmarks |

### 11.8 v2 Review Additions (QA, Bench, Ecosystem)

*Reviewed by: Gemini (AI Assistant), 2025-12-08*
*Full review: `docs/specifications/reviews/apr-tooling-review-toyota-way-v2.md`*

| New Module | Toyota Way Principle | Implementation |
|------------|---------------------|----------------|
| `aprender::qa` | **P5** Jidoka (stop the line) | `Severity::Blocker` prevents registry publish |
| `aprender::bench` | **P3** Pull systems | Pareto frontier pulls smallest viable model |
| `no_std` bare metal | **P4** Heijunka (leveling) | `MAX_STACK_BYTES` eliminates heap jitter |
| WASM playground | **P14** Hansei (reflection) | Browser-based model experimentation |
| Sovereign Stack | **P1** Long-term philosophy | 10+ year format stability, PQC-ready |

**New Academic Validation (v2)**:
- Goodfellow FGSM, Carlini-Wagner PGD → `aprender::qa` adversarial tests
- Deb NSGA-II → `aprender::bench` Pareto optimization
- Haas WASM → Section 8 sandbox safety
- Parnas modularity → Section 9 ecosystem boundaries

### 11.9 Roadmap: v1.1 Recommendations

| Recommendation | Rationale | Priority |
|----------------|-----------|----------|
| **Formal verification** (`kani`/`haniwa`) | Prove `no_std` kernels panic-free | P1 |
| **Differential Privacy accounting** | Formal ε-δ tracking in metadata | P2 |
| **WASM SIMD128 fallback** | Universal browser accessibility | P2 |
| **Model Cards auto-generation** | Reduce documentation burden | P3 |

```rust
/// Future: Formal verification target for no_std inference
#[cfg(kani)]
#[kani::proof]
fn verify_linear_predict_no_panic() {
    let weights: [f32; 4] = kani::any();
    let input: [f32; 4] = kani::any();
    let mut output: [f32; 1] = [0.0];

    // Prove: no panics for any valid input
    let result = LinearModel::predict_no_std(&weights, &input, &mut output);
    kani::assert(result.is_ok(), "predict_no_std must not panic");
}
```

---

## Appendix A: Quality Gate Commands

```bash
# Validate .apr file integrity
apr-validate model.apr

# Full quality assessment
apr-score model.apr --output report.json

# Compare model versions
apr-diff model_v1.apr model_v2.apr --output diff.json

# Inspect without loading
apr-inspect model.apr --format json

# Bundle models for distribution
apr-bundle --input models/ --output bundle.rs --compression zstd

# Convert to WASM-friendly format
apr-convert model.apr --target wasm --optimize-size
```

---

## Appendix B: Configuration Schema

```yaml
# apr-config.yaml
loading:
  mode: eager  # eager | mapped_demand | streaming | lazy_section
  max_memory_bytes: 104857600  # 100 MB
  verification: standard  # unsafe_skip | checksum_only | standard | paranoid
  prefetch:
    enabled: true
    strategy: sequential  # sequential | tree | stride
    lookahead: 4

caching:
  l1_size_bytes: 67108864  # 64 MB
  l2_size_bytes: 1073741824  # 1 GB
  eviction_policy: arc  # lru | lfu | arc | clock | fixed
  ttl_seconds: 3600

scoring:
  strict_mode: false  # Treat warnings as errors
  dimensions:
    accuracy_performance: 25
    generalization_robustness: 20
    model_complexity: 15
    documentation_provenance: 15
    reproducibility: 15
    security_safety: 10

wasm:
  max_memory_bytes: 52428800  # 50 MB
  enable_threading: false
  optimize_size: true
```

---

**END OF SPECIFICATION**

*This document follows Toyota Way principles and NASA software safety standards. All implementations must pass Tier 3 quality gates before production deployment.*

*Contact: noah@paiml.com | Pragmatic AI Labs | github.com/paiml/aprender*
