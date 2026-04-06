//! # Realizar
//!
//! Pure Rust, portable, high-performance ML library with unified CPU/GPU/WASM support.
//!
//! Realizar (Spanish: "to accomplish, to achieve") provides a unified API for machine learning
//! operations that automatically dispatches to the optimal backend based on data size,
//! operation complexity, and available hardware.
//!
//! ## Features
//!
//! - **Unified API**: Single interface for CPU SIMD, GPU, and WASM execution
//! - **Native Integration**: First-class support for `trueno` and `aprender`
//! - **Memory Safe**: Zero unsafe code in public API, leveraging Rust's type system
//! - **Production Ready**: EXTREME TDD, 85%+ coverage, zero tolerance for defects
//!
//! ## Example
//!
//! ```rust
//! use realizar::Tensor;
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![3, 3], vec![
//!     1.0, 2.0, 3.0,
//!     4.0, 5.0, 6.0,
//!     7.0, 8.0, 9.0,
//! ]).expect("test");
//!
//! // Check tensor properties
//! assert_eq!(a.shape(), &[3, 3]);
//! assert_eq!(a.ndim(), 2);
//! assert_eq!(a.size(), 9);
//! ```
//!
//! ## Future Operations (Phase 1+)
//!
//! ```rust,ignore
//! // Element-wise operations (SIMD-accelerated) - Coming in Phase 1
//! let sum = a.add(&b).expect("test");
//!
//! // Matrix multiplication (GPU-accelerated for large matrices) - Coming in Phase 2
//! let product = a.matmul(&b).expect("test");
//! ```
//!
//! ## Architecture
//!
//! Realizar is built on top of:
//! - **Trueno**: Low-level compute primitives with SIMD/GPU/WASM backends
//! - **Aprender**: High-level ML algorithms (will be refactored to use Realizar)
//!
//! ## Quality Standards
//!
//! Following EXTREME TDD methodology:
//! - Test Coverage: ≥85%
//! - Mutation Score: ≥80%
//! - TDG Score: ≥90/100
//! - Clippy Warnings: 0 (enforced)
//! - Cyclomatic Complexity: ≤10 per function

#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_comparisons,
    unused_mut,
    unused_assignments,
    unused_doc_comments
)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::identity_op)]
#![allow(clippy::no_effect)]
#![allow(clippy::erasing_op)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::iter_skip_zero)]
#![allow(clippy::expect_fun_call)]
#![allow(clippy::filter_next)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::iter_skip_next)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::items_after_test_module)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(clippy::write_literal)]
// Multiple crate versions are acceptable for dependencies
// #![warn(clippy::cargo)]

// Clippy allows (MUST come after deny/warn to override them)
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::large_stack_arrays)] // Test data
#![allow(clippy::cast_possible_wrap)] // u64 -> i64 for timestamps is safe
#![allow(clippy::cast_precision_loss)] // usize -> f32 precision loss is acceptable
#![allow(clippy::cast_possible_truncation)] // u128 -> u64 etc for metrics is safe
#![allow(clippy::cast_sign_loss)] // Metrics conversions are safe
#![allow(clippy::too_many_lines)] // Some handlers are naturally long
#![allow(clippy::must_use_candidate)] // Not all methods need #[must_use]
#![allow(clippy::doc_markdown)] // Allow technical terms without backticks
#![allow(clippy::redundant_clone)] // Sometimes clarity > performance
#![allow(clippy::uninlined_format_args)] // Prefer explicit format args
#![allow(clippy::single_match_else)] // Sometimes clearer than if-let
#![allow(clippy::unnecessary_to_owned)] // Allow explicit .to_string()
#![allow(clippy::single_char_pattern)] // Allow "x" instead of 'x' in contains()
#![allow(clippy::missing_panics_doc)] // Allow missing Panics doc sections
#![allow(clippy::missing_errors_doc)] // Allow missing Errors doc sections (common in math code)
#![allow(clippy::items_after_statements)] // Allow const/type definitions after statements
#![allow(clippy::unused_self)] // Allow unused self in methods for API consistency
#![allow(clippy::cloned_instead_of_copied)] // Allow cloned() even for Copy types
#![allow(clippy::needless_pass_by_value)] // Allow pass-by-value where it's clearer
#![allow(clippy::unnecessary_wraps)] // Allow wrapping in Result/Option for API consistency
#![allow(clippy::if_not_else)] // Allow if !condition { } else { }
#![allow(clippy::manual_let_else)] // Allow manual let-else patterns
#![allow(clippy::float_cmp)] // Allow float comparisons in tests
#![allow(clippy::cast_lossless)] // Allow i32 to f64 casts
#![allow(clippy::approx_constant)] // Allow approximate PI
#![allow(clippy::manual_range_contains)] // Allow manual range checks
#![allow(clippy::same_item_push)] // Allow pushing same items in tests
#![allow(clippy::similar_names)] // Allow similar variable names in test code
#![allow(clippy::unreadable_literal)] // Allow literals without separators in test code
#![allow(clippy::useless_vec)] // Allow vec![] where slice would work in tests
#![allow(clippy::ignore_without_reason)] // Allow #[ignore] without explicit reason
#![allow(clippy::cast_ptr_alignment)] // Allow unaligned SIMD pointer casts (loadu/storeu are safe)
#![allow(clippy::ptr_as_ptr)] // Allow pointer cast style in SIMD code
#![allow(clippy::struct_excessive_bools)] // Allow structs with multiple bool fields
#![allow(clippy::match_same_arms)] // Allow match arms with same bodies for clarity
#![allow(clippy::assertions_on_constants)] // Allow assert!(true) in tests
#![allow(clippy::format_push_string)] // Allow format! with push_str for clarity
#![allow(clippy::upper_case_acronyms)] // Allow VLLM, APR, GGUF, ONNX etc.
#![allow(clippy::struct_field_names)] // Allow field names with common suffix (_ms, _hash)
#![allow(clippy::if_same_then_else)] // Allow if/else with same block for clarity
#![allow(clippy::format_collect)] // Allow map().collect() with format! inside
#![allow(clippy::no_effect_underscore_binding)] // Allow underscore-prefixed bindings
#![allow(clippy::too_many_arguments)] // Allow functions with >7 args
#![allow(clippy::needless_range_loop)] // Allow for i in 0..len style loops
#![allow(clippy::trivially_copy_pass_by_ref)] // Allow &self on small Copy types
#![allow(clippy::used_underscore_items)] // Allow using _prefixed items
#![allow(clippy::field_reassign_with_default)] // Allow field reassign after default

// Contract assertions from YAML (pv codegen)
#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;
#[cfg(feature = "server")]
pub mod api;
/// Aprender .apr format support (PRIMARY inference format)
///
/// The .apr format is the native format for the sovereign AI stack.
/// GGUF and safetensors are supported as fallback formats.
pub mod apr;
/// APR Transformer format for WASM-compatible LLM inference
///
/// Provides F32 transformer weights for fair APR vs GGUF comparison.
/// Designed for WASM compatibility - no SIMD requirements.
pub mod apr_transformer;
/// Per-architecture required weight roles (GH-279).
///
/// UCBD §4 / GH-279: Compile-time enforcement that every loader
/// provides all tensors required by the target architecture.
pub mod arch_requirements;
/// Audit trail and provenance logging
///
/// Per spec §12: Comprehensive audit record for every inference request.
/// Implements GDPR Article 13/14 and SOC 2 compliance requirements.
/// - Full provenance tracking (model hash, distillation lineage)
/// - Latency breakdown (preprocessing, inference, postprocessing)
/// - Quality gates (Jidoka: NaN check, confidence check)
pub mod audit;
/// Benchmark harness for model runner comparison
///
/// Implements the benchmark specification v1.1 with Toyota Way engineering principles:
/// - Dynamic CV-based stop-rule (Hoefler & Belli)
/// - Thermal throttling protocol
/// - ITL variance measurement
/// - KV-cache fragmentation detection
/// - KL-Divergence quality validation
pub mod bench;
/// Preflight validation protocol for deterministic benchmarking
///
/// Per spec v1.0.1, implements Toyota Way principles:
/// - Jidoka: Fail-fast validation, stop on anomaly
/// - Poka-yoke: Error-proofing through type-safe configurations
/// - Genchi Genbutsu: Verify actual system state
///
/// References:
/// - Hoefler & Belli SC'15: CV-based stopping
/// - Vitek & Kalibera EMSOFT'11: Reproducibility requirements
pub mod bench_preflight;
/// Benchmark visualization for inference comparison (PAR-040)
///
/// Creates 2×3 grid visualizations comparing APR vs Ollama vs llama.cpp
/// and generates profiling logs suitable for chat paste debugging.
pub mod bench_viz;
/// ComputeBrick architecture for token-centric, self-verifying inference
///
/// Per spec: Qwen2.5-Coder Showcase Demo v3.0.0
/// Implements 5-layer brick hierarchy with Toyota Way engineering:
/// - Jidoka: Every brick has stop-the-line assertions
/// - Poka-Yoke: Token budgets enforce performance contracts
/// - Genchi Genbutsu: Statistical benchmarking with CV < 5%
/// - Mieruka: Visual progress via TUI integration
pub mod brick;
pub mod cache;
/// GH-280: Kernel capability gate — contract-driven GPU admission control.
///
/// Models declare required operations via `ArchConstraints`; GPU backends
/// declare supported operations. Mismatch = refuse at load time.
pub mod capability;
/// Chat template engine for model-specific message formatting
///
/// Supports ChatML (Qwen2, Yi), LLaMA2 (TinyLlama, Vicuna),
/// Mistral, Phi, Alpaca, and Raw formats.
/// Auto-detects format from model name.
pub mod chat_template;
/// CLI command implementations (extracted for testability)
#[cfg(feature = "cli")]
pub mod cli;
/// GGUF to APR Transformer converter
///
/// Converts GGUF models to APR format for fair comparison.
/// All weights are dequantized to F32 for WASM compatibility.
pub mod contract_gate;
pub mod convert;
/// CUDA PTX generation for NVIDIA GPUs
///
/// Provides native CUDA kernel generation and execution via trueno-gpu.
/// - Pure Rust PTX generation (no LLVM, no nvcc)
/// - Hand-optimized kernels: GEMM, Softmax, LayerNorm, Attention, Q4K
/// - FlashAttention-style tiled attention
/// - Full CUDA runtime via trueno-gpu driver (context, stream, memory)
#[cfg(feature = "cuda")]
#[allow(
    clippy::borrow_as_ptr,
    clippy::ptr_as_ptr,
    clippy::many_single_char_names,
    clippy::manual_div_ceil
)]
pub mod cuda;
pub mod error;
/// Model explainability (SHAP, Attention)
///
/// Per spec §13: Model explainability for APR classical ML models.
/// Implements SHAP TreeExplainer for tree ensembles and KernelSHAP for any model.
/// - TreeSHAP: O(TLD) complexity for tree-based models
/// - KernelSHAP: Model-agnostic with weighted linear regression
/// - Feature importance: Top-k features by absolute SHAP value
pub mod explain;
/// Test fixtures for model loading with RAII-based cleanup.
///
/// Provides standardized test fixtures for GGUF, SafeTensors, and APR formats
/// with automatic temporary file cleanup via TempDir.
#[cfg(test)]
pub mod fixtures;
/// Unified model format detection (APR, GGUF, SafeTensors)
///
/// Per spec §3: Format Support Matrix - auto-detect from magic bytes.
/// APR is first-class, GGUF and SafeTensors are backwards-compatible.
pub mod format;
pub mod generate;
pub mod gguf;
/// GPU acceleration module (Phase 4: ≥100 tok/s target)
///
/// Implements GPU-accelerated matrix operations via Trueno's wgpu backend.
/// - GPU matmul shader for large matrix multiplications
/// - Hybrid CPU/GPU scheduling based on workload size
/// - Automatic fallback to SIMD when GPU unavailable
#[cfg(feature = "gpu")]
#[allow(clippy::similar_names)] // GPU code has intentionally similar kv_head/k_head names
pub mod gpu;
/// Grammar-constrained generation for structured output
///
/// Implements GBNF-style grammar constraints for LLM generation.
/// - JSON schema validation
/// - Custom grammar rules (GBNF format)
/// - Token masking for efficient constrained generation
/// - State machine for tracking grammar state
pub mod grammar;
/// HTTP client for real model server benchmarking
///
/// Implements actual HTTP calls to external servers (vLLM, Ollama, llama.cpp).
/// **NO MOCK DATA** - measures real network latency and inference timing.
#[cfg(feature = "bench-http")]
pub mod http_client;
/// High-level inference API for CLI tools
///
/// Per spec APR-CLI-DELEGATE-001: All inference in `apr run` and `apr chat`
/// delegates to this module. This eliminates ~1800 lines of duplicated code.
///
/// # Example
///
/// ```rust,ignore
/// use realizar::infer::{InferenceConfig, run_inference};
///
/// let result = run_inference(&InferenceConfig::new("model.gguf")
///     .with_prompt("Hello!"))?;
/// println!("{}", result.text);
/// ```
pub mod infer;
/// SIMD-accelerated inference engine using trueno
///
/// Provides high-performance transformer inference competing with llama.cpp.
/// Uses trueno's SIMD primitives for matrix operations.
pub mod inference;
/// Inference tracing for debugging LLM pipelines
///
/// Per spec APR-TRACE-001: Toyota Way Genchi Genbutsu (Go and See) + Jidoka.
/// Provides step-by-step visualization of the inference pipeline:
/// - ENCODE: Tokenization with OOV detection
/// - EMBED: Token embedding lookup
/// - TRANSFORMER: Layer-by-layer processing
/// - LM_HEAD: Final projection to logits
/// - SAMPLE: Token sampling
/// - DECODE: Token to text decoding with garbage detection (APR-TOK-001)
pub mod inference_trace;
pub mod layers;
pub mod memory;
#[cfg(feature = "server")]
pub mod metrics;
/// Unified model loader for APR, GGUF, and SafeTensors
///
/// Per spec §3.2 and §5: Combines format detection with model loading.
/// Supports all 18 APR model types.
pub mod model_loader;
#[cfg(test)]
mod model_loader_tests_02;
#[cfg(test)]
mod model_loader_tests_load_error;
pub mod moe;
/// Observability: metrics, tracing, and A/B testing
///
/// Safe numeric casts for observability metrics:
/// - Duration microseconds: u128 -> u64 (durations under 584,942 years won't overflow)
/// - Timestamps: u128 -> u64 (Unix epoch nanoseconds/microseconds fit in u64 until ~2554)
/// - Percentages: integer -> f64 (exact for values under 2^53)
#[cfg(feature = "server")]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_sign_loss)]
pub mod observability;
/// PagedAttention KV cache management
///
/// Per spec §8.1: Efficient KV cache management based on vLLM's PagedAttention.
/// Reference: [4] Kwon et al. (2023) "Efficient Memory Management for LLM Serving"
/// - Physical pages: Fixed-size memory blocks for KV cache
/// - Page tables: Logical to physical mapping per sequence
/// - Copy-on-Write: Efficient prefix sharing between sequences
pub mod paged_kv;
/// Multi-GPU and Distributed Inference
///
/// Per spec §10: Implements parallelism strategies for 70B+ model inference.
/// Reference: [11] Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter LMs"
/// - Tensor Parallelism (TP): Split tensors across GPUs within node (2-8 GPUs)
/// - Pipeline Parallelism (PP): Split layers across GPUs/nodes (2-64 GPUs)
/// - Data Parallelism (DP): Replicate model, split batches
/// - ZeRO-Inference: Memory offload to CPU
pub mod parallel;
/// PTX Parity Validation — GH-219
///
/// Validates that batched GPU kernels maintain structural parity with their
/// single-vector reference implementations. Exposed as `apr qa` Gate 6.
pub mod ptx_parity;
pub mod quantize;
#[cfg(feature = "server")]
pub mod registry;
#[cfg(all(test, feature = "server"))]
mod registry_tests;
pub mod safetensors;
/// SafeTensors CUDA inference (PMAT-116)
///
/// Direct GPU loading for HuggingFace SafeTensors models.
/// Achieves GGUF GPU parity (200+ tok/s).
#[cfg(feature = "cuda")]
pub mod safetensors_cuda;
/// SafeTensors inference support (PAR-301)
///
/// Converts HuggingFace SafeTensors models to AprTransformer for inference.
/// Requires config.json and tokenizer.json in the same directory.
pub mod safetensors_infer;
/// Continuous batching scheduler
///
/// Per spec §8: Implements continuous batching for LLM serving based on vLLM/Orca.
/// Reference: [8] Yu et al. (2022) "Orca: A Distributed Serving System"
/// - Iteration-level scheduling: New requests join batch at any iteration
/// - Preemption: Low-priority requests can be preempted for high-priority
/// - Memory-aware: Respects KV cache limits when scheduling
pub mod scheduler;
#[cfg(feature = "aprender-serve")]
pub mod serve;
/// Speculative decoding for LLM inference acceleration
///
/// Per spec §8.3: Implements speculative decoding based on SGLang/DeepMind research.
/// Reference: [9] Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
/// - Draft model: Small model generates K candidate tokens
/// - Target model: Verifies all K tokens in single forward pass
/// - Rejection sampling: Maintains exact target distribution
/// - Speedup: Up to 3x with well-matched draft/target pairs
pub mod speculative;
pub mod stats;
pub mod tensor;
/// GH-311: Contract-driven tensor name resolution (tensor-names-v1.yaml codegen)
pub mod tensor_names;
/// Model fixture testing infrastructure with PyTorch-style patterns.
///
/// Provides standardized testing for model formats (GGUF, APR, SafeTensors)
/// across devices (CPU, CUDA) with combinatorial coverage and Popperian falsification.
/// Per spec: docs/specifications/model-fixture-setup-teardown.md
#[cfg(test)]
pub mod testing;
/// TUI monitoring for inference performance
pub mod tui;
pub mod viz;
/// Model warm-up and pre-loading
pub mod warmup;

/// AWS Lambda handler for aprender model serving
#[cfg(feature = "lambda")]
pub mod lambda;
/// Multi-target deployment support (Lambda, Docker, WASM)
pub mod target;
pub mod tokenizer;
/// Pacha URI scheme support for model loading
pub mod uri;

// Re-exports for convenience
pub use arch_requirements::*;
pub use error::{RealizarError, Result};
pub use infer::{
    run_batch_inference, run_inference, BatchInferenceConfig, BatchPrompt, BatchResult, BatchStats,
    InferenceConfig, InferenceResult, PreparedTokens,
};
pub use inference_trace::{InferenceTracer, ModelInfo, TraceConfig, TraceStep};
pub use ptx_parity::{KernelDimensions, PtxParityReport};
#[cfg(not(target_arch = "wasm32"))]
pub use safetensors::MappedSafeTensorsModel;
pub use safetensors::SafetensorsConfig;
#[cfg(not(target_arch = "wasm32"))]
pub use safetensors::ShardedSafeTensorsModel;
pub use safetensors::ValidatedAprTransformer;
pub use tensor::Tensor;

// UCBD §4: Canonical normalization functions (ONE PATH)
pub use gguf::ops::{rms_norm, rms_norm_into};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        // VERSION is a compile-time constant from CARGO_PKG_VERSION, so it's never empty
        assert!(VERSION.starts_with("0."));
        assert!(VERSION.len() >= 3); // At least "0.x"
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn test_version_parts() {
        let parts: Vec<&str> = VERSION.split('.').collect();
        assert!(parts.len() >= 2); // At least major.minor
    }

    #[test]
    fn test_tensor_reexport() {
        // Test that Tensor is properly re-exported
        let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("t");
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.size(), 4);
    }

    #[test]
    fn test_error_reexport() {
        // Test that RealizarError is properly re-exported
        let err = RealizarError::InvalidShape {
            reason: "test".to_string(),
        };
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_result_type() {
        // Test that Result type alias works
        fn test_fn() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(test_fn().expect("expected value"), 42);
    }

    #[test]
    fn test_inference_config_reexport() {
        // Test InferenceConfig is properly re-exported
        let config = InferenceConfig::new("/dev/null");
        // Just verify the config can be created
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("InferenceConfig"));
    }

    #[test]
    fn test_trace_step_reexport() {
        // Test TraceStep is properly re-exported
        let step = TraceStep::Tokenize;
        assert!(format!("{:?}", step).contains("Tokenize"));
    }

    #[test]
    fn test_trace_config_reexport() {
        // Test TraceConfig is properly re-exported
        let config = TraceConfig::default();
        assert!(!config.enabled);
    }
}
