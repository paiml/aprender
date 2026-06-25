//! Aprender Model Format (.apr)
//!
//! Binary format for ML model serialization with built-in quality (Jidoka):
//! - CRC32 checksum (integrity)
//! - Ed25519 signatures (provenance)
//! - AES-256-GCM encryption (confidentiality)
//! - Zstd compression (efficiency)
//! - Quantization (`Q8_0`, `Q4_0`, `Q4_1` - GGUF compatible)
//! - Streaming/mmap (JIT loading)
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Header (32 bytes, fixed)                │
//! ├─────────────────────────────────────────┤
//! │ Metadata (variable, MessagePack)        │
//! ├─────────────────────────────────────────┤
//! │ Chunk Index (if STREAMING flag)         │
//! ├─────────────────────────────────────────┤
//! │ Salt + Nonce (if ENCRYPTED flag)        │
//! ├─────────────────────────────────────────┤
//! │ Payload (variable, compressed)          │
//! ├─────────────────────────────────────────┤
//! │ Signature Block (if SIGNED flag)        │
//! ├─────────────────────────────────────────┤
//! │ Checksum (4 bytes, CRC32)               │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::{save, load, ModelType, SaveOptions};
//! use aprender::linear_model::LinearRegression;
//!
//! let model = LinearRegression::new();
//! // ... train model ...
//!
//! // Save with compression
//! save(&model, ModelType::LinearRegression, "model.apr", SaveOptions::default())?;
//!
//! // Load with verification
//! let loaded: LinearRegression = load("model.apr", ModelType::LinearRegression)?;
//! ```

// Imports needed by test modules via `use super::*`
// These were the original mod.rs imports before PMAT-198 extraction.
// The production code lives in submodules now, but tests use `use super::*`
// and need these types in scope.
#[allow(unused_imports)]
use crate::error::{AprenderError, Result};
#[allow(unused_imports)]
use serde::{de::DeserializeOwned, Deserialize, Serialize};
#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::fs::File;
#[cfg(feature = "format-compression")]
#[allow(unused_imports)]
use std::io::Cursor;
#[allow(unused_imports)]
use std::io::{BufReader, BufWriter, Read, Write};
#[allow(unused_imports)]
use std::path::Path;

// Quantization module (spec §6.2)
#[cfg(feature = "format-quantize")]
pub mod quantize;

// GH-386: AVX2 SIMD fast paths for Q4_0/Q8_0 dequant.
#[cfg(feature = "format-quantize")]
mod quantize_simd;

// Homomorphic encryption module (spec: homomorphic-encryption-spec.md)
#[cfg(feature = "format-homomorphic")]
pub mod homomorphic;

// Weight comparison module (GH-121, HuggingFace/SafeTensors comparison)
pub mod compare;

// APR format module (GH-119, 64-byte alignment, JSON metadata, sharding)
pub mod v2;

// GGUF export module (spec §7.2)
pub mod gguf;

// ONNX format reader (GH-238)
pub mod onnx;

// Hex dump and data flow visualization (GH-122, Toyota Principle 12: Genchi Genbutsu)
pub mod hexdump;

// Model card module (spec §11)
pub mod model_card;

// Validation module (spec §11 - 100-Point QA Checklist)
#[allow(clippy::case_sensitive_file_extension_comparisons)]
pub mod validation;

// Converter types module (PMAT-197 - File size reduction)
pub mod converter_types;

// Converter module (spec §13 - Import/Convert Pipeline)
#[allow(
    clippy::unnecessary_wraps,
    clippy::type_complexity,
    clippy::trivially_copy_pass_by_ref,
    clippy::explicit_iter_loop,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::map_unwrap_or,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::uninlined_format_args,
    clippy::derivable_impls
)]
pub mod converter;

// Lint module (spec §4.11 - Best Practices & Conventions)
#[allow(
    clippy::struct_excessive_bools,
    clippy::field_reassign_with_default,
    clippy::uninlined_format_args,
    dead_code
)]
pub mod lint;

// Sharded model import module (GH-127 - multi-tensor repos, streaming import)
pub mod sharded;

// Golden trace verification (spec §7.6.3 - prove model authenticity)
pub mod golden;

// Rosetta Stone - Universal Model Format Converter (PMAT-ROSETTA-001)
// Bidirectional conversion: GGUF ↔ APR ↔ SafeTensors
pub mod rosetta;

// Rosetta ML Diagnostics (ROSETTA-ML-001)
// ML-powered format conversion diagnostics using aprender's own algorithms
pub mod rosetta_ml;

// Type definitions (spec §2-§9, PMAT-198)
pub mod types;

// F16 safety constants and helpers (GH-186 - prevent NaN propagation)
pub mod f16_safety;

// Core I/O operations (save, load, inspect, PMAT-198)
pub mod core_io;

// Tensor listing library (TOOL-APR-001 - reads actual tensor index)
pub mod tensors;

// Model diff library (TOOL-APR-002 - format-agnostic comparison)
pub mod diff;

// Tensor Layout Contract - THE SOURCE OF TRUTH (LAYOUT-CONTRACT-001)
// ALL tooling that deals with tensor shapes/layouts MUST use this module.
// See: contracts/tensor-layout-v1.yaml and docs/specifications/qwen2.5-coder-showcase-demo.md §E.8
pub mod layout_contract;

// Validated Tensor Types - Compile-Time Contract Enforcement (PMAT-235)
// Implements Poka-Yoke (mistake-proofing) via newtype pattern.
// Makes it IMPOSSIBLE to use unvalidated tensor data at compile time.
// See: contracts/tensor-layout-v1.yaml §type_enforcement
pub mod validated_tensors;

// Validated Classification Types - Classification Fine-Tuning Contract
// Poka-Yoke types for classifier logits, labels, and weights.
// See: contracts/classification-finetune-v1.yaml
pub mod validated_classification;

// Model Family Contract Types (PMAT-241)
// Compiler-enforced model family contracts: trait, config types, registry.
// See: contracts/model-families/*.yaml and
// docs/specifications/compiler-enforced-model-types-model-oracle.md
pub mod model_family;

// Model Family YAML Contract Loader (PMAT-242)
// Runtime YAML parser for model family contracts (no external deps).
// Fallback path; build.rs codegen (PMAT-250) is preferred.
pub mod model_family_loader;

// SHIP-TWO-001 AC-SHIP1-010 / FALSIFY-SHIP-010 algorithm-level PARTIAL
// discharge: pure decision rules for the published-artifact ship gate
// (SHA-256 byte-identity + manifest URL well-formedness).
// See: contracts/publish-manifest-v1.yaml v1.4.0 GATE-PM-010.
pub mod ship_010;

// Special tokens registry contract falsification (FALSIFY-ST-001..006)
#[cfg(test)]
mod special_tokens_contract_falsify;

// Model metadata bounds contract falsification (FALSIFY-MB-001..006)
#[cfg(test)]
mod metadata_bounds_contract_falsify;

// Tokenizer-vocabulary contract falsification (FALSIFY-TV-001..006)
#[cfg(test)]
mod tokenizer_vocab_contract_falsify;

// Embedding contract falsification (FALSIFY-EM-001..004, FALSIFY-EMB-001..007)
// Refs: embedding-lookup-v1.yaml, embedding-algebra-v1.yaml (PMAT-339, PMAT-340)
#[cfg(test)]
mod embedding_contract_falsify;

// Classification contract falsification (FALSIFY-CLASS-001..006)
// Refs: classification-finetune-v1.yaml
#[cfg(test)]
mod classification_contract_falsify;

// Digital signatures (spec §4.2, PMAT-198)
#[cfg(feature = "format-signing")]
pub mod signing;

// Encryption operations (spec §4.1, PMAT-198)
#[cfg(feature = "format-encryption")]
pub mod encryption;

// Formal verification: Kani proofs for APR format invariants
#[cfg(any(kani, test))]
mod kani_proofs;

// Formal verification: Verus-compatible specification contracts
pub mod verification_specs;

// Test factory - Pygmy model builders (T-COV-95)
// Implements the "Active Pygmy" pattern for creating minimal valid models in memory
#[cfg(test)]
pub mod test_factory;

// Re-export golden trace types
pub use golden::{
    verify_logits, GoldenTrace, GoldenTraceSet, GoldenVerifyReport, LogitStats, TraceVerifyResult,
};

// Re-export model card types
pub use model_card::{ModelCard, TrainingDataInfo};

// Re-export validation types (spec §11 - 100-Point QA Checklist)
pub use validation::{
    AprHeader, AprValidator, Category, CheckStatus, TensorStats, ValidationCheck, ValidationReport,
};

// Re-export Poka-yoke types (APR-POKA-001 - Toyota Way mistake-proofing)
#[allow(deprecated)]
pub use validation::no_validation_result;
pub use validation::{fail_no_validation_rules, Gate, PokaYoke, PokaYokeResult};

// Re-export converter types (spec §13 - Import/Convert Pipeline)
pub use converter::{
    apr_convert, apr_export, apr_import, apr_merge, streaming_quantize_peak_estimate, AprConverter,
    Architecture, ConvertOptions, ConvertReport, EvolutionaryMergeConfig, EvolutionaryMergeResult,
    ExportFormat, ExportOptions, ExportReport, ImportError, ImportOptions, MergeOptions,
    MergeReport, MergeStrategy, QuantizationType, Source, TensorExpectation, ValidationConfig,
};

// Re-export lint types (spec §4.11 - Best Practices & Conventions)
pub use lint::{
    lint_apr_file, lint_model, lint_model_file, LintCategory, LintIssue, LintLevel, LintReport,
    ModelLintInfo, TensorLintInfo,
};

// Re-export sharded import types (GH-127 - multi-tensor repos)
pub use sharded::{
    estimate_shard_memory, get_shard_files, is_sharded_model, CacheStats, CachedShard, ImportPhase,
    ImportProgress, ImportReport, ShardCache, ShardIndex, ShardedImportConfig, ShardedImporter,
};

// Re-export Rosetta Stone types (PMAT-ROSETTA-001 - Universal Model Format Converter)
pub use rosetta::{
    ConversionOptions, ConversionPath, ConversionReport, FormatType, InspectionReport,
    RosettaStone, TensorInfo, VerificationReport,
};
// Note: rosetta::TensorStats intentionally not re-exported to avoid conflict with validation::TensorStats
// Use aprender::format::rosetta::TensorStats directly if needed

// Re-export Rosetta ML Diagnostics types (ROSETTA-ML-001)
pub use rosetta_ml::{
    AndonLevel, AnomalyDetector, CanaryFile, CategorySummary, ConversionCategory,
    ConversionDecision, ConversionIssue, ErrorPattern, ErrorPatternLibrary, FixAction,
    HanseiReport, JidokaViolation, PatternSource, Priority, Regression, Severity, TarantulaTracker,
    TensorCanary, TensorFeatures, Trend, WilsonScore,
};

// Re-export tensor listing types (TOOL-APR-001 - reads actual tensor index)
// Note: TensorListInfo used instead of TensorInfo to avoid conflict with rosetta::TensorInfo
pub use tensors::{
    format_size, is_valid_apr_magic, list_tensors, list_tensors_from_bytes,
    TensorInfo as TensorListInfo, TensorListOptions, TensorListResult,
};

// Re-export diff types (TOOL-APR-002 - format-agnostic comparison)
pub use diff::{diff_inspections, diff_models, DiffCategory, DiffEntry, DiffOptions, DiffReport};

// Re-export layout contract types (LAYOUT-CONTRACT-001 - Source of Truth)
pub use layout_contract::{
    block_sizes, contract, validate_ffn_shape_symmetry, validation_rules, ContractError,
    LayoutContract, TensorContract,
};

// Re-export validated tensor types (PMAT-235 - Compile-Time Contract Enforcement)
// Implements Poka-Yoke: makes invalid tensor states unrepresentable
// RowMajor: PMAT-248 layout marker (PhantomData zero-cost enforcement)
pub use validated_tensors::{
    ContractValidationError, RowMajor, TensorStats as ValidatedTensorStats, ValidatedEmbedding,
    ValidatedVector, ValidatedWeight,
};

// Re-export validated classification types (classification-finetune-v1 contract)
pub use validated_classification::{
    ValidatedClassLogits, ValidatedClassifierWeight, ValidatedSafetyLabel,
};

// Re-export quantization types when feature is enabled
#[cfg(feature = "format-quantize")]
pub use quantize::{
    dequantize, quantize as quantize_data, Q4_0Quantizer, Q8_0Quantizer, QuantType,
    QuantizationInfo, QuantizedBlock, QuantizedTensor, Quantizer, BLOCK_SIZE,
};

// Re-export homomorphic encryption types when feature is enabled
#[cfg(feature = "format-homomorphic")]
pub use homomorphic::{
    Ciphertext, HeContext, HeGaloisKeys, HeParameters, HePublicKey, HeRelinKeys, HeScheme,
    HeSecretKey, Plaintext, SecurityLevel,
};

// Re-export signing types when feature is enabled
#[cfg(feature = "format-signing")]
pub use ed25519_dalek::{SigningKey, VerifyingKey};

/// Ed25519 signature size in bytes
#[cfg(feature = "format-signing")]
pub const SIGNATURE_SIZE: usize = 64;

/// Ed25519 public key size in bytes
#[cfg(feature = "format-signing")]
pub const PUBLIC_KEY_SIZE: usize = 32;

/// Argon2id salt size in bytes (spec §4.1.2)
#[cfg(feature = "format-encryption")]
pub const SALT_SIZE: usize = 16;

/// AES-GCM nonce size in bytes
#[cfg(feature = "format-encryption")]
pub const NONCE_SIZE: usize = 12;

/// AES-256 key size in bytes
#[cfg(feature = "format-encryption")]
pub const KEY_SIZE: usize = 32;

/// X25519 public key size in bytes (spec §4.1.3)
#[cfg(feature = "format-encryption")]
pub const X25519_PUBLIC_KEY_SIZE: usize = 32;

/// Recipient public key hash size for identification (spec §4.1.3)
#[cfg(feature = "format-encryption")]
pub const RECIPIENT_HASH_SIZE: usize = 8;

/// HKDF info string for X25519 key derivation (spec §4.1.3)
#[cfg(feature = "format-encryption")]
pub const HKDF_INFO: &[u8] = b"apr-v1-encrypt";

// Re-export X25519 types when feature is enabled
#[cfg(feature = "format-encryption")]
pub use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret as X25519SecretKey};

/// Magic number: "APRN" in ASCII (0x4150524E)
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x4E];

/// Current format version (1.0)
pub const FORMAT_VERSION: (u8, u8) = (1, 0);

/// Header size in bytes
pub const HEADER_SIZE: usize = 32;

/// Maximum uncompressed size (1GB safety limit)
pub const MAX_UNCOMPRESSED_SIZE: u32 = 1024 * 1024 * 1024;

// FALSIFY-SHIP-003 / AC-SHIP1-003 — per-layer cosine similarity threshold
// verdict fn for `apr convert --quantize q4_k_m` round-trip quality.
// See contracts/qwen2-e2e-verification-v1.yaml FALSIFY-QW2E-SHIP-003 and
// docs/specifications/aprender-train/ship-two-models-spec.md §4.2 AC-SHIP1-003.
pub mod ship_003;

// FALSIFY-SHIP-004 / AC-SHIP1-004 — GGUF export boundary verdict fns:
// llama-cli exit code + GGUF magic bytes + GGUF version.
// See contracts/qwen2-e2e-verification-v1.yaml FALSIFY-QW2E-SHIP-004 and
// docs/specifications/aprender-train/ship-two-models-spec.md §4.2 AC-SHIP1-004.
pub mod ship_004;

// FALSIFY-SHIP-001 / AC-SHIP1-001 — safetensors load boundary verdict fns:
// Result<Model, _> → bool, safetensors header size invariant, JSON-object
// open-brace byte. See contracts/qwen2-e2e-verification-v1.yaml
// FALSIFY-QW2E-SHIP-001 and docs/specifications/aprender-train/ship-two-
// models-spec.md §4.2 AC-SHIP1-001.
pub mod ship_001;

// FALSIFY-SHIP-023 / AC-SHIP1-023 — two-day HumanEval pass@1 drift verdict:
// pair-of-runs drift ≤ 1.2 pp with symmetric `.abs()` combinator + input
// well-formedness guards. See contracts/qwen2-e2e-verification-v1.yaml
// FALSIFY-QW2E-SHIP-023 and docs/specifications/aprender-train/ship-two-
// models-spec.md §7.1 FALSIFY-SHIP-023.
pub mod ship_023;

// FALSIFY-SHIP-024 / AC-SHIP1-024 — adversarial-suite runtime-invariant
// verdict: suite-size floor ≥ 50 AND panic_count == 0 AND nan_count == 0.
// See contracts/qwen2-e2e-verification-v1.yaml FALSIFY-QW2E-SHIP-024 and
// docs/specifications/aprender-train/ship-two-models-spec.md §7.1
// FALSIFY-SHIP-024.
pub mod ship_024;

// SHIP-TWO-001 §6 Compound Ship Gates — aggregate / cross-cutting PARTIAL
// algorithm-level discharges. Each module binds one §6 compound-gate row
// to one pure verdict fn + mutation survey. Authoritative contract:
// contracts/compound-ship-gates-v1.yaml v1.0.0.

// FALSIFY-APR-GGUF-PARITY — per-layer ffn_swigl ratio gate for SHIP-007.
pub mod apr_gguf_forward_parity;

// FALSIFY-APR-DISTILL-TRAIN-005 — precompute byte-determinism gate.
pub mod distill_train_005;

// INV-DATA-006 — dataset-thestack-python disjoint train/val splits.
pub mod data_inv_006;

// FALSIFY-APR-DISTILL-TRAIN-002 — KL loss decreases over epochs gate.
pub mod distill_train_002;

// FALSIFY-QA-002 + 006 — apr-cli error-exit honesty (exit != 0 on missing file / error output).
pub mod qa_002_006;

// FALSIFY-QA-004 — apr-cli no NaN/Inf in user output (zero-tolerance scan).
pub mod qa_004;

// FALSIFY-QA-001 — apr-cli all 58 commands respond to --help.
pub mod qa_001;

// FALSIFY-PUB-CLI-002 + 004 — cargo install/check exit codes (shared verdict).
pub mod pub_cli_002_004;

// FALSIFY-PUB-CLI-001 — apr-cli default features contain no forbidden substrings.
pub mod pub_cli_001;

// FALSIFY-PUB-CLI-003 — apr --help line count > 50 (all 58 commands listed).
pub mod pub_cli_003;

// FALSIFY-APR-001..003 — apr-format-invariants-v1 PARTIAL_ALGORITHM_LEVEL discharge.
pub mod aprfi_001_003;

// FALSIFY-APR-PULL-DATASET-001 — apr pull dataset --help shows both flags + exits 0.
pub mod pull_dataset_001;

// FALSIFY-APR-PULL-DATASET-005 — apr pull <model> --dry-run backward compat.
pub mod pull_dataset_005;

// FALSIFY-APR-PULL-DATASET-003 — apr pull dataset no-match glob fails fast.
pub mod pull_dataset_003;

// FALSIFY-APR-PULL-DATASET-004 — license allowlist drops disallowed rows.
pub mod pull_dataset_004;

// FALSIFY-APR-PULL-DATASET-002 — apr pull dataset --include glob exact match count.
pub mod pull_dataset_002;

// FALSIFY-APR-DISTILL-TRAIN-009 — distill student val_loss < from-scratch baseline.
pub mod distill_train_009;

// INV-BPE-001 — tokenizer-bpe vocab range + paired-model match.
pub mod bpe_inv_001;

// INV-BPE-006 — tokenizer-bpe encode determinism (cross-process bit-identical IDs).
pub mod bpe_inv_006;

// INV-BPE-003 — tokenizer-bpe round-trip byte-equality on 10K held-out docs.
pub mod bpe_inv_003;

// INV-BPE-002 — tokenizer-bpe four required special tokens distinct + in range.
pub mod bpe_inv_002;

// FALSIFY-PROF10-003 — apr profile graphed vs ungraphed sanity inequality.
pub mod prof10_003;

// FALSIFY-SUB-FFN-005 — sub-FFN telemetry per-layer line count.
pub mod sub_ffn_005;

// FALSIFY-APR-TOK-PAR-002 — parallel BPE 80% efficiency floor.
pub mod tok_par_002;

// INV-DATA-004 — dataset-thestack-python train range + val floor.
pub mod data_inv_004;

// INV-DATA-007 — dataset-thestack-python UTF-8 + NFC round-trip.
pub mod data_inv_007;

// INV-DATA-001 — dataset-thestack-python license whitelist (zero-tolerance).
pub mod data_inv_001;

// INV-DATA-002 — dataset-thestack-python PII scrub zero-match invariant.
pub mod data_inv_002;

// INV-DATA-003 — dataset-thestack-python Jaccard dedup floor (< 0.85).
pub mod data_inv_003;

// INV-DATA-005 — dataset-thestack-python corpus_sha256 reproducibility.
pub mod data_inv_005;

// INV-PRETOK-003 — pretokenize-bin manifest sum=actual invariant.
pub mod pretok_inv_003;

// INV-PRETOK-002 — pretokenize-bin shard u32-alignment invariant.
pub mod pretok_inv_002;

// INV-PRETOK-001 — pretokenize-bin token id < vocab_size invariant.
pub mod pretok_inv_001;

// FALSIFY-APR-DISTILL-TRAIN-006 — stage train resumes from precompute cache.
pub mod distill_train_006;

// FALSIFY-APR-DISTILL-TRAIN-001 — real-training (not stub) tensor-diff gate.
pub mod distill_train_001;

// GATE-SHIP-001 — MODEL-1 aggregate-AND over 10 AC-SHIP1-* booleans.
pub mod gate_ship_001;

// GATE-SHIP-002 — MODEL-2 aggregate-AND over 12 AC-SHIP2-* booleans.
pub mod gate_ship_002;

// GATE-SHIP-003 — Golden Output byte-identity across quantize round-trip.
pub mod gate_ship_003;

// GATE-SHIP-004 — HumanEval bitwise-identical determinism (two seed=0 runs).
pub mod gate_ship_004;

// GATE-SHIP-005 — License metadata non-empty ASCII-printable byte-equal.
pub mod gate_ship_005;

// GATE-SHIP-006 — GGUF round-trip first-token probability delta ≤ 1e-3.
pub mod gate_ship_006;

// GATE-SHIP-007 — Zero-tolerance .unwrap() count threshold on new code.
pub mod gate_ship_007;

// GATE-SHIP-008 — Contract-density ratio threshold on new public fns.
pub mod gate_ship_008;

// GATE-SHIP-009 — CI aggregate-AND over 3 required checks (fmt / clippy / test).
pub mod gate_ship_009;

// GATE-SHIP-010 — Zero-tolerance security-advisory count threshold.
pub mod gate_ship_010;

// GATE-SHIP-011 — PMAT TDG score inclusive-floor threshold (≥ 90.0 / A-).
pub mod gate_ship_011;

// GATE-SHIP-012 — Line-coverage percentage inclusive-floor threshold (≥ 95.0).
pub mod gate_ship_012;

// Re-export types (PMAT-198 - backward compatibility)
pub use types::*;

// Re-export core I/O (PMAT-198 - backward compatibility)
pub use core_io::*;

// APR-2231: re-export the sovereign `apr-format` leaf so external consumers can
// reach it via `aprender_core::format::apr_format::*` while the bulk migration
// (Stage 2) lands. `aprender-core` From-wraps `apr_format::AprFormatError` into
// `AprenderError` (see `crate::error`), so the leaf's results compose with `?`.
pub use apr_format;

// Re-export signing functions (PMAT-198 - backward compatibility)
#[cfg(feature = "format-signing")]
pub use signing::*;

// Re-export encryption functions (PMAT-198 - backward compatibility)
#[cfg(feature = "format-encryption")]
pub use encryption::*;

#[cfg(test)]
mod tests;
