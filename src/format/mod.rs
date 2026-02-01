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

// Homomorphic encryption module (spec: homomorphic-encryption-spec.md)
#[cfg(feature = "format-homomorphic")]
pub mod homomorphic;

// Weight comparison module (GH-121, HuggingFace/SafeTensors comparison)
pub mod compare;

// APR format module (GH-119, 64-byte alignment, JSON metadata, sharding)
pub mod v2;

// GGUF export module (spec §7.2)
pub mod gguf;

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

// Core I/O operations (save, load, inspect, PMAT-198)
pub mod core_io;

// Tensor listing library (TOOL-APR-001 - reads actual tensor index)
pub mod tensors;

// Model diff library (TOOL-APR-002 - format-agnostic comparison)
pub mod diff;

// Digital signatures (spec §4.2, PMAT-198)
#[cfg(feature = "format-signing")]
pub mod signing;

// Encryption operations (spec §4.1, PMAT-198)
#[cfg(feature = "format-encryption")]
pub mod encryption;

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
    apr_convert, apr_export, apr_import, apr_merge, AprConverter, Architecture, ConvertOptions,
    ConvertReport, ExportFormat, ExportOptions, ExportReport, ImportError, ImportOptions,
    MergeOptions, MergeReport, MergeStrategy, QuantizationType, Source, TensorExpectation,
    ValidationConfig,
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

// Re-export types (PMAT-198 - backward compatibility)
pub use types::*;

// Re-export core I/O (PMAT-198 - backward compatibility)
pub use core_io::*;

// Re-export signing functions (PMAT-198 - backward compatibility)
#[cfg(feature = "format-signing")]
pub use signing::*;

// Re-export encryption functions (PMAT-198 - backward compatibility)
#[cfg(feature = "format-encryption")]
pub use encryption::*;

#[cfg(test)]
mod tests;
