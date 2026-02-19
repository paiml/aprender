#![allow(clippy::disallowed_methods)]
//! Popperian Falsification Tests — Compiler-Enforced Model Types & Model Oracle
//!
//! Per Popper (1959), each validation rule must make a prediction that could be
//! proven false. If a falsification test finds a counterexample, the implementation
//! is broken.
//!
//! Spec reference: docs/specifications/compiler-enforced-model-types-model-oracle.md §7
//!
//! Test IDs map to spec:
//! - FALSIFY-MFC-001..003: Model Family Contracts (§7.1)
//! - FALSIFY-ORC-001..004: Oracle CLI (§7.2)
//! - FALSIFY-CMP-001..003: Compiler Enforcement (§7.3)
//! - FALSIFY-BGN-001..002: Build-Time Codegen (§7.4)
//! - FALSIFY-ALG-001..009: Algebraic Invariants (§7.6)

use aprender::format::model_family::build_default_registry;

// Build-time generated constants
use aprender::format::model_family::{
    BERT_BASE_HIDDEN_DIM, BERT_VENDOR, DEEPSEEK_VENDOR, GEMMA_VENDOR, KNOWN_FAMILIES,
    LLAMA_8B_HIDDEN_DIM, LLAMA_8B_NUM_LAYERS, LLAMA_VENDOR, MISTRAL_VENDOR, PHI_VENDOR,
    QWEN2_0_5B_HIDDEN_DIM, QWEN2_0_5B_NUM_LAYERS, QWEN2_VENDOR, WHISPER_VENDOR,
};

// Validated tensor types for FALSIFY-CMP-001
use aprender::format::validated_tensors::{RowMajor, ValidatedWeight};

include!("includes/falsification_model_oracle_part_01.rs");
include!("includes/falsification_model_oracle_part_02.rs");
include!("includes/falsification_model_oracle_part_03.rs");
include!("includes/falsification_model_oracle_part_04.rs");
include!("includes/falsification_model_oracle_part_05.rs");
include!("includes/falsification_model_oracle_part_06.rs");
include!("includes/falsification_model_oracle_part_07.rs");
include!("includes/falsification_model_oracle_part_08.rs");
include!("includes/falsification_model_oracle_part_08b.rs");
include!("includes/falsification_model_oracle_part_09.rs");
include!("includes/falsification_model_oracle_part_10.rs");
