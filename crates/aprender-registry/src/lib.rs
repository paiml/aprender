// Clippy configuration for pacha crate
// Allow precision loss in size calculations (intentional for human-readable output)
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
// Allow similar names in complex algorithms
#![allow(clippy::similar_names)]
// Allow single char patterns where appropriate
#![allow(clippy::single_char_pattern)]
// Allow map().unwrap_or() pattern
#![allow(clippy::map_unwrap_or)]
// Allow unnested or-patterns for readability
#![allow(clippy::unnested_or_patterns)]
// Allow long literals (byte sequences, magic numbers)
#![allow(clippy::unreadable_literal)]
// Allow redundant closures for clarity
#![allow(clippy::redundant_closure)]
#![allow(clippy::redundant_closure_for_method_calls)]
// Allow lifetime elision choices
#![allow(clippy::needless_lifetimes)]
// Allow Result wrapping for API consistency
#![allow(clippy::unnecessary_wraps)]
// Allow default trait usage patterns
#![allow(clippy::default_trait_access)]
// Allow format string style choices
#![allow(clippy::uninlined_format_args)]
// Allow consecutive replace for readability
#![allow(clippy::collapsible_str_replace)]
// Dead code warnings help identify unused functions
// Individual items use #[allow(dead_code)] where needed
// Doc backticks optional
#![allow(clippy::doc_markdown)]
// Allow unused async for future implementation
#![allow(clippy::unused_async)]
// Allow missing docs for internal items
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
// Prefer expect() over unwrap() for better error messages
// unwrap_used is enforced via clippy.toml disallowed-methods
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
// Allow case-sensitive file extension checks (intentional)
#![allow(clippy::case_sensitive_file_extension_comparisons)]
// Allow manual Default implementations
#![allow(clippy::derivable_impls)]
// Allow field assignment patterns
#![allow(clippy::field_reassign_with_default)]
// Allow from_str method name (not trait impl)
#![allow(clippy::should_implement_trait)]
// Allow identical match arms for clarity
#![allow(clippy::match_same_arms)]
// Allow format in collect patterns
#![allow(clippy::format_collect)]
// Allow HashMap patterns
#![allow(clippy::map_entry)]
// Allow map_or patterns
#![allow(clippy::option_if_let_else)]
// Allow unused self for API consistency
#![allow(clippy::unused_self)]
// Allow pass-by-value for small types
#![allow(clippy::needless_pass_by_value)]
// Allow map_or simplification choices
#![allow(clippy::unnecessary_map_or)]
//! Pacha: Model, Data and Recipe Registry
//!
//! Pacha provides a unified registry for machine learning artifacts—models,
//! datasets, and training recipes—with full lineage tracking, semantic
//! versioning, and cryptographic integrity.
//!
//! # Quick Start
//!
//! ```no_run
//! use pacha::prelude::*;
//!
//! // Open or create the registry
//! let registry = Registry::open_default()?;
//!
//! // Register a model
//! let model_data = std::fs::read("model.apr")?;
//! let card = ModelCard::builder()
//!     .description("Fraud detection model")
//!     .metrics([("auc", 0.95), ("f1", 0.88)])
//!     .build();
//!
//! registry.register_model(
//!     "fraud-detector",
//!     &ModelVersion::new(1, 0, 0),
//!     &model_data,
//!     card,
//! )?;
//!
//! // Retrieve the model
//! let model = registry.get_model("fraud-detector", &ModelVersion::new(1, 0, 0))?;
//! println!("Model stage: {}", model.stage);
//! # Ok::<(), pacha::error::PachaError>(())
//! ```
//!
//! # Architecture
//!
//! Pacha consists of three main registries:
//!
//! - **Model Registry** - `.apr` format files with metadata, metrics, and lineage
//! - **Data Registry** - `.ald` format files with schema and provenance
//! - **Recipe Registry** - TOML configs with hyperparameters and environment specs
//!
//! # Storage
//!
//! Pacha uses content-addressed storage with BLAKE3 hashing for:
//! - Deduplication across versions
//! - Tamper detection
//! - Efficient delta storage
//!
//! Registry metadata is stored in `SQLite` at `~/.pacha/registry.db`.
#[macro_use]
#[allow(unused_macros, unused_variables)]
mod generated_contracts;
pub mod aliases;
pub mod cache;
pub mod catalog;
pub mod cli;
pub mod crypto;
pub mod data;
pub mod error;
pub mod experiment;
pub mod fetcher;
pub mod format;
pub mod lineage;
pub mod manifest;
pub mod model;
pub mod prelude;
pub mod recipe;
pub mod registry;
pub mod remote;
pub mod resolver;
pub mod signing;
pub mod storage;
pub mod uri;
pub use error::{PachaError, Result};
pub use registry::{Registry, RegistryConfig, StorageStats};
