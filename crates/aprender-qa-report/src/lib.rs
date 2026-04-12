//! APR QA Report Generator
//!
//! Implements Popperian falsification scoring and report generation.
//! Produces MQS (Model Qualification Score) with Toyota-style gateway checks.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Allow common patterns
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unused_self)]
#![allow(clippy::format_push_string)]
#![allow(clippy::needless_raw_string_hashes)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::single_match_else)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::imprecise_flops)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::option_if_let_else)]
// Allow common patterns in test code
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]
#![cfg_attr(test, allow(clippy::float_cmp))]

pub mod certificate;
pub mod certification_data;
pub mod defect_map;
pub mod error;
pub mod evidence_export;
pub mod html;
pub mod junit;
pub mod markdown;
pub mod mqs;
pub mod popperian;
pub mod proof_status;
pub mod ticket;

pub use certificate::{Certificate, CertificateGenerator, CertificationStatus};
pub use certification_data::{
    lookup_family, lookup_model, read_models_csv, write_models_csv, CertificationRow, ModelStatus,
    SizeCategory,
};
pub use error::{Error, Result};
pub use evidence_export::{
    EvidenceExport, EvidenceExportBuilder, ExportSummary, GateResult, ModelMeta, MqsExport,
    PlaybookMeta,
};
pub use markdown::{generate_evidence_detail, generate_index_entry, generate_rag_markdown};
pub use mqs::{GatewayResult, MqsCalculator, MqsScore};
pub use popperian::PopperianScore;
pub use proof_status::{proof_bonus_for_class, read_proof_status, ExternalProofStatus, ProofBonus};
