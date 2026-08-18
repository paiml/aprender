//! trueno-ptx-debug: Pure Rust PTX debugging and static analysis tool
//!
//! This crate provides static analysis capabilities for PTX (Parallel Thread Execution)
//! code, implementing the Popperian falsification framework for systematic bug detection.
//!
//! # Architecture
//!
//! The analyzer is structured into several components:
//! - **Parser**: Lexer and AST construction for PTX source
//! - **Analyzer**: Static analysis passes (type, control flow, data flow, address space)
//! - **Bugs**: Bug pattern definitions and registry
//! - **Falsification**: 100-point Popperian testing framework
//! - **Output**: Report generation (HTML, FKR tests)
//!
//! # Philosophy
//!
//! Pure Rust PTX Analysis - Zero CUDA SDK Dependency.
//! Following Popper's falsificationism: we cannot prove PTX correct, but we can
//! systematically attempt to falsify it.

#![warn(missing_docs)]
// Allow these clippy lints for the PTX debug tool (parser/analyzer code)
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::unused_self)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::similar_names)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_else)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::if_not_else)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::format_push_string)]
#![allow(missing_docs)]

pub mod analyzer;
pub mod bugs;
pub mod cli;
pub mod falsification;
pub mod output;
pub mod parser;

// Re-export key types
pub use analyzer::{AddressSpaceValidator, ControlFlowAnalyzer, DataFlowAnalyzer, TypeChecker};
pub use bugs::{BugClass, BugRegistry, Severity};
pub use falsification::{FalsificationRegistry, FalsificationReport, TestResult};
pub use parser::{Lexer, ParseError, Parser, PtxModule};
