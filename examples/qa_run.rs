#![allow(clippy::disallowed_methods)]
//! QA Example: apr run Falsification Suite (PMAT-QA-RUST-001 + PMAT-QA-MATRIX-001)
//!
//! Popperian falsification tests for `apr run` command with full matrix support.
//!
//! # CRITICAL: Same-Model Comparison Protocol (PMAT-SHOWCASE-METHODOLOGY-001)
//!
//! **Class A (Quantized):** GGUF Q4_K_M vs APR Q4_K (converted from same GGUF)
//! **Class B (Full Precision):** SafeTensors F32 vs APR F32 (converted from same SafeTensors)
//!
//! NEVER compare different quantizations (e.g., Q4_K vs F32) - this is a FATAL DEFECT.
//!
//! # Canonical Model
//!
//! ```text
//! GGUF: hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
//! ```
//!
//! # Test Classes
//!
//! ## Class A: Quantized (60 points)
//! | Cell | Backend | Format | Model Source |
//! |------|---------|--------|--------------|
//! | A1 | CPU | GGUF Q4_K | HF GGUF |
//! | A2 | CPU | APR Q4_K | Converted from GGUF |
//! | A3 | GPU | GGUF Q4_K | HF GGUF |
//! | A4 | GPU | APR Q4_K | Converted from GGUF |
//!
//! ## Class B: Full Precision (40 points) - SLOWER, memory-bound
//! | Cell | Backend | Format | Model Source |
//! |------|---------|--------|--------------|
//! | B1 | CPU | SafeTensors F32 | HF SafeTensors |
//! | B2 | CPU | APR F32 | Converted from SafeTensors |
//! | B3 | GPU | SafeTensors F32 | HF SafeTensors |
//! | B4 | GPU | APR F32 | Converted from SafeTensors |
//!
//! # Usage
//!
//! ```bash
//! # Class A only (quantized - recommended)
//! cargo run --example qa_run -- --class quantized --matrix
//!
//! # Class B only (full precision - slow)
//! cargo run --example qa_run -- --class full-precision --matrix
//!
//! # Both classes
//! cargo run --example qa_run -- --class all --matrix
//!
//! # Single cell
//! cargo run --example qa_run -- --backend gpu --format gguf
//!
//! # With tracing
//! cargo run --example qa_run -- --backend gpu --format gguf --trace
//! ```
//!
//! # Tracing (ALL must work for run/chat/serve)
//!
//! - `--trace`: Step-by-step timing with [TRACE-CACHE] messages
//! - `--trace-level layer`: Per-layer breakdown (Attention, FFN, Norm)
//! - `--profile`: Roofline analysis (memory vs compute bound)
//!
//! # Citations
//!
//! - Popper, K. R. (1959). The Logic of Scientific Discovery. Routledge.
//! - PMAT-SHOWCASE-METHODOLOGY-001: Same-Model Comparison Protocol

use std::env;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

include!("includes/qa_run.rs");
include!("includes/pmat-qa-protocol-001.rs");
include!("includes/verify_result.rs");
include!("includes/parsed_args.rs");
include!("includes/ollama.rs");
