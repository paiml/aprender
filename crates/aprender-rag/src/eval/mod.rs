//! Evaluation framework for RAG retrieval quality (PMAT-015)
//!
//! World-class RAG evaluation using LLM-as-judge on actual chunk content
//! and synthetic ground truth generated from the corpus itself.
//!
//! # Architecture
//!
//! Split pipeline — trueno-rag handles data, Claude Code handles LLM work:
//! - `eval sample` — Sample chunks from index (no API needed)
//! - `eval retrieve` — Run queries against index (no API needed)
//! - `eval metrics` — Compute IR metrics from judgments (no API needed)
//! - Claude Code `/eval-generate` skill — Generate questions from sampled chunks
//! - Claude Code `/eval-judge` skill — Judge relevance of retrieved chunks
//!
//! Optional direct API mode (requires `ANTHROPIC_API_KEY`):
//! - `eval generate` — Sample + generate questions via Claude API
//! - `eval judge` — Judge + compute metrics via Claude API

pub mod client;
pub mod domain;
pub mod generate;
pub mod judge;
pub mod metrics;
pub mod types;

pub use client::AnthropicClient;
pub use domain::classify_domain;
pub use generate::GroundTruthGenerator;
pub use judge::RelevanceJudge;
pub use metrics::compute_metrics_from_judgments;
pub use types::{
    EvalConfig, GroundTruthEntry, JudgeCache, JudgeCacheEntry, JudgeVerdict, JudgmentEntry,
    RetrievalResultEntry,
};
