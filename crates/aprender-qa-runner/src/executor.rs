//! Playbook executor
//!
//! Executes playbooks with parallel execution and failure handling.

#![allow(clippy::cast_possible_truncation)]

use crate::command::{CommandRunner, RealCommandRunner};
use crate::conversion::{resolve_model_path, ConversionConfig, ConversionExecutor};
use crate::diagnostics::FailFastReporter;
use crate::error::Result;
use crate::evidence::{Evidence, EvidenceCollector, Outcome, PerformanceMetrics};
use crate::integrity;
use crate::layout_contract::{load_contract_from, validate_model, DEFAULT_CONTRACT_PATH};
use crate::playbook::{OllamaParityConfig, Playbook};
use apr_qa_gen::{
    Backend, Format, HfParityOracle, Modality, ModelId, Oracle, QaScenario, Tolerance,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Parse timing in milliseconds from command output (e.g., "Completed in 1.5s" -> 1500.0)
fn parse_timing_ms(output: &str) -> Option<f64> {
    // Match "Completed in X.Xs" or "X.Xs" pattern
    for line in output.lines() {
        let lower = line.to_lowercase();
        if let Some(pos) = lower.find("completed in ") {
            let after = &lower[pos + 13..];
            if let Some(s_pos) = after.find('s') {
                if let Ok(secs) = after[..s_pos].trim().parse::<f64>() {
                    return Some(secs * 1000.0);
                }
            }
        }
    }
    None
}

/// Parse throughput in tok/s from JSON output (e.g., `"throughput_tps":25.0`)
fn parse_throughput(output: &str) -> Option<f64> {
    // Match "throughput_tps":N.N in JSON
    if let Some(pos) = output.find("\"throughput_tps\":") {
        let after = &output[pos + 17..];
        let end = after.find(|c: char| !c.is_ascii_digit() && c != '.')?;
        after[..end].parse::<f64>().ok()
    } else {
        None
    }
}

/// Failure handling policy (Jidoka)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailurePolicy {
    /// Stop entire pipeline on any failure
    StopOnFirst,
    /// Stop on P0 failures, continue on P1/P2
    #[default]
    StopOnP0,
    /// Collect all failures, report at end
    CollectAll,
    /// Stop on first failure with enhanced tracing (§12.5.3)
    /// Designed for debugging and GitHub ticket creation.
    /// Equivalent to StopOnFirst but signals tracing infrastructure
    /// to emit comprehensive diagnostics.
    FailFast,
}

impl FailurePolicy {
    /// Returns true if this policy should emit enhanced tracing on failure.
    #[must_use]
    pub fn emit_diagnostic(&self) -> bool {
        matches!(self, Self::FailFast)
    }

    /// Returns true if execution should stop on any failure.
    #[must_use]
    pub fn stops_on_any_failure(&self) -> bool {
        matches!(self, Self::StopOnFirst | Self::FailFast)
    }
}

/// Execution configuration
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExecutionConfig {
    /// Failure handling policy
    pub failure_policy: FailurePolicy,
    /// Default timeout in milliseconds
    pub default_timeout_ms: u64,
    /// Maximum parallel workers
    pub max_workers: usize,
    /// Dry run (don't actually execute commands)
    pub dry_run: bool,
    /// Path to the model file
    pub model_path: Option<String>,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Run P0 format conversion tests (CRITICAL - should be true by default)
    pub run_conversion_tests: bool,
    /// Run profile CI assertions
    pub run_profile_ci: bool,
    /// Run Golden Rule Test (convert → inference → diff)
    /// This is the single most important invariant: converted models
    /// MUST produce the same output as the original. (Five Whys: GH-190)
    pub run_golden_rule_test: bool,
    /// Path to golden reference JSON for the model
    pub golden_reference_path: Option<String>,
    /// Path to playbook lock file for integrity checks (§3.1)
    pub lock_file_path: Option<String>,
    /// Path to the playbook YAML file (for integrity hash verification)
    pub playbook_file_path: Option<String>,
    /// Check playbook integrity against lock file (§3.1)
    pub check_integrity: bool,
    /// Warn about implicit format/backend skips (§3.3)
    pub warn_implicit_skips: bool,
    /// Run HF parity verification against golden corpus
    pub run_hf_parity: bool,
    /// Path to HF golden corpus directory (e.g., "../hf-ground-truth-corpus/oracle")
    pub hf_parity_corpus_path: Option<String>,
    /// HF parity model family (e.g., "qwen2.5-coder-1.5b/v1")
    pub hf_parity_model_family: Option<String>,
    /// Output directory for conversion test artifacts (ISO-OUT-001)
    /// Defaults to "output/" - keeps test artifacts isolated from source models
    pub output_dir: Option<String>,
    /// Run contract invariant tests I-2 through I-5 (GH-190/191 Five-Whys)
    pub run_contract_tests: bool,
    /// Run ollama parity tests (GH-6/AC-2)
    pub run_ollama_parity: bool,
    /// Metadata-only mode: skip inference, only verify config.json + SafeTensors headers
    /// Used by dim-smoke tier for rapid model qualification.
    pub metadata_only: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            failure_policy: FailurePolicy::default(),
            default_timeout_ms: 60_000,
            max_workers: 4,
            dry_run: false,
            model_path: None,
            no_gpu: false,
            run_conversion_tests: true, // P0 CRITICAL: Always run by default
            run_profile_ci: false,      // Only enable for CI pipelines
            run_golden_rule_test: true, // v1.3.1: Golden Rule (Five Whys GH-190)
            golden_reference_path: None,
            lock_file_path: None,
            playbook_file_path: None,
            check_integrity: false,
            warn_implicit_skips: false,
            run_hf_parity: false,
            hf_parity_corpus_path: None,
            hf_parity_model_family: None,
            output_dir: Some("output".to_string()), // ISO-OUT-001: Default to isolated output
            run_contract_tests: true, // v1.4.0: Contract invariants (GH-190/191 Five-Whys)
            run_ollama_parity: false, // GH-6/AC-2: Opt-in, requires ollama binary
            metadata_only: false,
        }
    }
}

/// Executor for running playbooks
pub struct Executor {
    config: ExecutionConfig,
    collector: EvidenceCollector,
    command_runner: Arc<dyn CommandRunner>,
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor")
            .field("config", &self.config)
            .field("collector", &self.collector)
            .field("command_runner", &"<dyn CommandRunner>")
            .finish()
    }
}

include!("executor_lifecycle.rs");
include!("scenario.rs");
