//! APR QA CLI Library
//!
//! Library functions for the APR QA CLI tool.

#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
// Allow common patterns in test code
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]

pub mod cli;

use aprender_qa_gen::models::ModelMetadata;
use aprender_qa_gen::{ModelId, ModelRegistry, ScenarioGenerator};
use aprender_qa_report::{
    html::HtmlDashboard,
    junit::JunitReport,
    mqs::MqsCalculator,
    popperian::PopperianCalculator,
    ticket::{generate_structured_tickets, TicketGenerator, UpstreamTicket},
};
use aprender_qa_runner::{
    Evidence, EvidenceCollector, ExecutionConfig, ExecutionResult, Executor, FailurePolicy,
    Playbook, ToolExecutor,
};
use std::path::Path;

/// Result of a CLI operation
#[derive(Debug)]
pub enum CliResult {
    /// Operation succeeded
    Success(String),
    /// Operation failed with error
    Error(String),
}

impl CliResult {
    /// Returns true if the result is a success
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    /// Returns the message
    pub fn message(&self) -> &str {
        match self {
            Self::Success(msg) | Self::Error(msg) => msg,
        }
    }
}

/// Configuration for running a playbook
#[derive(Debug, Clone)]
pub struct PlaybookRunConfig {
    /// Failure policy (stop-on-first, stop-on-p0, collect-all)
    pub failure_policy: String,
    /// Dry run mode - don't execute, just show what would be done
    pub dry_run: bool,
    /// Maximum parallel workers
    pub workers: usize,
    /// Path to model file
    pub model_path: Option<String>,
    /// Timeout per test in milliseconds
    pub timeout: u64,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Skip P0 format conversion tests
    pub skip_conversion_tests: bool,
    /// Run APR tool coverage tests
    pub run_tool_tests: bool,
    /// Run profile CI assertions (throughput, latency)
    pub run_profile_ci: bool,
    /// Run HF parity verification against golden corpus
    pub run_hf_parity: bool,
    /// Path to HF golden corpus directory
    pub hf_parity_corpus_path: Option<String>,
    /// HF parity model family (e.g., "qwen2.5-coder-1.5b/v1")
    pub hf_parity_model_family: Option<String>,
    /// Metadata-only mode (dimensional checks only, no inference)
    pub metadata_only: bool,
}

impl Default for PlaybookRunConfig {
    fn default() -> Self {
        Self {
            failure_policy: "stop-on-p0".to_string(),
            dry_run: false,
            workers: 4,
            model_path: None,
            timeout: 60000,
            no_gpu: false,
            skip_conversion_tests: false,
            run_tool_tests: false,
            run_profile_ci: false,
            run_hf_parity: false,
            hf_parity_corpus_path: None,
            hf_parity_model_family: None,
            metadata_only: false,
        }
    }
}

/// Parse failure policy string to enum
pub fn parse_failure_policy(policy: &str) -> Result<FailurePolicy, String> {
    match policy {
        "stop-on-first" => Ok(FailurePolicy::StopOnFirst),
        "stop-on-p0" => Ok(FailurePolicy::StopOnP0),
        "collect-all" => Ok(FailurePolicy::CollectAll),
        "fail-fast" => Ok(FailurePolicy::FailFast),
        _ => Err(format!("Unknown failure policy: {policy}")),
    }
}

/// Load a playbook from a file path
pub fn load_playbook(path: &Path) -> Result<Playbook, String> {
    Playbook::from_file(path).map_err(|e| format!("Error loading playbook: {e}"))
}

/// Run tool tests and return results
pub fn execute_tool_tests(
    model_path: &str,
    no_gpu: bool,
    timeout: u64,
    include_serve: bool,
) -> Vec<aprender_qa_runner::ToolTestResult> {
    let executor = ToolExecutor::new(model_path.to_string(), no_gpu, timeout);
    executor.execute_all_with_serve(include_serve)
}

/// Generate scenarios for a model
pub fn generate_model_scenarios(model_id: &str, count: usize) -> Vec<aprender_qa_gen::QaScenario> {
    let parts: Vec<&str> = model_id.split('/').collect();
    let (org, name) = if parts.len() >= 2 {
        (parts[0], parts[1])
    } else {
        ("unknown", model_id)
    };

    let model = ModelId::new(org, name);
    let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(count);
    generator.generate()
}

/// Format scenarios as YAML
pub fn scenarios_to_yaml(scenarios: &[aprender_qa_gen::QaScenario]) -> Result<String, String> {
    let mut output = String::new();
    for scenario in scenarios {
        match serde_yaml::to_string(scenario) {
            Ok(yaml) => {
                output.push_str("---\n");
                output.push_str(&yaml);
            }
            Err(e) => return Err(format!("Error serializing scenario: {e}")),
        }
    }
    Ok(output)
}

/// Format scenarios as JSON
pub fn scenarios_to_json(scenarios: &[aprender_qa_gen::QaScenario]) -> Result<String, String> {
    serde_json::to_string_pretty(scenarios).map_err(|e| format!("Error serializing scenarios: {e}"))
}

/// Parse evidence from JSON string
pub fn parse_evidence(json: &str) -> Result<Vec<Evidence>, String> {
    serde_json::from_str(json).map_err(|e| format!("Error parsing evidence JSON: {e}"))
}

/// Create an evidence collector from evidence list
pub fn collect_evidence(evidence: Vec<Evidence>) -> EvidenceCollector {
    let mut collector = EvidenceCollector::new();
    for e in evidence {
        collector.add(e);
    }
    collector
}

/// Calculate MQS score from evidence
pub fn calculate_mqs_score(
    model_id: &str,
    collector: &EvidenceCollector,
) -> Result<aprender_qa_report::mqs::MqsScore, String> {
    let calculator = MqsCalculator::new();
    calculator
        .calculate(model_id, collector)
        .map_err(|e| format!("Error calculating MQS: {e}"))
}

/// Calculate Popperian score from evidence
pub fn calculate_popperian_score(
    model_id: &str,
    collector: &EvidenceCollector,
) -> aprender_qa_report::popperian::PopperianScore {
    let calculator = PopperianCalculator::new();
    calculator.calculate(model_id, collector)
}

/// Generate HTML report
pub fn generate_html_report(
    title: &str,
    mqs_score: &aprender_qa_report::mqs::MqsScore,
    popperian_score: &aprender_qa_report::popperian::PopperianScore,
    collector: &EvidenceCollector,
) -> Result<String, String> {
    let dashboard = HtmlDashboard::new(title.to_string());
    dashboard
        .generate(mqs_score, popperian_score, collector)
        .map_err(|e| format!("Error generating HTML: {e}"))
}

/// Generate JUnit XML report
pub fn generate_junit_report(
    model_id: &str,
    collector: &EvidenceCollector,
    mqs_score: &aprender_qa_report::mqs::MqsScore,
) -> Result<String, String> {
    let junit = JunitReport::new(model_id);
    junit
        .generate(collector, mqs_score)
        .map_err(|e| format!("Error generating JUnit: {e}"))
}

/// List all models from registry
pub fn list_all_models() -> Vec<ModelMetadata> {
    let registry = ModelRegistry::with_defaults();
    registry.all().into_iter().cloned().collect()
}

/// Filter models by size
pub fn filter_models_by_size(models: &[ModelMetadata], size_filter: &str) -> Vec<ModelMetadata> {
    models
        .iter()
        .filter(|model| {
            let size_str = format!("{:?}", model.size).to_lowercase();
            size_str.contains(&size_filter.to_lowercase())
        })
        .cloned()
        .collect()
}

/// Generate tickets from evidence
pub fn generate_tickets_from_evidence(
    evidence: &[Evidence],
    repo: &str,
    black_swans_only: bool,
    min_occurrences: usize,
) -> Vec<UpstreamTicket> {
    let mut generator = TicketGenerator::new(repo).with_min_occurrences(min_occurrences);

    if black_swans_only {
        generator = generator.black_swans_only();
    }

    generator.generate_from_evidence(evidence)
}

/// Format ticket for display
pub fn format_ticket_for_display(ticket: &UpstreamTicket, repo: &str) -> String {
    format!(
        "--- {} ---\nPriority: {}\nCategory: {}\nLabels: {}\n\ngh command:\n  {}\n",
        ticket.title,
        ticket.priority,
        ticket.category,
        ticket.labels.join(", "),
        ticket.to_gh_command(repo)
    )
}

/// Build execution config from run config
pub fn build_execution_config(config: &PlaybookRunConfig) -> Result<ExecutionConfig, String> {
    let policy = parse_failure_policy(&config.failure_policy)?;

    Ok(ExecutionConfig {
        failure_policy: policy,
        dry_run: config.dry_run,
        max_workers: config.workers,
        model_path: config.model_path.clone(),
        default_timeout_ms: config.timeout,
        no_gpu: config.no_gpu,
        run_conversion_tests: !config.skip_conversion_tests,
        run_profile_ci: config.run_profile_ci,
        run_golden_rule_test: true,
        golden_reference_path: None,
        lock_file_path: None,
        playbook_file_path: None,
        check_integrity: false,
        warn_implicit_skips: false,
        run_hf_parity: config.run_hf_parity,
        hf_parity_corpus_path: config.hf_parity_corpus_path.clone(),
        hf_parity_model_family: config.hf_parity_model_family.clone(),
        output_dir: Some("output".to_string()), // ISO-OUT-001: Isolated output directory
        run_contract_tests: !config.metadata_only,
        run_ollama_parity: false,
        metadata_only: config.metadata_only,
    })
}

/// Execute a playbook with the given configuration
pub fn execute_playbook(
    playbook: &Playbook,
    config: ExecutionConfig,
) -> Result<ExecutionResult, String> {
    let mut executor = Executor::with_config(config);
    executor
        .execute(playbook)
        .map_err(|e| format!("Execution failed: {e}"))
}

/// Certification tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CertTier {
    /// Dimensional smoke: dimension-only verification via kernel equivalence
    DimensionalSmoke,
    /// Tier 1: Smoke test
    Smoke,
    /// Tier 2: MVP - all formats/backends/modalities
    Mvp,
    /// Tier 3: Quick check
    #[default]
    Quick,
    /// Tier 4: Standard certification
    Standard,
    /// Tier 5: Deep certification
    Deep,
}

impl std::str::FromStr for CertTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dim-smoke" | "dimensional-smoke" => Ok(Self::DimensionalSmoke),
            "smoke" => Ok(Self::Smoke),
            "mvp" => Ok(Self::Mvp),
            "quick" => Ok(Self::Quick),
            "standard" => Ok(Self::Standard),
            "deep" => Ok(Self::Deep),
            _ => Err(format!(
                "Unknown tier: {s}. Use: dim-smoke, smoke, mvp, quick, standard, deep"
            )),
        }
    }
}

impl CertTier {
    /// Get the playbook suffix for this tier
    #[must_use]
    pub const fn playbook_suffix(self) -> &'static str {
        match self {
            Self::DimensionalSmoke => "-dim-smoke",
            Self::Smoke => "-smoke",
            Self::Mvp => "-mvp",
            Self::Quick => "-quick",
            Self::Standard | Self::Deep => "",
        }
    }
}

/// Configuration for certification runs
#[derive(Debug, Clone)]
pub struct CertificationConfig {
    /// Certification tier
    pub tier: CertTier,
    /// Model cache directory (contains gguf/apr/safetensors subdirs)
    pub model_cache: Option<std::path::PathBuf>,
    /// Path to apr binary
    pub apr_binary: String,
    /// Output directory for artifacts
    pub output_dir: std::path::PathBuf,
    /// Dry run mode
    pub dry_run: bool,
}

impl Default for CertificationConfig {
    fn default() -> Self {
        Self {
            tier: CertTier::Quick,
            model_cache: None,
            apr_binary: "apr".to_string(),
            output_dir: std::path::PathBuf::from("certifications"),
            dry_run: false,
        }
    }
}

/// Result of certifying a single model
#[derive(Debug, Clone)]
pub struct ModelCertificationResult {
    /// Model ID
    pub model_id: String,
    /// Whether certification succeeded
    pub success: bool,
    /// MQS score (0-1000)
    pub mqs_score: u32,
    /// Grade (A, B, C, D, F)
    pub grade: String,
    /// Pass rate as percentage
    pub pass_rate: f64,
    /// Gateway failures (if any)
    pub gateway_failed: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Build an ExecutionConfig for certification
///
/// This is the canonical way to build an ExecutionConfig for certification.
pub fn build_certification_config(
    tier: CertTier,
    model_cache_path: Option<String>,
) -> ExecutionConfig {
    build_certification_config_with_policy(tier, model_cache_path, false)
}

/// Build an ExecutionConfig for certification with fail-fast option
pub fn build_certification_config_with_policy(
    tier: CertTier,
    model_cache_path: Option<String>,
    fail_fast: bool,
) -> ExecutionConfig {
    if matches!(tier, CertTier::DimensionalSmoke) {
        return build_dimensional_smoke_config(model_cache_path);
    }

    let failure_policy = if fail_fast {
        FailurePolicy::FailFast
    } else {
        FailurePolicy::CollectAll
    };
    ExecutionConfig {
        failure_policy,
        dry_run: false,
        max_workers: 4,
        model_path: model_cache_path,
        default_timeout_ms: 60000,
        no_gpu: false,
        run_conversion_tests: true,
        run_profile_ci: matches!(tier, CertTier::Mvp | CertTier::Standard | CertTier::Deep),
        run_golden_rule_test: true,
        golden_reference_path: None,
        lock_file_path: None,
        playbook_file_path: None,
        check_integrity: false,
        warn_implicit_skips: false,
        run_hf_parity: false,
        hf_parity_corpus_path: None,
        hf_parity_model_family: None,
        output_dir: Some("output".to_string()), // ISO-OUT-001: Isolated output directory
        run_contract_tests: true,
        run_ollama_parity: false,
        metadata_only: false,
    }
}

// Certification functions: dim-smoke config, playbook paths, bootstrap,
// certify, lock, auto-tickets
include!("lib_certification.rs");

#[cfg(test)]
#[path = "lib_tests_a.rs"]
mod tests_a;

#[cfg(test)]
#[path = "lib_tests_b.rs"]
mod tests_b;

#[cfg(test)]
#[path = "lib_tests_c.rs"]
mod tests_c;

#[cfg(test)]
#[path = "lib_tests_d.rs"]
mod tests_d;

#[cfg(test)]
#[path = "lib_tests_e.rs"]
mod tests_e;
