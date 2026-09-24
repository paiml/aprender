//! Command surface for the APR QA CLI.
//!
//! Holds the clap parser (`Cli`), the subcommand enum (`Commands`), and the
//! `dispatch` entry point. These live in the library rather than in the
//! `apr-qa` binary so any host binary — notably the single `apr` binary — can
//! parse into `Commands` and dispatch without shelling out to `apr-qa`.
//!
//! The `apr-qa` binary is now a shim over [`run`].

// Relocated from src/main.rs, which carried these as crate-level allows.
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]

use crate::{
    bootstrap_playbook_from_contract, build_certification_config_with_policy,
    build_execution_config, calculate_mqs_score, calculate_popperian_score, collect_evidence,
    execute_auto_tickets, execute_playbook, filter_models_by_size, generate_html_report,
    generate_junit_report, generate_lock_file, generate_model_scenarios,
    generate_tickets_from_evidence, list_all_models, load_playbook, parse_evidence,
    parse_failure_policy, playbook_path_for_model, scenarios_to_json, scenarios_to_yaml, CertTier,
    PlaybookRunConfig,
};
use aprender_qa_report::{MqsScore, PopperianScore};
use aprender_qa_runner::ToolExecutor;
use aprender_qa_runner::{Evidence, EvidenceCollector};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "apr-qa")]
#[command(about = "APR Model QA Playbook Runner", long_about = None)]
#[command(version)]
pub struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Commands,
}

// CertTier enum now comes from the crate root

#[derive(Subcommand, Clone, Debug)]
pub enum Commands {
    /// Certify models against the verification matrix
    Certify {
        /// Certify all models in registry
        #[arg(long)]
        all: bool,

        /// Certify by model family (e.g., "qwen-coder", "llama")
        #[arg(long)]
        family: Option<String>,

        /// Certification tier (dim-smoke, smoke, quick, standard, deep)
        #[arg(long, default_value = "quick")]
        tier: String,

        /// Kernel equivalence class (A-F) for batch dim-smoke certification
        #[arg(long)]
        kernel_class: Option<String>,

        /// Specific model IDs to certify
        #[arg(value_name = "MODEL")]
        models: Vec<String>,

        /// Output directory for certification artifacts
        #[arg(short, long, default_value = "certifications")]
        output: PathBuf,

        /// Dry run (show what would be certified without running)
        #[arg(long)]
        dry_run: bool,

        /// Model cache directory (contains GGUF/APR/SafeTensors files)
        /// Structure: <cache>/<model-name>/<format>/<file>
        #[arg(long)]
        model_cache: Option<PathBuf>,

        /// Path to apr binary for real inference
        #[arg(long, default_value = "apr")]
        apr_binary: String,

        /// Auto-generate structured tickets from failures (§3.6)
        #[arg(long)]
        auto_ticket: bool,

        /// Repository for auto-ticket creation (e.g., "paiml/aprender")
        #[arg(long, default_value = "paiml/aprender")]
        ticket_repo: String,

        /// Disable playbook integrity checks (§3.1)
        #[arg(long)]
        no_integrity_check: bool,

        /// Stop on first failure with enhanced diagnostics (§12.5.3)
        #[arg(long)]
        fail_fast: bool,

        /// Enhance failures with batuta oracle context (§12.1.1)
        /// Generates falsification checklists and enriched metrics
        #[arg(long)]
        oracle_enhance: bool,
    },

    /// Run a playbook
    Run {
        /// Path to playbook YAML file
        #[arg(value_name = "PLAYBOOK")]
        playbook: PathBuf,

        /// Output directory for reports
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Failure policy (stop-on-first, stop-on-p0, collect-all, fail-fast)
        #[arg(long, default_value = "stop-on-p0")]
        failure_policy: String,

        /// Stop on first failure with enhanced diagnostics (§12.5.3)
        /// Equivalent to --failure-policy fail-fast
        /// Emits comprehensive trace output for debugging and GitHub ticket creation
        #[arg(long)]
        fail_fast: bool,

        /// Dry run: execute playbook but skip evidence persistence and Jidoka non-zero exit.
        /// Useful for inspecting results without producing artifacts.
        #[arg(long)]
        dry_run: bool,

        /// Maximum parallel workers
        #[arg(long, default_value = "4")]
        workers: usize,

        /// Path to model file
        #[arg(long)]
        model_path: Option<String>,

        /// Timeout per test in milliseconds
        #[arg(long, default_value = "60000")]
        timeout: u64,

        /// Disable GPU acceleration (use CPU only)
        #[arg(long)]
        no_gpu: bool,

        /// Skip P0 format conversion tests (NOT RECOMMENDED - these are critical)
        #[arg(long)]
        skip_conversion_tests: bool,

        /// Run APR tool coverage tests (inspect, validate, bench, check, trace, profile)
        #[arg(long)]
        run_tool_tests: bool,

        /// Run profile CI assertions (throughput, latency thresholds)
        #[arg(long)]
        profile_ci: bool,

        /// Enable HF parity verification against golden corpus
        #[arg(long)]
        hf_parity: bool,

        /// Path to HF golden corpus directory
        #[arg(long, default_value = "../hf-ground-truth-corpus/oracle")]
        hf_corpus_path: String,

        /// HF parity model family (e.g., "qwen2.5-coder-1.5b/v1")
        #[arg(long)]
        hf_model_family: Option<String>,

        /// Disable playbook integrity checks (§3.1)
        #[arg(long)]
        no_integrity_check: bool,

        /// Metadata-only mode (dimensional checks, no inference)
        #[arg(long)]
        metadata_only: bool,
    },

    /// Run APR tool coverage tests
    Tools {
        /// Path to model file
        #[arg(value_name = "MODEL_PATH")]
        model_path: PathBuf,

        /// Disable GPU acceleration
        #[arg(long)]
        no_gpu: bool,

        /// Output directory for results
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Include serve lifecycle test (F-INTEG-003)
        #[arg(long)]
        include_serve: bool,
    },

    /// Generate scenarios for a model
    Generate {
        /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        #[arg(value_name = "MODEL")]
        model: String,

        /// Number of scenarios per combination
        #[arg(short, long, default_value = "100")]
        count: usize,

        /// Output format (yaml, json)
        #[arg(short, long, default_value = "yaml")]
        format: String,
    },

    /// Calculate MQS score from evidence
    Score {
        /// Path to evidence JSON file
        #[arg(value_name = "EVIDENCE")]
        evidence: PathBuf,

        /// Model ID for the score
        #[arg(short, long)]
        model: String,
    },

    /// Generate report from execution results
    Report {
        /// Path to evidence JSON file
        #[arg(value_name = "EVIDENCE")]
        evidence: PathBuf,

        /// Output directory
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Report formats to generate (html, junit, all)
        #[arg(long, default_value = "all")]
        formats: String,

        /// Model ID
        #[arg(short, long)]
        model: String,
    },

    /// List available models in registry
    List {
        /// Filter by size category (small, medium, large, xlarge)
        #[arg(short, long)]
        size: Option<String>,
    },

    /// Lock playbook hashes for integrity verification (§3.1)
    LockPlaybooks {
        /// Directory containing playbook YAML files
        #[arg(value_name = "DIR", default_value = "playbooks")]
        dir: PathBuf,

        /// Output lock file path
        #[arg(short, long, default_value = "playbooks/playbook.lock.yaml")]
        output: PathBuf,
    },

    /// Generate upstream tickets from failures
    Tickets {
        /// Path to evidence JSON file
        #[arg(value_name = "EVIDENCE")]
        evidence: PathBuf,

        /// Target repository (e.g., "paiml/aprender")
        #[arg(short, long, default_value = "paiml/aprender")]
        repo: String,

        /// Only generate tickets for black swan events
        #[arg(long)]
        black_swans_only: bool,

        /// Minimum occurrences before creating ticket
        #[arg(long, default_value = "1")]
        min_occurrences: usize,

        /// Ticket generation mode (F-TICKET-004)
        /// - create: Generate ticket files (default)
        /// - draft: Only print ticket content without creating files
        #[arg(long, default_value = "create")]
        ticket_mode: String,
    },

    /// Verify model output parity against HuggingFace golden corpus
    ///
    /// Implements Popperian falsification: any divergence beyond tolerance
    /// falsifies the hypothesis that the implementation is equivalent to HuggingFace.
    Parity {
        /// Model family (e.g., "qwen2.5-coder-1.5b")
        #[arg(short, long)]
        model_family: String,

        /// Path to golden corpus directory
        #[arg(short, long, default_value = "../hf-ground-truth-corpus/oracle")]
        corpus_path: PathBuf,

        /// SafeTensors file containing logits to verify
        #[arg(short, long)]
        logits_file: Option<PathBuf>,

        /// Prompt used to generate the logits
        #[arg(short, long)]
        prompt: Option<String>,

        /// Tolerance level (fp32, fp16, int8, int4)
        #[arg(short, long, default_value = "fp32")]
        tolerance: String,

        /// List available golden outputs for the model
        #[arg(long)]
        list: bool,

        /// Verify all golden outputs against themselves (sanity check)
        #[arg(long)]
        self_check: bool,
    },

    /// Export certification data to models.csv (PMAT-264)
    ///
    /// Scans evidence directory and updates models.csv with MQS scores,
    /// grades, and certification status for oracle consumption.
    ExportCsv {
        /// Directory containing evidence JSON files
        #[arg(short, long, default_value = "docs/certifications/evidence")]
        evidence_dir: PathBuf,

        /// Output CSV file path
        #[arg(short, long, default_value = "docs/certifications/models.csv")]
        output: PathBuf,

        /// Append to existing CSV (instead of overwrite)
        #[arg(long)]
        append: bool,
    },

    /// Export evidence to schema-compliant JSON (PMAT-265)
    ///
    /// Exports test run results to the standard evidence JSON format
    /// consumed by the oracle for certification lookup.
    ExportEvidence {
        /// Path to source evidence or execution result JSON
        #[arg(value_name = "SOURCE")]
        source: PathBuf,

        /// Output directory for evidence files
        #[arg(short, long, default_value = "docs/certifications/evidence")]
        output_dir: PathBuf,

        /// Model HF repo ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        #[arg(short, long)]
        model: String,

        /// Model family (e.g., "qwen2")
        #[arg(long)]
        family: String,

        /// Model size (e.g., "0.5b")
        #[arg(long)]
        size: String,

        /// Playbook name
        #[arg(long)]
        playbook_name: String,

        /// Certification tier (smoke, mvp, full)
        #[arg(long, default_value = "mvp")]
        tier: String,
    },

    /// Bootstrap an architecture-aware playbook from family contract
    ///
    /// Generates a playbook with architecture-specific prompts that stress-test
    /// the exact kernel operations each model family exercises (GQA, RoPE, etc.)
    Bootstrap {
        /// Model family name (e.g., "qwen2", "llama", "falcon")
        #[arg(value_name = "FAMILY")]
        family: String,

        /// Model size variant (e.g., "1.5b", "7b", "0.5b")
        #[arg(value_name = "SIZE")]
        size: String,

        /// HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        #[arg(long)]
        hf_repo: String,

        /// Certification tier
        #[arg(long, default_value = "mvp")]
        tier: String,

        /// Output path for generated playbook YAML
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Path to family contracts directory
        #[arg(long, default_value = "../aprender/contracts/model-families")]
        contracts_path: PathBuf,

        /// Dry run: print YAML to stdout instead of writing to file
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate model against tensor layout contract (Issue #4)
    ///
    /// Checks that an APR model file conforms to the tensor layout contract
    /// from aprender (tensor-layout-v1.yaml). This prevents GH-202 style bugs
    /// where wrong tensor shapes cause garbage output.
    ValidateContract {
        /// Path to APR model file to validate (not a HuggingFace model ID)
        #[arg(value_name = "MODEL_PATH")]
        model_path: PathBuf,

        /// Path to tensor layout contract YAML
        /// Defaults to ../aprender/contracts/tensor-layout-v1.yaml
        #[arg(long)]
        contract_path: Option<PathBuf>,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Only check critical tensors (lm_head, etc.)
        #[arg(long)]
        critical_only: bool,
    },

    /// Verify kernel coverage across HuggingFace architectures.
    ///
    /// Discovers which kernel operations each model architecture requires,
    /// checks implementation status in the sovereign stack (trueno/realizar),
    /// and generates upstream tickets for gaps. (Spec §20)
    KernelCoverage {
        /// Check specific architecture (e.g., "qwen2", "llama", "phi")
        #[arg(long)]
        architecture: Option<String>,

        /// Check all known architectures
        #[arg(long)]
        all: bool,

        /// Check all models in the registry (100+ models)
        #[arg(long)]
        models: bool,

        /// Verify binding claims against actual source code in sibling repos
        #[arg(long)]
        verify: bool,

        /// Generate upstream ticket markdown for gaps
        #[arg(long)]
        file_tickets: bool,

        /// Directory for ticket files
        #[arg(long, default_value = "docs/tickets")]
        output_dir: PathBuf,

        /// Path to trueno repo (default: ../trueno)
        #[arg(long, default_value = "../trueno")]
        trueno_path: PathBuf,

        /// Path to realizar repo (default: ../realizar)
        #[arg(long, default_value = "../realizar")]
        realizar_path: PathBuf,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Path to provable-contracts/contracts directory (arch-constraints YAML)
        #[arg(long, default_value = "../provable-contracts/contracts")]
        contracts_path: PathBuf,

        /// Path to kernel bindings YAML
        #[arg(long, default_value = "playbooks/kernel-bindings.yaml")]
        bindings_path: PathBuf,
    },
}

/// Parse the process arguments and run the selected subcommand.
///
/// Installs the Jidoka SIGINT handler first so child processes are reaped on
/// Ctrl-C, then hands the parsed subcommand to [`dispatch`].
///
/// # Panics
///
/// Does not panic. Argument errors are reported by clap, which exits the
/// process; subcommand failures exit non-zero via [`dispatch`].
pub fn run() {
    setup_signal_handler();

    let cli = Cli::parse();

    dispatch(cli.command);
}

/// Run one already-parsed subcommand.
///
/// Taken by value because every arm destructures the variant and forwards the
/// owned fields to its handler, exactly as `fn main` did.
///
/// # Panics
///
/// Does not panic. Handlers report failures on stderr and terminate the
/// process with a non-zero status via `std::process::exit`.
#[allow(clippy::too_many_lines)]
pub fn dispatch(command: Commands) {
    match command {
        Commands::Certify {
            all,
            family,
            tier,
            kernel_class,
            models,
            output,
            dry_run,
            model_cache,
            apr_binary,
            auto_ticket,
            ticket_repo,
            no_integrity_check,
            fail_fast,
            oracle_enhance,
        } => {
            run_certification(
                all,
                family,
                &tier,
                kernel_class,
                &models,
                &output,
                dry_run,
                model_cache,
                &apr_binary,
                auto_ticket,
                &ticket_repo,
                no_integrity_check,
                fail_fast,
                oracle_enhance,
            );
        }
        Commands::Run {
            playbook,
            output,
            failure_policy,
            fail_fast,
            dry_run,
            workers,
            model_path,
            timeout,
            no_gpu,
            skip_conversion_tests,
            run_tool_tests,
            profile_ci,
            hf_parity,
            hf_corpus_path,
            hf_model_family,
            no_integrity_check,
            metadata_only,
        } => {
            // --fail-fast flag overrides --failure-policy
            let effective_policy = if fail_fast {
                "fail-fast".to_string()
            } else {
                failure_policy
            };
            run_playbook(
                &playbook,
                &output,
                &effective_policy,
                dry_run,
                workers,
                model_path,
                timeout,
                no_gpu,
                skip_conversion_tests,
                run_tool_tests,
                profile_ci,
                hf_parity,
                &hf_corpus_path,
                hf_model_family,
                no_integrity_check,
                metadata_only,
            );
        }
        Commands::Tools {
            model_path,
            no_gpu,
            output,
            include_serve,
        } => {
            run_tool_tests(&model_path, no_gpu, &output, include_serve);
        }
        Commands::Generate {
            model,
            count,
            format,
        } => {
            generate_scenarios(&model, count, &format);
        }
        Commands::Score { evidence, model } => {
            calculate_score(&evidence, &model);
        }
        Commands::Report {
            evidence,
            output,
            formats,
            model,
        } => {
            generate_report(&evidence, &output, &formats, &model);
        }
        Commands::List { size } => {
            list_models(size.as_deref());
        }
        Commands::LockPlaybooks { dir, output } => match generate_lock_file(&dir, &output) {
            Ok(0) => {
                eprintln!("Error: No playbook files found in {}", dir.display());
                std::process::exit(1);
            }
            Ok(count) => println!("Locked {count} playbook(s) → {}", output.display()),
            Err(e) => {
                eprintln!("Error generating lock file: {e}");
                std::process::exit(1);
            }
        },
        Commands::Tickets {
            evidence,
            repo,
            black_swans_only,
            min_occurrences,
            ticket_mode,
        } => {
            generate_tickets(
                &evidence,
                &repo,
                black_swans_only,
                min_occurrences,
                &ticket_mode,
            );
        }
        Commands::Parity {
            model_family,
            corpus_path,
            logits_file,
            prompt,
            tolerance,
            list,
            self_check,
        } => {
            run_parity_check(
                &model_family,
                &corpus_path,
                logits_file.as_deref(),
                prompt.as_deref(),
                &tolerance,
                list,
                self_check,
            );
        }
        Commands::ExportCsv {
            evidence_dir,
            output,
            append,
        } => {
            export_csv(&evidence_dir, &output, append);
        }
        Commands::ExportEvidence {
            source,
            output_dir,
            model,
            family,
            size,
            playbook_name,
            tier,
        } => {
            export_evidence(
                &source,
                &output_dir,
                &model,
                &family,
                &size,
                &playbook_name,
                &tier,
            );
        }
        Commands::Bootstrap {
            family,
            size,
            hf_repo,
            tier,
            output,
            contracts_path,
            dry_run,
        } => {
            run_bootstrap(
                &family,
                &size,
                &hf_repo,
                &tier,
                output.as_deref(),
                &contracts_path,
                dry_run,
            );
        }
        Commands::ValidateContract {
            model_path,
            contract_path,
            format,
            critical_only,
        } => {
            validate_contract_command(
                &model_path,
                contract_path.as_deref(),
                &format,
                critical_only,
            );
        }
        Commands::KernelCoverage {
            architecture,
            all,
            models,
            verify,
            file_tickets,
            output_dir,
            trueno_path,
            realizar_path,
            format,
            contracts_path,
            bindings_path,
        } => {
            kernel_coverage_command(
                architecture.as_deref(),
                all,
                models,
                verify,
                file_tickets,
                &output_dir,
                &trueno_path,
                &realizar_path,
                &format,
                &contracts_path,
                &bindings_path,
            );
        }
    }
}

include!("configuration.rs");
include!("main_display_results.rs");
include!("main_tickets_and_parity.rs");
