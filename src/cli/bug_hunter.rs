//! Bug Hunter CLI Module
//!
//! Command-line interface for proactive bug hunting.
//!
//! Implements BH-01 to BH-15 from the Popperian Falsification Checklist.

use crate::ansi_colors::Colorize;
use crate::bug_hunter::{
    hunt, hunt_ensemble, hunt_with_spec, hunt_with_ticket, CrashBucketingMode, DefectCategory,
    Finding, FindingSeverity, HuntConfig, HuntMode, HuntResult, LocalizationStrategy, SbflFormula,
};
use clap::{Subcommand, ValueEnum};
use std::path::PathBuf;

#[path = "bug_hunter_output.rs"]
mod bug_hunter_output;
use bug_hunter_output::{
    handle_diff_command, handle_trend_command, handle_triage_command, output_result, output_sarif,
    output_stack_issue, output_stack_json, output_stack_text, CrateStats,
};

/// Bug Hunter subcommands.
#[derive(Subcommand, Debug)]
pub enum BugHunterCommand {
    /// LLM-augmented static analysis (LLIFT pattern)
    ///
    /// Runs clippy and pattern detection, optionally filtering with LLM.
    Analyze {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories to scan
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Enable LLM filtering of false positives
        #[arg(long)]
        llm_filter: bool,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Maximum findings to report
        #[arg(long, default_value = "50")]
        max_findings: usize,

        /// Enable PMAT quality-weighted suspiciousness (BH-21)
        #[arg(long)]
        pmat_quality: bool,

        /// Quality weight factor (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        quality_weight: f64,

        /// Use PMAT to scope targets by quality (BH-22)
        #[arg(long)]
        pmat_scope: bool,

        /// PMAT query string for scoping (BH-22)
        #[arg(long)]
        pmat_query: Option<String>,

        /// Coverage file (lcov format) for hotpath weighting
        #[arg(long)]
        coverage: Option<PathBuf>,

        /// Coverage weight factor for hotpath weighting (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        coverage_weight: f64,

        /// Path to provable-contracts directory (BH-26)
        #[arg(long)]
        contracts: Option<PathBuf>,

        /// Auto-discover provable-contracts in sibling directories (BH-26)
        #[arg(long)]
        contracts_auto: bool,

        /// Path to tiny-model-ground-truth directory (BH-27)
        #[arg(long)]
        model_parity: Option<PathBuf>,

        /// Auto-discover tiny-model-ground-truth in sibling directories (BH-27)
        #[arg(long)]
        model_parity_auto: bool,
    },

    /// SBFL without failing tests (SBEST pattern)
    ///
    /// Uses stack traces or coverage data for fault localization.
    Hunt {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Stack trace file to analyze
        #[arg(long)]
        stack_trace: Option<PathBuf>,

        /// Coverage file (lcov format)
        #[arg(long)]
        coverage: Option<PathBuf>,

        /// SBFL formula to use
        #[arg(long, value_enum, default_value = "ochiai")]
        formula: SbflFormulaArg,

        /// Fault localization strategy (BH-16 to BH-19)
        #[arg(long, value_enum, default_value = "sbfl")]
        strategy: LocalizationStrategyArg,

        /// Crash bucketing mode (BH-20)
        #[arg(long, value_enum, default_value = "none")]
        crash_bucket: CrashBucketArg,

        /// Number of top suspicious locations
        #[arg(long, default_value = "10")]
        top_n: usize,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Mutation-based invariant falsification (FDV pattern)
    ///
    /// Identifies mutation testing targets and weak test coverage.
    Falsify {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target files or directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Minimum mutation kill rate to pass
        #[arg(long, default_value = "80")]
        min_kill_rate: u8,

        /// Mutation timeout in seconds
        #[arg(long, default_value = "30")]
        timeout: u64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Targeted unsafe Rust fuzzing (FourFuzz pattern)
    ///
    /// Identifies unsafe blocks and generates fuzzing targets.
    Fuzz {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Focus on unsafe blocks only
        #[arg(long)]
        target_unsafe: bool,

        /// Fuzzing duration in seconds (for actual fuzzing)
        #[arg(long, default_value = "60")]
        duration: u64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Hybrid concolic + SBFL (COTTONTAIL pattern)
    ///
    /// Deep analysis of complex conditionals and path coverage.
    DeepHunt {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Coverage file (lcov format)
        #[arg(long)]
        coverage: Option<PathBuf>,

        /// Enable concolic execution analysis
        #[arg(long)]
        concolic: bool,

        /// Use SBFL ensemble
        #[arg(long)]
        sbfl_ensemble: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Run all modes and combine results (ensemble approach)
    Ensemble {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Target directories
        #[arg(long, default_value = "src")]
        target: Vec<String>,

        /// Minimum suspiciousness threshold
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Enable PMAT quality-weighted suspiciousness (BH-21)
        #[arg(long)]
        pmat_quality: bool,

        /// Quality weight factor (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        quality_weight: f64,

        /// PMAT query string for scoping (BH-22)
        #[arg(long)]
        pmat_query: Option<String>,

        /// Path to provable-contracts directory (BH-26)
        #[arg(long)]
        contracts: Option<PathBuf>,

        /// Auto-discover provable-contracts (BH-26)
        #[arg(long)]
        contracts_auto: bool,

        /// Path to tiny-model-ground-truth directory (BH-27)
        #[arg(long)]
        model_parity: Option<PathBuf>,

        /// Auto-discover tiny-model-ground-truth (BH-27)
        #[arg(long)]
        model_parity_auto: bool,
    },

    /// Spec-driven bug hunting (BH-11)
    ///
    /// Hunt bugs guided by a specification file.
    Spec {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Specification file path
        #[arg(long, short = 's')]
        spec: PathBuf,

        /// Filter to specific section (e.g., "Authentication")
        #[arg(long)]
        section: Option<String>,

        /// Update spec with findings (BH-14: Bidirectional Linking)
        #[arg(long)]
        update_spec: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Minimum suspiciousness threshold
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,

        /// Quick mode: skip clippy, only do pattern matching
        #[arg(long, short = 'q')]
        quick: bool,

        /// Enable PMAT quality-weighted suspiciousness (BH-21/BH-25)
        #[arg(long)]
        pmat_quality: bool,

        /// Quality weight factor (0.0-1.0, default 0.5)
        #[arg(long, default_value = "0.5")]
        quality_weight: f64,

        /// PMAT query string for scoping
        #[arg(long)]
        pmat_query: Option<String>,
    },

    /// Ticket-scoped bug hunting (BH-12)
    ///
    /// Hunt bugs focused on PMAT work ticket areas.
    Ticket {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// PMAT ticket reference (ID or file path)
        #[arg(long, short = 't')]
        ticket: String,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Minimum suspiciousness threshold
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,
    },

    /// Cross-stack bug analysis for Sovereign AI Stack
    ///
    /// Scans trueno, aprender, realizar, entrenar, repartir in parallel
    /// and generates a consolidated cross-stack report.
    Stack {
        /// Base directory containing stack crates (default: parent of cwd)
        #[arg(long)]
        base: Option<PathBuf>,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.7")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Crates to scan (default: trueno,aprender,realizar,entrenar,repartir)
        #[arg(long, value_delimiter = ',')]
        crates: Option<Vec<String>>,

        /// Generate GitHub issue body
        #[arg(long)]
        issue: bool,

        /// Path to provable-contracts directory (BH-26)
        #[arg(long)]
        contracts: Option<PathBuf>,

        /// Auto-discover provable-contracts (BH-26)
        #[arg(long)]
        contracts_auto: bool,

        /// Path to tiny-model-ground-truth directory (BH-27)
        #[arg(long)]
        model_parity: Option<PathBuf>,

        /// Auto-discover tiny-model-ground-truth (BH-27)
        #[arg(long)]
        model_parity_auto: bool,
    },

    /// Show only new findings compared to a baseline
    ///
    /// Compares current findings against a git branch or time period.
    Diff {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Base git branch/commit to compare against (e.g., "main", "HEAD~5")
        #[arg(long)]
        base: Option<String>,

        /// Time period to compare against (e.g., "7d", "2w", "1m")
        #[arg(long)]
        since: Option<String>,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.3")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,

        /// Save current findings as new baseline
        #[arg(long)]
        save_baseline: bool,
    },

    /// Show tech debt trends over time
    Trend {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Number of weeks to show
        #[arg(long, default_value = "12")]
        weeks: usize,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },

    /// Group related findings by root cause
    Triage {
        /// Project path to analyze
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Minimum suspiciousness threshold (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        min_suspiciousness: f64,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: BugHunterOutputFormat,
    },
}

/// Output format for bug hunter results.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BugHunterOutputFormat {
    /// Human-readable text
    Text,
    /// JSON output
    Json,
    /// SARIF format (Static Analysis Results Interchange Format)
    Sarif,
    /// Markdown table
    Markdown,
}

/// SBFL formula argument.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SbflFormulaArg {
    Tarantula,
    Ochiai,
    Dstar2,
    Dstar3,
}

impl From<SbflFormulaArg> for SbflFormula {
    fn from(arg: SbflFormulaArg) -> Self {
        match arg {
            SbflFormulaArg::Tarantula => SbflFormula::Tarantula,
            SbflFormulaArg::Ochiai => SbflFormula::Ochiai,
            SbflFormulaArg::Dstar2 => SbflFormula::DStar2,
            SbflFormulaArg::Dstar3 => SbflFormula::DStar3,
        }
    }
}

/// Localization strategy CLI argument (BH-16 to BH-19).
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum LocalizationStrategyArg {
    /// Spectrum-Based Fault Localization only
    #[default]
    Sbfl,
    /// Mutation-Based Fault Localization (BH-16)
    Mbfl,
    /// Causal inference (BH-17)
    Causal,
    /// Multi-channel combination (BH-19)
    MultiChannel,
    /// Hybrid SBFL + MBFL
    Hybrid,
}

impl From<LocalizationStrategyArg> for LocalizationStrategy {
    fn from(arg: LocalizationStrategyArg) -> Self {
        match arg {
            LocalizationStrategyArg::Sbfl => LocalizationStrategy::Sbfl,
            LocalizationStrategyArg::Mbfl => LocalizationStrategy::Mbfl,
            LocalizationStrategyArg::Causal => LocalizationStrategy::Causal,
            LocalizationStrategyArg::MultiChannel => LocalizationStrategy::MultiChannel,
            LocalizationStrategyArg::Hybrid => LocalizationStrategy::Hybrid,
        }
    }
}

/// Crash bucketing mode CLI argument (BH-20).
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum CrashBucketArg {
    /// No bucketing
    #[default]
    None,
    /// Stack trace similarity
    StackTrace,
    /// Semantic root cause analysis
    Semantic,
}

impl From<CrashBucketArg> for CrashBucketingMode {
    fn from(arg: CrashBucketArg) -> Self {
        match arg {
            CrashBucketArg::None => CrashBucketingMode::None,
            CrashBucketArg::StackTrace => CrashBucketingMode::StackTrace,
            CrashBucketArg::Semantic => CrashBucketingMode::Semantic,
        }
    }
}

/// Handle bug hunter command.
pub fn handle_bug_hunter_command(command: BugHunterCommand) -> Result<(), String> {
    match command {
        BugHunterCommand::Analyze {
            path,
            target,
            llm_filter,
            min_suspiciousness,
            format,
            max_findings,
            pmat_quality,
            quality_weight,
            pmat_scope,
            pmat_query,
            coverage,
            coverage_weight,
            contracts,
            contracts_auto,
            model_parity,
            model_parity_auto,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Analyze,
                targets: target.into_iter().map(PathBuf::from).collect(),
                min_suspiciousness,
                max_findings,
                llm_filter,
                use_pmat_quality: pmat_quality,
                quality_weight,
                pmat_scope,
                pmat_query,
                coverage_path: coverage,
                coverage_weight,
                contracts_path: contracts,
                contracts_auto,
                model_parity_path: model_parity,
                model_parity_auto,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Hunt {
            path,
            stack_trace: _,
            coverage,
            formula,
            strategy,
            crash_bucket,
            top_n,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::Hunt,
                targets: vec![PathBuf::from("src")],
                max_findings: top_n,
                sbfl_formula: formula.into(),
                coverage_path: coverage,
                localization_strategy: strategy.into(),
                crash_bucketing: crash_bucket.into(),
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Falsify { path, target, min_kill_rate: _, timeout, format } => {
            let config = HuntConfig {
                mode: HuntMode::Falsify,
                targets: target.into_iter().map(PathBuf::from).collect(),
                mutation_timeout_secs: timeout,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Fuzz { path, target, target_unsafe: _, duration, format } => {
            let config = HuntConfig {
                mode: HuntMode::Fuzz,
                targets: target.into_iter().map(PathBuf::from).collect(),
                fuzz_duration_secs: duration,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::DeepHunt {
            path,
            target,
            coverage,
            concolic: _,
            sbfl_ensemble: _,
            format,
        } => {
            let config = HuntConfig {
                mode: HuntMode::DeepHunt,
                targets: target.into_iter().map(PathBuf::from).collect(),
                coverage_path: coverage,
                ..Default::default()
            };
            let result = hunt(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Ensemble {
            path,
            target,
            min_suspiciousness,
            format,
            pmat_quality,
            quality_weight,
            pmat_query,
            contracts,
            contracts_auto,
            model_parity,
            model_parity_auto,
        } => {
            let config = HuntConfig {
                targets: target.into_iter().map(PathBuf::from).collect(),
                min_suspiciousness,
                use_pmat_quality: pmat_quality,
                quality_weight,
                pmat_query,
                contracts_path: contracts,
                contracts_auto,
                model_parity_path: model_parity,
                model_parity_auto,
                ..Default::default()
            };
            let result = hunt_ensemble(&path, config);
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Spec {
            path,
            spec,
            section,
            update_spec,
            format,
            min_suspiciousness,
            quick,
            pmat_quality,
            quality_weight,
            pmat_query,
        } => {
            let config = HuntConfig {
                min_suspiciousness,
                // Quick mode does pattern-only scan, no clippy/coverage
                mode: if quick { HuntMode::Quick } else { HuntMode::Analyze },
                use_pmat_quality: pmat_quality,
                quality_weight,
                pmat_query,
                ..Default::default()
            };
            let (result, mut parsed_spec) =
                hunt_with_spec(&path, &spec, section.as_deref(), config)?;

            output_result(&result, format);

            // Update spec file if requested (BH-14)
            if update_spec {
                let findings_by_claim: Vec<(String, Vec<_>)> =
                    parsed_spec.claims.iter().map(|c| (c.id.clone(), Vec::new())).collect();
                if let Ok(updated_content) = parsed_spec.update_with_findings(&findings_by_claim) {
                    if let Err(e) = parsed_spec.write_updated(&updated_content) {
                        eprintln!("Warning: Failed to update spec: {}", e);
                    } else {
                        println!("\nSpec updated: {}", spec.display());
                    }
                }
            }

            Ok(())
        }

        BugHunterCommand::Ticket { path, ticket, format, min_suspiciousness } => {
            let config = HuntConfig { min_suspiciousness, ..Default::default() };
            let result = hunt_with_ticket(&path, &ticket, config)?;
            output_result(&result, format);
            Ok(())
        }

        BugHunterCommand::Stack {
            base,
            min_suspiciousness,
            format,
            crates,
            issue,
            contracts,
            contracts_auto,
            model_parity,
            model_parity_auto,
        } => handle_stack_command(
            base,
            min_suspiciousness,
            format,
            crates,
            issue,
            HuntConfig {
                contracts_path: contracts,
                contracts_auto,
                model_parity_path: model_parity,
                model_parity_auto,
                ..Default::default()
            },
        ),

        BugHunterCommand::Diff { path, base, since, min_suspiciousness, format, save_baseline } => {
            handle_diff_command(path, base, since, min_suspiciousness, format, save_baseline)
        }

        BugHunterCommand::Trend { path, weeks, format } => {
            handle_trend_command(path, weeks, format)
        }

        BugHunterCommand::Triage { path, min_suspiciousness, format } => {
            handle_triage_command(path, min_suspiciousness, format)
        }
    }
}

/// Handle the cross-stack bug analysis command.
fn handle_stack_command(
    base: Option<PathBuf>,
    min_suspiciousness: f64,
    format: BugHunterOutputFormat,
    crates: Option<Vec<String>>,
    generate_issue: bool,
    hunt_overrides: HuntConfig,
) -> Result<(), String> {
    use std::collections::HashMap;

    // Default stack crates
    let default_crates = vec![
        "trueno".to_string(),
        "aprender".to_string(),
        "realizar".to_string(),
        "entrenar".to_string(),
        "repartir".to_string(),
    ];
    let crate_list = crates.unwrap_or(default_crates);

    // Determine base directory
    let base_dir = base.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from(".."))
    });

    eprintln!(
        "{}",
        format!("Scanning {} crates in {}...", crate_list.len(), base_dir.display()).dimmed()
    );

    // Scan each crate in parallel
    let results: Vec<(String, HuntResult)> = std::thread::scope(|s| {
        let handles: Vec<_> = crate_list
            .iter()
            .map(|crate_name| {
                let base = &base_dir;
                let overrides = &hunt_overrides;
                let min_susp = min_suspiciousness;
                s.spawn(move || {
                    let crate_path = base.join(crate_name);
                    if !crate_path.exists() {
                        eprintln!("  {} {} (not found)", "[SKIP]".yellow(), crate_name);
                        return None;
                    }
                    eprintln!("  {} {}...", "[SCAN]".dimmed(), crate_name);
                    let config = HuntConfig {
                        mode: HuntMode::Analyze,
                        min_suspiciousness: min_susp,
                        contracts_path: overrides.contracts_path.clone(),
                        contracts_auto: overrides.contracts_auto,
                        model_parity_path: overrides.model_parity_path.clone(),
                        model_parity_auto: overrides.model_parity_auto,
                        ..Default::default()
                    };
                    let result = hunt(&crate_path, config);
                    Some((crate_name.clone(), result))
                })
            })
            .collect();

        handles.into_iter().filter_map(|h| h.join().ok().flatten()).collect()
    });

    if results.is_empty() {
        return Err("No crates found to analyze".to_string());
    }

    // Aggregate stats
    let mut total_findings = 0usize;
    let mut total_critical = 0usize;
    let mut total_high = 0usize;
    let mut by_category: HashMap<DefectCategory, usize> = HashMap::new();
    let mut crate_stats: Vec<CrateStats> = Vec::new();

    for (crate_name, result) in &results {
        let stats = &result.stats;
        total_findings += stats.total_findings;

        let critical = stats.by_severity.get(&FindingSeverity::Critical).copied().unwrap_or(0);
        let high = stats.by_severity.get(&FindingSeverity::High).copied().unwrap_or(0);
        total_critical += critical;
        total_high += high;

        for (cat, count) in &stats.by_category {
            *by_category.entry(*cat).or_insert(0) += count;
        }

        crate_stats.push(CrateStats {
            name: crate_name.clone(),
            total: stats.total_findings,
            critical,
            high,
            gpu: stats.by_category.get(&DefectCategory::GpuKernelBugs).copied().unwrap_or(0),
            debt: stats.by_category.get(&DefectCategory::HiddenDebt).copied().unwrap_or(0),
            test: stats.by_category.get(&DefectCategory::TestDebt).copied().unwrap_or(0),
            silent: stats.by_category.get(&DefectCategory::SilentDegradation).copied().unwrap_or(0),
            memory: stats.by_category.get(&DefectCategory::MemorySafety).copied().unwrap_or(0),
            contract: stats.by_category.get(&DefectCategory::ContractGap).copied().unwrap_or(0),
            parity: stats.by_category.get(&DefectCategory::ModelParityGap).copied().unwrap_or(0),
        });
    }

    match format {
        BugHunterOutputFormat::Json => {
            output_stack_json(&crate_stats, &results);
        }
        BugHunterOutputFormat::Markdown | BugHunterOutputFormat::Text => {
            output_stack_text(
                &crate_stats,
                total_findings,
                total_critical,
                total_high,
                &by_category,
            );
            if generate_issue {
                println!();
                output_stack_issue(&crate_stats, &results);
            }
        }
        BugHunterOutputFormat::Sarif => {
            // Combine all findings into one SARIF output
            let mut all_findings: Vec<Finding> = Vec::new();
            for (_, result) in &results {
                all_findings.extend(result.findings.clone());
            }
            if let Some((_, first_result)) = results.first() {
                let mut combined = first_result.clone();
                combined.findings = all_findings;
                output_sarif(&combined);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbfl_formula_conversion() {
        assert_eq!(SbflFormula::from(SbflFormulaArg::Tarantula), SbflFormula::Tarantula);
        assert_eq!(SbflFormula::from(SbflFormulaArg::Ochiai), SbflFormula::Ochiai);
        assert_eq!(SbflFormula::from(SbflFormulaArg::Dstar2), SbflFormula::DStar2);
        assert_eq!(SbflFormula::from(SbflFormulaArg::Dstar3), SbflFormula::DStar3);
    }

    #[test]
    fn test_handle_analyze_command() {
        let cmd = BugHunterCommand::Analyze {
            path: PathBuf::from("."),
            target: vec!["src".to_string()],
            llm_filter: false,
            min_suspiciousness: 0.5,
            format: BugHunterOutputFormat::Json,
            max_findings: 10,
            pmat_quality: false,
            quality_weight: 0.5,
            pmat_scope: false,
            pmat_query: None,
            coverage: None,
            coverage_weight: 0.5,
            contracts: None,
            contracts_auto: false,
            model_parity: None,
            model_parity_auto: false,
        };
        // Should not panic
        let _ = handle_bug_hunter_command(cmd);
    }
}
