//! `pv` -- provable-contracts CLI.
//!
//! The command surface lives here rather than in `main.rs` so that something
//! other than the `pv` binary can reach it: `apr pv <cmd>` and `pv <cmd>` now
//! call the same [`dispatch`].

pub mod cli;
pub mod commands;
pub mod contract_walk;
pub mod json_obj;
pub mod query_args;

// `PathBuf` is imported rather than fully qualified because the `#[cfg(test)]`
// include modules below open with `use super::*` and take it from here — as
// they did from `main.rs` before the command surface moved into this library.
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use cli::Commands;

/// Glance form, printed by `pv -V`. One line, and it names the tool.
///
/// clap renders `{name} {version}`, so this yields
/// `pv 0.63.0 (aprender provable-contracts verifier)`. The bare semver stays the
/// SECOND whitespace field because `scripts/pv_bin.sh` reads it positionally to
/// prove a resolved binary was built from HEAD.
const SHORT_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (aprender provable-contracts verifier)"
);

/// Full form, printed by `pv --version`.
///
/// Four things claim the name `pv` on a developer box: `pv(1)` the pipe viewer
/// from every distro, the `pv` crate on crates.io (also a pipe viewer, first
/// published 2019), this binary, and — until #2553 — the aprender facade. The
/// operator settled that this tool KEEPS the name (2026-08-21), which makes this
/// string the mitigation the project relies on, so it rules the others out by
/// name rather than merely describing itself. See #2559 and
/// `tests/version_identity.rs`.
const LONG_VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (aprender provable-contracts verifier)\n",
    "crate aprender-contracts-cli — ",
    env!("CARGO_PKG_REPOSITORY"),
    "\n",
    "Verifies YAML contracts under contracts/; run `pv --help` for the command surface.\n",
    "This is NOT pv(1), the pipe viewer (distro package `pv`, or the `pv` crate on crates.io)."
);

/// Top-level CLI argument parser for the `pv` command
#[derive(Parser)]
#[command(
    name = "pv",
    about = "provable-contracts — papers to provable Rust kernels",
    version = SHORT_VERSION,
    long_version = LONG_VERSION
)]
pub struct Cli {
    /// The command to run
    #[command(subcommand)]
    pub command: Commands,

    /// Suppress non-essential output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

/// Dispatch a parsed CLI subcommand to its handler.
///
/// Public so `apr pv` executes the identical code path rather than a copy of
/// it. The `pv` binary keeps shipping under its own name -- that is a decided
/// design, not an oversight -- but the command surface lived in a `main.rs`,
/// importable by nothing, so `apr pv` could not exist.
///
/// # Errors
/// Propagates whatever the selected subcommand returns.
#[allow(clippy::too_many_lines)]
pub fn dispatch(command: Commands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Explain {
            contract,
            format,
            binding,
        } => commands::explain::run(&contract, binding.as_deref(), &format),
        Commands::Validate { contract } => commands::validate::run(&contract),
        Commands::CheckParity { contract } => commands::check_parity::run(&contract),
        Commands::Scaffold {
            contract,
            r#trait,
            output,
        } => commands::scaffold::run(&contract, r#trait, output.as_deref()),
        Commands::ExtractPytorch { target, output } => {
            commands::extract::run(&target, output.as_deref())
        }
        Commands::Codegen {
            contract_dir,
            output,
        } => commands::codegen::run(&contract_dir, output.as_deref()),
        Commands::Kani { contract } => commands::kani::run(&contract),
        Commands::Probar { contract, binding } => {
            commands::probar::run(&contract, binding.as_deref())
        }
        Commands::Status { contract } => commands::status::run(&contract),
        Commands::Audit {
            contract, binding, ..
        } => commands::audit::run(&contract, binding.as_deref()),
        Commands::Diff { old, new } => commands::diff::run(&old, &new),
        Commands::Coverage {
            contract_dir,
            binding,
            fuzz,
            reverse,
            enforcement,
        } => commands::coverage::run(
            &contract_dir,
            binding.as_deref(),
            fuzz,
            reverse.as_deref(),
            enforcement.as_deref(),
        ),
        Commands::Generate {
            contract,
            output,
            binding,
            readme,
            ci,
        } => commands::generate::run(&contract, &output, binding.as_deref(), readme, ci),
        Commands::Graph {
            contract_dir,
            format,
        } => match commands::graph::GraphFormat::from_str(&format) {
            Ok(fmt) => commands::graph::run(&contract_dir, fmt),
            Err(e) => Err(e.into()),
        },
        Commands::Equations { contract, format } => {
            match commands::equations::OutputFormat::from_str(&format) {
                Ok(fmt) => commands::equations::run(&contract, fmt),
                Err(e) => Err(e.into()),
            }
        }
        Commands::Lean {
            contract,
            output_dir,
        } => commands::lean::run(&contract, output_dir.as_deref()),
        Commands::LeanStatus { path } => commands::lean_status::run(&path),
        Commands::ProofStatus {
            path,
            binding,
            verify_bindings,
            format,
            table,
            kind,
        } => commands::proof_status::run(
            &path,
            binding.as_deref(),
            verify_bindings.as_deref(),
            &format,
            table,
            kind.as_deref(),
        ),
        Commands::Lint {
            contract_dir,
            min_score,
            binding,
            format,
            severity,
            strict,
            suppress,
            suppress_rule,
            suppress_file,
            rule,
            config,
            diff_ref,
            trend,
            show_trend,
            no_cache,
            cache_stats,
            coverage,
            min_coverage,
            crate_dir,
            min_level,
            explain,
            watch,
            strict_test_binding,
            ..
        } => {
            if let Some(ref rule_id) = explain {
                commands::lint::explain_rule(rule_id);
                return Ok(());
            }
            commands::lint::run(
                &contract_dir,
                binding.as_deref(),
                min_score,
                format.as_deref(),
                severity.as_deref(),
                strict,
                suppress.as_deref(),
                suppress_rule.as_deref(),
                suppress_file.as_deref(),
                &rule,
                config.as_deref(),
                diff_ref.as_deref(),
                trend,
                show_trend,
                no_cache,
                cache_stats,
                coverage,
                min_coverage,
                crate_dir.as_deref(),
                min_level.as_deref(),
                watch,
                strict_test_binding,
            )
        }
        Commands::Score {
            path,
            binding,
            format,
            min_score,
            summary,
            top_gaps,
            weights,
            pvscore,
            ..
        } => commands::score::run(
            &path,
            binding.as_deref(),
            &format,
            min_score,
            summary,
            top_gaps,
            weights.as_deref(),
            pvscore,
        ),
        Commands::Query(q) => commands::query::run(&commands::query::QueryCliParams {
            contract_dir: &q.contract_dir,
            query_str: &q.query,
            regex: q.regex,
            literal: q.literal,
            case_sensitive: q.case_sensitive,
            limit: q.limit,
            obligation: q.obligation.as_deref(),
            min_score: q.min_score,
            min_level: q.min_level,
            depends_on: q.depends_on.as_deref(),
            depended_by: q.depended_by.as_deref(),
            unproven: q.unproven,
            show_score: q.score,
            show_graph: q.graph,
            show_paper: q.paper,
            show_proof_status: q.proof_status,
            show_binding: q.binding_info,
            binding_gaps: q.binding_gaps,
            show_diff: q.diff,
            show_pagerank: q.pagerank,
            show_call_sites: q.call_sites,
            show_violations: q.violations,
            show_coverage_map: q.coverage_map,
            project_filter: q.project.as_deref(),
            include_project: q.include_project.as_deref(),
            tier: q.tier,
            class: q.class,
            kind: q.kind.as_deref(),
            all_projects: q.all_projects,
            rebuild_index: q.rebuild_index,
            binding: q.binding.as_deref(),
            format: &q.format,
            exit_code: q.exit_code,
        }),
        Commands::Invariants { contract } => commands::invariants::run(&contract),
        Commands::Coq { contract } => commands::coq::run(&contract),
        Commands::Fuzz { contract } => commands::fuzz::run(&contract),
        Commands::Mirai { contract } => commands::mirai::run(&contract),
        Commands::Flux { contract } => commands::flux::run(&contract),
        Commands::Tla { contract_dir } => commands::tla::run(&contract_dir),
        Commands::Book {
            contract_dir,
            output,
            update_summary,
            summary_path,
        } => commands::book::run(
            &contract_dir,
            &output,
            update_summary,
            summary_path.as_deref(),
        ),
        Commands::Infer {
            crate_dir,
            binding,
            contract_dir,
            top,
        } => commands::infer::run(&crate_dir, &binding, &contract_dir, top),
        Commands::Unlock { contract, reason } => commands::unlock::run(&contract, &reason),
        Commands::Roofline {
            contract_dir,
            params,
            bits,
            hardware,
            format,
        } => commands::roofline::run(&contract_dir, params, bits, &hardware, &format),
        Commands::Pipeline { pipeline, format } => commands::pipeline::run(&pipeline, &format),
        Commands::Kaizen {
            contract_dir,
            src_root,
            repo,
            dry_run,
            codegen,
            fix,
            json,
            min_score,
        } => {
            let default_root = PathBuf::from("..");
            let root = src_root.as_deref().unwrap_or(&default_root);
            commands::kaizen::run(
                &contract_dir,
                root,
                repo.as_deref(),
                dry_run || !fix, // default to dry-run unless --fix
                codegen || fix,  // --fix implies --codegen
                fix,
                json,
                min_score,
            )
        }
        Commands::VerifyBindings {
            binding,
            output,
            crate_name,
        } => commands::verify_bindings::run(&binding, output.as_deref(), crate_name.as_deref()),
        Commands::Certify {
            contract_dir,
            config,
            output,
        } => commands::certify::run(&contract_dir, config.as_deref(), output.as_deref()),
        Commands::VerifyStructure {
            contract_dir,
            config,
            model,
        } => commands::verify_structure::run(&contract_dir, config.as_deref(), model.as_deref()),
        Commands::VerifyPipeline {
            contract_dir,
            format,
        } => {
            commands::verify_pipeline::run(&contract_dir, &format);
            Ok(())
        }
        Commands::Migrate {
            contract_dir,
            dry_run,
        } => commands::migrate::run(&contract_dir, dry_run),
    }
}

/// Parse `argv` and run one command, exiting non-zero on failure. This is the
/// whole of the standalone `pv` binary.
pub fn run() {
    let cli = Cli::parse();
    let _ = (cli.quiet, cli.verbose); // Flags accepted; used by subcommands via Cli struct

    if let Err(e) = dispatch(cli.command) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "../tests/includes/version_identity_unit.rs"]
mod version_identity_unit;

#[cfg(test)]
#[path = "../tests/includes/dispatch_tests.rs"]
mod dispatch_tests;

#[cfg(test)]
#[path = "../tests/includes/dispatch_query_tests.rs"]
mod dispatch_query_tests;
