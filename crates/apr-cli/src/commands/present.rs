//! `apr present`, `apr top` and `apr score` — the three capabilities that used
//! to ship as the standalone `presentar`, `ptop` and `score` binaries.
//!
//! Every function here is a thin adapter: it maps parsed clap arguments onto
//! the library entry point the old binary's `main` called, and translates the
//! result into apr's error type. No logic was copied — `presentar`'s seven
//! verbs live in `aprender_present_cli`, and `ptop`/`score` live in
//! `presentar_terminal::ptop::run` and `presentar_terminal::tools::score`.

use crate::error::{CliError, Result};
use crate::PresentCommands;
use presentar_terminal::ptop::PtopOptions;
use presentar_terminal::tools::score::{OutputFormat, ScoreOptions};
use std::path::PathBuf;

/// Run the ptop system monitor (`apr top`).
///
/// # Errors
///
/// Returns [`CliError::Io`] if the terminal cannot be prepared or a frame
/// cannot be written.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub fn run_top(opts: &PtopOptions) -> Result<()> {
    presentar_terminal::ptop::run(opts).map_err(CliError::Io)
}

/// Score a Rust TUI crate (`apr score`).
///
/// `ci` reproduces the `score` binary's contract exactly: a run below the
/// threshold exits 1. Without `--ci`, a failing score is still reported in
/// full and the command succeeds — the report *is* the deliverable.
///
/// # Errors
///
/// Returns [`CliError::ValidationFailed`] when the path is not a Rust crate,
/// or when the report cannot be serialised into the requested format.
pub fn run_score(opts: &ScoreOptions, ci: bool) -> Result<()> {
    let passed = presentar_terminal::tools::score::run(opts).map_err(CliError::ValidationFailed)?;
    if ci && !passed {
        // F-PMAT-007 / F-PMAT-008: the `score` binary's documented CI exit
        // code is 1. Preserved verbatim rather than remapped onto apr's
        // error-class codes, because CI scripts assert on this number.
        std::process::exit(1);
    }
    Ok(())
}

/// Dispatch `apr present <SUBCOMMAND>`.
///
/// # Errors
///
/// Never returns `Err`: like the `presentar` binary, each verb prints its own
/// diagnostics and exits 1 on failure. The `Result` keeps the signature
/// uniform with the rest of the dispatch table.
pub fn dispatch_present(command: &PresentCommands) -> Result<()> {
    match command {
        PresentCommands::Serve { port, dir, watch } => {
            aprender_present_cli::serve(*port, dir.clone(), *watch);
        }
        PresentCommands::Bundle {
            output,
            no_optimize,
        } => aprender_present_cli::bundle(output.clone(), *no_optimize),
        PresentCommands::New { name } => aprender_present_cli::new_project(name),
        PresentCommands::Check { manifest } => aprender_present_cli::check_manifest(manifest),
        PresentCommands::Score {
            manifest,
            format,
            badge,
        } => aprender_present_cli::compute_score(manifest, format, badge.as_ref()),
        PresentCommands::Gate {
            manifest,
            min_grade,
            min_score,
            strict,
        } => aprender_present_cli::run_gates(manifest, min_grade, *min_score, *strict),
        PresentCommands::Deploy {
            source,
            target,
            bucket,
            distribution,
            region,
            dry_run,
            skip_build,
        } => aprender_present_cli::deploy(
            source,
            target,
            bucket.as_deref(),
            distribution.as_deref(),
            region,
            *dry_run,
            *skip_build,
        ),
    }
    Ok(())
}

/// Build [`PtopOptions`] from the parsed `apr top` arguments.
///
/// Exists so the mapping is testable: a dropped field here is exactly the
/// #2418 failure — an argument the help text advertises that never reaches
/// the implementation.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
#[must_use]
pub fn top_options(
    refresh: u64,
    deterministic: bool,
    no_color: bool,
    render_once: bool,
    width: u16,
    height: u16,
    config: Option<&std::path::Path>,
    dump_config: bool,
    qa_timing: bool,
    explode: Option<&str>,
) -> PtopOptions {
    PtopOptions {
        refresh,
        deterministic,
        no_color,
        render_once,
        width,
        height,
        config: config.map(PathBuf::from),
        dump_config,
        qa_timing,
        explode: explode.map(str::to_owned),
    }
}

/// Build [`ScoreOptions`] from the parsed `apr score` arguments.
#[must_use]
pub fn score_options(
    path: &std::path::Path,
    output: OutputFormat,
    quiet: bool,
    verbose: bool,
    threshold: u32,
    no_color: bool,
    config: Option<&std::path::Path>,
) -> ScoreOptions {
    ScoreOptions {
        path: path.to_path_buf(),
        output,
        quiet,
        verbose,
        threshold,
        no_color,
        config: config.map(PathBuf::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `apr top` flag must land on the matching `PtopOptions` field.
    ///
    /// Each value here is deliberately different from that field's default,
    /// so a mapping that dropped or crossed a field cannot pass.
    #[test]
    fn top_options_carries_every_flag() {
        let opts = top_options(
            250,
            true,
            true,
            true,
            200,
            60,
            Some(std::path::Path::new("/tmp/ptop.yaml")),
            true,
            true,
            Some("gpu"),
        );
        assert_eq!(opts.refresh, 250);
        assert!(opts.deterministic);
        assert!(opts.no_color);
        assert!(opts.render_once);
        assert_eq!(opts.width, 200);
        assert_eq!(opts.height, 60);
        assert_eq!(opts.config, Some(PathBuf::from("/tmp/ptop.yaml")));
        assert!(opts.dump_config);
        assert!(opts.qa_timing);
        assert_eq!(opts.explode.as_deref(), Some("gpu"));
    }

    /// The absent-optional case maps to `None`, not to a default path/panel.
    #[test]
    fn top_options_absent_optionals_stay_none() {
        let opts = top_options(1000, false, false, false, 120, 40, None, false, false, None);
        assert_eq!(opts.config, None);
        assert_eq!(opts.explode, None);
    }

    /// Every `apr score` flag must land on the matching `ScoreOptions` field.
    #[test]
    fn score_options_carries_every_flag() {
        let opts = score_options(
            std::path::Path::new("/tmp/some-crate"),
            OutputFormat::Yaml,
            true,
            true,
            42,
            true,
            Some(std::path::Path::new("/tmp/weights.yaml")),
        );
        assert_eq!(opts.path, PathBuf::from("/tmp/some-crate"));
        assert_eq!(opts.output, OutputFormat::Yaml);
        assert!(opts.quiet);
        assert!(opts.verbose);
        assert_eq!(opts.threshold, 42);
        assert!(opts.no_color);
        assert_eq!(opts.config, Some(PathBuf::from("/tmp/weights.yaml")));
    }

    /// `--config` absent must stay absent — the `score` binary falls back to
    /// its built-in weights, it does not read a file from a default path.
    #[test]
    fn score_options_absent_config_stays_none() {
        let opts = score_options(
            std::path::Path::new("."),
            OutputFormat::Text,
            false,
            false,
            80,
            false,
            None,
        );
        assert_eq!(opts.config, None);
    }
}
