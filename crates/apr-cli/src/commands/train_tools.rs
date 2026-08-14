//! Dispatch for the five `apr train` tools rehomed from standalone binaries.
//!
//! APR-MONO Rule 1 says `apr` is the only user-facing binary. Five published
//! crates still shipped their own:
//!
//! | was                        | is now                |
//! |----------------------------|-----------------------|
//! | `aprender-train-bench`     | `apr train bench`     |
//! | `aprender-train-distill`   | `apr train distill`   |
//! | `aprender-train-inspect`   | `apr train inspect`   |
//! | `aprender-train-lora`      | `apr train lora`      |
//! | `aprender-train-shell`     | `apr train shell`     |
//!
//! This module is a *dispatcher only*. Every arm calls the `::cli::run_*`
//! function in the owning crate — the same entry point the deleted `main`
//! called. No logic was copied into apr-cli; where logic was stranded in a
//! `main.rs` (the distill export writers, the inspect memory formula) it moved
//! into the owning crate's library, not here.

use crate::error::CliError;
use crate::{Cli, TrainToolArgs};

/// Build the `entrenar_common::Cli` the tool libraries expect.
///
/// The deleted binaries flattened `entrenar_common::CommonArgs`, which carried
/// `--format`, `--quiet`, `--verbose` and `--no-color`. On `apr`, `--quiet` and
/// `--verbose` are already global flags, so only `--format` and `--no-color`
/// are redeclared per subcommand; this folds all four back together.
///
/// `apr --json` is honoured as an override for `--format json`, matching how
/// every other `apr` subcommand treats the global flag.
pub(crate) fn common_cli(args: &TrainToolArgs, cli: &Cli) -> entrenar_common::Cli {
    let verbosity = if cli.quiet {
        0
    } else if cli.verbose {
        2
    } else {
        1
    };

    let format = if cli.json {
        entrenar_common::OutputFormat::Json
    } else {
        args.format.parse().unwrap_or_default()
    };

    entrenar_common::Cli {
        format,
        verbosity,
        color: !args.no_color,
    }
}

/// Map a tool-library error onto apr's error type.
///
/// The tools' errors are already user-facing prose (`EntrenarError` renders a
/// message plus a suggestion), so the mapping preserves the text rather than
/// re-wording it.
fn map_err(e: entrenar_common::EntrenarError) -> CliError {
    CliError::ValidationFailed(e.to_string())
}

/// Dispatch `apr train bench <verb>`.
pub(crate) fn dispatch_bench(
    action: &crate::TrainBenchCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    use crate::TrainBenchCommands as C;
    match action {
        C::Temperature {
            start,
            end,
            step,
            runs,
            common,
        } => entrenar_bench::cli::run_temperature(
            *start,
            *end,
            *step,
            *runs,
            &common_cli(common, cli),
        )
        .map_err(map_err),
        C::Alpha {
            start,
            end,
            step,
            runs,
            common,
        } => entrenar_bench::cli::run_alpha(*start, *end, *step, *runs, &common_cli(common, cli))
            .map_err(map_err),
        C::Compare {
            strategies,
            runs,
            common,
        } => entrenar_bench::cli::run_compare(strategies, *runs, &common_cli(common, cli))
            .map_err(map_err),
        C::Ablation { config, common } => {
            entrenar_bench::cli::run_ablation(config.as_deref(), &common_cli(common, cli))
                .map_err(map_err)
        }
        C::CostPerformance {
            gpu,
            results,
            common,
        } => entrenar_bench::cli::run_cost_performance(
            gpu,
            results.as_deref(),
            &common_cli(common, cli),
        )
        .map_err(map_err),
        C::Recommend {
            max_gpu_hours,
            max_cost,
            min_accuracy,
            max_memory,
            gpu,
            common,
        } => entrenar_bench::cli::run_recommend(
            *max_gpu_hours,
            *max_cost,
            *min_accuracy,
            *max_memory,
            gpu,
            &common_cli(common, cli),
        )
        .map_err(map_err),
    }
}

/// Dispatch `apr train distill <verb>`.
pub(crate) fn dispatch_distill(
    action: &crate::TrainDistillCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    use crate::TrainDistillCommands as C;
    match action {
        C::Run {
            config,
            output,
            dry_run,
            common,
        } => entrenar_distill::cli::run_pipeline(
            config,
            output.clone(),
            *dry_run,
            &common_cli(common, cli),
        )
        .map_err(map_err),
        C::Estimate {
            teacher,
            student,
            batch_size,
            seq_len,
            common,
        } => entrenar_distill::cli::run_estimate(
            teacher,
            student.clone(),
            *batch_size,
            *seq_len,
            &common_cli(common, cli),
        )
        .map_err(map_err),
        C::Validate { config, common } => {
            entrenar_distill::cli::run_validate(config, &common_cli(common, cli)).map_err(map_err)
        }
        C::Export {
            input,
            format,
            output,
            quantize,
            no_color,
        } => {
            // `export` owns `--format` for the MODEL format, so it cannot also
            // flatten TrainToolArgs. Build the display config from the global
            // flags plus this command's `--no-color`.
            let tool_args = TrainToolArgs {
                format: "table".to_string(),
                no_color: *no_color,
            };
            entrenar_distill::cli::run_export(
                input,
                format,
                output,
                quantize,
                &common_cli(&tool_args, cli),
            )
            .map_err(map_err)
        }
    }
}

/// Dispatch `apr train inspect <verb>`.
pub(crate) fn dispatch_inspect(
    action: &crate::TrainInspectCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    use crate::TrainInspectCommands as C;
    match action {
        C::Info { path, common } => {
            entrenar_inspect::cli::run_info(path, &common_cli(common, cli)).map_err(map_err)
        }
        // The deleted binary's own `--verbose` is `apr`'s global `--verbose`
        // here — same spelling, same meaning (list every tensor).
        C::Layers { path, common } => {
            entrenar_inspect::cli::run_layers(path, cli.verbose, &common_cli(common, cli))
                .map_err(map_err)
        }
        C::Memory {
            path,
            batch_size,
            seq_len,
            common,
        } => {
            entrenar_inspect::cli::run_memory(path, *batch_size, *seq_len, &common_cli(common, cli))
                .map_err(map_err)
        }
        C::Validate {
            path,
            strict,
            common,
        } => {
            // The deleted binary called `process::exit(1)` on an invalid model.
            // Returning an error keeps the same non-zero exit without a
            // `process::exit` inside a library, and prints a reason.
            let valid =
                entrenar_inspect::cli::run_validate(path, *strict, &common_cli(common, cli))
                    .map_err(map_err)?;
            if valid {
                Ok(())
            } else {
                Err(CliError::ValidationFailed(format!(
                    "{} failed integrity validation (see report above)",
                    path.display()
                )))
            }
        }
        C::Convert {
            input,
            to,
            output,
            quantize,
            common,
        } => entrenar_inspect::cli::run_convert(
            input,
            to,
            output,
            quantize,
            &common_cli(common, cli),
        )
        .map_err(map_err),
        C::Compare {
            model1,
            model2,
            common,
        } => entrenar_inspect::cli::run_compare(model1, model2, &common_cli(common, cli))
            .map_err(map_err),
    }
}

/// Dispatch `apr train lora <verb>`.
pub(crate) fn dispatch_lora(
    action: &crate::TrainLoraCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    use crate::TrainLoraCommands as C;
    match action {
        C::Plan {
            model,
            vram,
            method,
            common,
        } => entrenar_lora::cli::run_plan(model, *vram, method, &common_cli(common, cli))
            .map_err(map_err),
        C::Compare {
            model,
            vram,
            common,
        } => {
            entrenar_lora::cli::run_compare(model, *vram, &common_cli(common, cli)).map_err(map_err)
        }
        C::Merge {
            base,
            adapter,
            output,
            scale,
            common,
        } => entrenar_lora::cli::run_merge(base, adapter, output, *scale, &common_cli(common, cli))
            .map_err(map_err),
        C::Inspect { path, common } => {
            entrenar_lora::cli::run_inspect(path, &common_cli(common, cli)).map_err(map_err)
        }
    }
}

/// Dispatch `apr train shell`.
///
/// With `--command`, parses and runs one shell command and returns. Without it,
/// enters the interactive REPL. `--session` pre-loads a saved session in both
/// modes; a session that fails to load falls back to an empty one, exactly as
/// the deleted binary did.
pub(crate) fn dispatch_shell(
    session: Option<&std::path::Path>,
    command: Option<&str>,
    format: &str,
    no_color: bool,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    // `--format` / `--no-color` are accepted for parity with the deleted
    // binary's CommonArgs surface. The REPL renders its own prompts, so the
    // value is validated and carried, not used to re-shape output.
    let _tool_cli = common_cli(
        &TrainToolArgs {
            format: format.to_string(),
            no_color,
        },
        cli,
    );

    let mut state = match session {
        Some(path) => entrenar_shell::cli::load_session_or_default(path).0,
        None => entrenar_shell::SessionState::new(),
    };

    match command {
        Some(cmd) => {
            entrenar_shell::cli::run_single_command(cmd, &mut state).map_err(map_err)?;
            Ok(())
        }
        None => entrenar_shell::cli::run_interactive(state).map_err(map_err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Parse `apr` args on a 16 MB stack.
    ///
    /// Same reason as `crate::parsing::parse_cli`: clap's generated parser for
    /// the full `apr` command tree overflows the 2 MB default test-thread stack
    /// in debug builds. Calling `Cli::try_parse_from` directly from a test
    /// aborts the whole test binary with SIGABRT, not a failed assertion.
    fn try_parse(args: &[&str]) -> std::result::Result<Cli, clap::error::Error> {
        let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || Cli::try_parse_from(owned))
            .expect("spawn parse thread")
            .join()
            .expect("join parse thread")
    }

    fn parse(args: &[&str]) -> Cli {
        try_parse(args).expect("apr CLI should parse these arguments")
    }

    fn train_command(cli: &Cli) -> &crate::TrainCommands {
        match cli.command.as_ref() {
            crate::Commands::Extended(crate::ExtendedCommands::Train { command }) => command,
            other => panic!("expected an `apr train` command, parsed {other:?}"),
        }
    }

    // ── common_cli: the four CommonArgs knobs the binaries carried ──────────

    #[test]
    fn common_cli_maps_apr_quiet_to_verbosity_zero() {
        let cli = parse(&[
            "apr",
            "--quiet",
            "train",
            "lora",
            "inspect",
            "a.safetensors",
        ]);
        let args = TrainToolArgs {
            format: "table".into(),
            no_color: false,
        };
        assert_eq!(common_cli(&args, &cli).verbosity, 0);
    }

    #[test]
    fn common_cli_maps_apr_verbose_to_verbosity_two() {
        let cli = parse(&[
            "apr",
            "--verbose",
            "train",
            "lora",
            "inspect",
            "a.safetensors",
        ]);
        let args = TrainToolArgs {
            format: "table".into(),
            no_color: false,
        };
        assert_eq!(common_cli(&args, &cli).verbosity, 2);
    }

    #[test]
    fn common_cli_default_verbosity_is_one() {
        let cli = parse(&["apr", "train", "lora", "inspect", "a.safetensors"]);
        let args = TrainToolArgs {
            format: "table".into(),
            no_color: false,
        };
        assert_eq!(common_cli(&args, &cli).verbosity, 1);
    }

    #[test]
    fn common_cli_carries_every_format_value() {
        let cli = parse(&["apr", "train", "lora", "inspect", "a.safetensors"]);
        for (text, expected) in [
            ("table", entrenar_common::OutputFormat::Table),
            ("text", entrenar_common::OutputFormat::Table),
            ("json", entrenar_common::OutputFormat::Json),
            ("compact", entrenar_common::OutputFormat::Compact),
            ("line", entrenar_common::OutputFormat::Compact),
        ] {
            let args = TrainToolArgs {
                format: text.into(),
                no_color: false,
            };
            assert_eq!(
                common_cli(&args, &cli).format,
                expected,
                "--format {text} must map to {expected:?}"
            );
        }
    }

    #[test]
    fn common_cli_global_json_overrides_format() {
        let cli = parse(&["apr", "--json", "train", "lora", "inspect", "a.safetensors"]);
        let args = TrainToolArgs {
            format: "table".into(),
            no_color: false,
        };
        assert_eq!(
            common_cli(&args, &cli).format,
            entrenar_common::OutputFormat::Json
        );
    }

    #[test]
    fn common_cli_carries_no_color() {
        let cli = parse(&["apr", "train", "lora", "inspect", "a.safetensors"]);
        let colored = TrainToolArgs {
            format: "table".into(),
            no_color: false,
        };
        let plain = TrainToolArgs {
            format: "table".into(),
            no_color: true,
        };
        assert!(common_cli(&colored, &cli).color);
        assert!(!common_cli(&plain, &cli).color);
    }

    // ── argument reachability: every flag the deleted binaries accepted ─────
    //
    // #2418 shipped three times in this repo: a rehomed command quietly lost an
    // argument. These parse the full flag set of each migrated subcommand and
    // assert the parsed value, so a dropped or renamed flag fails the build.

    #[test]
    fn bench_temperature_carries_all_four_sweep_args() {
        let cli = parse(&[
            "apr",
            "train",
            "bench",
            "temperature",
            "--start",
            "2.0",
            "--end",
            "6.0",
            "--step",
            "0.25",
            "--runs",
            "7",
            "--format",
            "json",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action:
                    crate::TrainBenchCommands::Temperature {
                        start,
                        end,
                        step,
                        runs,
                        common,
                    },
            } => {
                assert_eq!(*start, 2.0);
                assert_eq!(*end, 6.0);
                assert_eq!(*step, 0.25);
                assert_eq!(*runs, 7);
                assert_eq!(common.format, "json");
            }
            other => panic!("expected train bench temperature, got {other:?}"),
        }
    }

    #[test]
    fn bench_temperature_defaults_match_the_deleted_binary() {
        let cli = parse(&["apr", "train", "bench", "temperature"]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action:
                    crate::TrainBenchCommands::Temperature {
                        start,
                        end,
                        step,
                        runs,
                        ..
                    },
            } => {
                assert_eq!(*start, 1.0);
                assert_eq!(*end, 8.0);
                assert_eq!(*step, 0.5);
                assert_eq!(*runs, 3);
            }
            other => panic!("expected train bench temperature, got {other:?}"),
        }
    }

    #[test]
    fn bench_alpha_defaults_match_the_deleted_binary() {
        let cli = parse(&["apr", "train", "bench", "alpha"]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action:
                    crate::TrainBenchCommands::Alpha {
                        start,
                        end,
                        step,
                        runs,
                        ..
                    },
            } => {
                assert_eq!(*start, 0.1);
                assert_eq!(*end, 0.9);
                assert_eq!(*step, 0.1);
                assert_eq!(*runs, 3);
            }
            other => panic!("expected train bench alpha, got {other:?}"),
        }
    }

    #[test]
    fn bench_compare_splits_strategies_on_commas() {
        let cli = parse(&[
            "apr",
            "train",
            "bench",
            "compare",
            "--strategies",
            "kd,progressive",
            "--runs",
            "9",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action:
                    crate::TrainBenchCommands::Compare {
                        strategies, runs, ..
                    },
            } => {
                assert_eq!(strategies, &["kd".to_string(), "progressive".to_string()]);
                assert_eq!(*runs, 9);
            }
            other => panic!("expected train bench compare, got {other:?}"),
        }
    }

    #[test]
    fn bench_ablation_keeps_its_short_config_flag() {
        let cli = parse(&["apr", "train", "bench", "ablation", "-c", "base.yaml"]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action: crate::TrainBenchCommands::Ablation { config, .. },
            } => assert_eq!(
                config.as_deref(),
                Some(std::path::Path::new("base.yaml")),
                "-c must still reach the ablation config argument"
            ),
            other => panic!("expected train bench ablation, got {other:?}"),
        }
    }

    #[test]
    fn bench_cost_performance_carries_gpu_and_results() {
        let cli = parse(&[
            "apr",
            "train",
            "bench",
            "cost-performance",
            "--gpu",
            "v100",
            "--results",
            "r.json",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action: crate::TrainBenchCommands::CostPerformance { gpu, results, .. },
            } => {
                assert_eq!(gpu, "v100");
                assert_eq!(results.as_deref(), Some(std::path::Path::new("r.json")));
            }
            other => panic!("expected train bench cost-performance, got {other:?}"),
        }
    }

    #[test]
    fn bench_recommend_carries_all_four_constraints() {
        let cli = parse(&[
            "apr",
            "train",
            "bench",
            "recommend",
            "--max-gpu-hours",
            "12",
            "--max-cost",
            "40",
            "--min-accuracy",
            "0.85",
            "--max-memory",
            "48",
            "--gpu",
            "t4",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Bench {
                action:
                    crate::TrainBenchCommands::Recommend {
                        max_gpu_hours,
                        max_cost,
                        min_accuracy,
                        max_memory,
                        gpu,
                        ..
                    },
            } => {
                assert_eq!(*max_gpu_hours, Some(12.0));
                assert_eq!(*max_cost, Some(40.0));
                assert_eq!(*min_accuracy, Some(0.85));
                assert_eq!(*max_memory, Some(48.0));
                assert_eq!(gpu, "t4");
            }
            other => panic!("expected train bench recommend, got {other:?}"),
        }
    }

    #[test]
    fn distill_run_keeps_short_config_output_and_dry_run() {
        let cli = parse(&[
            "apr",
            "train",
            "distill",
            "run",
            "-c",
            "d.yaml",
            "-o",
            "out",
            "--dry-run",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Distill {
                action:
                    crate::TrainDistillCommands::Run {
                        config,
                        output,
                        dry_run,
                        ..
                    },
            } => {
                assert_eq!(config, std::path::Path::new("d.yaml"));
                assert_eq!(output.as_deref(), Some(std::path::Path::new("out")));
                assert!(*dry_run);
            }
            other => panic!("expected train distill run, got {other:?}"),
        }
    }

    #[test]
    fn distill_estimate_carries_teacher_student_batch_and_seq() {
        let cli = parse(&[
            "apr",
            "train",
            "distill",
            "estimate",
            "--teacher",
            "Qwen/Qwen2.5-7B",
            "--student",
            "Qwen/Qwen2.5-0.5B",
            "--batch-size",
            "8",
            "--seq-len",
            "1024",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Distill {
                action:
                    crate::TrainDistillCommands::Estimate {
                        teacher,
                        student,
                        batch_size,
                        seq_len,
                        ..
                    },
            } => {
                assert_eq!(teacher, "Qwen/Qwen2.5-7B");
                assert_eq!(student.as_deref(), Some("Qwen/Qwen2.5-0.5B"));
                assert_eq!(*batch_size, 8);
                assert_eq!(*seq_len, 1024);
            }
            other => panic!("expected train distill estimate, got {other:?}"),
        }
    }

    #[test]
    fn distill_export_keeps_short_flags_and_quantize() {
        let cli = parse(&[
            "apr",
            "train",
            "distill",
            "export",
            "-i",
            "student.safetensors",
            "-f",
            "gguf",
            "-o",
            "student.gguf",
            "--quantize",
            "q4_0",
            "--no-color",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Distill {
                action:
                    crate::TrainDistillCommands::Export {
                        input,
                        format,
                        output,
                        quantize,
                        no_color,
                    },
            } => {
                assert_eq!(input, std::path::Path::new("student.safetensors"));
                assert_eq!(format, "gguf");
                assert_eq!(output, std::path::Path::new("student.gguf"));
                assert_eq!(quantize, "q4_0");
                assert!(*no_color);
            }
            other => panic!("expected train distill export, got {other:?}"),
        }
    }

    /// `apr train distill export -f` names the MODEL format, so it must NOT be
    /// shadowed by the display `--format` that every sibling subcommand
    /// flattens. Accepting `--format json` here would silently produce a
    /// SafeTensors-vs-JSON mix-up.
    #[test]
    fn distill_export_format_is_the_model_format_not_the_display_format() {
        let cli = parse(&[
            "apr",
            "train",
            "distill",
            "export",
            "-i",
            "s.safetensors",
            "-o",
            "s.apr",
            "--format",
            "apr",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Distill {
                action: crate::TrainDistillCommands::Export { format, .. },
            } => assert_eq!(
                format, "apr",
                "--format must reach the model-format argument"
            ),
            other => panic!("expected train distill export, got {other:?}"),
        }
    }

    /// The deleted binary's `apr train inspect layers -v` listed every tensor.
    /// That flag is now `apr`'s global `--verbose`; both spellings must still
    /// reach the listing, in either position.
    #[test]
    fn inspect_layers_verbose_comes_from_the_global_flag() {
        for argv in [
            ["apr", "train", "inspect", "layers", "m.st", "--verbose"],
            ["apr", "-v", "train", "inspect", "layers", "m.st"],
        ] {
            let cli = parse(&argv);
            assert!(
                cli.verbose,
                "{argv:?} must set the verbose flag that drives the tensor listing"
            );
            match train_command(&cli) {
                crate::TrainCommands::Inspect {
                    action: crate::TrainInspectCommands::Layers { path, .. },
                } => assert_eq!(path, std::path::Path::new("m.st")),
                other => panic!("expected train inspect layers, got {other:?}"),
            }
        }
    }

    #[test]
    fn inspect_memory_keeps_short_batch_and_seq_flags() {
        let cli = parse(&[
            "apr",
            "train",
            "inspect",
            "memory",
            "model.safetensors",
            "-b",
            "4",
            "-s",
            "256",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Inspect {
                action:
                    crate::TrainInspectCommands::Memory {
                        path,
                        batch_size,
                        seq_len,
                        ..
                    },
            } => {
                assert_eq!(path, std::path::Path::new("model.safetensors"));
                assert_eq!(*batch_size, 4);
                assert_eq!(*seq_len, 256);
            }
            other => panic!("expected train inspect memory, got {other:?}"),
        }
    }

    #[test]
    fn inspect_validate_carries_strict() {
        let cli = parse(&[
            "apr",
            "train",
            "inspect",
            "validate",
            "model.safetensors",
            "--strict",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Inspect {
                action: crate::TrainInspectCommands::Validate { strict, .. },
            } => assert!(*strict),
            other => panic!("expected train inspect validate, got {other:?}"),
        }
    }

    #[test]
    fn inspect_convert_keeps_short_to_and_output_flags() {
        let cli = parse(&[
            "apr",
            "train",
            "inspect",
            "convert",
            "in.safetensors",
            "-t",
            "gguf",
            "-o",
            "out.gguf",
            "--quantize",
            "q8_0",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Inspect {
                action:
                    crate::TrainInspectCommands::Convert {
                        input,
                        to,
                        output,
                        quantize,
                        ..
                    },
            } => {
                assert_eq!(input, std::path::Path::new("in.safetensors"));
                assert_eq!(to, "gguf");
                assert_eq!(output, std::path::Path::new("out.gguf"));
                assert_eq!(quantize, "q8_0");
            }
            other => panic!("expected train inspect convert, got {other:?}"),
        }
    }

    #[test]
    fn inspect_compare_takes_two_positional_models() {
        let cli = parse(&["apr", "train", "inspect", "compare", "a.st", "b.st"]);
        match train_command(&cli) {
            crate::TrainCommands::Inspect {
                action: crate::TrainInspectCommands::Compare { model1, model2, .. },
            } => {
                assert_eq!(model1, std::path::Path::new("a.st"));
                assert_eq!(model2, std::path::Path::new("b.st"));
            }
            other => panic!("expected train inspect compare, got {other:?}"),
        }
    }

    #[test]
    fn lora_plan_carries_model_vram_and_method() {
        let cli = parse(&[
            "apr", "train", "lora", "plan", "--model", "7B", "--vram", "24", "--method", "qlora",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Lora {
                action:
                    crate::TrainLoraCommands::Plan {
                        model,
                        vram,
                        method,
                        ..
                    },
            } => {
                assert_eq!(model, "7B");
                assert_eq!(*vram, 24.0);
                assert_eq!(method, "qlora");
            }
            other => panic!("expected train lora plan, got {other:?}"),
        }
    }

    /// The deleted binary declared `-m` twice — auto-derived for `--model` and
    /// explicitly for `--method`. That is a clap error, not a preference, so
    /// exactly one of them can keep it. `-m` is `--method` across `apr`
    /// (`apr tune -m`, `apr finetune -m`); this pins that choice so a later
    /// edit cannot quietly hand `-m` back to `--model`.
    #[test]
    fn lora_plan_short_m_is_method_not_model() {
        let cli = parse(&[
            "apr", "train", "lora", "plan", "--model", "7B", "--vram", "24", "-m", "qlora",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Lora {
                action:
                    crate::TrainLoraCommands::Plan {
                        model,
                        vram,
                        method,
                        ..
                    },
            } => {
                assert_eq!(method, "qlora", "-m must set --method");
                assert_eq!(model, "7B", "--model must be untouched by -m");
                assert_eq!(*vram, 24.0);
            }
            other => panic!("expected train lora plan, got {other:?}"),
        }
    }

    /// `-v` belongs to `apr`'s global `--verbose`. If `--vram` ever reclaimed
    /// it, `apr train lora plan -v` would stop meaning "verbose" and start
    /// eating the next token as a VRAM figure.
    #[test]
    fn lora_plan_short_v_is_global_verbose_not_vram() {
        let cli = parse(&[
            "apr", "train", "lora", "plan", "--model", "7B", "--vram", "24", "-v",
        ]);
        assert!(cli.verbose, "-v must set the global verbose flag");
        match train_command(&cli) {
            crate::TrainCommands::Lora {
                action: crate::TrainLoraCommands::Plan { vram, .. },
            } => assert_eq!(*vram, 24.0, "-v must not have been read as --vram"),
            other => panic!("expected train lora plan, got {other:?}"),
        }
    }

    #[test]
    fn lora_compare_defaults_vram_to_twentyfour() {
        let cli = parse(&["apr", "train", "lora", "compare", "--model", "13B"]);
        match train_command(&cli) {
            crate::TrainCommands::Lora {
                action: crate::TrainLoraCommands::Compare { model, vram, .. },
            } => {
                assert_eq!(model, "13B");
                assert_eq!(*vram, 24.0);
            }
            other => panic!("expected train lora compare, got {other:?}"),
        }
    }

    #[test]
    fn lora_merge_carries_scale() {
        // `apr finetune --merge` has no --scale; losing it here would silently
        // drop the only way to reach a scaled adapter merge.
        let cli = parse(&[
            "apr",
            "train",
            "lora",
            "merge",
            "-b",
            "base.st",
            "-a",
            "adapter.st",
            "-o",
            "out.st",
            "-s",
            "0.5",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Lora {
                action:
                    crate::TrainLoraCommands::Merge {
                        base,
                        adapter,
                        output,
                        scale,
                        ..
                    },
            } => {
                assert_eq!(base, std::path::Path::new("base.st"));
                assert_eq!(adapter, std::path::Path::new("adapter.st"));
                assert_eq!(output, std::path::Path::new("out.st"));
                assert_eq!(*scale, 0.5);
            }
            other => panic!("expected train lora merge, got {other:?}"),
        }
    }

    #[test]
    fn lora_merge_defaults_scale_to_one() {
        let cli = parse(&[
            "apr",
            "train",
            "lora",
            "merge",
            "-b",
            "base.st",
            "-a",
            "adapter.st",
            "-o",
            "out.st",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Lora {
                action: crate::TrainLoraCommands::Merge { scale, .. },
            } => assert_eq!(*scale, 1.0),
            other => panic!("expected train lora merge, got {other:?}"),
        }
    }

    #[test]
    fn shell_keeps_short_session_and_command_flags() {
        let cli = parse(&[
            "apr",
            "train",
            "shell",
            "-s",
            "sess.json",
            "-c",
            "help",
            "--no-color",
        ]);
        match train_command(&cli) {
            crate::TrainCommands::Shell {
                session,
                command,
                no_color,
                ..
            } => {
                assert_eq!(session.as_deref(), Some(std::path::Path::new("sess.json")));
                assert_eq!(command.as_deref(), Some("help"));
                assert!(*no_color);
            }
            other => panic!("expected train shell, got {other:?}"),
        }
    }

    // ── refusals: invalid input must be rejected, not accepted ──────────────

    #[test]
    fn bench_recommend_refuses_non_numeric_max_cost() {
        // Asserting is_ok() on invalid input would lock the defect in.
        assert!(
            try_parse(&["apr", "train", "bench", "recommend", "--max-cost", "cheap"]).is_err(),
            "a non-numeric --max-cost must be refused by the parser"
        );
    }

    #[test]
    fn lora_plan_refuses_missing_required_vram() {
        assert!(
            try_parse(&["apr", "train", "lora", "plan", "--model", "7B"]).is_err(),
            "--vram is required by `apr train lora plan` and its absence must be refused"
        );
    }

    #[test]
    fn distill_run_refuses_missing_required_config() {
        assert!(
            try_parse(&["apr", "train", "distill", "run"]).is_err(),
            "--config is required by `apr train distill run` and its absence must be refused"
        );
    }

    #[test]
    fn inspect_info_refuses_missing_positional_path() {
        assert!(
            try_parse(&["apr", "train", "inspect", "info"]).is_err(),
            "the model path is required by `apr train inspect info`"
        );
    }
}
