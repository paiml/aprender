// APR-MONO: the standalone binaries `cbtop`, `cgp`, `trueno-db`,
// `trueno-ptx-debug`, `probador` and this crate's own `apr-corpus-ingest` were
// removed; their capabilities are reached as `apr compute`, `apr perf`,
// `apr db`, `apr ptx-debug`, `apr probar <cmd>` and `apr corpus-ingest`.
//
// These tests are the ledger for that move. Each asserts that a command the
// removed binary accepted still parses HERE, with the same arguments arriving
// at the same fields — the failure mode being guarded is #2418, an argument
// that survives in the help text but never reaches the handler.

/// Every re-homed capability must appear as a top-level `apr` subcommand.
///
/// A capability that vanishes in a rename is the one outcome this whole
/// exercise exists to prevent, so it gets its own assertion.
#[test]
fn every_rehomed_binary_has_a_top_level_subcommand() {
    for argv in [
        vec!["apr", "compute", "bench"],
        vec!["apr", "perf", "doctor"],
        vec!["apr", "db", "serve", "--config", "db.yaml"],
        vec!["apr", "ptx-debug", "analyze", "k.ptx"],
        vec!["apr", "corpus-ingest", "plan"],
        vec!["apr", "probar", "test"],
    ] {
        let label = argv.join(" ");
        assert!(
            parse_cli(argv).is_ok(),
            "`{label}` must parse — the capability was re-homed, not deleted"
        );
    }
}

/// `apr compute top` carries every flag the standalone `cbtop` no-subcommand
/// mode carried, by the same short and long names, into the same fields.
#[test]
fn compute_top_arguments_reach_the_command() {
    let cli = parse_cli(vec![
        "apr",
        "compute",
        "top",
        "-r",
        "250",
        "-b",
        "cuda",
        "-w",
        "attention",
        "-s",
        "4096",
        "--headless",
        "--format",
        "json",
        "--duration",
        "9",
    ])
    .expect("`apr compute top` with every flag must parse");

    let Commands::Extended(ExtendedCommands::Compute {
        command:
            ::cbtop::cli::ComputeCommand::Top {
                refresh,
                backend,
                workload,
                size,
                headless,
                format,
                duration,
                ..
            },
    }) = *cli.command
    else {
        panic!("expected `apr compute top`");
    };
    assert_eq!(refresh, 250);
    assert_eq!(backend, "cuda");
    assert_eq!(workload, "attention");
    assert_eq!(size, 4096);
    assert!(headless);
    assert_eq!(format, "json");
    assert_eq!(duration, 9);
}

/// `apr perf profile kernel` carries all four of its arguments.
#[test]
fn perf_profile_kernel_arguments_reach_the_command() {
    let cli = parse_cli(vec![
        "apr",
        "perf",
        "profile",
        "kernel",
        "--name",
        "gemm_q4k",
        "--size",
        "1024",
        "--roofline",
        "--metrics",
        "sm__cycles_elapsed",
    ])
    .expect("`apr perf profile kernel` must parse");

    let Commands::Extended(ExtendedCommands::Perf {
        command:
            ::cgp::cli::Commands::Profile {
                target:
                    ::cgp::cli::ProfileTarget::Kernel {
                        name,
                        size,
                        roofline,
                        metrics,
                    },
            },
    }) = *cli.command
    else {
        panic!("expected `apr perf profile kernel`");
    };
    assert_eq!(name, "gemm_q4k");
    assert_eq!(size, 1024);
    assert!(roofline);
    assert_eq!(metrics.as_deref(), Some("sm__cycles_elapsed"));
}

/// `apr db serve` requires `--config`. It was the standalone binary's only
/// argument and it was required there; a defaulted config would silently serve
/// from a directory the operator never named.
#[test]
fn db_serve_without_config_is_refused() {
    let err = parse_cli(vec!["apr", "db", "serve"])
        .expect_err("`apr db serve` with no --config must be refused");
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
}

/// `apr ptx-debug analyze` carries `--min-score`, `--html` and `--json`.
#[test]
fn ptx_debug_analyze_arguments_reach_the_command() {
    let cli = parse_cli(vec![
        "apr",
        "ptx-debug",
        "analyze",
        "kernel.ptx",
        "--falsify",
        "--min-score",
        "95",
        "--html",
        "report.html",
        "--json",
    ])
    .expect("`apr ptx-debug analyze` with every flag must parse");

    let Commands::Extended(ExtendedCommands::PtxDebug {
        command:
            ::trueno_ptx_debug::cli::PtxDebugCommand::Analyze {
                file,
                falsify,
                min_score,
                html,
                json,
            },
    }) = *cli.command
    else {
        panic!("expected `apr ptx-debug analyze`");
    };
    assert_eq!(file, PathBuf::from("kernel.ptx"));
    assert!(falsify);
    assert!((min_score - 95.0).abs() < f64::EPSILON);
    assert_eq!(html, Some(PathBuf::from("report.html")));
    assert!(json);
}

/// `apr ptx-debug gen-fkr` keeps the standalone binary's `-o` short flag.
#[test]
fn ptx_debug_gen_fkr_keeps_short_output_flag() {
    let cli = parse_cli(vec!["apr", "ptx-debug", "gen-fkr", "k.ptx", "-o", "t.rs"])
        .expect("`apr ptx-debug gen-fkr -o` must parse");

    let Commands::Extended(ExtendedCommands::PtxDebug {
        command: ::trueno_ptx_debug::cli::PtxDebugCommand::GenFkr { file, output },
    }) = *cli.command
    else {
        panic!("expected `apr ptx-debug gen-fkr`");
    };
    assert_eq!(file, PathBuf::from("k.ptx"));
    assert_eq!(output, Some(PathBuf::from("t.rs")));
}

/// `apr corpus-ingest` keeps both subcommands and both of `plan`'s defaults.
#[test]
fn corpus_ingest_subcommands_and_defaults_survive() {
    let cli = parse_cli(vec!["apr", "corpus-ingest", "plan"])
        .expect("`apr corpus-ingest plan` must parse");
    let Commands::Extended(ExtendedCommands::CorpusIngest {
        command: CorpusIngestCommands::Plan {
            contract,
            output_dir,
        },
    }) = *cli.command
    else {
        panic!("expected `apr corpus-ingest plan`");
    };
    assert_eq!(
        contract,
        PathBuf::from(commands::corpus_ingest::DEFAULT_CONTRACT_PATH)
    );
    assert_eq!(
        output_dir,
        PathBuf::from(commands::corpus_ingest::DEFAULT_OUTPUT_DIR)
    );

    let cli = parse_cli(vec!["apr", "corpus-ingest", "validate-contract", "c.yaml"])
        .expect("`apr corpus-ingest validate-contract` must parse");
    let Commands::Extended(ExtendedCommands::CorpusIngest {
        command: CorpusIngestCommands::ValidateContract { path },
    }) = *cli.command
    else {
        panic!("expected `apr corpus-ingest validate-contract`");
    };
    assert_eq!(path, PathBuf::from("c.yaml"));
}

/// `apr corpus-ingest validate-contract` needs its positional path; without it
/// the command would have to invent one.
#[test]
fn corpus_ingest_validate_contract_without_a_path_is_refused() {
    let err = parse_cli(vec!["apr", "corpus-ingest", "validate-contract"])
        .expect_err("validate-contract with no path must be refused");
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
}

/// Every subcommand the standalone `probador` binary shipped is reachable as
/// `apr probar <cmd>`, alongside apr's own `tensor`.
#[test]
fn every_probador_subcommand_is_reachable_under_apr_probar() {
    // `tensor` is apr's own; the rest are probador's, flattened in.
    assert!(parse_cli(vec!["apr", "probar", "tensor", "m.apr"]).is_ok());

    // Each entry is a minimal INVOCATION, not just a name: `record` and the
    // media checks take required positionals, so a bare name would prove only
    // that clap knows the word, not that the subcommand is usable.
    for argv in [
        vec!["apr", "probar", "test"],
        vec!["apr", "probar", "record", "my_test"],
        vec!["apr", "probar", "report"],
        vec!["apr", "probar", "coverage"],
        vec!["apr", "probar", "init"],
        vec!["apr", "probar", "config"],
        vec!["apr", "probar", "serve"],
        vec!["apr", "probar", "build"],
        vec!["apr", "probar", "watch"],
        vec!["apr", "probar", "playbook", "pb.yaml"],
        vec!["apr", "probar", "comply"],
        vec!["apr", "probar", "av-sync", "check", "v.mp4"],
        vec!["apr", "probar", "audio", "check", "v.mp4"],
        vec!["apr", "probar", "video", "check", "v.mp4"],
        // `animation check` takes TWO positionals: timeline and observed.
        vec![
            "apr",
            "probar",
            "animation",
            "check",
            "timeline.json",
            "observed.json",
        ],
        vec!["apr", "probar", "stress"],
        // `llm test` requires both --config and --url.
        vec![
            "apr",
            "probar",
            "llm",
            "test",
            "--config",
            "llm.yaml",
            "--url",
            "http://localhost:8080",
        ],
    ] {
        let label = argv.join(" ");
        assert!(
            parse_cli(argv).is_ok(),
            "`{label}` must parse — it was a probador subcommand"
        );
    }
}

/// `apr probar coverage` must not explode on apr's global `--json`, and its
/// own JSON-file output must still be reachable.
///
/// probador spelled the latter `--json <FILE>`; apr's root declares a GLOBAL
/// `--json: bool`, and a global propagates onto every subcommand. Both then
/// claim the id `json` with different types, which clap does NOT catch in
/// `debug_assert` — it panics at PARSE time. So `apr probar coverage` aborted
/// on every invocation until the file flag was respelled `--json-out`.
#[test]
fn probar_coverage_json_flags_do_not_collide() {
    // apr's global boolean must parse here without panicking.
    let cli = parse_cli(vec!["apr", "probar", "coverage", "--json"])
        .expect("apr's global --json must parse under `probar coverage`");
    assert!(cli.json, "apr's global --json must be the boolean");

    // And probador's file output must still be reachable.
    let cli = parse_cli(vec![
        "apr",
        "probar",
        "coverage",
        "--json-out",
        "cov.json",
    ])
    .expect("`--json-out FILE` must parse");
    let Commands::Extended(ExtendedCommands::Probar {
        command: ProbarSubcommand::Probador(probador::Commands::Coverage(args)),
        ..
    }) = *cli.command
    else {
        panic!("expected `apr probar coverage`");
    };
    assert_eq!(
        args.json,
        Some(PathBuf::from("cov.json")),
        "--json-out must carry the coverage JSON path"
    );
}

/// `apr probar comply migrate --version <V>` must reach the handler.
///
/// The root sets `propagate_version = true`, which pushes clap's auto
/// `--version` onto every subcommand and collided with this real argument.
/// The collision shipped undetected in the standalone binary; assert the
/// argument works so the fix cannot silently regress.
#[test]
fn probar_comply_migrate_keeps_its_own_version_argument() {
    let cli = parse_cli(vec![
        "apr",
        "probar",
        "comply",
        "migrate",
        "--version",
        "2.0",
    ])
    .expect("`apr probar comply migrate --version 2.0` must parse");

    let Commands::Extended(ExtendedCommands::Probar {
        command: ProbarSubcommand::Probador(probador::Commands::Comply(args)),
        ..
    }) = *cli.command
    else {
        panic!("expected `apr probar comply`");
    };
    let Some(probador::ComplySubcommand::Migrate(migrate)) = args.subcommand else {
        panic!("expected `comply migrate`");
    };
    assert_eq!(
        migrate.version.as_deref(),
        Some("2.0"),
        "--version must carry the migration target, not be eaten by clap's \
         auto-generated version flag"
    );
}

/// A probador subcommand must parse into the flattened variant, not into
/// apr's own `tensor`. Flattening that silently collapsed everything into one
/// variant would pass the reachability test above while losing the arguments.
#[test]
fn probador_subcommands_parse_into_the_flattened_variant() {
    let cli = parse_cli(vec!["apr", "probar", "test"]).expect("`apr probar test` must parse");
    let Commands::Extended(ExtendedCommands::Probar { command, .. }) = *cli.command else {
        panic!("expected `apr probar`");
    };
    assert!(
        matches!(command, ProbarSubcommand::Probador(_)),
        "`apr probar test` must land in the probador variant, not apr's `tensor`"
    );
}

/// `probador`'s `--color` was a global flag on ITS root command, not on any
/// subcommand, so flattening only the subcommands would have dropped it.
/// It must still be reachable, on every subcommand, with all three values.
#[test]
fn probador_global_color_flag_survives_the_flatten() {
    for (value, expected) in [
        ("auto", probador::ColorArg::Auto),
        ("always", probador::ColorArg::Always),
        ("never", probador::ColorArg::Never),
    ] {
        let cli = parse_cli(vec!["apr", "probar", "test", "--color", value])
            .unwrap_or_else(|e| panic!("`apr probar test --color {value}` must parse: {e}"));
        let Commands::Extended(ExtendedCommands::Probar { color, .. }) = *cli.command else {
            panic!("expected `apr probar`");
        };
        assert_eq!(
            std::mem::discriminant(&color),
            std::mem::discriminant(&expected),
            "--color {value} must reach the command as {expected:?}"
        );
    }

    // It is global, so it also attaches to apr's own `tensor` subcommand.
    assert!(
        parse_cli(vec!["apr", "probar", "tensor", "m.apr", "--color", "never"]).is_ok(),
        "--color must be global across `apr probar` subcommands"
    );

    // And an unknown value is refused rather than silently falling back.
    assert!(
        parse_cli(vec!["apr", "probar", "test", "--color", "chartreuse"]).is_err(),
        "an unknown --color value must be refused"
    );
}

/// An unknown `apr probar` subcommand is refused rather than silently treated
/// as one of the known ones.
#[test]
fn unknown_probar_subcommand_is_refused() {
    let err = parse_cli(vec!["apr", "probar", "no-such-subcommand"])
        .expect_err("an unknown probar subcommand must be refused");
    assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
}

/// `apr compute` and `apr cbtop` are DIFFERENT tools that share an acronym.
/// `--model-path` belongs to `apr cbtop` alone; `--workload` to `apr compute`
/// alone. Each must reject the other's flag, or the two have been conflated.
#[test]
fn compute_and_cbtop_do_not_share_an_argument_surface() {
    assert!(
        parse_cli(vec!["apr", "cbtop", "--model-path", "m.gguf"]).is_ok(),
        "`apr cbtop --model-path` is the ComputeBrick tool's own flag"
    );
    assert!(
        parse_cli(vec!["apr", "compute", "top", "--model-path", "m.gguf"]).is_err(),
        "`apr compute` must not accept `apr cbtop`'s --model-path"
    );
    assert!(
        parse_cli(vec!["apr", "compute", "top", "--workload", "gemm"]).is_ok(),
        "`apr compute top --workload` is the compute-backend tool's own flag"
    );
    assert!(
        parse_cli(vec!["apr", "cbtop", "--workload", "gemm"]).is_err(),
        "`apr cbtop` must not accept `apr compute`'s --workload"
    );
}
