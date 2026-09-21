//! #3745 S1: the surface emitter, read back.

use super::*;
use clap::{Arg, ArgAction};

/// Run `f` on a thread with the walk's stack. The full `apr` tree does not fit
/// a default test-thread stack.
fn big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(WALK_STACK)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("join")
}

/// S1.1: a flag added to the clap tree appears in the JSON, under its command
/// path, with a role. The emitter reads the tree it is given; it keeps no list
/// that a new flag could be missing from.
#[test]
fn the_emitter_sees_a_flag_added_to_the_clap_tree() {
    let (before, after) = big_stack(|| {
        let root = Cli::command();
        let before = emit_from(&root);
        let mutated = root.mut_subcommand("run", |run| {
            run.arg(
                Arg::new("zz-dummy")
                    .long("zz-dummy")
                    .action(ArgAction::SetTrue),
            )
        });
        (before, emit_from(&mutated))
    });
    let run_before = before.command(&["run"]).expect("run is on the surface");
    assert!(
        run_before.arg("zz-dummy").is_none(),
        "the dummy flag must be absent before it is added"
    );

    let run = after.command(&["run"]).expect("run is on the surface");
    let dummy = run
        .arg("zz-dummy")
        .expect("a flag added to the clap tree must appear in the surface");
    assert_eq!(dummy.long.as_deref(), Some("zz-dummy"));
    assert_eq!(dummy.value_type, "flag");
    assert_eq!(dummy.role, "mode");
    let json = after.to_json();
    assert!(
        json.contains("\"zz-dummy\""),
        "the dummy flag must be in the JSON text too"
    );
}

#[test]
fn the_surface_is_versioned_and_deterministic() {
    let (a, b) = big_stack(|| (emit().to_json(), emit().to_json()));
    assert_eq!(a, b, "two emissions from one binary must be byte-identical");
    let v: serde_json::Value = serde_json::from_str(&a).expect("the surface is JSON");
    assert_eq!(v["schema"], SCHEMA);
    assert_eq!(v["schema"], "apr-cli-surface/v1.1");
    assert_eq!(v["binary"]["name"], "apr");
    let roles: Vec<&str> = v["roles"]
        .as_array()
        .expect("roles")
        .iter()
        .filter_map(|r| r.as_str())
        .collect();
    assert_eq!(
        roles,
        [
            "model",
            "prompt",
            "input-file",
            "backend",
            "mode",
            "other",
            "unknown"
        ]
    );
    let paths: Vec<Vec<String>> = v["commands"]
        .as_array()
        .expect("commands")
        .iter()
        .map(|c| serde_json::from_value(c["path"].clone()).expect("path is a string array"))
        .collect();
    let mut sorted = paths.clone();
    sorted.sort();
    assert_eq!(paths, sorted, "commands are sorted by path");
}

/// clap's generated `help` subcommand and `--help`/`--version` flags are not
/// apr's declared surface; emitting them would owe a release cell per group.
#[test]
fn clap_generated_help_and_version_are_not_emitted() {
    let s = big_stack(emit);
    // `help` is never an apr-declared id. `version` can be (`data x registry
    // delete --version` names a registry entry's version), so the generated
    // `--version` is checked where apr declares no such flag: the root and `run`.
    for c in &s.commands {
        assert!(
            c.args.iter().all(|a| a.id != "help"),
            "`{} --help` is clap-generated",
            c.key
        );
    }
    assert!(s
        .global_args
        .iter()
        .all(|a| a.id != "help" && a.id != "version"));
    assert!(
        s.command(&["run"]).expect("run").arg("version").is_none(),
        "propagated --version"
    );
    // `serve` has subcommands and does not disable clap's help subcommand.
    let serve = s.command(&["serve"]).expect("serve");
    assert!(!serve.leaf);
    assert!(
        s.command(&["serve", "help"]).is_none(),
        "clap's generated `serve help` must not be emitted"
    );
}

#[test]
fn globals_are_emitted_once_at_the_root() {
    let s = big_stack(emit);
    let ids: Vec<&str> = s.global_args.iter().map(|a| a.id.as_str()).collect();
    assert!(
        ids.contains(&"json") && ids.contains(&"quiet") && ids.contains(&"verbose"),
        "{ids:?}"
    );
    let run = s.command(&["run"]).expect("run");
    assert!(
        run.arg("json").is_none(),
        "a propagated global is not repeated per command"
    );
}

#[test]
fn the_backend_role_comes_from_the_backend_arg_group() {
    let s = big_stack(emit);
    for path in [&["run"][..], &["chat"][..], &["serve", "run"][..]] {
        let c = s
            .command(path)
            .unwrap_or_else(|| panic!("{path:?} is on the surface"));
        let b = c
            .arg("backend")
            .unwrap_or_else(|| panic!("{path:?} has --backend"));
        assert_eq!(b.role, "backend", "{path:?} --backend");
        assert_eq!(b.marker, Some("BackendArg"));
        assert_eq!(
            b.values,
            crate::BACKEND_VALUES.to_vec(),
            "{path:?} --backend values"
        );
    }
}

/// The `surface` verb itself is hidden from `--help` but present on the surface,
/// so S3 derives it as a verb and S2 owes it a runs cell.
#[test]
fn the_surface_verb_is_hidden_and_listed() {
    let s = big_stack(emit);
    let c = s.command(&["surface"]).expect("surface lists itself");
    assert!(c.hidden);
    assert!(c.leaf);
    assert!(!c.foreign);
}

/// Every foreign CLI the walker lists is found mounted exactly once, and every
/// command under a mount is marked foreign (the mount point included).
#[test]
fn foreign_mounts_are_found_by_type() {
    let s = big_stack(emit);
    for mount in [
        &["pv"][..],
        &["data", "x"][..],
        &["sim"][..],
        &["rag"][..],
        &["zram"][..],
        &["cgp"][..],
    ] {
        let c = s
            .command(mount)
            .unwrap_or_else(|| panic!("{mount:?} is on the surface"));
        assert!(c.foreign, "{mount:?} is a foreign mount");
        for sub in &c.subcommands {
            let mut p: Vec<&str> = mount.to_vec();
            p.push(sub);
            assert!(
                s.command(&p).is_some_and(|e| e.foreign),
                "{p:?} is under a foreign mount"
            );
        }
    }
    let data = s.command(&["data"]).expect("data");
    assert!(
        !data.foreign,
        "`data` is apr-cli's own; only its `x` arm is alimentar's"
    );
    assert!(!s.command(&["run"]).expect("run").foreign);
}

// ── classification case table ────────────────────────────────────────────────

/// One row per way an argument can be built. `expect` is (value_type, role,
/// marker). The table is the specification of `classify`: a class of argument
/// with no row here is a class nobody decided.
#[test]
fn classify_case_table() {
    use batuta_common::cli_roles::{
        ConfigPath, DirPath, EncodeText, FreeText, InputFile, ModelPath, ModelRef, OutputPath,
        PromptText,
    };
    #[derive(clap::Parser)]
    struct Probe {
        model_path: ModelPath,
        #[arg(long)]
        model_ref: Option<ModelRef>,
        #[arg(long)]
        prompt: Option<PromptText>,
        #[arg(long)]
        input: Option<InputFile>,
        #[arg(long)]
        output: Option<OutputPath>,
        #[arg(long)]
        dir: Option<DirPath>,
        #[arg(long)]
        config: Option<ConfigPath>,
        #[arg(long)]
        name: Option<FreeText>,
        #[arg(long)]
        encoded: Option<EncodeText>,
        #[arg(long)]
        raw_path: Option<PathBuf>,
        #[arg(long)]
        raw_text: Option<String>,
        #[arg(long, value_parser = ["a", "b"])]
        listed: Option<String>,
        #[arg(long)]
        flag: bool,
        #[arg(long, action = ArgAction::Count)]
        level: u8,
        #[arg(long)]
        n: Option<usize>,
        #[arg(long)]
        t: Option<f32>,
        #[arg(long, value_enum)]
        fmt: Option<crate::CodeOutputFormat>,
        #[arg(long)]
        c: Option<char>,
        #[command(flatten)]
        backend: BackendArg,
    }
    let rows: &[(&str, (&str, &str, Option<&str>))] = &[
        ("model_path", ("path", "model", Some("ModelPath"))),
        ("model_ref", ("text", "model", Some("ModelRef"))),
        ("prompt", ("text", "prompt", Some("PromptText"))),
        ("input", ("path", "input-file", Some("InputFile"))),
        ("output", ("path", "other", Some("OutputPath"))),
        ("dir", ("path", "other", Some("DirPath"))),
        ("config", ("path", "other", Some("ConfigPath"))),
        ("name", ("text", "other", Some("FreeText"))),
        // Encoder/scorer input: a model input, but not a prompt (v1.1).
        ("encoded", ("text", "other", Some("EncodeText"))),
        // The two rows the marker guard exists for.
        ("raw_path", ("path", "unknown", None)),
        ("raw_text", ("text", "unknown", None)),
        ("listed", ("enum", "mode", None)),
        ("flag", ("flag", "mode", None)),
        ("level", ("count", "mode", None)),
        ("n", ("int", "other", None)),
        ("t", ("float", "other", None)),
        ("fmt", ("enum", "mode", None)),
        ("c", ("other", "other", None)),
        ("backend", ("enum", "backend", Some("BackendArg"))),
    ];
    let cmd = <Probe as CommandFactory>::command();
    let root = clap::Command::new("apr").subcommand(cmd.name("probe"));
    let s = emit_from(&root);
    let probe = s.command(&["probe"]).expect("probe");
    assert_eq!(
        probe.args.len(),
        rows.len(),
        "every arg of the probe has a row"
    );
    for (id, (value_type, role, marker)) in rows {
        let a = probe.arg(id).unwrap_or_else(|| panic!("probe has {id}"));
        assert_eq!(
            (a.value_type, a.role, a.marker),
            (*value_type, *role, *marker),
            "row {id}"
        );
    }
}

/// The verb writes [`emit`]'s document and nothing else. The binary-level pin
/// (the spawned `apr surface --json` equals `emit()`) is the integration test
/// `tests/surface_binary_pin.rs`.
#[test]
fn the_verb_writes_exactly_what_emit_returns() {
    let (written, emitted) = big_stack(|| {
        let mut buf = Vec::new();
        write_surface(&mut buf).expect("write to a Vec");
        (buf, emit().to_json())
    });
    assert_eq!(written, format!("{emitted}\n").into_bytes());
}

/// `apr surface` parses to the hidden verb, with and without `--json`/`--quiet`.
#[test]
fn apr_surface_parses_to_the_surface_verb() {
    use clap::Parser;
    for argv in [
        &["apr", "surface"][..],
        &["apr", "surface", "--json"][..],
        &["apr", "surface", "--quiet"][..],
    ] {
        let cli = big_stack({
            let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
            move || {
                Cli::try_parse_from(argv).map(|c| matches!(*c.command, crate::Commands::Surface))
            }
        });
        assert_eq!(
            cli.ok(),
            Some(true),
            "{argv:?} must parse to Commands::Surface"
        );
    }
}

/// v1.1 `generates`: one row per way a command can be built. A command
/// generates iff it has a `PromptText` argument or carries the
/// `ServesGeneration` marker; a name — the arg's or the group's — never counts.
#[test]
fn generates_case_table() {
    use batuta_common::cli_roles::{ModelPath, PromptText, ServesGeneration};
    let model = || Arg::new("model").value_parser(clap::value_parser!(ModelPath));
    let prompt_arg =
        |p: clap::builder::ValueParser| Arg::new("prompt").long("prompt").value_parser(p);
    let root = clap::Command::new("apr")
        .subcommand(
            clap::Command::new("with-prompt")
                .arg(model())
                .arg(prompt_arg(clap::value_parser!(PromptText).into())),
        )
        .subcommand(
            clap::Command::new("marked")
                .arg(model())
                .group(ServesGeneration::group()),
        )
        .subcommand(clap::Command::new("model-only").arg(model()))
        .subcommand(
            clap::Command::new("lookalike-group")
                .arg(model())
                .group(clap::ArgGroup::new("ServesGeneration").multiple(true)),
        )
        .subcommand(
            clap::Command::new("raw-prompt-name")
                .arg(model())
                .arg(prompt_arg(clap::value_parser!(String))),
        );
    let s = emit_from(&root);
    let rows = [
        ("with-prompt", true),
        ("marked", true),
        ("model-only", false),
        ("lookalike-group", false),
        ("raw-prompt-name", false),
    ];
    assert_eq!(s.commands.len(), rows.len(), "every command has a row");
    for (name, want) in rows {
        let c = s.command(&[name]).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(c.generates, want, "row {name}");
    }
}
