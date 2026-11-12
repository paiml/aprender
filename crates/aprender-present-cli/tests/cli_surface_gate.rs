//! ONT-10 surface gate for the `aprender-present` binary: `contracts/bin-aprender-present-cli--aprender-present-v1.yaml` (aprender#4079).
//!
//! Every assertion runs the binary built from this tree (`CARGO_BIN_EXE_aprender-present`), which
//! cannot fall back to a stale copy on PATH. A test that only asserted success on good
//! input would pass against `fn main() {}`, so each invariant here is one the binary
//! can actually fail: a rejected input must exit non-zero AND must not be a panic.

use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_aprender-present");

/// An argument list the parser must refuse as an unknown subcommand or extra positional.
const UNKNOWN_SUB: &[&str] = &["no-such-subcommand"];

/// The pinned surface. Adding or removing a subcommand edits this list and the
/// contract's FALSIFY-BIN-*-004 prediction together.
const SUBCOMMANDS: &[&str] = &[
    "serve", "bundle", "new", "check", "score", "gate", "deploy", "plot",
];

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn the binary under test")
}

fn help_section(header: &str) -> Vec<String> {
    // `-h` is clap's one-line-per-entry form; `--help` may be the multi-line long form.
    let out = run(&["-h"]);
    assert!(out.status.success(), "-h must exit 0");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip_while(|l| !l.starts_with(header))
        .skip(1)
        .take_while(|l| !l.is_empty() && !l.ends_with(':'))
        .map(str::to_owned)
        .collect()
}

fn advertised() -> Vec<String> {
    help_section("Commands:")
        .iter()
        .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
        .filter(|c| c != "help")
        .collect()
}

/// Three inputs no command can process: a missing path, an empty file, garbage bytes.
fn unusable_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("aprender-present-surface-gate");
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let empty = dir.join("empty.bin");
    let garbage = dir.join("garbage.bin");
    std::fs::write(&empty, b"").expect("write empty fixture");
    std::fs::write(&garbage, b"not a model {{{\x00\x01\xff").expect("write garbage fixture");
    vec![
        dir.join("does-not-exist").display().to_string(),
        empty.display().to_string(),
        garbage.display().to_string(),
    ]
}

fn assert_rejected(args: &[&str]) {
    let out = run(args);
    let code = out.status.code();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        matches!(code, Some(c) if c != 0 && c != 101) && !stderr.contains("panicked"),
        "`aprender-present {}` must reject its input with a clean non-zero exit, got {code:?}\nstderr: {stderr}",
        args.join(" ")
    );
}

fn assert_usage_error(args: &[&str]) {
    let out = run(args);
    assert_eq!(
        out.status.code(),
        Some(2),
        "`aprender-present {}` must be a usage error (exit 2)\nstderr: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn version_reports_the_crate_version() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = format!("aprender-present {}", env!("CARGO_PKG_VERSION"));
    assert!(
        stdout.starts_with(&want),
        "--version printed {stdout:?}, not {want:?}"
    );
}

#[test]
fn advertised_subcommands_are_the_pinned_surface() {
    assert_eq!(advertised(), SUBCOMMANDS);
}

#[test]
fn every_advertised_subcommand_is_reachable() {
    let cmds = advertised();
    assert!(
        !cmds.is_empty(),
        "--help advertised no subcommands; the parser is broken"
    );
    for cmd in &cmds {
        let out = run(&[cmd, "--help"]);
        assert!(
            out.status.success(),
            "`aprender-present {cmd} --help` exited {:?}",
            out.status.code()
        );
    }
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    assert_usage_error(UNKNOWN_SUB);
}

#[test]
fn unknown_flag_is_a_usage_error() {
    assert_usage_error(&["--no-such-flag"]);
}

/// Usage errors the contract names: missing required values and values outside an
/// enumerated set must be refused by the parser, not reach the command.
#[test]
fn malformed_arguments_are_usage_errors() {
    let bad = unusable_inputs();
    let b = &bad[0];
    assert_usage_error(&["score", b.as_str(), "--no-such-flag"]);
    assert_usage_error(&["plot"]);
    assert_usage_error(&["plot", "--data", b.as_str()]);
    assert_usage_error(&["plot", "--spec", b.as_str(), "--data", b.as_str()]);
    assert_usage_error(&[
        "plot",
        "--data",
        b.as_str(),
        "--aes",
        "x=x,y=y",
        "--output",
        "png",
    ]);
}

#[test]
fn unusable_input_is_rejected() {
    for b in &unusable_inputs() {
        assert_rejected(&["check", b.as_str()]);
        assert_rejected(&["score", b.as_str()]);
        assert_rejected(&["gate", b.as_str()]);
        assert_rejected(&["plot", "--data", b.as_str(), "--aes", "x=x,y=y"]);
        assert_rejected(&["plot", "--spec", b.as_str()]);
    }
}

// ── plot (PLOT-GG-001, aprender#6909) ────────────────────────────────────────────

/// Row 1's ggplot2 fixture data and the break labels ggplot2 4.0.3 computed for it.
const ROW1_CSV: &str = include_str!("../../aprender-viz/fixtures/ggplot2/geom_point/data.csv");
const ROW1_EXPECTED: &str =
    include_str!("../../aprender-viz/fixtures/ggplot2/geom_point/expected.json");
const ORACLE_X_LABELS: &[&str] = &["5", "10"];
const ORACLE_Y_LABELS: &[&str] = &["-2.5", "0.0", "2.5", "5.0", "7.5"];
/// `aprender-present plot --data data.csv --aes x=x,y=y` at 80x24 with stdout not a terminal.
const ROW1_GOLDEN: &str = include_str!("golden/plot-row1-point-80x24.txt");

fn row1_csv() -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("aprender-present-plot");
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let path = dir.join("row1.csv");
    std::fs::write(&path, format!("# ggplot2 row 1 fixture\n{ROW1_CSV}")).expect("write csv");
    path.display().to_string()
}

#[test]
fn oracle_labels_are_the_fixture_labels() {
    let quoted = |ls: &[&str]| {
        ls.iter()
            .map(|l| format!("\"{l}\""))
            .collect::<Vec<_>>()
            .join(", ")
    };
    for labels in [ORACLE_X_LABELS, ORACLE_Y_LABELS] {
        let want = format!("\"labels\": [{}]", quoted(labels));
        assert!(ROW1_EXPECTED.contains(&want), "expected.json lacks {want}");
    }
}

#[test]
fn plot_row1_matches_the_golden_render() {
    let csv = row1_csv();
    let out = Command::new(BIN)
        .args([
            "plot", "--data", &csv, "--aes", "x=x,y=y", "--geom", "point",
        ])
        .env_remove("NO_COLOR")
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(stdout, ROW1_GOLDEN);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 24);
    assert!(
        !stdout.contains('\x1b'),
        "piped output must carry no colour"
    );
    assert_eq!(
        lines[23].split_whitespace().collect::<Vec<_>>(),
        ORACLE_X_LABELS
    );
    let mut y: Vec<&str> = lines
        .iter()
        .filter_map(|l| l.split_once('┤').map(|(label, _)| label.trim()))
        .collect();
    y.reverse();
    assert_eq!(y, ORACLE_Y_LABELS);
    assert_eq!(stdout.matches('●').count(), 10);
}

#[test]
fn plot_svg_carries_the_oracle_break_labels() {
    let csv = row1_csv();
    let out = run(&[
        "plot", "--data", &csv, "--aes", "x=x,y=y", "--output", "svg",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let svg = String::from_utf8_lossy(&out.stdout);
    for label in ORACLE_X_LABELS.iter().chain(ORACLE_Y_LABELS) {
        assert!(
            svg.contains(&format!(">{label}</text>")),
            "SVG lacks break label {label}"
        );
    }
}

#[test]
fn plot_refuses_what_it_cannot_draw() {
    let csv = row1_csv();
    for (args, want) in [
        (
            vec!["--aes", "x=x,y=y", "--geom", "line"],
            "unsupported geom 'line'; supported: point",
        ),
        (
            vec!["--aes", "x=x,y=nope"],
            "unknown column 'nope'; columns: x, y",
        ),
    ] {
        let mut argv = vec!["plot", "--data", csv.as_str()];
        argv.extend(args);
        assert_rejected(&argv);
        let stderr = String::from_utf8_lossy(&run(&argv).stderr).into_owned();
        assert!(stderr.contains(want), "stderr {stderr:?} lacks {want:?}");
    }
}
