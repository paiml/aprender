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

/// Write row 1 under a directory named for the calling test. Tests run in parallel, and one
/// shared path let a test read the file while another truncated it ("no header row").
fn row1_csv(test: &str) -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("aprender-present-plot")
        .join(test);
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
    let csv = row1_csv("golden");
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
    let csv = row1_csv("svg");
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
    let csv = row1_csv("refuses");
    for (args, want) in [
        (
            vec!["--aes", "x=x,y=y", "--geom", "bar"],
            "unsupported geom 'bar'; supported: point, line, path, step, hline, vline, abline",
        ),
        // Exit 1 with the bounds, where it used to abort (exit 134) on an 80 GB allocation.
        (
            vec![
                "--aes", "x=x,y=y", "--width", "100000", "--height", "100000",
            ],
            "terminal plot size must be 20x8 to 1000x1000 cells, got 100000x100000",
        ),
        (
            vec!["--aes", "x=x,y=y", "--output", "svg", "--width", "100000"],
            "svg plot size must be 1x1 to 4096x4096 pixels, got 100000x600",
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

// ── plot: the line family and the reference lines (PLOT-GG-001, aprender#6909) ────────────

/// Write `body` to `<tmp>/<test>/<name>` and return the path.
fn fixture(test: &str, name: &str, body: &str) -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("aprender-present-plot-lines")
        .join(test);
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write fixture");
    path.display().to_string()
}

/// The diagonal 0,0 / 5,5 / 10,10.
const DIAG: &str = "x,y\n0,0\n5,5\n10,10\n";

fn svg_of(args: &[&str]) -> String {
    let mut argv = vec!["plot", "--output", "svg"];
    argv.extend_from_slice(args);
    let out = run(&argv);
    assert!(
        out.status.success(),
        "{argv:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn attr(line: &str, name: &str) -> f64 {
    let key = format!("{name}=\"");
    let rest = &line[line.find(&key).expect("attribute present") + key.len()..];
    rest[..rest.find('"').expect("closing quote")]
        .parse()
        .expect("numeric attribute")
}

/// Every `<polyline>` of the SVG, as pixel vertices.
fn polylines(svg: &str) -> Vec<Vec<(f64, f64)>> {
    svg.lines()
        .filter(|l| l.trim_start().starts_with("<polyline"))
        .map(|l| {
            let pts = &l[l.find("points=\"").expect("points") + 8..];
            pts[..pts.find('"').expect("quote")]
                .split(' ')
                .map(|p| {
                    let (x, y) = p.split_once(',').expect("x,y");
                    (x.parse().expect("x"), y.parse().expect("y"))
                })
                .collect()
        })
        .collect()
}

/// The pixel of the axis tick labelled `label`: a `<line>` followed by its `<text>`. `x` picks
/// the horizontal axis (ticks below the panel, `y2 > y1` and a vertical tick).
fn tick_px(svg: &str, label: &str, x: bool) -> f64 {
    let lines: Vec<&str> = svg.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if l.contains(&format!(">{label}</text>")) && i > 0 {
            let prev = lines[i - 1];
            let horizontal = (attr(prev, "x1") - attr(prev, "x2")).abs() < 1e-9;
            if horizontal == x {
                return attr(prev, if x { "x1" } else { "y1" });
            }
        }
    }
    panic!("no {} tick labelled {label}", if x { "x" } else { "y" });
}

/// Data -> pixel on the plot's own axis, from its 0.0 and 10.0 ticks (independent of the renderer's
/// scale code).
struct Axes {
    x0: f64,
    x10: f64,
    y0: f64,
    y10: f64,
}

impl Axes {
    fn of(svg: &str) -> Self {
        Self {
            x0: tick_px(svg, "0.0", true),
            x10: tick_px(svg, "10.0", true),
            y0: tick_px(svg, "0.0", false),
            y10: tick_px(svg, "10.0", false),
        }
    }
    fn px(&self, v: f64) -> f64 {
        self.x0 + v * (self.x10 - self.x0) / 10.0
    }
    fn py(&self, v: f64) -> f64 {
        self.y0 + v * (self.y10 - self.y0) / 10.0
    }
    fn dx(&self, p: f64) -> f64 {
        (p - self.x0) * 10.0 / (self.x10 - self.x0)
    }
    fn dy(&self, p: f64) -> f64 {
        (p - self.y0) * 10.0 / (self.y10 - self.y0)
    }
}

fn assert_path(got: &[(f64, f64)], want: &[(f64, f64)], ax: &Axes) {
    assert_eq!(got.len(), want.len(), "vertex count: {got:?}");
    for (g, w) in got.iter().zip(want) {
        let (ex, ey) = (ax.px(w.0), ax.py(w.1));
        assert!(
            (g.0 - ex).abs() < 0.01 && (g.1 - ey).abs() < 0.01,
            "vertex {g:?} != ({ex}, {ey}) for data {w:?}"
        );
    }
}

#[test]
fn plot_line_svg_draws_the_data_at_axis_pixels() {
    let csv = fixture("line", "d.csv", DIAG);
    let svg = svg_of(&["--data", &csv, "--aes", "x=x,y=y", "--geom", "line"]);
    let lines = polylines(&svg);
    assert_eq!(lines.len(), 1);
    assert_path(
        &lines[0],
        &[(0., 0.), (5., 5.), (10., 10.)],
        &Axes::of(&svg),
    );
    assert!(svg.contains("stroke-width=\"1.423\""), "linewidth 0.5mm");
}

#[test]
fn plot_path_keeps_row_order_and_line_sorts_by_x() {
    let csv = fixture("order", "u.csv", "x,y\n5,5\n0,0\n10,10\n");
    let path = svg_of(&["--data", &csv, "--aes", "x=x,y=y", "--geom", "path"]);
    assert_path(
        &polylines(&path)[0],
        &[(5., 5.), (0., 0.), (10., 10.)],
        &Axes::of(&path),
    );
    let line = svg_of(&["--data", &csv, "--aes", "x=x,y=y", "--geom", "line"]);
    assert_path(
        &polylines(&line)[0],
        &[(0., 0.), (5., 5.), (10., 10.)],
        &Axes::of(&line),
    );
}

#[test]
fn plot_step_directions_match_the_staircases() {
    let csv = fixture("step", "d.csv", DIAG);
    for (dir, want) in [
        (
            None,
            vec![(0., 0.), (5., 0.), (5., 5.), (10., 5.), (10., 10.)],
        ),
        (
            Some("hv"),
            vec![(0., 0.), (5., 0.), (5., 5.), (10., 5.), (10., 10.)],
        ),
        (
            Some("vh"),
            vec![(0., 0.), (0., 5.), (5., 5.), (5., 10.), (10., 10.)],
        ),
        (
            Some("mid"),
            vec![
                (0., 0.),
                (2.5, 0.),
                (2.5, 5.),
                (7.5, 5.),
                (7.5, 10.),
                (10., 10.),
            ],
        ),
    ] {
        let mut args = vec!["--data", csv.as_str(), "--aes", "x=x,y=y", "--geom", "step"];
        if let Some(d) = dir {
            args.extend(["--direction", d]);
        }
        let svg = svg_of(&args);
        assert_path(&polylines(&svg)[0], &want, &Axes::of(&svg));
    }
}

#[test]
fn plot_reference_lines_sit_at_their_values() {
    let csv = fixture("refs", "d.csv", DIAG);
    let svg = svg_of(&[
        "--data", &csv, "--aes", "x=x,y=y", "--geom", "line", "--hline", "2", "--vline", "3",
        "--abline", "0.5,1",
    ]);
    let ax = Axes::of(&svg);
    let lines = polylines(&svg);
    assert_eq!(lines.len(), 4, "data line + hline + vline + abline");
    // hline: constant y at 2, spanning the panel left to right.
    let h = &lines[1];
    assert_eq!(h.len(), 2);
    assert!((h[0].1 - ax.py(2.0)).abs() < 0.01 && (h[1].1 - ax.py(2.0)).abs() < 0.01);
    assert!(
        h[0].0 < ax.px(0.0) && h[1].0 > ax.px(10.0),
        "spans the panel"
    );
    // vline: constant x at 3, spanning the panel top to bottom.
    let v = &lines[2];
    assert_eq!(v.len(), 2);
    assert!((v[0].0 - ax.px(3.0)).abs() < 0.01 && (v[1].0 - ax.px(3.0)).abs() < 0.01);
    assert!((v[0].1 - v[1].1).abs() > 400.0, "spans the panel");
    // abline: every vertex satisfies y = 0.5 x + 1 in data units.
    for &(px, py) in &lines[3] {
        let (x, y) = (ax.dx(px), ax.dy(py));
        assert!(
            (y - (0.5 * x + 1.0)).abs() < 1e-2,
            "({x}, {y}) off the line"
        );
    }
}

#[test]
fn plot_hline_widens_the_axis_but_abline_does_not() {
    let csv = fixture("train", "d.csv", DIAG);
    let base = ["--data", csv.as_str(), "--aes", "x=x,y=y", "--geom", "line"];
    let with_h = svg_of(&[&base[..], &["--hline", "20"]].concat());
    assert!(with_h.contains(">20</text>"), "hline 20 trains y");
    let with_v = svg_of(&[&base[..], &["--vline", "20"]].concat());
    assert!(with_v.contains(">20</text>"), "vline 20 trains x");
    let with_a = svg_of(&[&base[..], &["--abline", "1,100"]].concat());
    assert!(!with_a.contains(">20</text>"), "abline does not train");
    assert_eq!(
        polylines(&with_a).len(),
        1,
        "an abline off the panel draws nothing"
    );
}

#[test]
fn plot_groups_and_missing_values_break_the_line() {
    let csv = fixture(
        "groups",
        "d.csv",
        "x,y,g\n0,0,a\n5,5,a\n10,10,a\n0,10,b\n5,NA,b\n10,0,b\n",
    );
    let out = run(&[
        "plot",
        "--output",
        "svg",
        "--data",
        &csv,
        "--aes",
        "x=x,y=y,group=g",
        "--geom",
        "line",
    ]);
    assert!(out.status.success());
    let svg = String::from_utf8_lossy(&out.stdout);
    let ax = Axes::of(&svg);
    let lines = polylines(&svg);
    // group a: one run of 3; group b: the NA splits it into two single points -> nothing to draw.
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_path(&lines[0], &[(0., 0.), (5., 5.), (10., 10.)], &ax);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("Removed 1 rows") && err.contains("geom_line"),
        "{err}"
    );
}

#[test]
fn plot_path_with_interleaved_groups_draws_each_group_as_r_does() {
    // ggplot2 orders GeomPath rows by group before drawing: 2 groups interleaved in the file are
    // the same 2 polylines as the file sorted by group (the review's repro drew 0 and exited 0).
    let il = fixture(
        "il",
        "d.csv",
        "x,y,g\n0,0,a\n0,10,b\n5,5,a\n5,5,b\n10,10,a\n10,0,b\n",
    );
    let sorted = fixture(
        "il_sorted",
        "d.csv",
        "x,y,g\n0,0,a\n5,5,a\n10,10,a\n0,10,b\n5,5,b\n10,0,b\n",
    );
    let draw = |csv: &str| {
        let out = run(&[
            "plot",
            "--output",
            "svg",
            "--data",
            csv,
            "--aes",
            "x=x,y=y,group=g",
            "--geom",
            "path",
        ]);
        assert!(out.status.success());
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let (a, b) = (draw(&il), draw(&sorted));
    let ax = Axes::of(&a);
    let lines = polylines(&a);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_path(&lines[0], &[(0., 0.), (5., 5.), (10., 10.)], &ax);
    assert_path(&lines[1], &[(0., 10.), (5., 5.), (10., 0.)], &ax);
    assert_eq!(polylines(&a), polylines(&b));
}

#[test]
fn plot_titles_still_draw_with_lines() {
    let csv = fixture("labs", "d.csv", DIAG);
    let svg = svg_of(&[
        "--data", &csv, "--aes", "x=x,y=y", "--geom", "line", "--title", "Hello", "--xlab", "XL",
        "--ylab", "YL", "--hline", "1",
    ]);
    for t in ["Hello", "XL", "YL"] {
        assert!(svg.contains(&format!(">{t}</text>")), "title {t} missing");
    }
}

#[test]
fn plot_lines_terminal_golden() {
    let csv = fixture("term", "d.csv", DIAG);
    let render = |extra: &[&str]| {
        let mut argv = vec![
            "plot",
            "--data",
            csv.as_str(),
            "--aes",
            "x=x,y=y",
            "--width",
            "40",
            "--height",
            "12",
        ];
        argv.extend_from_slice(extra);
        let out = Command::new(BIN)
            .args(&argv)
            .env_remove("NO_COLOR")
            .output()
            .expect("spawn");
        assert!(out.status.success(), "{argv:?}");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let line = render(&["--geom", "line"]);
    assert_eq!(line.lines().count(), 12);
    assert!(
        line.matches('·').count() >= 12,
        "a joined line, not 3 points:\n{line}"
    );
    assert!(!line.contains('●'));
    // hline fills the whole panel width on its label row.
    let h = render(&["--geom", "line", "--hline", "2.5"]);
    let row = h
        .lines()
        .find(|l| l.starts_with("2.5 ┤") || l.contains(" 2.5 ┤"))
        .expect("2.5 row");
    let cells = row.split_once('┤').expect("axis").1;
    assert!(
        cells.chars().all(|c| c != ' '),
        "hline row has a gap: {row:?}"
    );
    // vline: one column filled on every row.
    let v = render(&["--geom", "line", "--vline", "5"]);
    let body: Vec<&str> = v.lines().take(11).collect();
    let col = body[0].chars().position(|c| c == '·').expect("vline top");
    assert!(
        body.iter()
            .take(10)
            .all(|l| l.chars().nth(col) == Some('·')),
        "vline column broken:\n{v}"
    );
    // The same command twice is the same bytes.
    assert_eq!(render(&["--geom", "step"]), render(&["--geom", "step"]));
}

#[test]
fn plot_lines_usage_errors_are_exit_2_and_data_errors_exit_1() {
    let csv = fixture("errs", "d.csv", DIAG);
    let spec = |name: &str, body: &str| fixture("errs", name, &body.replace("DATA", &csv));
    let unknown_key = spec(
        "k.yaml",
        "data: DATA\naes: {x: x, y: y}\nlayers:\n  - geom: line\n    bogus: 1\n",
    );
    let no_yint = spec(
        "h.yaml",
        "data: DATA\naes: {x: x, y: y}\nlayers:\n  - geom: hline\n",
    );
    let flip = spec(
        "f.yaml",
        "data: DATA\naes: {x: x, y: y}\nlayers:\n  - geom: line\ncoord:\n  type: flip\n",
    );
    let only_refs = spec(
        "r.yaml",
        "data: DATA\naes: {x: x, y: y}\nlayers:\n  - geom: hline\n    yintercept: 1\n",
    );
    let base = ["plot", "--data", csv.as_str(), "--aes", "x=x,y=y"];
    let flags = |extra: &[&str]| {
        [&base[..], extra]
            .concat()
            .iter()
            .map(|s| (*s).to_owned())
            .collect::<Vec<_>>()
    };
    for (argv, want) in [
        (
            flags(&["--geom", "bar"]),
            "unsupported geom 'bar'; supported: point, line, path, step, hline, vline, abline",
        ),
        (flags(&["--direction", "vh"]), "applies to geom step only"),
        (
            flags(&["--geom", "line", "--direction", "vh"]),
            "applies to geom step only",
        ),
        (
            flags(&["--geom", "step", "--direction", "zz"]),
            "unsupported direction 'zz'",
        ),
        (flags(&["--abline", "1"]), "expected SLOPE,INTERCEPT"),
        (flags(&["--abline", "a,b"]), "expected SLOPE,INTERCEPT"),
        (flags(&["--hline", "nan"]), "must be a finite number"),
        (flags(&["--vline", "inf"]), "must be a finite number"),
        (flags(&["--hline", "1e300"]), "must be a finite number"),
        (flags(&["--abline", "1,1e300"]), "must be a finite number"),
        (flags(&["--geom", "hline"]), "needs 'yintercept'"),
        (flags(&["--geom", "vline"]), "needs 'xintercept'"),
        (flags(&["--geom", "abline"]), "needs"),
        (flags(&["--vline", "abc"]), "invalid value"),
        (
            vec![
                "plot".into(),
                "--spec".into(),
                csv.clone(),
                "--hline".into(),
                "1".into(),
            ],
            "cannot be used with",
        ),
        (
            vec!["plot".into(), "--spec".into(), unknown_key],
            "unknown field",
        ),
        (
            vec!["plot".into(), "--spec".into(), no_yint],
            "needs 'yintercept'",
        ),
        (vec!["plot".into(), "--spec".into(), flip], "flip"),
        (
            vec!["plot".into(), "--spec".into(), only_refs],
            "data layer",
        ),
    ] {
        let out = run(&argv.iter().map(String::as_str).collect::<Vec<_>>());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{argv:?}: {stderr}");
        assert!(
            stderr.contains(want),
            "{argv:?}: stderr {stderr:?} lacks {want:?}"
        );
        assert!(!stderr.contains("panicked"));
    }
    // A data error is exit 1, not a usage error.
    for argv in [
        vec![
            "plot",
            "--data",
            "/nonexistent/x.csv",
            "--aes",
            "x=x,y=y",
            "--geom",
            "line",
        ],
        vec![
            "plot",
            "--data",
            csv.as_str(),
            "--aes",
            "x=x,y=nope",
            "--geom",
            "line",
        ],
    ] {
        let out = run(&argv);
        assert_eq!(out.status.code(), Some(1), "{argv:?}");
    }
}

#[test]
fn plot_extreme_inputs_never_panic() {
    let big = fixture(
        "fuzz",
        "big.csv",
        "x,y\n0,0\n3e38,-3e38\n1e-38,1e-38\n10,10\n",
    );
    for extra in [
        vec!["--geom", "line"],
        vec!["--geom", "path"],
        vec!["--geom", "step", "--direction", "mid"],
        vec!["--geom", "line", "--abline", "1e30,1e30"],
        vec!["--geom", "line", "--hline", "3e38", "--vline", "-3e38"],
        vec!["--geom", "line", "--abline", "0,0", "--hline", "0"],
    ] {
        for target in ["term", "svg"] {
            let mut argv = vec![
                "plot",
                "--data",
                big.as_str(),
                "--aes",
                "x=x,y=y",
                "--output",
                target,
            ];
            argv.extend(extra.iter().copied());
            let out = run(&argv);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                matches!(out.status.code(), Some(0..=2)) && !stderr.contains("panicked"),
                "{argv:?}: {:?} {stderr}",
                out.status.code()
            );
        }
    }
}
