//! `aprender-present plot`: draw a plot from a CSV file and a ggplot2-like spec (PLOT-GG-001,
//! paiml/aprender#6909).
//!
//! Two front ends build the same plot. Flags: `--data file.csv --aes x=a,y=b --geom point`.
//! A YAML spec (`--spec plot.yaml`) that mirrors the grammar of graphics: `data`, `aes`,
//! `layers` (`geom`, `stat`, `position`), `scales`, `coord`, `facet`, `labs`, `theme`.
//! Both end in one `trueno_viz::grammar::GGPlot`, which is the only engine: this module parses
//! and validates, it does not compute a single break or position.
//!
//! What the engine cannot draw yet is refused by name with the supported values listed, and the
//! process exits non-zero. Nothing falls back to something else silently.
//!
//! The CSV reader is local on purpose. The one in-tree CSV path is arrow-csv inside
//! aprender-data, which would pull Arrow into a CLI that needs a header row, comma splitting with
//! RFC 4180 quotes, and `#` comment lines. Values must fit in `f32` (the engine's column type).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use trueno_viz::grammar::{Aes, Coord, DataFrame, GGPlot, Geom, TerminalPlot};

/// Geoms the plot verb draws end to end.
pub const GEOMS: &[&str] = &["point"];
/// Stats a layer may name.
pub const STATS: &[&str] = &["identity"];
/// Positions a layer may name.
pub const POSITIONS: &[&str] = &["identity"];
/// Coordinate systems.
pub const COORDS: &[&str] = &["cartesian"];
/// Facets (`null` is ggplot2's `facet_null()`, one panel).
pub const FACETS: &[&str] = &["null"];
/// Themes (`grey` is ggplot2's default `theme_grey()`).
pub const THEMES: &[&str] = &["grey", "gray"];
/// Scale types for the x and y positions.
pub const SCALES: &[&str] = &["continuous"];
/// Aesthetics the plot verb maps.
pub const AESTHETICS: &[&str] = &["x", "y"];
/// Terminal size when stdout is not a terminal, or when a terminal does not say its size.
pub const DEFAULT_TERM_SIZE: (u32, u32) = (80, 24);
/// SVG size in pixels when none is given (ggplot2's 800x600 preview in aprender-viz).
pub const DEFAULT_SVG_SIZE: (u32, u32) = (800, 600);

/// Where the plot goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Output {
    /// Unicode text for a terminal.
    Term,
    /// An SVG document.
    Svg,
}

/// Arguments of `aprender-present plot`.
#[derive(Debug, Clone, clap::Args)]
pub struct PlotArgs {
    /// YAML plot spec (data, aes, layers, scales, coord, facet, labs, theme)
    #[arg(long, conflicts_with_all = ["data", "aes", "geom"])]
    pub spec: Option<PathBuf>,

    /// CSV file with a header row (`#` lines are skipped)
    #[arg(long, required_unless_present = "spec")]
    pub data: Option<PathBuf>,

    /// Aesthetic mappings, e.g. x=height,y=weight
    #[arg(long, required_unless_present = "spec")]
    pub aes: Option<String>,

    /// Geom of the single layer (supported: point) [default: point]
    #[arg(long)]
    pub geom: Option<String>,

    /// Plot title (overrides the spec's labs.title)
    #[arg(long)]
    pub title: Option<String>,

    /// x-axis label (overrides labs.x)
    #[arg(long)]
    pub xlab: Option<String>,

    /// y-axis label (overrides labs.y)
    #[arg(long)]
    pub ylab: Option<String>,

    /// Width: cells for term, pixels for svg
    #[arg(long)]
    pub width: Option<u32>,

    /// Height: lines for term, pixels for svg
    #[arg(long)]
    pub height: Option<u32>,

    /// Render target
    #[arg(long, value_enum, default_value_t = Output::Term)]
    pub output: Output,

    /// Write to this file instead of stdout
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Never emit colour escape sequences
    #[arg(long)]
    pub no_color: bool,
}

/// The process facts that decide size and colour, read once so tests can set them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Env {
    /// stdout is a terminal.
    pub stdout_is_tty: bool,
    /// `$COLUMNS`.
    pub columns: Option<String>,
    /// `$LINES`.
    pub lines: Option<String>,
    /// `$NO_COLOR` is set to a non-empty value (<https://no-color.org>).
    pub no_color: bool,
}

impl Env {
    /// Read the facts from this process.
    pub fn from_process() -> Self {
        use std::io::IsTerminal;
        Self {
            stdout_is_tty: std::io::stdout().is_terminal(),
            columns: std::env::var("COLUMNS").ok(),
            lines: std::env::var("LINES").ok(),
            no_color: std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()),
        }
    }
}

/// The YAML spec. Unknown keys are errors, so a typo is not silently ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// CSV path, relative to the spec file.
    pub data: PathBuf,
    /// Global aesthetic mappings.
    pub aes: BTreeMap<String, String>,
    /// Layers, drawn in order.
    pub layers: Vec<LayerSpec>,
    /// Position scales, `x:`/`y:` -> scale type.
    #[serde(default)]
    pub scales: BTreeMap<String, String>,
    /// Coordinate system.
    #[serde(default)]
    pub coord: Option<CoordSpec>,
    /// Facet (only `null`).
    #[serde(default)]
    pub facet: Option<String>,
    /// Labels.
    #[serde(default)]
    pub labs: Labs,
    /// Theme name.
    #[serde(default)]
    pub theme: Option<String>,
}

/// One layer of the spec.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerSpec {
    /// `point`.
    pub geom: String,
    /// `identity` when absent.
    #[serde(default)]
    pub stat: Option<String>,
    /// `identity` when absent.
    #[serde(default)]
    pub position: Option<String>,
}

/// `coord:` of the spec.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordSpec {
    /// `cartesian`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Zoom limits on x, as `coord_cartesian(xlim = )`.
    #[serde(default)]
    pub xlim: Option<[f64; 2]>,
    /// Zoom limits on y.
    #[serde(default)]
    pub ylim: Option<[f64; 2]>,
}

/// `labs:` of the spec.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Labs {
    /// Title.
    pub title: Option<String>,
    /// x-axis label.
    pub x: Option<String>,
    /// y-axis label.
    pub y: Option<String>,
}

/// Run the verb: build the plot, write it to `--out` or `stdout`, warnings to `stderr`.
///
/// # Errors
///
/// A message for the user: unreadable or malformed input, an unknown column, a non-numeric
/// column, or anything the engine does not support yet (with the supported values).
pub fn run(
    args: &PlotArgs,
    env: &Env,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), String> {
    let spec = match &args.spec {
        Some(path) => load_spec(path)?,
        None => spec_from_flags(args)?,
    };
    let text = render(&spec, args, env, stderr)?;
    match &args.out {
        Some(path) => {
            std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
        }
        None => stdout
            .write_all(text.as_bytes())
            .map_err(|e| format!("cannot write stdout: {e}")),
    }
}

fn load_spec(path: &Path) -> Result<Spec, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read spec {}: {e}", path.display()))?;
    let mut spec: Spec = serde_saphyr::from_str(&text)
        .map_err(|e| format!("invalid plot spec {}: {e}", path.display()))?;
    if spec.data.is_relative() {
        if let Some(dir) = path.parent() {
            spec.data = dir.join(&spec.data);
        }
    }
    Ok(spec)
}

fn spec_from_flags(args: &PlotArgs) -> Result<Spec, String> {
    // clap guarantees both are present without --spec.
    let data = args.data.clone().unwrap_or_default();
    let aes = parse_aes(args.aes.as_deref().unwrap_or_default())?;
    let geom = args.geom.clone().unwrap_or_else(|| "point".into());
    Ok(Spec {
        data,
        aes,
        layers: vec![LayerSpec {
            geom,
            stat: None,
            position: None,
        }],
        ..Spec::default()
    })
}

/// Parse `x=col,y=col` into a mapping.
///
/// # Errors
///
/// A pair without `=`, an empty side, or the same aesthetic twice.
pub fn parse_aes(s: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for pair in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (k, v) = pair
            .split_once('=')
            .map(|(k, v)| (k.trim(), v.trim()))
            .filter(|(k, v)| !k.is_empty() && !v.is_empty())
            .ok_or_else(|| format!("--aes: expected aesthetic=column, got '{pair}'"))?;
        if out.insert(k.to_string(), v.to_string()).is_some() {
            return Err(format!("--aes: aesthetic '{k}' is mapped twice"));
        }
    }
    Ok(out)
}

fn refuse(what: &str, got: &str, supported: &[&str]) -> String {
    format!(
        "unsupported {what} '{got}'; supported: {}",
        supported.join(", ")
    )
}

fn check(what: &str, got: Option<&str>, supported: &[&str]) -> Result<(), String> {
    match got {
        Some(g) if !supported.contains(&g) => Err(refuse(what, g, supported)),
        _ => Ok(()),
    }
}

/// Validate the spec against what the engine draws, then build and render the plot.
fn render(
    spec: &Spec,
    args: &PlotArgs,
    env: &Env,
    stderr: &mut dyn Write,
) -> Result<String, String> {
    if spec.layers.is_empty() {
        return Err("the plot has no layers; add one, e.g. geom: point".into());
    }
    for layer in &spec.layers {
        check("geom", Some(&layer.geom), GEOMS)?;
        check("stat", layer.stat.as_deref(), STATS)?;
        check("position", layer.position.as_deref(), POSITIONS)?;
    }
    for (aes, scale) in &spec.scales {
        check("scale aesthetic", Some(aes), AESTHETICS)?;
        check("scale", Some(scale), SCALES)?;
    }
    check("facet", spec.facet.as_deref(), FACETS)?;
    check("theme", spec.theme.as_deref(), THEMES)?;
    for aes in spec.aes.keys() {
        check("aesthetic", Some(aes), AESTHETICS)?;
    }
    let col = |a: &str| {
        spec.aes
            .get(a)
            .ok_or_else(|| format!("aes needs a mapping for '{a}' (e.g. {a}=column)"))
    };
    let (xcol, ycol) = (col("x")?, col("y")?);

    let table = read_csv(&spec.data)?;
    let (xs, ys, dropped) = numeric_xy(&table, xcol, ycol)?;
    if dropped > 0 {
        // ggplot2's wording for the same event.
        let _ = writeln!(stderr, "warning: Removed {dropped} rows containing missing values or values outside the scale range (geom_point).");
    }
    let mut df = DataFrame::new();
    df.add_column_f32(xcol, &xs);
    df.add_column_f32(ycol, &ys);

    let mut plot = GGPlot::new().data(df).aes(Aes::new().x(xcol).y(ycol));
    for _ in &spec.layers {
        plot = plot.geom(Geom::point());
    }
    if let Some(c) = &spec.coord {
        check("coord", Some(&c.kind), COORDS)?;
        let mut coord = Coord::cartesian();
        if let Some([lo, hi]) = c.xlim {
            coord = coord.xlim(lo, hi);
        }
        if let Some([lo, hi]) = c.ylim {
            coord = coord.ylim(lo, hi);
        }
        plot = plot.coord(coord);
    }
    let labs = [
        (&args.title, &spec.labs.title),
        (&args.xlab, &spec.labs.x),
        (&args.ylab, &spec.labs.y),
    ]
    .map(|(flag, spec)| flag.clone().or_else(|| spec.clone()));
    let [title, xlab, ylab] = labs;
    if let Some(t) = title {
        plot = plot.title(&t);
    }
    if let Some(t) = xlab {
        plot = plot.xlab(&t);
    }
    if let Some(t) = ylab {
        plot = plot.ylab(&t);
    }

    match args.output {
        Output::Term => {
            let (w, h) = term_size(args, env);
            let color = !args.no_color && !env.no_color && env.stdout_is_tty && args.out.is_none();
            let built = plot.build().map_err(|e| e.to_string())?;
            built
                .to_terminal(TerminalPlot {
                    width: w,
                    height: h,
                    color,
                })
                .map_err(|e| e.to_string())
        }
        Output::Svg => {
            let w = args.width.unwrap_or(DEFAULT_SVG_SIZE.0);
            let h = args.height.unwrap_or(DEFAULT_SVG_SIZE.1);
            let built = plot.dimensions(w, h).build().map_err(|e| e.to_string())?;
            built.to_svg().map_err(|e| e.to_string())
        }
    }
}

/// Terminal size: flags, then `$COLUMNS`/`$LINES` when stdout is a terminal, then 80x24.
fn term_size(args: &PlotArgs, env: &Env) -> (u32, u32) {
    let from_env = |v: &Option<String>| {
        v.as_deref()
            .filter(|_| env.stdout_is_tty)
            .and_then(|s| s.trim().parse::<u32>().ok())
    };
    let w = args
        .width
        .or_else(|| from_env(&env.columns))
        .unwrap_or(DEFAULT_TERM_SIZE.0);
    let h = args
        .height
        .or_else(|| from_env(&env.lines))
        .unwrap_or(DEFAULT_TERM_SIZE.1);
    (w, h)
}

/// A CSV file: header names and the rows below it, each with one field per column.
#[derive(Debug, PartialEq, Eq)]
pub struct Table {
    /// Header names.
    pub columns: Vec<String>,
    /// `(1-based line number, fields)` per data row.
    pub rows: Vec<(usize, Vec<String>)>,
}

fn read_csv(path: &Path) -> Result<Table, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read data {}: {e}", path.display()))?;
    parse_csv(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse CSV text: blank lines and lines starting with `#` are skipped; the first other line is
/// the header. Fields may be double-quoted, with `""` for a quote inside.
///
/// # Errors
///
/// No header, an unterminated quote, or a row whose field count differs from the header's.
pub fn parse_csv(text: &str) -> Result<Table, String> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim_end_matches('\r')))
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
    let (hline, header) = lines.next().ok_or("no header row")?;
    let columns = split_fields(header).map_err(|e| format!("line {hline}: {e}"))?;
    let mut rows = Vec::new();
    for (n, line) in lines {
        let fields = split_fields(line).map_err(|e| format!("line {n}: {e}"))?;
        if fields.len() != columns.len() {
            return Err(format!(
                "line {n}: {} fields, the header has {}",
                fields.len(),
                columns.len()
            ));
        }
        rows.push((n, fields));
    }
    Ok(Table { columns, rows })
}

fn split_fields(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            (true, '"') => quoted = false,
            (false, '"') if cur.trim().is_empty() => {
                cur.clear();
                quoted = true;
            }
            (false, ',') => fields.push(std::mem::take(&mut cur).trim().to_string()),
            _ => cur.push(c),
        }
    }
    if quoted {
        return Err("unterminated quote".into());
    }
    fields.push(cur.trim().to_string());
    Ok(fields)
}

/// A cell ggplot2 would read as missing.
fn is_missing(s: &str) -> bool {
    matches!(s, "" | "NA" | "NaN" | "nan" | "null" | "NULL")
}

/// The mapped x and y columns as `f32`, rows with a missing value in either dropped; also the
/// number dropped.
///
/// # Errors
///
/// A column not in the header (the message lists the columns), or a value that is neither a
/// number nor missing (the column is not numeric).
pub fn numeric_xy(
    table: &Table,
    xcol: &str,
    ycol: &str,
) -> Result<(Vec<f32>, Vec<f32>, usize), String> {
    let index = |name: &str| {
        table.columns.iter().position(|c| c == name).ok_or_else(|| {
            format!(
                "unknown column '{name}'; columns: {}",
                table.columns.join(", ")
            )
        })
    };
    let (xi, yi) = (index(xcol)?, index(ycol)?);
    let (mut xs, mut ys, mut dropped) = (Vec::new(), Vec::new(), 0);
    for (line, fields) in &table.rows {
        let value = |i: usize, name: &str| -> Result<Option<f32>, String> {
            let s = fields[i].as_str();
            if is_missing(s) {
                return Ok(None);
            }
            match s.parse::<f32>() {
                Ok(v) if v.is_finite() => Ok(Some(v)),
                Ok(_) => Ok(None),
                Err(_) => Err(format!(
                    "column '{name}' is not numeric: line {line} has '{s}' (geom_point needs a continuous x and y)"
                )),
            }
        };
        match (value(xi, xcol)?, value(yi, ycol)?) {
            (Some(x), Some(y)) => {
                xs.push(x);
                ys.push(y);
            }
            _ => dropped += 1,
        }
    }
    if xs.is_empty() {
        return Err(format!(
            "no rows with a numeric '{xcol}' and '{ycol}' to plot"
        ));
    }
    Ok((xs, ys, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        plot: PlotArgs,
    }

    fn parse(argv: &[&str]) -> Result<PlotArgs, clap::Error> {
        Cli::try_parse_from(std::iter::once("plot").chain(argv.iter().copied())).map(|c| c.plot)
    }

    fn tmp(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plot-rs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let p = dir.join(name);
        std::fs::write(&p, body).expect("write tmp");
        p
    }

    const ROW1: &str = "# row 1 fixture\nx,y\n1.5,-2.25\n2.25,0.5\n3,1.75\n4.75,3\n6.5,2.5\n\
                        8.125,4.125\n9.5,5.5\n11,4.75\n12.25,6.25\n13.75,7.5\n";

    fn go(args: &PlotArgs, env: &Env) -> (Result<(), String>, String, String) {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let r = run(args, env, &mut out, &mut err);
        (
            r,
            String::from_utf8_lossy(&out).into(),
            String::from_utf8_lossy(&err).into(),
        )
    }

    fn flags(csv: &Path, extra: &[&str]) -> PlotArgs {
        let mut v = vec!["--data", csv.to_str().expect("utf8"), "--aes", "x=x,y=y"];
        v.extend_from_slice(extra);
        parse(&v).expect("parses")
    }

    #[test]
    fn args_parse_with_defaults() {
        let a = parse(&["--data", "d.csv", "--aes", "x=a,y=b"]).expect("parses");
        assert_eq!(a.output, Output::Term);
        assert!(a.geom.is_none() && !a.no_color && a.out.is_none());
        let a = parse(&[
            "--data",
            "d.csv",
            "--aes",
            "x=a,y=b",
            "--geom",
            "point",
            "--title",
            "T",
            "--xlab",
            "X",
            "--ylab",
            "Y",
            "--width",
            "60",
            "--height",
            "20",
            "--output",
            "svg",
            "--out",
            "o.svg",
            "--no-color",
        ])
        .expect("parses");
        assert_eq!(
            (a.width, a.height, a.output),
            (Some(60), Some(20), Output::Svg)
        );
        assert!(a.no_color);
        assert!(parse(&["--spec", "p.yaml"]).is_ok());
    }

    #[test]
    fn args_reject_missing_and_conflicting() {
        assert!(parse(&[]).is_err(), "needs --spec or --data/--aes");
        assert!(parse(&["--data", "d.csv"]).is_err(), "needs --aes");
        assert!(parse(&["--spec", "p.yaml", "--data", "d.csv"]).is_err());
        assert!(parse(&["--spec", "p.yaml", "--geom", "point"]).is_err());
        assert!(parse(&["--data", "d", "--aes", "x=a,y=b", "--output", "png"]).is_err());
    }

    #[test]
    fn aes_parses_and_rejects_malformed() {
        let m = parse_aes(" x = a , y=b ,").expect("parses");
        assert_eq!(m.get("x").map(String::as_str), Some("a"));
        assert_eq!(m.get("y").map(String::as_str), Some("b"));
        assert!(parse_aes("x")
            .expect_err("no =")
            .contains("expected aesthetic=column"));
        assert!(parse_aes("x=").is_err());
        assert!(parse_aes("x=a,x=b")
            .expect_err("twice")
            .contains("mapped twice"));
    }

    #[test]
    fn csv_skips_comments_and_blank_lines_and_honours_quotes() {
        let t = parse_csv("# c\n\n\"a, b\",c\r\n\"say \"\"hi\"\"\", 2\n  # indented\n")
            .expect("parses");
        assert_eq!(t.columns, ["a, b", "c"]);
        assert_eq!(
            t.rows,
            vec![(4, vec!["say \"hi\"".to_string(), "2".to_string()])]
        );
        assert!(parse_csv("# only\n\n")
            .expect_err("empty")
            .contains("no header"));
        assert!(parse_csv("a,b\n1\n")
            .expect_err("short")
            .contains("line 2: 1 fields, the header has 2"));
        assert!(parse_csv("\"a,b\n")
            .expect_err("quote")
            .contains("line 1: unterminated quote"));
        assert!(parse_csv("a,b\n\"1,2\n")
            .expect_err("quote")
            .contains("line 2"));
    }

    #[test]
    fn numeric_columns_drop_missing_and_refuse_text() {
        let t = parse_csv("x,y,name\n1,2,a\nNA,3,b\n4,,c\n5,inf,d\n6,7,e\n").expect("parses");
        let (xs, ys, dropped) = numeric_xy(&t, "x", "y").expect("numeric");
        assert_eq!((xs, ys, dropped), (vec![1.0, 6.0], vec![2.0, 7.0], 3));
        let e = numeric_xy(&t, "x", "z").expect_err("unknown");
        assert_eq!(e, "unknown column 'z'; columns: x, y, name");
        let e = numeric_xy(&t, "name", "y").expect_err("text");
        assert!(
            e.contains("column 'name' is not numeric: line 2 has 'a'"),
            "{e}"
        );
        let t = parse_csv("x,y\nNA,1\n").expect("parses");
        assert!(numeric_xy(&t, "x", "y")
            .expect_err("none")
            .contains("no rows"));
    }

    #[test]
    fn renders_row1_with_the_oracle_break_labels() {
        let csv = tmp("row1.csv", ROW1);
        let (r, out, err) = go(&flags(&csv, &[]), &Env::default());
        assert_eq!(r, Ok(()));
        assert!(err.is_empty(), "{err}");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 24);
        assert_eq!(
            lines[23].split_whitespace().collect::<Vec<_>>(),
            ["5", "10"]
        );
        assert_eq!(out.matches('●').count(), 10);
        assert!(!out.contains('\x1b'));
    }

    #[test]
    fn colour_only_on_a_tty_without_opt_outs() {
        let csv = tmp("row1c.csv", ROW1);
        let tty = Env {
            stdout_is_tty: true,
            ..Env::default()
        };
        assert!(go(&flags(&csv, &[]), &tty).1.contains('\x1b'));
        assert!(!go(&flags(&csv, &["--no-color"]), &tty).1.contains('\x1b'));
        let nc = Env {
            no_color: true,
            ..tty.clone()
        };
        assert!(!go(&flags(&csv, &[]), &nc).1.contains('\x1b'));
        assert!(!go(&flags(&csv, &[]), &Env::default()).1.contains('\x1b'));
    }

    #[test]
    fn size_from_flags_then_tty_env_then_default() {
        let csv = tmp("row1s.csv", ROW1);
        let env = Env {
            stdout_is_tty: true,
            columns: Some("50".into()),
            lines: Some(" 12 ".into()),
            no_color: true,
        };
        let out = go(&flags(&csv, &[]), &env).1;
        assert_eq!(out.lines().count(), 12);
        let out = go(&flags(&csv, &["--width", "40", "--height", "10"]), &env).1;
        assert_eq!(out.lines().count(), 10);
        let piped = Env {
            stdout_is_tty: false,
            ..env.clone()
        };
        assert_eq!(go(&flags(&csv, &[]), &piped).1.lines().count(), 24);
        let junk = Env {
            columns: Some("wide".into()),
            lines: None,
            ..env
        };
        assert_eq!(go(&flags(&csv, &[]), &junk).1.lines().count(), 24);
    }

    #[test]
    fn labels_from_flags_override_the_spec() {
        let csv = tmp("row1l.csv", ROW1);
        let spec = tmp(
            "labs.yaml",
            &format!(
                "data: {}\naes: {{x: x, y: y}}\nlayers: [{{geom: point, stat: identity, position: identity}}]\n\
                 scales: {{x: continuous, y: continuous}}\ncoord: {{type: cartesian}}\nfacet: \"null\"\n\
                 theme: grey\nlabs: {{title: Spec title, x: spec x, y: spec y}}\n",
                csv.display()
            ),
        );
        let args = parse(&[
            "--spec",
            spec.to_str().expect("utf8"),
            "--title",
            "Flag title",
        ])
        .expect("parses");
        let (r, out, _) = go(&args, &Env::default());
        assert_eq!(r, Ok(()));
        assert!(
            out.contains("Flag title") && !out.contains("Spec title"),
            "{out}"
        );
        assert!(out.contains("spec x") && out.contains("spec y"), "{out}");
        let args = parse(&[
            "--spec",
            spec.to_str().expect("utf8"),
            "--xlab",
            "fx",
            "--ylab",
            "fy",
        ])
        .expect("parses");
        let out = go(&args, &Env::default()).1;
        assert!(
            out.contains("Spec title") && out.contains("fx") && out.contains("fy"),
            "{out}"
        );
    }

    #[test]
    fn spec_data_is_relative_to_the_spec_and_coord_limits_apply() {
        let _ = tmp("rel.csv", ROW1);
        let spec = tmp(
            "rel.yaml",
            "data: rel.csv\naes: {x: x, y: y}\nlayers: [{geom: point}]\n\
             coord: {type: cartesian, xlim: [0, 7], ylim: [0, 3]}\n",
        );
        let args = parse(&["--spec", spec.to_str().expect("utf8")]).expect("parses");
        let (r, out, _) = go(&args, &Env::default());
        assert_eq!(r, Ok(()));
        // x <= 7 and 0 <= y <= 3: (2.25, 0.5), (3, 1.75), (4.75, 3), (6.5, 2.5).
        assert_eq!(out.matches('●').count(), 4, "{out}");
    }

    #[test]
    fn unsupported_grammar_is_refused_with_the_supported_values() {
        let csv = tmp("row1u.csv", ROW1);
        let e = go(&flags(&csv, &["--geom", "line"]), &Env::default())
            .0
            .expect_err("refused");
        assert_eq!(e, "unsupported geom 'line'; supported: point");
        let cases = [
            (
                "layers: [{geom: point, stat: bin}]",
                "unsupported stat 'bin'; supported: identity",
            ),
            (
                "layers: [{geom: point, position: jitter}]",
                "unsupported position 'jitter'; supported: identity",
            ),
            (
                "layers: [{geom: point}]\nscales: {x: log10}",
                "unsupported scale 'log10'; supported: continuous",
            ),
            (
                "layers: [{geom: point}]\nscales: {colour: continuous}",
                "unsupported scale aesthetic 'colour'; supported: x, y",
            ),
            (
                "layers: [{geom: point}]\nfacet: wrap",
                "unsupported facet 'wrap'; supported: null",
            ),
            (
                "layers: [{geom: point}]\ntheme: bw",
                "unsupported theme 'bw'; supported: grey, gray",
            ),
            (
                "layers: [{geom: point}]\ncoord: {type: polar}",
                "unsupported coord 'polar'; supported: cartesian",
            ),
            (
                "layers: []",
                "the plot has no layers; add one, e.g. geom: point",
            ),
        ];
        for (i, (body, want)) in cases.iter().enumerate() {
            let spec = tmp(
                &format!("u{i}.yaml"),
                &format!("data: {}\naes: {{x: x, y: y}}\n{body}\n", csv.display()),
            );
            let args = parse(&["--spec", spec.to_str().expect("utf8")]).expect("parses");
            assert_eq!(
                go(&args, &Env::default()).0.err().as_deref(),
                Some(*want),
                "{body}"
            );
        }
        let colour = parse(&[
            "--data",
            csv.to_str().expect("utf8"),
            "--aes",
            "x=x,y=y,colour=x",
        ])
        .expect("parses");
        assert_eq!(
            go(&colour, &Env::default()).0.err().as_deref(),
            Some("unsupported aesthetic 'colour'; supported: x, y")
        );
        let no_y = parse(&["--data", csv.to_str().expect("utf8"), "--aes", "x=x"]).expect("parses");
        assert_eq!(
            go(&no_y, &Env::default()).0.err().as_deref(),
            Some("aes needs a mapping for 'y' (e.g. y=column)")
        );
    }

    #[test]
    fn bad_inputs_are_errors_not_panics() {
        let env = Env::default();
        let missing = parse(&["--data", "/nonexistent/d.csv", "--aes", "x=x,y=y"]).expect("parses");
        assert!(go(&missing, &env)
            .0
            .expect_err("io")
            .starts_with("cannot read data /nonexistent/d.csv"));
        let nospec = parse(&["--spec", "/nonexistent/p.yaml"]).expect("parses");
        assert!(go(&nospec, &env)
            .0
            .expect_err("io")
            .starts_with("cannot read spec"));
        let typo = tmp(
            "typo.yaml",
            "data: d.csv\naes: {x: x, y: y}\nlayers: [{geom: point}]\nlabz: {}\n",
        );
        let typo = parse(&["--spec", typo.to_str().expect("utf8")]).expect("parses");
        assert!(go(&typo, &env)
            .0
            .expect_err("yaml")
            .starts_with("invalid plot spec"));
        let csv = tmp("unk.csv", ROW1);
        let unk =
            parse(&["--data", csv.to_str().expect("utf8"), "--aes", "x=x,y=w"]).expect("parses");
        assert_eq!(
            go(&unk, &env).0.err().as_deref(),
            Some("unknown column 'w'; columns: x, y")
        );
        let bad = tmp("bad.csv", "x,y\n1\n");
        let bad =
            parse(&["--data", bad.to_str().expect("utf8"), "--aes", "x=x,y=y"]).expect("parses");
        assert!(go(&bad, &env)
            .0
            .expect_err("csv")
            .ends_with("line 2: 1 fields, the header has 2"));
        let tiny = flags(&csv, &["--width", "10"]);
        assert!(go(&tiny, &env)
            .0
            .expect_err("size")
            .contains("at least 20x8"));
        let unwritable = flags(&csv, &["--out", "/nonexistent/dir/o.txt"]);
        assert!(go(&unwritable, &env)
            .0
            .expect_err("out")
            .starts_with("cannot write /nonexistent/dir/o.txt"));
        let svg_zero = flags(&csv, &["--output", "svg", "--width", "0", "--height", "0"]);
        assert!(go(&svg_zero, &env).0.is_err());
    }

    #[test]
    fn missing_rows_warn_like_ggplot2() {
        let csv = tmp("na.csv", "x,y\n1,2\nNA,3\n4,5\n");
        let (r, out, err) = go(&flags(&csv, &[]), &Env::default());
        assert_eq!(r, Ok(()));
        assert_eq!(out.matches('●').count(), 2);
        assert!(
            err.starts_with("warning: Removed 1 rows containing missing values"),
            "{err}"
        );
    }

    #[test]
    fn svg_goes_to_out_with_the_oracle_break_labels() {
        let csv = tmp("row1v.csv", ROW1);
        let out_path = csv.with_extension("svg");
        let args = flags(
            &csv,
            &["--output", "svg", "--out", out_path.to_str().expect("utf8")],
        );
        let (r, stdout, _) = go(
            &args,
            &Env {
                stdout_is_tty: true,
                ..Env::default()
            },
        );
        assert_eq!(r, Ok(()));
        assert!(stdout.is_empty());
        let svg = std::fs::read_to_string(&out_path).expect("svg written");
        assert!(
            svg.starts_with("<svg") || svg.starts_with("<?xml"),
            "{}",
            &svg[..40.min(svg.len())]
        );
        for label in ["5", "10", "-2.5", "0.0", "2.5", "5.0", "7.5"] {
            assert!(
                svg.contains(&format!(">{label}</text>")),
                "missing break label {label}"
            );
        }
    }

    #[test]
    fn stdout_write_failure_is_an_error() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("closed"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let csv = tmp("row1b.csv", ROW1);
        let r = run(
            &flags(&csv, &[]),
            &Env::default(),
            &mut Broken,
            &mut Vec::new(),
        );
        assert_eq!(r, Err("cannot write stdout: closed".into()));
    }

    #[test]
    fn env_from_process_reads_without_panicking() {
        let e = Env::from_process();
        assert_eq!(
            e.no_color,
            std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty())
        );
    }
}
