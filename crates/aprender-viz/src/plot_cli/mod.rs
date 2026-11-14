//! The library core of `apr plot` (#6909): a CSV file's numeric columns drawn as ggplot2's row 1,
//! `ggplot(df, aes(x, y)) + geom_point()`, to a terminal or SVG; [`run`] can ask for nothing else.
//! The scales are trained on the `f64` values, as ggplot2 trains on doubles (the engine stores
//! `f32`). Compared with R 4.6.2 / ggplot2 4.0.3: row 1's oracle and the 17 `geom_point_edges`
//! sets, whose expanded edge lands on a break. Refused by column and line: a nonzero magnitude
//! below 1e-5 or from 1e16 up, an infinite value, and a value `f32` moves by more than
//! [`MAX_F32_ERROR_OF_RANGE`] of its column's range.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::grammar::{
    Aes, Coord, DataFrame, GGPlot, Geom, TerminalPlot, Theme, MAX_TERMINAL_HEIGHT,
    MAX_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT, MIN_TERMINAL_WIDTH,
};

/// Terminal size in cells when no width or height is given.
pub const DEFAULT_TERM_SIZE: (u32, u32) = (80, 24);
/// SVG size in pixels when no width or height is given.
pub const DEFAULT_SVG_SIZE: (u32, u32) = (800, 600);
/// Largest SVG side in pixels (the smallest is 1): the engine allocates a width x height buffer.
pub const MAX_SVG_SIZE: u32 = 4096;
/// Smallest nonzero magnitude drawn.
pub const MIN_MAGNITUDE: f64 = 1e-5;
/// Magnitudes at or above this are refused.
pub const MAX_MAGNITUDE: f64 = 1e16;
/// The most a value may move when stored as `f32`, as a fraction of its column's range: points
/// are placed from `f32` values, so `16777217,16777219` would be drawn where the data is not.
pub const MAX_F32_ERROR_OF_RANGE: f64 = 1e-4;

/// Where the plot goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Output {
    /// Unicode text for a terminal.
    #[default]
    Term,
    /// An SVG document.
    Svg,
}

/// What to plot and how.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlotArgs {
    /// CSV file with a header row (`#` lines are skipped).
    pub data: PathBuf,
    /// Column mapped to x.
    pub x: String,
    /// Column mapped to y.
    pub y: String,
    /// Width: cells for term, pixels for svg.
    pub width: Option<u32>,
    /// Height: lines for term, pixels for svg.
    pub height: Option<u32>,
    /// Render target.
    pub output: Output,
    /// Never emit colour escape sequences.
    pub no_color: bool,
}

/// The process facts that decide colour, read once so tests can set them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Env {
    /// stdout is a terminal.
    pub stdout_is_tty: bool,
    /// `$NO_COLOR` is set to a non-empty value (<https://no-color.org>).
    pub no_color: bool,
}

impl Env {
    /// Read the facts from this process.
    pub fn from_process() -> Self {
        use std::io::IsTerminal;
        let no_color = !std::env::var_os("NO_COLOR").unwrap_or_default().is_empty();
        Self { stdout_is_tty: std::io::stdout().is_terminal(), no_color }
    }
}

/// Draw the plot to `stdout`, warnings to `stderr`.
///
/// # Errors
///
/// A message for the user: a size out of bounds, unreadable or malformed CSV, an unknown or
/// non-numeric column, a value refused by column and line, or an engine error.
pub fn run(
    args: &PlotArgs,
    env: &Env,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), String> {
    let text = render(args, env, stderr)?;
    stdout.write_all(text.as_bytes()).map_err(|e| format!("cannot write stdout: {e}"))
}

/// The `(min, max)` of `values`.
fn domain(values: &[f64]) -> (f64, f64) {
    values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

/// Row 1's plot: the data frame holds `f32`, the scales are trained on the `f64` domains.
fn build_plot(args: &PlotArgs, (xs, ys): (&[f64], &[f64])) -> GGPlot {
    let narrow = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let mut df = DataFrame::new();
    df.add_column_f32(&args.x, &narrow(xs));
    df.add_column_f32(&args.y, &narrow(ys));
    GGPlot::new()
        .data(df)
        .aes(Aes::new().x(&args.x).y(&args.y))
        .geom(Geom::point())
        .coord(Coord::cartesian())
        .theme(Theme::grey())
        .trained_on(domain(xs), domain(ys))
}

/// Check the size, read the data, then build and render the plot.
fn render(args: &PlotArgs, env: &Env, stderr: &mut dyn Write) -> Result<String, String> {
    let (default, color) = match args.output {
        Output::Term => (DEFAULT_TERM_SIZE, !args.no_color && !env.no_color && env.stdout_is_tty),
        Output::Svg => (DEFAULT_SVG_SIZE, false),
    };
    let size = (args.width.unwrap_or(default.0), args.height.unwrap_or(default.1));
    // Before the data is read: a size the engine cannot allocate must not cost a read first.
    check_size(args.output, size)?;
    let (xs, ys, dropped) = numeric_xy(&read_csv(&args.data)?, &args.x, &args.y)?;
    if dropped > 0 {
        // ggplot2's wording for the same event, singular for one row.
        let rows = if dropped == 1 { "row" } else { "rows" };
        let _ = writeln!(stderr, "warning: Removed {dropped} {rows} containing missing values or values outside the scale range (geom_point).");
    }
    let plot = build_plot(args, (&xs, &ys));
    let target = TerminalPlot { width: size.0, height: size.1, color };
    match args.output {
        Output::Term => plot.build().and_then(|b| b.to_terminal(target)),
        Output::Svg => plot.dimensions(size.0, size.1).build().and_then(|b| b.to_svg()),
    }
    .map_err(|e| e.to_string())
}

/// A size inside the target's bounds: 20x8 to 1000x1000 cells, or 1x1 to 4096x4096 pixels.
fn check_size(output: Output, (w, h): (u32, u32)) -> Result<(), String> {
    let (kind, unit, (a, b), (c, d)) = match output {
        Output::Term => (
            "terminal",
            "cells",
            (MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT),
            (MAX_TERMINAL_WIDTH, MAX_TERMINAL_HEIGHT),
        ),
        Output::Svg => ("svg", "pixels", (1, 1), (MAX_SVG_SIZE, MAX_SVG_SIZE)),
    };
    let msg = format!("{kind} plot size must be {a}x{b} to {c}x{d} {unit}, got {w}x{h}");
    ((a..=c).contains(&w) && (b..=d).contains(&h)).then_some(()).ok_or(msg)
}

/// A CSV file: header names, then `(1-based line number, fields)` per data row.
#[derive(Debug, PartialEq, Eq)]
struct Table {
    columns: Vec<String>,
    rows: Vec<(usize, Vec<String>)>,
}

fn read_csv(path: &Path) -> Result<Table, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read data {}: {e}", path.display()))?;
    parse_csv(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse CSV text: a BOM, blank lines and lines starting with `#` are skipped; the first other
/// line is the header. Fields may be double-quoted, with `""` for a quote inside.
fn parse_csv(text: &str) -> Result<Table, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
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
        let (got, want) = (fields.len(), columns.len());
        if got != want {
            return Err(format!("line {n}: {got} fields, the header has {want}"));
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

/// Zero, or a magnitude from [`MIN_MAGNITUDE`] up to but excluding [`MAX_MAGNITUDE`]: far inside
/// f32's normal range, so `as f32` keeps the magnitude.
fn in_drawn_range(v: f64) -> bool {
    let m = v.abs();
    m == 0.0 || (MIN_MAGNITUDE..MAX_MAGNITUDE).contains(&m)
}

/// Refuse a column whose `f32` storage moves a value by more than [`MAX_F32_ERROR_OF_RANGE`] of
/// the column's range. `values` are `(line, text, value)`.
fn check_f32_fidelity(name: &str, values: &[(usize, String, f64)]) -> Result<(), String> {
    let (lo, hi) = domain(&values.iter().map(|v| v.2).collect::<Vec<f64>>());
    let allowed = MAX_F32_ERROR_OF_RANGE * (hi - lo);
    let worst = values
        .iter()
        .map(|(line, text, v)| (line, text, *v, (f64::from(*v as f32) - v).abs()))
        .filter(|w| w.3 > allowed)
        .max_by(|a, b| a.3.total_cmp(&b.3));
    match worst {
        None => Ok(()),
        Some((line, text, v, err)) => Err(format!(
            "column '{name}' line {line} has '{text}', which the engine stores as {} (binary32), \
             a change of {err:e} against a column range of {:e} (at most {MAX_F32_ERROR_OF_RANGE:e})",
            f64::from(v as f32),
            hi - lo
        )),
    }
}

/// The x values, the y values and the number of rows dropped.
type Xy = (Vec<f64>, Vec<f64>, usize);

/// The mapped columns as `f64`, without the rows missing either value. A cell is missing when it
/// is empty, `NA` or a NaN (`NaN`, `nan`), the cells R's `read.csv` reads as `NA`/`NaN` in a
/// numeric column; `null` is text, as it is to R. An unknown column, text, an infinite value
/// (`Inf`, `-inf`, `infinity`, or `1e400`, which overflows) and a magnitude outside the drawn
/// range are refused; all but the first by column and line.
fn numeric_xy(table: &Table, xcol: &str, ycol: &str) -> Result<Xy, String> {
    let index = |name: &str| {
        table.columns.iter().position(|c| c == name).ok_or_else(|| {
            format!("unknown column '{name}'; columns: {}", table.columns.join(", "))
        })
    };
    let (xi, yi) = (index(xcol)?, index(ycol)?);
    let (mut xt, mut yt, mut dropped) = (Vec::new(), Vec::new(), 0);
    for (line, fields) in &table.rows {
        let value = |i: usize, name: &str| -> Result<Option<f64>, String> {
            let s = fields[i].as_str();
            match s.parse::<f64>() {
                _ if matches!(s, "" | "NA") => Ok(None),
                Ok(v) if v.is_nan() => Ok(None),
                Ok(v) if v.is_infinite() => Err(format!(
                    "column '{name}' line {line} has '{s}', which is infinite; ggplot2 would draw \
                     it at the panel edge, which apr plot does not do yet"
                )),
                Ok(v) if !in_drawn_range(v) => Err(format!(
                    "column '{name}' line {line} has '{s}', outside the drawn magnitudes (zero, \
                     or {MIN_MAGNITUDE:e} up to but excluding {MAX_MAGNITUDE:e})"
                )),
                Ok(v) => Ok(Some(v)),
                Err(_) => Err(format!(
                    "column '{name}' is not numeric: line {line} has '{s}' (geom_point needs a continuous x and y)"
                )),
            }
        };
        match (value(xi, xcol)?, value(yi, ycol)?) {
            (Some(x), Some(y)) => {
                xt.push((*line, fields[xi].clone(), x));
                yt.push((*line, fields[yi].clone(), y));
            }
            _ => dropped += 1,
        }
    }
    if xt.is_empty() {
        return Err(format!("no rows with a numeric '{xcol}' and '{ycol}' to plot"));
    }
    check_f32_fidelity(xcol, &xt)?;
    check_f32_fidelity(ycol, &yt)?;
    let col = |t: &[(usize, String, f64)]| t.iter().map(|v| v.2).collect();
    Ok((col(&xt), col(&yt), dropped))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        (r, String::from_utf8_lossy(&out).into(), String::from_utf8_lossy(&err).into())
    }

    fn args(csv: &Path) -> PlotArgs {
        PlotArgs { data: csv.to_path_buf(), x: "x".into(), y: "y".into(), ..PlotArgs::default() }
    }

    fn svg(csv: &Path) -> PlotArgs {
        PlotArgs { output: Output::Svg, ..args(csv) }
    }

    fn xy(csv: &str) -> Result<Xy, String> {
        numeric_xy(&parse_csv(csv)?, "x", "y")
    }

    fn refusal(csv: &str, starts: &str) {
        let e = xy(csv).expect_err(csv);
        assert!(e.starts_with(starts), "{e}");
    }

    /// The x labels (`text-anchor="middle"`) and y labels (`"end"`) of an SVG, space-joined.
    fn tick_labels(svg: &str) -> (String, String) {
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for (head, label) in svg.split("</text>").filter_map(|t| t.rsplit_once('>')) {
            if head.contains("text-anchor=\"middle\"") {
                xs.push(label);
            } else if head.contains("text-anchor=\"end\"") {
                ys.push(label);
            }
        }
        (xs.join(" "), ys.join(" "))
    }

    /// B3: every `geom_point_edges` set (an expanded range edge on a candidate break) gets
    /// ggplot2's labels.
    #[test]
    fn edge_labels_match_ggplot2() {
        let tsv = include_str!("../../fixtures/ggplot2/geom_point_edges/expected.tsv");
        assert_eq!(tsv.lines().count(), 17);
        for f in tsv.lines().map(|l| l.split('\t').collect::<Vec<&str>>()) {
            let rows: String =
                f[1].split(',').zip(f[2].split(',')).map(|(x, y)| format!("{x},{y}\n")).collect();
            let (r, out, _) =
                go(&svg(&tmp(&format!("{}.csv", f[0]), &format!("x,y\n{rows}"))), &Env::default());
            assert_eq!(r, Ok(()), "{}", f[0]);
            assert_eq!(tick_labels(&out), (f[3].to_string(), f[4].to_string()), "{}", f[0]);
        }
    }

    #[test]
    fn csv_skips_bom_comments_and_blank_lines_and_unquotes() {
        let t = parse_csv("\u{feff}# c\n\nx, \"y\"\r\n1,\"a,\"\"b\"\"\"\n").expect("parses");
        assert_eq!(t.columns, ["x", "y"]);
        assert_eq!(t.rows, [(4, vec!["1".to_string(), "a,\"b\"".to_string()])]);
        assert_eq!(parse_csv("x\nab\"c\n").expect("mid-field quote").rows[0].1, ["ab\"c"]);
        assert_eq!(parse_csv("# only\n"), Err("no header row".into()));
        assert_eq!(parse_csv("x,y\n1\n"), Err("line 2: 1 fields, the header has 2".into()));
        assert_eq!(parse_csv("x\n\"a\n"), Err("line 2: unterminated quote".into()));
        assert_eq!(parse_csv("\"x\n"), Err("line 1: unterminated quote".into()));
    }

    /// B4 and N8: only empty, NA and NaN cells are missing; an infinite value is refused by
    /// column and line, and `null` is text.
    #[test]
    fn numeric_columns_drop_missing_and_refuse_the_rest_by_line() {
        let ok = xy("x,y\n1,2\n,3\nNA,4\nNaN,5\n2,nan\n0,-1e-5\n").expect("numeric");
        assert_eq!(ok, (vec![1.0, 0.0], vec![2.0, -1e-5], 4));
        for (csv, starts) in [
            ("x,y\n1,1\n2,Inf\n", "column 'y' line 3 has 'Inf', which is infinite"),
            ("x,y\n1,1\n2,-inf\n", "column 'y' line 3 has '-inf', which is infinite"),
            ("x,y\n1,1\n2,infinity\n", "column 'y' line 3 has 'infinity', which is infinite"),
            ("x,y\n1,1\n2,1e400\n", "column 'y' line 3 has '1e400', which is infinite"),
            ("x,y\nnull,1\n", "column 'x' is not numeric: line 2 has 'null'"),
            ("x,y\nNULL,1\n", "column 'x' is not numeric: line 2 has 'NULL'"),
            ("x,y\n1,1\n9e-6,2\n", "column 'x' line 3 has '9e-6', outside"),
            ("x,y\n1,1\n-1e16,2\n", "column 'x' line 3 has '-1e16', outside"),
        ] {
            refusal(csv, starts);
        }
        assert!(xy("x,y\n1e-5,9.999e15\n2,1\n").is_ok());
        assert_eq!(xy("x,y\nNA,1\n"), Err("no rows with a numeric 'x' and 'y' to plot".into()));
        let e = numeric_xy(&parse_csv("x,y\n1,1\n").expect("csv"), "x", "z");
        assert_eq!(e, Err("unknown column 'z'; columns: x, y".into()));
    }

    /// A column f32 moves by more than 1e-4 of its range is refused; 2.4e-5 of it is drawn.
    #[test]
    fn values_f32_cannot_hold_are_refused_by_name() {
        refusal(
            "x,y\n16777217,1\n16777219,2\n",
            "column 'x' line 3 has '16777219', which the engine stores as 16777220",
        );
        refusal("x,y\n1,99999.99\n2,100000.01\n", "column 'y' line");
        assert!(xy("x,y\n1000.1,1\n1001.1,2\n").is_ok() && xy("x,y\n1,1\n1,2\n").is_ok());
    }

    /// N9: ggplot2's warning, singular for one row.
    #[test]
    fn missing_rows_warn_as_ggplot2_does() {
        assert_eq!(go(&args(&tmp("none.csv", ROW1)), &Env::default()).2, "");
        for (extra, n, rows) in [("NA,1\n", 1, "row"), ("NA,1\n2,\n", 2, "rows")] {
            let csv = tmp(&format!("missing{n}.csv"), &format!("{ROW1}{extra}"));
            let (r, _, err) = go(&args(&csv), &Env::default());
            assert_eq!(r, Ok(()));
            assert_eq!(err, format!("warning: Removed {n} {rows} containing missing values or values outside the scale range (geom_point).\n"));
        }
    }

    #[test]
    fn colour_only_on_a_tty_and_size_from_flags() {
        let csv = tmp("row1.csv", ROW1);
        let tty = Env { stdout_is_tty: true, no_color: false };
        let plain = go(&args(&csv), &Env::default()).1;
        assert!(!plain.contains('\u{1b}') && plain.lines().count() == 24, "{plain}");
        assert!(go(&args(&csv), &tty).1.contains('\u{1b}'));
        assert!(!go(&PlotArgs { no_color: true, ..args(&csv) }, &tty).1.contains('\u{1b}'));
        assert!(!go(&args(&csv), &Env { no_color: true, ..tty }).1.contains('\u{1b}'));
        let small = go(&PlotArgs { width: Some(40), height: Some(12), ..args(&csv) }, &tty).1;
        assert_eq!(small.lines().count(), 12, "{small}");
        let doc = go(&PlotArgs { width: Some(320), height: Some(200), ..svg(&csv) }, &tty).1;
        assert!(doc.contains("width=\"320\"") && doc.contains("height=\"200\""), "{doc}");
        assert!(go(&svg(&csv), &tty).1.contains("width=\"800\" height=\"600\""));
    }

    #[test]
    fn sizes_are_bounded_and_checked_before_the_read() {
        let fits = |o, w, h| check_size(o, (w, h)).is_ok();
        assert!(fits(Output::Term, 20, 8) && fits(Output::Term, 1000, 1000));
        assert!(!fits(Output::Term, 19, 8) && !fits(Output::Term, 20, 7));
        assert!(!fits(Output::Term, 1001, 8) && !fits(Output::Term, 20, 1001));
        assert!(fits(Output::Svg, 1, 1) && fits(Output::Svg, 4096, 4096));
        assert!(!fits(Output::Svg, 0, 1) && !fits(Output::Svg, 1, 0));
        assert!(!fits(Output::Svg, 4097, 1) && !fits(Output::Svg, 1, 4097));
        let refused = |a: &PlotArgs| go(a, &Env::default()).0.expect_err("refused");
        let gone = svg(Path::new("/nonexistent/plot.csv"));
        let e = refused(&PlotArgs { width: Some(4097), ..gone.clone() });
        assert_eq!(e, "svg plot size must be 1x1 to 4096x4096 pixels, got 4097x600");
        let e = refused(&PlotArgs { height: Some(7), ..args(&gone.data) });
        assert_eq!(e, "terminal plot size must be 20x8 to 1000x1000 cells, got 80x7");
        assert!(refused(&gone).starts_with("cannot read data /nonexistent/plot.csv"));
        let bad = refused(&args(&tmp("bad.csv", "x,y\n1\n")));
        assert!(bad.ends_with("bad.csv: line 2: 1 fields, the header has 2"), "{bad}");
    }

    #[test]
    fn write_and_engine_failures_are_messages() {
        let mut full: &mut [u8] = &mut [];
        let r = run(&args(&tmp("full.csv", ROW1)), &Env::default(), &mut full, &mut Vec::new());
        assert_eq!(r, Err("cannot write stdout: failed to write whole buffer".into()));
        let wide = tmp("wide.csv", "x,y\n0,-70368744177664\n1,-70368828063744\n");
        let e =
            go(&PlotArgs { width: Some(20), height: Some(8), ..args(&wide) }, &Env::default()).0;
        assert!(e.expect_err("no room").contains("no room for the panel"));
    }

    #[test]
    fn env_is_read_from_the_process() {
        std::env::set_var("NO_COLOR", "1"); // nothing else in this crate reads it
        assert!(Env::from_process().no_color);
    }
}
