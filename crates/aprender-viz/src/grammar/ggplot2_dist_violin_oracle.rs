//! ggplot2 4.0.3 oracle for the `geom_violin` row (PLOT-GG-001, paiml/aprender#6909):
//! `stat_ydensity` + `GeomViolin` + `position_dodge` through [`GGPlot::layout`]. Fixtures are
//! committed under `fixtures/ggplot2/geom_violin/` (made by R 4.6.2 with ggplot2 4.0.3); R is
//! never run here. Tolerance and helpers are those of `ggplot2_oracle.rs`: 1e-9 relative with a
//! 1e-12 absolute floor, the same for every case.

use serde_json::Value;

use super::ggplot2_oracle::{assert_close_slice, check_axis, close, nums, strs, ABS_TOL, REL_TOL};
use super::{Aes, BuiltLayout, DataFrame, GGPlot, Geom, Layer, PositionScale, ViolinScale};

/// A CSV with numeric and text columns; `NA` is a missing number. Numbers must be exact in f32.
pub(super) fn frame_from_csv(csv: &str) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::trim).collect())
        .collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[c]).collect();
        let numeric = cells.iter().all(|s| *s == "NA" || s.parse::<f64>().is_ok());
        if numeric {
            let col: Vec<f32> = cells
                .iter()
                .map(|s| {
                    let v: f64 = if *s == "NA" { f64::NAN } else { s.parse().expect("number") };
                    let f = v as f32;
                    assert!(
                        v.is_nan() || f64::from(f) == v,
                        "fixture value {v} is not exact in f32"
                    );
                    f
                })
                .collect();
            df.add_column_f32(name.trim(), &col);
        } else {
            df.add_column_str(name.trim(), &cells);
        }
    }
    df
}

/// A fixture column; `null` (R's `NA`) is `NaN`.
fn nums_na(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| if e.is_null() { f64::NAN } else { e.as_f64().expect("number") })
        .collect()
}

fn run(csv: &str, aes: Aes, geom: Geom) -> BuiltLayout {
    GGPlot::new()
        .data(frame_from_csv(csv))
        .aes(aes)
        .layer(Layer::new(geom))
        .build()
        .expect("builds")
        .layout()
        .expect("layout")
}

fn close_or_both_missing(g: f64, w: f64) -> bool {
    (g.is_nan() && w.is_nan()) || close(g, w)
}

/// The three figures reported for a case: the largest relative error over every value (zero
/// against zero counts as no error; a value R floors to zero where this port holds rounding noise
/// counts as 1), the largest absolute error, and the largest share of the tolerance used, the
/// error divided by `1e-9 * max(|got|, |want|) + 1e-12` (the rule of [`close`]; at most 1 passes).
type Figures = (f64, f64, f64);

/// The [`Figures`] of one value pair.
fn err(g: f64, w: f64) -> Figures {
    if g.is_nan() && w.is_nan() {
        return (0.0, 0.0, 0.0);
    }
    let (abs, mag) = ((g - w).abs(), g.abs().max(w.abs()));
    (if mag > 0.0 { abs / mag } else { 0.0 }, abs, abs / (REL_TOL * mag + ABS_TOL))
}

fn worse(a: Figures, b: Figures) -> Figures {
    (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2))
}

fn check_x_axis(case: &str, got: &PositionScale, want: &Value) -> Figures {
    let kind = want["kind"].as_str().expect("axis kind");
    assert_eq!(got.levels.is_some(), kind == "discrete", "{case}.x kind");
    if let Some(levels) = &got.levels {
        assert_eq!(levels, &strs(&want["limits"]), "{case}.x levels");
    }
    let range = [got.continuous_range.0, got.continuous_range.1];
    let w_range = nums(&want["continuous_range"]);
    assert_close_slice(&format!("{case}.x.continuous_range"), &range, &w_range);
    assert_close_slice(&format!("{case}.x.breaks"), &got.breaks, &nums(&want["breaks"]));
    assert_eq!(got.labels, strs(&want["labels"]), "{case}.x labels");
    assert_eq!(
        got.n_breaks_censored as u64,
        want["n_breaks_censored"].as_u64().expect("count"),
        "{case}.x censored"
    );
    if kind == "continuous" {
        let lim = [got.limits.0, got.limits.1];
        assert_close_slice(&format!("{case}.x.limits"), &lim, &nums(&want["limits"]));
    }
    range.iter().zip(&w_range).map(|(g, w)| err(*g, *w)).fold((0.0, 0.0, 0.0), worse)
}

/// Compare every column of the layer and both axes; returns the [`Figures`] over all of them.
fn check_columns(case: &str, got: &BuiltLayout, want: &Value) -> Figures {
    let (g, w) = (&got.layers[0], &want["layers"][0]);
    assert_eq!(w["geom"], "GeomViolin", "{case}");
    let v = g.violin.as_ref().expect("a violin layer carries its columns");
    let n = g.x.len();
    let width = vec![v.width; n];
    let columns: [(&str, &[f64]); 15] = [
        ("x", &g.x),
        ("y", &g.y),
        ("ymax", &g.y),
        ("density", &v.density),
        ("scaled", &v.scaled),
        ("ndensity", &v.scaled),
        ("count", &v.count),
        ("wdensity", &v.wdensity),
        ("n", &v.n),
        ("violinwidth", &v.violinwidth),
        ("width", &width),
        ("xmin", &v.xmin),
        ("xmax", &v.xmax),
        ("order", &v.order),
        ("quantile", &v.quantile),
    ];
    let mut worst = (0.0_f64, 0.0_f64, 0.0_f64);
    for (name, got_col) in columns {
        let want_col = nums_na(&w[name]);
        assert_eq!(got_col.len(), want_col.len(), "{case}.{name}: length");
        for (i, (a, b)) in got_col.iter().zip(&want_col).enumerate() {
            assert!(close_or_both_missing(*a, *b), "{case}.{name}[{i}]: got {a:?} want {b:?}");
            worst = worse(worst, err(*a, *b));
        }
    }
    worst = worse(worst, check_x_axis(case, &got.x, &want["panel"]["x"]));
    check_axis(&format!("{case}.y"), &got.y, &want["panel"]["y"]);
    assert_eq!(got.y.levels, None, "{case}.y is continuous");
    worst
}

fn check(case: &str, json: &str, csv: &str, aes: Aes, geom: Geom) -> Figures {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "{case}: not the pinned ggplot2");
    assert_eq!(want["versions"]["collate"], "C", "{case}: fixture not made under C collation");
    check_columns(case, &run(csv, aes, geom), &want)
}

fn gy() -> Aes {
    Aes::new().x("g").y("y")
}

fn gyf() -> Aes {
    gy().fill("f")
}

/// A committed fixture file of this row.
macro_rules! fixture {
    ($file:literal) => {
        include_str!(concat!("../../fixtures/ggplot2/geom_violin/", $file))
    };
}

/// One case: its name, `expected_<name>.json`, the data and the plot; prints the figure the PR
/// body quotes under `--nocapture`.
fn case(name: &str, json: &str, csv: &str, aes: Aes, geom: Geom) {
    let (rel, abs, share) = check(name, json, csv, aes, geom);
    println!(
        "violin {name} vs R: relative {rel:e} (all values) absolute {abs:e} tolerance share {share:e}"
    );
}

#[test]
fn single() {
    case(
        "single",
        fixture!("expected_single.json"),
        fixture!("data_single.csv"),
        gy(),
        Geom::gg_violin(),
    );
}

#[test]
fn three() {
    case(
        "three",
        fixture!("expected_three.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin(),
    );
}

#[test]
fn count() {
    case(
        "count",
        fixture!("expected_count.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().violin_scale(ViolinScale::Count),
    );
}

#[test]
fn width() {
    case(
        "width",
        fixture!("expected_width.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().violin_scale(ViolinScale::Width),
    );
}

#[test]
fn trim_false() {
    case(
        "trim_false",
        fixture!("expected_trim_false.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().trim(false),
    );
}

#[test]
fn trim_adjust() {
    case(
        "trim_adjust",
        fixture!("expected_trim_adjust.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().trim(false).adjust(0.5),
    );
}

#[test]
fn adjust_half() {
    case(
        "adjust_half",
        fixture!("expected_adjust_half.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().adjust(0.5),
    );
}

#[test]
fn width_half() {
    case(
        "width_half",
        fixture!("expected_width_half.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().width(0.5),
    );
}

// Wider than the spacing of the levels: ggplot2 warns about overlap and widens the x range.
#[test]
fn width_wide() {
    case(
        "width_wide",
        fixture!("expected_width_wide.json"),
        fixture!("data_three.csv"),
        gy(),
        Geom::gg_violin().width(2.0),
    );
}

#[test]
fn fill() {
    case(
        "fill",
        fixture!("expected_fill.json"),
        fixture!("data_three.csv"),
        gyf(),
        Geom::gg_violin(),
    );
}

// x = a holds three fill groups and x = b two: each x splits its own width.
#[test]
fn dodge3() {
    case(
        "dodge3",
        fixture!("expected_dodge3.json"),
        fixture!("data_dodge3.csv"),
        gyf(),
        Geom::gg_violin(),
    );
}

#[test]
fn numeric_x() {
    case(
        "numeric_x",
        fixture!("expected_numeric_x.json"),
        fixture!("data_three.csv"),
        Aes::new().x("xn").y("y").group("g"),
        Geom::gg_violin(),
    );
}

#[test]
fn na() {
    case("na", fixture!("expected_na.json"), fixture!("data_na.csv"), gy(), Geom::gg_violin());
}

// A group of one row is dropped (its level stays on the axis); a group of two is kept.
#[test]
fn tiny() {
    case(
        "tiny",
        fixture!("expected_tiny.json"),
        fixture!("data_tiny.csv"),
        gy(),
        Geom::gg_violin(),
    );
}

// Constant y: every grid point is a quantile, so only the three quantile rows remain.
#[test]
fn constant() {
    case(
        "constant",
        fixture!("expected_constant.json"),
        fixture!("data_constant.csv"),
        gy(),
        Geom::gg_violin(),
    );
}

#[test]
fn constant_only() {
    case(
        "constant_only",
        fixture!("expected_constant_only.json"),
        fixture!("data_constant_only.csv"),
        gy(),
        Geom::gg_violin(),
    );
}

/// `csv` with column `col` replaced by `f(row index, value)` for every data row, as R's `edit` in
/// `generate.R` does (`inf` is how Rust prints an infinity, and parses it back).
fn edit_csv(csv: &str, col: &str, f: impl Fn(usize, f64) -> f64) -> String {
    let mut lines = csv.lines();
    let header = lines.next().expect("csv has a header");
    let at = header.split(',').position(|h| h.trim() == col).expect("column in the csv");
    let mut out = vec![header.to_string()];
    for (i, line) in lines.filter(|l| !l.trim().is_empty()).enumerate() {
        let mut cells: Vec<String> = line.split(',').map(|c| c.trim().to_string()).collect();
        let v: f64 = cells[at].parse().expect("numeric cell");
        cells[at] = f(i, v).to_string();
        out.push(cells.join(","));
    }
    out.join("\n")
}

/// `data_three.csv` with the same edit `generate.R` applies (its `edited` list).
fn three_edited(col: &str, f: impl Fn(usize, f64) -> f64) -> String {
    edit_csv(fixture!("data_three.csv"), col, f)
}

fn xn_gaps(csv: &str) -> String {
    // `xn = c(2^-30, 2^-29, 3 * 2^-30)[match(g, c('a','b','c'))]`: the first cell of each group.
    let gaps = [(1.0 / 1_073_741_824.0), (1.0 / 536_870_912.0), 3.0 * (1.0 / 1_073_741_824.0)];
    let mut lines = csv.lines();
    let header = lines.next().expect("csv has a header").to_string();
    let mut out = vec![header];
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let mut cells: Vec<String> = line.split(',').map(|c| c.trim().to_string()).collect();
        let at = ["a", "b", "c"].iter().position(|g| *g == cells[0]).expect("group a, b or c");
        cells[3] = gaps[at].to_string();
        out.push(cells.join(","));
    }
    out.join("\n")
}

fn xn_gy() -> Aes {
    Aes::new().x("xn").y("y").group("g")
}

#[test]
fn small_unit_8() {
    let csv = three_edited("y", |_, v| v * (1.0 / 256.0));
    case("small_unit_8", fixture!("expected_small_unit_8.json"), &csv, gy(), Geom::gg_violin());
}

#[test]
fn large_unit() {
    let csv = three_edited("y", |_, v| v * 16384.0);
    case("large_unit", fixture!("expected_large_unit.json"), &csv, gy(), Geom::gg_violin());
}

#[test]
fn offset() {
    let csv = three_edited("y", |_, v| (v * 16.0).floor() / 16.0 + 1e6);
    case("offset", fixture!("expected_offset.json"), &csv, gy(), Geom::gg_violin());
}

// R's `resolution()` is 1 when no gap exceeds sqrt(eps): the width stays 0.9.
#[test]
fn tiny_x_gaps() {
    let csv = xn_gaps(fixture!("data_three.csv"));
    case("tiny_x_gaps", fixture!("expected_tiny_x_gaps.json"), &csv, xn_gy(), Geom::gg_violin());
}

// R drops infinite rows (`remove_missing(finite = TRUE)`) and keeps the group numbering.
#[test]
fn inf_y() {
    let csv = three_edited("y", |i, v| match i {
        1 => f64::INFINITY,
        14 => f64::NEG_INFINITY,
        _ => v,
    });
    case("inf_y", fixture!("expected_inf_y.json"), &csv, gy(), Geom::gg_violin());
}

#[test]
fn inf_x() {
    let csv = three_edited("xn", |i, v| match i {
        2 => f64::INFINITY,
        19 => f64::NEG_INFINITY,
        _ => v,
    });
    case("inf_x", fixture!("expected_inf_x.json"), &csv, xn_gy(), Geom::gg_violin());
}

/// A case R computes but this port refuses: the fixture must be the pinned one, and the refusal
/// must be the typed noise-floor error (never an `Ok` that differs from R).
fn refused(name: &str, json: &str, csv: &str, geom: Geom) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "{name}: not the pinned ggplot2");
    let err = GGPlot::new()
        .data(frame_from_csv(csv))
        .aes(gy())
        .layer(Layer::new(geom))
        .build()
        .expect("builds")
        .layout()
        .expect_err("refused");
    assert!(err.to_string().contains("too concentrated"), "{name}: {err}");
}

// y * 2^-14: R's tails are FFT rounding noise this port cannot reproduce, so the group is refused.
#[test]
fn small_unit_refused() {
    let csv = three_edited("y", |_, v| v * (1.0 / 16384.0));
    refused("small_unit", fixture!("expected_small_unit.json"), &csv, Geom::gg_violin());
}

#[test]
fn small_unit_adjust_refused() {
    let csv = three_edited("y", |_, v| v * (1.0 / 16384.0));
    refused(
        "small_unit_adjust",
        fixture!("expected_small_unit_adjust.json"),
        &csv,
        Geom::gg_violin().adjust(0.5),
    );
}
