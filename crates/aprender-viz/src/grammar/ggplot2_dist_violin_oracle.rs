//! ggplot2 4.0.3 oracle for the `geom_violin` row (PLOT-GG-001, paiml/aprender#6909):
//! `stat_ydensity` + `GeomViolin` + `position_dodge` through [`GGPlot::layout`]. Fixtures are
//! committed under `fixtures/ggplot2/geom_violin/` (made by R 4.6.2 with ggplot2 4.0.3); R is
//! never run here. Tolerance and helpers are those of `ggplot2_oracle.rs`: 1e-9 relative with a
//! 1e-12 absolute floor, the same for every case.

use serde_json::Value;

use super::ggplot2_oracle::{assert_close_slice, check_axis, close, nums, strs};
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

/// `(relative, absolute)` error of one value. The relative figure is only meaningful away from
/// zero, so values under 1e-6 contribute to the absolute figure alone.
fn err(g: f64, w: f64) -> (f64, f64) {
    if g.is_nan() && w.is_nan() {
        return (0.0, 0.0);
    }
    let (abs, mag) = ((g - w).abs(), g.abs().max(w.abs()));
    (if mag < 1e-6 { 0.0 } else { abs / mag }, abs)
}

fn worse(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0.max(b.0), a.1.max(b.1))
}

fn check_x_axis(case: &str, got: &PositionScale, want: &Value) -> (f64, f64) {
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
    range.iter().zip(&w_range).map(|(g, w)| err(*g, *w)).fold((0.0, 0.0), worse)
}

/// Compare every column of the layer and both axes; returns the largest relative error seen.
fn check_columns(case: &str, got: &BuiltLayout, want: &Value) -> (f64, f64) {
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
    let mut worst = (0.0_f64, 0.0_f64);
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

fn check(case: &str, json: &str, csv: &str, aes: Aes, geom: Geom) -> (f64, f64) {
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
    let (rel, abs) = check(name, json, csv, aes, geom);
    println!("violin {name} vs R: relative {rel:e} absolute {abs:e}");
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
