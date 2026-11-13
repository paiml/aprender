//! ggplot2 conformance oracle for the distribution rows (PLOT-GG-001, paiml/aprender#6909):
//! `geom_histogram`, `geom_freqpoly`, `geom_density`, `geom_boxplot`, `geom_errorbar`. Fixtures are
//! `fixtures/ggplot2/<row>/expected_<case>.json`, produced by ggplot2 4.0.3 / R 4.6 under C
//! collation (`dist_oracle.R` and each row's `generate.R`); R is never run here.
//!
//! Tolerance: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for every number; groups, panels, counts,
//! colours and the set of columns compare exactly. `NaN` matches `null` (R's `NA`) by position.
//! Every case is built through the public builder and read back with `layout()`.

use serde_json::Value;

use super::coord::Coord;
use super::{
    Aes, BuiltDist, BuiltLayout, Closed, DataFrame, Facet, GGPlot, Geom, Layer, Paint,
    PositionScale,
};
use crate::color::Rgba;
use crate::error::{Error, Result};

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
    }
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| if e.is_null() { f64::NAN } else { e.as_f64().expect("fixture number") })
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture string").to_string())
        .collect()
}

/// Parse a CSV into a frame. A column whose every non-`NA` cell parses as a number is numeric
/// (and each value must be exact in f32); any other column is text. `NA` in a numeric column is
/// `NaN`.
fn frame_from_csv(csv: &str) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::trim).collect())
        .collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[c]).collect();
        if cells.iter().all(|s| *s == "NA" || s.parse::<f64>().is_ok()) {
            let col: Vec<f32> = cells
                .iter()
                .map(|s| {
                    if *s == "NA" {
                        return f32::NAN;
                    }
                    let v: f64 = s.parse().expect("numeric cell");
                    let f = v as f32;
                    assert_eq!(f64::from(f), v, "fixture value {v} is not exact in f32");
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

macro_rules! fixture {
    ($row:literal, $case:literal) => {
        include_str!(concat!("../../fixtures/ggplot2/", $row, "/expected_", $case, ".json"))
    };
}

macro_rules! csv {
    ($row:literal, $file:literal) => {
        frame_from_csv(include_str!(concat!("../../fixtures/ggplot2/", $row, "/", $file, ".csv")))
    };
}

fn want(json: &str, row: &str, case: &str) -> Value {
    let v: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["collation"], "C", "fixture was not made under C collation");
    assert_eq!(v["row"], row);
    assert_eq!(v["case"], case);
    v
}

fn layout_of(data: DataFrame, aes: Aes, geom: Geom) -> Result<BuiltLayout> {
    GGPlot::new().data(data).aes(aes).layer(Layer::new(geom)).build()?.layout()
}

fn check_axis(name: &str, got: &PositionScale, w: &Value) {
    assert_close_slice(
        &format!("{name}.limits"),
        &[got.limits.0, got.limits.1],
        &nums(&w["limits"]),
    );
    assert_close_slice(
        &format!("{name}.continuous_range"),
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&w["continuous_range"]),
    );
    assert_close_slice(&format!("{name}.breaks"), &got.breaks, &nums(&w["breaks"]));
    assert_eq!(got.labels, strs(&w["labels"]), "{name}.labels");
    let censored = w["n_breaks_censored"].as_u64().expect("count");
    assert_eq!(got.n_breaks_censored as u64, censored, "{name}.n_breaks_censored");
}

fn rgba(p: Paint) -> Option<[f64; 4]> {
    match p {
        Paint::Fixed(c) => Some([c.r, c.g, c.b, c.a].map(f64::from)),
        Paint::Na | Paint::Mapped => None,
    }
}

fn check_paint(tag: &str, got: &[Paint], w: &Value) {
    match w {
        Value::Null => assert!(
            got.iter().all(|p| *p == Paint::Na),
            "{tag}: fixture has no colour, got {got:?}"
        ),
        Value::Array(rows) => {
            assert_eq!(rows.len(), got.len(), "{tag}: row count");
            for (i, (g, r)) in got.iter().zip(rows).enumerate() {
                match g {
                    Paint::Na => panic!("{tag}[{i}]: ggplot2 has a colour, got NA"),
                    // A mapped aesthetic goes through a palette this crate does not resolve.
                    Paint::Mapped => {}
                    Paint::Fixed(_) => {
                        assert_eq!(rgba(*g).map(|a| a.to_vec()), Some(nums(r)), "{tag}[{i}]");
                    }
                }
            }
        }
        other => panic!("{tag}: unexpected fixture value {other:?}"),
    }
}

/// Compare one layer against the fixture; `skip` names fixture columns this crate does not emit.
fn check_layer(tag: &str, got: &BuiltDist, w: &Value, geom: &str, skip: &[&str]) {
    assert_eq!(w["geom"], geom, "{tag}: R geom class");
    assert_eq!(got.geom, geom, "{tag}: built geom class");
    let obj = w.as_object().expect("layer is an object");
    let numeric: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|k| {
            !["geom", "panel", "group", "outliers", "colour_rgba", "fill_rgba"].contains(k)
                && !skip.contains(k)
        })
        .collect();
    let mut have: Vec<&str> = got.cols.iter().map(|(n, _)| *n).filter(|n| *n != "panel").collect();
    have.sort_unstable();
    let mut expect = numeric.clone();
    expect.sort_unstable();
    assert_eq!(have, expect, "{tag}: column set");
    for k in &numeric {
        assert_close_slice(&format!("{tag}.{k}"), got.col(k).expect("column"), &nums(&w[*k]));
    }
    let panel = nums(&w["panel"]);
    assert_eq!(panel.len(), got.nrow(), "{tag}: panel rows");
    assert_close_slice(&format!("{tag}.panel"), got.col("panel").expect("panel"), &panel);
    let groups: Vec<i64> = nums(&w["group"]).iter().map(|g| *g as i64).collect();
    assert_eq!(got.group, groups, "{tag}.group");
    if let Some(out) = w.get("outliers") {
        let want_out: Vec<Vec<f64>> = out.as_array().expect("array").iter().map(nums).collect();
        assert_eq!(got.outliers.len(), want_out.len(), "{tag}.outliers rows");
        for (i, (g, o)) in got.outliers.iter().zip(&want_out).enumerate() {
            assert_close_slice(&format!("{tag}.outliers[{i}]"), g, o);
        }
    } else {
        assert!(got.outliers.is_empty(), "{tag}: unexpected outliers");
    }
    check_paint(
        &format!("{tag}.colour"),
        &got.colour,
        obj.get("colour_rgba").unwrap_or(&Value::Null),
    );
    check_paint(&format!("{tag}.fill"), &got.fill, obj.get("fill_rgba").unwrap_or(&Value::Null));
}

fn run(
    json: &str,
    row: &str,
    case: &str,
    class: &str,
    skip: &[&str],
    data: DataFrame,
    aes: Aes,
    geom: Geom,
) {
    let w = want(json, row, case);
    let tag = format!("{row}/{case}");
    let got = layout_of(data, aes, geom).unwrap_or_else(|e| panic!("{tag}: {e}"));
    let layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), layers.len(), "{tag}: layer count");
    let d = got.layers[0].dist.as_ref().expect("a distribution layer");
    check_layer(&tag, d, &layers[0], class, skip);
    check_axis(&format!("{tag}.x"), &got.x, &w["panel"]["x"]);
    check_axis(&format!("{tag}.y"), &got.y, &w["panel"]["y"]);
}

fn ax() -> Aes {
    Aes::new().x("x")
}

// --- geom_histogram (stat_bin, stacked) -----------------------------------------------------

/// `width` of a histogram bar is GeomBar's 0.9 default, which has no meaning after `stat_bin`.
const HIST_SKIP: &[&str] = &["width"];

#[test]
fn hist_default() {
    run(
        fixture!("geom_histogram", "default"),
        "geom_histogram",
        "default",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram(),
    );
}
#[test]
fn hist_bins5() {
    run(
        fixture!("geom_histogram", "bins5"),
        "geom_histogram",
        "bins5",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram().bins(5),
    );
}
#[test]
fn hist_binwidth_boundary() {
    run(
        fixture!("geom_histogram", "binwidth_boundary"),
        "geom_histogram",
        "binwidth_boundary",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram().binwidth(2.5).boundary(0.0),
    );
}
#[test]
fn hist_binwidth_center() {
    run(
        fixture!("geom_histogram", "binwidth_center"),
        "geom_histogram",
        "binwidth_center",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram().binwidth(3.0).center(0.0),
    );
}
#[test]
fn hist_breaks() {
    run(
        fixture!("geom_histogram", "breaks"),
        "geom_histogram",
        "breaks",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram().breaks(&[0.0, 2.0, 4.0, 8.0, 16.0]),
    );
}
#[test]
fn hist_closed_right() {
    run(
        fixture!("geom_histogram", "closed_right"),
        "geom_histogram",
        "closed_right",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram().binwidth(2.0).boundary(0.0),
    );
}
#[test]
fn hist_closed_left() {
    run(
        fixture!("geom_histogram", "closed_left"),
        "geom_histogram",
        "closed_left",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax(),
        Geom::gg_histogram().binwidth(2.0).boundary(0.0).closed(Closed::Left),
    );
}
#[test]
fn hist_fill_two_groups() {
    run(
        fixture!("geom_histogram", "fill_two_groups"),
        "geom_histogram",
        "fill_two_groups",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax().fill("g"),
        Geom::gg_histogram().bins(4),
    );
}
#[test]
fn hist_fill_small_groups() {
    run(
        fixture!("geom_histogram", "fill_small_groups"),
        "geom_histogram",
        "fill_small_groups",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax().fill("h"),
        Geom::gg_histogram().binwidth(2.5).boundary(0.0),
    );
}
#[test]
fn hist_na_x() {
    run(
        fixture!("geom_histogram", "na_x"),
        "geom_histogram",
        "na_x",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data_na"),
        ax(),
        Geom::gg_histogram().binwidth(2.0).boundary(0.0),
    );
}
#[test]
fn hist_const_x() {
    run(
        fixture!("geom_histogram", "const_x"),
        "geom_histogram",
        "const_x",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data_const"),
        ax(),
        Geom::gg_histogram(),
    );
}
#[test]
fn hist_colour_alpha() {
    run(
        fixture!("geom_histogram", "colour_alpha"),
        "geom_histogram",
        "colour_alpha",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data"),
        ax().color_value(Rgba::new(0, 0, 255, 255)).alpha_value(0.25),
        Geom::gg_histogram().bins(5),
    );
}

// --- geom_freqpoly (stat_bin, pad) ----------------------------------------------------------

#[test]
fn freq_default() {
    run(
        fixture!("geom_freqpoly", "default"),
        "geom_freqpoly",
        "default",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax(),
        Geom::gg_freqpoly(),
    );
}
#[test]
fn freq_bins5() {
    run(
        fixture!("geom_freqpoly", "bins5"),
        "geom_freqpoly",
        "bins5",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax(),
        Geom::gg_freqpoly().bins(5),
    );
}
#[test]
fn freq_binwidth_boundary() {
    run(
        fixture!("geom_freqpoly", "binwidth_boundary"),
        "geom_freqpoly",
        "binwidth_boundary",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax(),
        Geom::gg_freqpoly().binwidth(2.5).boundary(0.0),
    );
}
#[test]
fn freq_breaks() {
    run(
        fixture!("geom_freqpoly", "breaks"),
        "geom_freqpoly",
        "breaks",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax(),
        Geom::gg_freqpoly().breaks(&[0.0, 2.0, 4.0, 8.0, 16.0]),
    );
}
#[test]
fn freq_closed_left() {
    run(
        fixture!("geom_freqpoly", "closed_left"),
        "geom_freqpoly",
        "closed_left",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax(),
        Geom::gg_freqpoly().binwidth(2.0).boundary(0.0).closed(Closed::Left),
    );
}
#[test]
fn freq_pad_false() {
    run(
        fixture!("geom_freqpoly", "pad_false"),
        "geom_freqpoly",
        "pad_false",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax(),
        Geom::gg_freqpoly().bins(5).pad(false),
    );
}
#[test]
fn freq_colour_two_groups() {
    run(
        fixture!("geom_freqpoly", "colour_two_groups"),
        "geom_freqpoly",
        "colour_two_groups",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax().color("g"),
        Geom::gg_freqpoly().bins(4),
    );
}
#[test]
fn freq_colour_small_groups() {
    run(
        fixture!("geom_freqpoly", "colour_small_groups"),
        "geom_freqpoly",
        "colour_small_groups",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data"),
        ax().color("h"),
        Geom::gg_freqpoly().binwidth(2.5).boundary(0.0),
    );
}
#[test]
fn freq_na_x() {
    run(
        fixture!("geom_freqpoly", "na_x"),
        "geom_freqpoly",
        "na_x",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data_na"),
        ax(),
        Geom::gg_freqpoly().binwidth(2.0).boundary(0.0),
    );
}
#[test]
fn freq_const_x() {
    run(
        fixture!("geom_freqpoly", "const_x"),
        "geom_freqpoly",
        "const_x",
        "GeomPath",
        &[],
        csv!("geom_freqpoly", "data_const"),
        ax(),
        Geom::gg_freqpoly(),
    );
}

// --- geom_density (stat_density) ------------------------------------------------------------

#[test]
fn dens_default() {
    run(
        fixture!("geom_density", "default"),
        "geom_density",
        "default",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax(),
        Geom::gg_density(),
    );
}
#[test]
fn dens_bw_number() {
    run(
        fixture!("geom_density", "bw_number"),
        "geom_density",
        "bw_number",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax(),
        Geom::gg_density().bw(0.75),
    );
}
#[test]
fn dens_adjust2() {
    run(
        fixture!("geom_density", "adjust2"),
        "geom_density",
        "adjust2",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax(),
        Geom::gg_density().adjust(2.0),
    );
}
#[test]
fn dens_n64() {
    run(
        fixture!("geom_density", "n64"),
        "geom_density",
        "n64",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax(),
        Geom::gg_density().n(64),
    );
}
#[test]
fn dens_fill_two_groups() {
    run(
        fixture!("geom_density", "fill_two_groups"),
        "geom_density",
        "fill_two_groups",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax().fill("g").alpha_value(0.5),
        Geom::gg_density(),
    );
}
#[test]
fn dens_na_x() {
    run(
        fixture!("geom_density", "na_x"),
        "geom_density",
        "na_x",
        "GeomDensity",
        &[],
        csv!("geom_density", "data_na"),
        ax(),
        Geom::gg_density(),
    );
}
#[test]
fn dens_trim_two_groups() {
    run(
        fixture!("geom_density", "trim_two_groups"),
        "geom_density",
        "trim_two_groups",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax().fill("g"),
        Geom::gg_density().trim(true),
    );
}
#[test]
fn dens_trim() {
    run(
        fixture!("geom_density", "trim"),
        "geom_density",
        "trim",
        "GeomDensity",
        &[],
        csv!("geom_density", "data"),
        ax(),
        Geom::gg_density().trim(true),
    );
}

// --- geom_boxplot (stat_boxplot, position_dodge2) -------------------------------------------

fn ay() -> Aes {
    Aes::new().y("v")
}

#[test]
fn box_y_only() {
    run(
        fixture!("geom_boxplot", "y_only"),
        "geom_boxplot",
        "y_only",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay(),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_fill_two_groups() {
    run(
        fixture!("geom_boxplot", "fill_two_groups"),
        "geom_boxplot",
        "fill_two_groups",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay().fill("g"),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_fill_dodge_small() {
    run(
        fixture!("geom_boxplot", "fill_dodge_small"),
        "geom_boxplot",
        "fill_dodge_small",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay().fill("k"),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_x_continuous_group() {
    run(
        fixture!("geom_boxplot", "x_continuous_group"),
        "geom_boxplot",
        "x_continuous_group",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay().x("x").group("g"),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_x_varies_in_group() {
    run(
        fixture!("geom_boxplot", "x_varies_in_group"),
        "geom_boxplot",
        "x_varies_in_group",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data_xvar"),
        Aes::new().x("x").y("v").group("g"),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_coef_3() {
    run(
        fixture!("geom_boxplot", "coef_3"),
        "geom_boxplot",
        "coef_3",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay(),
        Geom::gg_boxplot().coef(3.0),
    );
}
#[test]
fn box_coef_0() {
    run(
        fixture!("geom_boxplot", "coef_0"),
        "geom_boxplot",
        "coef_0",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay(),
        Geom::gg_boxplot().coef(0.0),
    );
}
#[test]
fn box_width_param() {
    run(
        fixture!("geom_boxplot", "width_param"),
        "geom_boxplot",
        "width_param",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data"),
        ay(),
        Geom::gg_boxplot().width(0.4),
    );
}
#[test]
fn box_na_y() {
    run(
        fixture!("geom_boxplot", "na_y"),
        "geom_boxplot",
        "na_y",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data_na"),
        ay(),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_na_y_groups() {
    run(
        fixture!("geom_boxplot", "na_y_groups"),
        "geom_boxplot",
        "na_y_groups",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data_na"),
        ay().fill("g"),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_two_rows() {
    run(
        fixture!("geom_boxplot", "two_rows"),
        "geom_boxplot",
        "two_rows",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data_two"),
        ay().fill("g"),
        Geom::gg_boxplot(),
    );
}

// --- geom_errorbar (identity) ---------------------------------------------------------------

fn ae() -> Aes {
    Aes::new().x("x").ymin("lo").ymax("hi")
}

#[test]
fn err_default() {
    run(
        fixture!("geom_errorbar", "default"),
        "geom_errorbar",
        "default",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data"),
        ae(),
        Geom::gg_errorbar(),
    );
}
#[test]
fn err_width_param() {
    run(
        fixture!("geom_errorbar", "width_param"),
        "geom_errorbar",
        "width_param",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data"),
        ae(),
        Geom::gg_errorbar().width(0.3),
    );
}
#[test]
fn err_colour_groups() {
    run(
        fixture!("geom_errorbar", "colour_groups"),
        "geom_errorbar",
        "colour_groups",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data"),
        ae().color("g"),
        Geom::gg_errorbar(),
    );
}
#[test]
fn err_linewidth_colour() {
    run(
        fixture!("geom_errorbar", "linewidth_colour"),
        "geom_errorbar",
        "linewidth_colour",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data"),
        ae().color_value(Rgba::new(255, 0, 0, 255)),
        Geom::gg_errorbar().linewidth(1.0),
    );
}
#[test]
fn err_na_bounds() {
    run(
        fixture!("geom_errorbar", "na_bounds"),
        "geom_errorbar",
        "na_bounds",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data_na"),
        ae(),
        Geom::gg_errorbar(),
    );
}
#[test]
fn err_one_row() {
    run(
        fixture!("geom_errorbar", "one_row"),
        "geom_errorbar",
        "one_row",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data_one"),
        ae(),
        Geom::gg_errorbar(),
    );
}
#[test]
fn err_ties_groups() {
    run(
        fixture!("geom_errorbar", "ties_groups"),
        "geom_errorbar",
        "ties_groups",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data_ties"),
        ae().color("g"),
        Geom::gg_errorbar(),
    );
}
#[test]
fn err_inverted() {
    run(
        fixture!("geom_errorbar", "inverted"),
        "geom_errorbar",
        "inverted",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data_ties"),
        Aes::new().x("x").ymin("hi").ymax("lo"),
        Geom::gg_errorbar(),
    );
}

// --- typed refusals -------------------------------------------------------------------------

fn all_geoms() -> Vec<(&'static str, Geom, Aes)> {
    vec![
        ("hist", Geom::gg_histogram(), ax()),
        ("freq", Geom::gg_freqpoly(), ax()),
        ("dens", Geom::gg_density(), ax()),
        ("box", Geom::gg_boxplot(), Aes::new().y("x")),
        ("err", Geom::gg_errorbar(), Aes::new().x("x").ymin("x").ymax("x")),
    ]
}

fn small() -> DataFrame {
    DataFrame::from_xy(&[1.0, 2.0, 3.0, 4.0, 9.0], &[1.0, 2.0, 3.0, 4.0, 9.0])
}

fn plot_with(geom: Geom, aes: Aes, data: DataFrame) -> GGPlot {
    GGPlot::new().data(data).aes(aes).layer(Layer::new(geom))
}

#[test]
fn non_default_coord_is_a_typed_error() {
    for (name, g, a) in all_geoms() {
        for coord in [
            Coord::cartesian().flip(),
            Coord::cartesian().xlim(0.0, 1.0),
            Coord::polar(),
            Coord::fixed(1.0),
        ] {
            let r = plot_with(g.clone(), a.clone(), small())
                .coord(coord)
                .build()
                .and_then(|p| p.layout());
            assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name}: {r:?}");
        }
    }
}

#[test]
fn facets_are_refused() {
    for (name, g, a) in all_geoms() {
        let mut d = small();
        d.add_column_str("f", &["a", "b", "a", "b", "a"]);
        let r = plot_with(g, a, d).facet(Facet::wrap("f", 2)).build().and_then(|p| p.layout());
        assert!(matches!(r, Err(Error::Rendering(_))), "{name}: {r:?}");
    }
}

#[test]
fn mixing_with_other_geoms_is_refused() {
    let p = plot_with(Geom::gg_histogram(), ax(), small())
        .layer(Layer::new(Geom::point()).aes(Aes::new().x("x").y("y")));
    assert!(matches!(p.build().and_then(|p| p.layout()), Err(Error::Rendering(_))));
}

#[test]
fn text_and_infinite_and_unknown_columns_are_refused() {
    let mut d = small();
    d.add_column_str("t", &["a", "b", "c", "d", "e"]);
    for col in ["t", "nope"] {
        let r = layout_of(d.clone(), Aes::new().x(col), Geom::gg_histogram());
        assert!(r.is_err(), "{col}: {r:?}");
    }
    assert!(matches!(
        layout_of(d.clone(), Aes::new().x("nope"), Geom::gg_density()),
        Err(Error::UnknownColumn(_))
    ));
    let inf = DataFrame::from_xy(&[1.0, f32::INFINITY, 3.0], &[1.0, 2.0, 3.0]);
    for (name, g, a) in all_geoms() {
        let r = layout_of(inf.clone(), a, g);
        assert!(matches!(r, Err(Error::Rendering(_))), "{name}: {r:?}");
    }
}

#[test]
fn empty_and_all_missing_data_is_refused() {
    for (name, g, a) in all_geoms() {
        let empty = DataFrame::from_xy(&[], &[]);
        let r = layout_of(empty, a.clone(), g.clone());
        assert!(r.is_err(), "{name} empty: {r:?}");
        let na = DataFrame::from_xy(&[f32::NAN, f32::NAN], &[f32::NAN, f32::NAN]);
        let r = layout_of(na, a, g);
        assert!(r.is_err(), "{name} all-NA: {r:?}");
    }
}

#[test]
fn flipped_orientation_and_bad_parameters_are_refused() {
    let y_only = Aes::new().y("y");
    for g in [Geom::gg_histogram(), Geom::gg_freqpoly(), Geom::gg_density()] {
        let r = layout_of(small(), y_only.clone(), g);
        assert!(matches!(r, Err(Error::Rendering(_) | Error::EmptyData)), "{r:?}");
    }
    assert!(matches!(
        layout_of(small(), Aes::new().x("x"), Geom::gg_boxplot()),
        Err(Error::Rendering(_))
    ));
    let bad: Vec<(&str, Geom)> = vec![
        ("bins 0", Geom::gg_histogram().bins(0)),
        ("binwidth 0", Geom::gg_histogram().binwidth(0.0)),
        ("binwidth nan", Geom::gg_histogram().binwidth(f64::NAN)),
        ("tiny binwidth", Geom::gg_histogram().binwidth(1e-12)),
        ("boundary and center", Geom::gg_histogram().boundary(0.0).center(0.0)),
        ("one break", Geom::gg_histogram().breaks(&[1.0])),
        ("repeated breaks", Geom::gg_histogram().breaks(&[1.0, 1.0, 2.0])),
        ("infinite break", Geom::gg_histogram().breaks(&[1.0, f64::INFINITY])),
        ("huge bins", Geom::gg_histogram().bins(usize::MAX)),
    ];
    for (name, g) in bad {
        let r = layout_of(small(), ax(), g);
        assert!(matches!(r, Err(Error::Rendering(_))), "{name}: {r:?}");
    }
    let dens: Vec<(&str, Geom)> = vec![
        ("n 0", Geom::gg_density().n(0)),
        ("n huge", Geom::gg_density().n(usize::MAX)),
        ("adjust 0", Geom::gg_density().adjust(0.0)),
        ("bw 0", Geom::gg_density().bw(0.0)),
        ("bw nan", Geom::gg_density().bw(f64::NAN)),
    ];
    for (name, g) in dens {
        let r = layout_of(small(), ax(), g);
        assert!(matches!(r, Err(Error::Rendering(_))), "{name}: {r:?}");
    }
    for coef in [-1.0, f64::NAN, f64::INFINITY] {
        let r = layout_of(small(), Aes::new().y("y"), Geom::gg_boxplot().coef(coef));
        assert!(matches!(r, Err(Error::Rendering(_))), "coef {coef}");
    }
}

#[test]
fn continuous_or_doubled_discrete_mappings_are_refused() {
    let r = layout_of(small(), ax().fill("y"), Geom::gg_histogram());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    let mut d = small();
    d.add_column_str("a", &["p", "q", "p", "q", "p"]);
    d.add_column_str("b", &["u", "u", "v", "v", "v"]);
    let r = layout_of(d.clone(), ax().fill("a").color("b"), Geom::gg_histogram());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    // The same column for both aesthetics is one grouping, not two.
    assert!(layout_of(d, ax().fill("a").color("a"), Geom::gg_histogram()).is_ok());
}

#[test]
fn degenerate_and_tiny_groups_are_typed_errors() {
    // A density of one constant: ggplot2 widens the x scale, this crate refuses.
    let flat = DataFrame::from_xy(&[5.0; 6], &[1.0; 6]);
    assert!(matches!(layout_of(flat, ax(), Geom::gg_density()), Err(Error::ScaleDomain(_))));
    // A group of one point has no bandwidth: ggplot2 returns an NA row.
    let mut one = DataFrame::from_xy(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    one.add_column_str("g", &["a", "a", "b"]);
    assert!(matches!(layout_of(one, ax().fill("g"), Geom::gg_density()), Err(Error::Rendering(_))));
    // An error bar of one point at one height has a degenerate y scale.
    let pt = DataFrame::from_xy(&[1.0], &[1.0]);
    let r = layout_of(pt, Aes::new().x("x").ymin("y").ymax("y"), Geom::gg_errorbar());
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
}

// --- fuzz: extreme and non-finite f32, empty data -------------------------------------------

/// SplitMix64: a fixed stream, so a failure replays.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f32(&mut self) -> f32 {
        const SPECIAL: [f32; 14] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::EPSILON,
            1e30,
            -1e30,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            16_777_216.0,
        ];
        match self.next() % 4 {
            0 => SPECIAL[(self.next() % SPECIAL.len() as u64) as usize],
            1 => f32::from_bits(self.next() as u32),
            2 => (self.next() % 2000) as f32 / 10.0 - 100.0,
            _ => (self.next() % 7) as f32,
        }
    }
}

/// Every layout returns `Ok` or a typed error, never panics or hangs, and never builds more rows
/// than the cap.
#[test]
fn fuzz_layout_never_panics_and_stays_bounded() {
    let start = std::time::Instant::now();
    let mut rng = Rng(0x00C0_FFEE);
    let mut ok = 0usize;
    for round in 0..400 {
        let n = (rng.next() % 12) as usize;
        let col = |rng: &mut Rng| (0..n).map(|_| rng.f32()).collect::<Vec<_>>();
        let (x, y, lo, hi) = (col(&mut rng), col(&mut rng), col(&mut rng), col(&mut rng));
        let mut d = DataFrame::new();
        d.add_column_f32("x", &x);
        d.add_column_f32("y", &y);
        d.add_column_f32("lo", &lo);
        d.add_column_f32("hi", &hi);
        let groups: Vec<&str> = (0..n).map(|i| ["a", "b", "c"][i % 3]).collect();
        d.add_column_str("g", &groups);
        let geoms: Vec<(Geom, Aes)> = vec![
            (Geom::gg_histogram(), ax().fill("g")),
            (Geom::gg_histogram().bins(1 + (round % 40)), ax()),
            (Geom::gg_histogram().binwidth(f64::from(rng.f32())), ax()),
            (Geom::gg_freqpoly().bins(7), ax().color("g")),
            (Geom::gg_density(), ax()),
            (Geom::gg_density().n(1 + round).adjust(f64::from(rng.f32().abs())), ax().fill("g")),
            (Geom::gg_boxplot(), Aes::new().y("y").x("x")),
            (Geom::gg_boxplot().coef(f64::from(rng.f32().abs())), Aes::new().y("y").fill("g")),
            (Geom::gg_errorbar(), Aes::new().x("x").ymin("lo").ymax("hi").color("g")),
        ];
        for (g, a) in geoms {
            if let Ok(l) = layout_of(d.clone(), a, g) {
                ok += 1;
                for layer in &l.layers {
                    let rows = layer.dist.as_ref().map_or(0, BuiltDist::nrow);
                    assert!(
                        rows <= super::dist_math::MAX_OUTPUT_ROWS,
                        "round {round}: {rows} rows"
                    );
                }
            }
        }
    }
    assert!(ok > 100, "the fuzz must reach the success path: only {ok} layouts built");
    assert!(start.elapsed().as_secs() < 120, "fuzz took {:?}", start.elapsed());
}

#[test]
fn bin_count_is_bounded_for_extreme_ranges() {
    let d = DataFrame::from_xy(&[f32::MIN, f32::MAX], &[0.0, 0.0]);
    let l =
        layout_of(d.clone(), ax(), Geom::gg_histogram()).expect("30 bins over the whole f32 range");
    assert_eq!(l.layers[0].dist.as_ref().expect("dist").nrow(), 30);
    for g in [Geom::gg_histogram().binwidth(1.0), Geom::gg_freqpoly().binwidth(1.0)] {
        assert!(matches!(layout_of(d.clone(), ax(), g), Err(Error::Rendering(_))));
    }
}

// --- more refusals and edge inputs ------------------------------------------------------------

#[test]
fn missing_x_mapping_is_refused_not_empty() {
    for g in [Geom::gg_histogram(), Geom::gg_freqpoly(), Geom::gg_density()] {
        let r = layout_of(small(), Aes::new().y("y"), g.clone());
        assert!(matches!(r, Err(Error::Rendering(_))), "y only: {r:?}");
        let r = layout_of(small(), Aes::new(), g);
        assert!(matches!(r, Err(Error::Rendering(_))), "no mapping: {r:?}");
    }
}

#[test]
fn errorbar_needs_its_mappings_and_a_sane_width() {
    let no_x = layout_of(small(), Aes::new().ymin("x").ymax("y"), Geom::gg_errorbar());
    assert!(matches!(no_x, Err(Error::Rendering(_))), "{no_x:?}");
    let a = Aes::new().x("x").ymin("x").ymax("y");
    for w in [-1.0_f32, f32::NAN, f32::INFINITY] {
        let r = layout_of(small(), a.clone(), Geom::gg_errorbar().width(w));
        assert!(matches!(r, Err(Error::Rendering(_))), "width {w}: {r:?}");
    }
}

#[test]
fn mapped_alpha_and_bad_linewidth_are_refused() {
    let r = layout_of(small(), ax().alpha("y"), Geom::gg_histogram());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    for lw in [-1.0, f64::NAN] {
        let r = layout_of(small(), ax(), Geom::gg_histogram().linewidth(lw));
        assert!(matches!(r, Err(Error::Rendering(_))), "linewidth {lw}: {r:?}");
    }
}

#[test]
fn a_layer_with_no_finite_x_is_empty_even_beside_a_good_one() {
    let na = DataFrame::from_xy(&[f32::NAN, f32::NAN], &[1.0, 2.0]);
    for g in [Geom::gg_histogram(), Geom::gg_freqpoly(), Geom::gg_density()] {
        let p = plot_with(g.clone(), ax(), small()).layer(Layer::new(g).data(na.clone()));
        let r = p.build().and_then(|p| p.layout());
        assert!(matches!(r, Err(Error::EmptyData)), "{r:?}");
    }
}

#[test]
fn output_row_count_is_capped_before_it_is_built() {
    let mut d = DataFrame::from_xy(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1.0; 6]);
    d.add_column_str("g", &["a", "b", "c", "d", "e", "f"]);
    let r = layout_of(d.clone(), ax().fill("g"), Geom::gg_histogram().bins(999_999));
    assert!(matches!(r, Err(Error::Rendering(_))), "hist: {r:?}");
    let r = layout_of(d, ax().fill("g"), Geom::gg_density().n(1 << 20));
    assert!(matches!(r, Err(Error::Rendering(_))), "dens: {r:?}");
}

#[test]
fn numeric_groups_sort_before_na() {
    // A numeric `group` column: numbers form their own groups and NA another, last.
    let mut g = DataFrame::from_xy(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0; 5]);
    g.add_column_f32("k", &[1.0, 2.0, f32::NAN, 1.0, 2.0]);
    let b = layout_of(g, ax().group("k"), Geom::gg_histogram().bins(2)).expect("builds");
    let mut seen: Vec<i64> = b.layers[0].dist.as_ref().expect("dist").group.clone();
    seen.dedup();
    assert_eq!(seen, vec![1, 2, 3], "1, 2 then NA last");
}

// --- #6976 review: per-layer widths, edge cases and refusals ----------------------------------

/// Several layers, each with its own data, against one fixture.
fn run_layers(
    json: &str,
    row: &str,
    case: &str,
    skips: &[&[&str]],
    layers: Vec<(DataFrame, Aes, Geom, &str)>,
) {
    let w = want(json, row, case);
    let tag = format!("{row}/{case}");
    let mut plot = GGPlot::new();
    for (d, a, g, _) in &layers {
        plot = plot.layer(Layer::new(g.clone()).data(d.clone()).aes(a.clone()));
    }
    let got = plot.build().and_then(|p| p.layout()).unwrap_or_else(|e| panic!("{tag}: {e}"));
    let want_layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), want_layers.len(), "{tag}: layer count");
    for (i, (l, (_, _, _, class))) in got.layers.iter().zip(&layers).enumerate() {
        let d = l.dist.as_ref().expect("a distribution layer");
        check_layer(&format!("{tag}#{i}"), d, &want_layers[i], class, skips[i]);
    }
    check_axis(&format!("{tag}.x"), &got.x, &w["panel"]["x"]);
    check_axis(&format!("{tag}.y"), &got.y, &w["panel"]["y"]);
}

fn eb() -> Aes {
    Aes::new().x("x").ymin("lo").ymax("hi")
}

#[test]
fn err_width_comes_from_each_layers_own_x() {
    run_layers(
        fixture!("geom_errorbar", "two_layers"),
        "geom_errorbar",
        "two_layers",
        &[&[], &[]],
        vec![
            (csv!("geom_errorbar", "data"), eb(), Geom::gg_errorbar(), "GeomErrorbar"),
            (csv!("geom_errorbar", "data_b"), eb(), Geom::gg_errorbar(), "GeomErrorbar"),
        ],
    );
}
#[test]
fn err_beside_a_histogram_ignores_the_histogram_x() {
    run_layers(
        fixture!("geom_errorbar", "hist_plus_errorbar"),
        "geom_errorbar",
        "hist_plus_errorbar",
        &[HIST_SKIP, &[]],
        vec![
            (csv!("geom_histogram", "data"), ax(), Geom::gg_histogram(), "GeomBar"),
            (csv!("geom_errorbar", "data_b"), eb(), Geom::gg_errorbar(), "GeomErrorbar"),
        ],
    );
}
#[test]
fn err_x_near_zero_uses_resolution_without_zero() {
    run(
        fixture!("geom_errorbar", "near_zero"),
        "geom_errorbar",
        "near_zero",
        "GeomErrorbar",
        &[],
        csv!("geom_errorbar", "data_near0"),
        eb(),
        Geom::gg_errorbar(),
    );
}
#[test]
fn box_x_near_zero_uses_resolution_with_zero() {
    run(
        fixture!("geom_boxplot", "x_near_zero"),
        "geom_boxplot",
        "x_near_zero",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data_near0"),
        Aes::new().x("x").y("v").group("x"),
        Geom::gg_boxplot(),
    );
}
#[test]
fn box_touching_boxes_stay_in_separate_clusters() {
    run(
        fixture!("geom_boxplot", "touching_boxes"),
        "geom_boxplot",
        "touching_boxes",
        "GeomBoxplot",
        &[],
        csv!("geom_boxplot", "data_touch"),
        Aes::new().x("x").y("v").group("x"),
        Geom::gg_boxplot().width(1.0),
    );
}
#[test]
fn hist_break_fuzz_is_1e_minus_8() {
    run(
        fixture!("geom_histogram", "fuzz_edge"),
        "geom_histogram",
        "fuzz_edge",
        "GeomBar",
        HIST_SKIP,
        csv!("geom_histogram", "data_fuzz"),
        ax(),
        Geom::gg_histogram().breaks(&[0.0, 1.0, 2.0, 3.0, 4.0]),
    );
}

use super::Stat;

#[test]
fn a_stat_other_than_the_geoms_own_is_refused() {
    let stats = [
        Stat::Identity,
        Stat::Count,
        Stat::Bin { bins: 5 },
        Stat::Density,
        Stat::Boxplot,
        Stat::Smooth,
        Stat::Bin2d { bins_x: 2, bins_y: 2 },
        Stat::Sum,
        Stat::Mean,
    ];
    // (geom name, geom, aes, the one stat it accepts)
    for (name, g, a) in all_geoms() {
        let own = match name {
            "dens" => Some(Stat::Density),
            "box" => Some(Stat::Boxplot),
            "err" => Some(Stat::Identity),
            _ => None,
        };
        for s in &stats {
            let r = layout_of(small(), a.clone(), g.clone().stat(s.clone()));
            if own.as_ref().is_some_and(|o| std::mem::discriminant(o) == std::mem::discriminant(s))
            {
                assert!(r.is_ok(), "{name} with its own stat {s:?}: {r:?}");
            } else {
                assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{name} {s:?}: {r:?}");
            }
        }
    }
}

#[test]
fn unused_aesthetics_are_refused_not_ignored() {
    let sized = Aes::new().x("x").size("y");
    for g in [Geom::gg_histogram(), Geom::gg_freqpoly(), Geom::gg_density()] {
        for a in [
            ax().y("y"),
            sized.clone(),
            ax().size_value(2.0),
            ax().shape("y"),
            ax().label("y"),
            ax().ymin("y"),
            ax().ymax("y"),
        ] {
            let r = layout_of(small(), a.clone(), g.clone());
            assert!(matches!(r, Err(Error::Rendering(_))), "{a:?}: {r:?}");
        }
    }
    // A box plot has no ymin/ymax mapping; an error bar keeps neither y nor a size.
    let r = layout_of(small(), Aes::new().y("y").ymin("x"), Geom::gg_boxplot());
    assert!(matches!(r, Err(Error::Rendering(_))), "box ymin: {r:?}");
    let r = layout_of(small(), Aes::new().y("y").ymax("x"), Geom::gg_boxplot());
    assert!(matches!(r, Err(Error::Rendering(_))), "box ymax: {r:?}");
    let r = layout_of(small(), eb_small().y("y"), Geom::gg_errorbar());
    assert!(matches!(r, Err(Error::Rendering(_))), "err y: {r:?}");
    // The aesthetics each layer does use still build.
    assert!(layout_of(small(), Aes::new().y("y").x("x"), Geom::gg_boxplot()).is_ok());
    assert!(layout_of(small(), eb_small(), Geom::gg_errorbar()).is_ok());
}

fn eb_small() -> Aes {
    Aes::new().x("x").ymin("x").ymax("y")
}

#[test]
fn ragged_columns_are_a_typed_error() {
    let mut d = DataFrame::new();
    d.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    d.add_column_f32("z", &[1.0, 2.0, 3.0]);
    d.add_column_f32("w", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    d.add_column_str("g", &["a", "b", "a"]);
    for (name, a, g) in [
        ("hist", Aes::new().x("z"), Geom::gg_histogram()),
        ("dens", Aes::new().x("z"), Geom::gg_density()),
        ("hist fill", ax().fill("g"), Geom::gg_histogram()),
        ("box y", Aes::new().y("z"), Geom::gg_boxplot()),
        ("box x", Aes::new().y("x").x("z"), Geom::gg_boxplot()),
        ("err ymin", Aes::new().x("x").ymin("z").ymax("w"), Geom::gg_errorbar()),
    ] {
        let r = layout_of(d.clone(), a, g);
        assert!(matches!(r, Err(Error::Rendering(_))), "{name}: {r:?}");
    }
}

/// Run `f` on a thread and fail the test, instead of hanging it, if it does not return.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs)).expect("layout() did not return in time")
}

#[test]
fn extreme_parameters_return_a_typed_error_not_a_hang() {
    let cases: Vec<(&str, Geom)> = vec![
        ("hist binwidth 1e300", Geom::gg_histogram().binwidth(1e300)),
        ("hist binwidth 1e200", Geom::gg_histogram().binwidth(1e200)),
        ("freqpoly binwidth 1e300", Geom::gg_freqpoly().binwidth(1e300)),
        ("dens bw 1e-300", Geom::gg_density().bw(1e-300)),
        ("dens adjust 1e-300", Geom::gg_density().adjust(1e-300)),
        ("dens bw 1e-150", Geom::gg_density().bw(1e-150)),
    ];
    for (name, g) in cases {
        let r = within(60, move || layout_of(small(), ax(), g));
        assert!(matches!(r, Err(Error::ScaleDomain(_) | Error::Rendering(_))), "{name}: {r:?}");
    }
    // Only the lower limit is huge: negative data in a bin that ends at the boundary 0.
    let neg = DataFrame::from_xy(&[-1.0, -2.0, -3.0], &[1.0, 2.0, 3.0]);
    let r = within(60, move || {
        layout_of(neg, ax(), Geom::gg_histogram().binwidth(1e300).boundary(0.0))
    });
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "one-sided low: {r:?}");
    // Just inside the bound the layout still builds.
    let r = within(60, || layout_of(small(), ax(), Geom::gg_density().bw(1e-90)));
    assert!(r.is_ok(), "bw 1e-90: {r:?}");
}

#[test]
fn every_coord_limit_and_facet_kind_is_refused() {
    for (name, g, a) in all_geoms() {
        for coord in
            [Coord::cartesian().ylim(0.0, 1.0), Coord::cartesian().xlim(0.0, 1.0).ylim(0.0, 1.0)]
        {
            let r = plot_with(g.clone(), a.clone(), small())
                .coord(coord)
                .build()
                .and_then(|p| p.layout());
            assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name} ylim: {r:?}");
        }
        for facet in [Facet::row("f"), Facet::col("f"), Facet::grid("f", "f"), Facet::wrap("f", 2)]
        {
            let mut d = small();
            d.add_column_str("f", &["a", "b", "a", "b", "a"]);
            let r = plot_with(g.clone(), a.clone(), d)
                .facet(facet.clone())
                .build()
                .and_then(|p| p.layout());
            assert!(matches!(r, Err(Error::Rendering(_))), "{name} {facet:?}: {r:?}");
        }
    }
}
