//! ggplot2 conformance oracle for the bar train (PLOT-GG-001, paiml/aprender#6909):
//! `stat_count` + `geom_bar`, `stat_identity` + `geom_col`, `position_stack`, `position_dodge`,
//! `position_fill`. Same pattern as `ggplot2_oracle.rs`: each fixture under
//! `fixtures/ggplot2/<row>/` is `ggplot_build()` output from ggplot2 4.0.3, committed; R is never
//! run here.
//!
//! # Tolerance
//!
//! Continuous values (`x`, `y`, `ymin`, `ymax`, `xmin`, `xmax`, `width`, `prop`, axis ranges):
//! `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12`. Inputs are exact in binary32 and both sides do a
//! handful of binary64 additions and divisions, so honest differences are a few ulp (~1e-16
//! relative); 1e-9 is still seven orders below the smallest default guarded here (a bar width of
//! 0.8 instead of 0.9 moves an edge by 0.05). `group`, `count`, labels and levels compare exactly.

use serde_json::Value;

use super::coord::Coord;
use super::stat::Stat;
use super::stat_count::{bar_layout, BarSpec, BuiltBars, DiscreteAxis};
use super::{Aes, DataFrame, Facet, GGPlot, Geom, Layer, Position, PositionScale, Preserve};
use crate::error::Error;

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
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
        .map(|e| e.as_f64().expect("fixture array holds numbers"))
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture array holds strings").to_string())
        .collect()
}

/// Parse a CSV: a column whose every cell is a number becomes numeric (and must be exact in f32),
/// any other column is text.
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
        let parsed: Option<Vec<f64>> = cells.iter().map(|s| s.parse().ok()).collect();
        match parsed {
            Some(vals) => {
                let f: Vec<f32> = vals.iter().map(|v| *v as f32).collect();
                for (v, g) in vals.iter().zip(&f) {
                    let same = f64::from(*g) == *v || (v.is_nan() && g.is_nan());
                    assert!(same, "fixture value {v} is not exact in f32");
                }
                df.add_column_f32(name.trim(), &f);
            }
            None => df.add_column_str(name.trim(), &cells),
        }
    }
    df
}

fn check_discrete(got: &DiscreteAxis, want: &Value) {
    assert_eq!(got.limits, strs(&want["limits"]), "x.limits");
    assert_close_slice(
        "x.continuous_range",
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&want["continuous_range"]),
    );
    assert_close_slice("x.breaks", &got.breaks, &nums(&want["breaks"]));
    assert_eq!(got.labels, strs(&want["labels"]), "x.labels");
}

fn check_continuous(got: &PositionScale, want: &Value) {
    assert_close_slice("y.limits", &[got.limits.0, got.limits.1], &nums(&want["limits"]));
    assert_close_slice(
        "y.continuous_range",
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&want["continuous_range"]),
    );
    assert_close_slice("y.breaks", &got.breaks, &nums(&want["breaks"]));
    assert_eq!(got.labels, strs(&want["labels"]), "y.labels");
    let censored = want["n_breaks_censored"].as_u64().expect("count");
    assert_eq!(got.n_breaks_censored as u64, censored, "y.n_breaks_censored");
}

/// Compare every layer column ggplot2 reports and aprender-viz computes, then the panel scales.
fn check_bars(got: &BuiltBars, want: &Value, expect_geom: &str) {
    check_bars_layer(got, want, 0, expect_geom);
    check_discrete(&got.x, &want["panel"]["x"]);
    check_continuous(&got.y, &want["panel"]["y"]);
}

/// The rows of layer `idx` of the fixture (no scales: those belong to the whole plot).
fn check_bars_layer(got: &BuiltBars, want: &Value, idx: usize, expect_geom: &str) {
    let layer = &want["layers"][idx];
    assert_eq!(layer["geom"], expect_geom);
    assert_eq!(
        want["versions"]["ggplot2"], "4.0.3",
        "fixture was not produced by the pinned ggplot2"
    );
    assert_eq!(
        want["collate"], "C",
        "fixture must be generated under LC_COLLATE=C (level order is bytewise)"
    );
    let col =
        |f: fn(&super::position::BarRow) -> f64| -> Vec<f64> { got.rows.iter().map(f).collect() };
    for (name, f) in [
        ("x", (|r| r.x) as fn(&super::position::BarRow) -> f64),
        ("y", |r| r.y),
        ("ymin", |r| r.ymin),
        ("ymax", |r| r.ymax),
        ("xmin", |r| r.xmin),
        ("xmax", |r| r.xmax),
        ("width", |r| r.width),
    ] {
        assert_close_slice(name, &col(f), &nums(&layer[name]));
    }
    let groups: Vec<f64> = got.rows.iter().map(|r| f64::from(r.group)).collect();
    assert_eq!(groups, nums(&layer["group"]), "group");
    if layer.get("count").is_some() {
        let count: Vec<f64> = got.rows.iter().map(|r| r.count.expect("count")).collect();
        let prop: Vec<f64> = got.rows.iter().map(|r| r.prop.expect("prop")).collect();
        assert_eq!(count, nums(&layer["count"]), "count");
        assert_close_slice("prop", &prop, &nums(&layer["prop"]));
    } else {
        assert!(got.rows.iter().all(|r| r.count.is_none()), "stat_identity has no count");
    }
    // ggplot2 orders a layer's rows by PANEL; the panel of each row must agree.
    let panels: Vec<String> = got.panel.iter().map(|p| (p + 1).to_string()).collect();
    assert_eq!(panels, strs(&layer["PANEL"]), "PANEL");
}

macro_rules! fixture {
    ($dir:literal, $file:literal) => {
        (
            include_str!(concat!("../../fixtures/ggplot2/", $dir, "/data.csv")),
            include_str!(concat!("../../fixtures/ggplot2/", $dir, "/", $file)),
        )
    };
}

fn run(fx: (&str, &str), spec: &BarSpec, geom: &str) {
    let want: Value = serde_json::from_str(fx.1).expect("fixture is valid JSON");
    let got = bar_layout(&frame_from_csv(fx.0), spec).expect("bar layer builds");
    check_bars(&got, &want, geom);
}

// --- stat_count + geom_bar ---------------------------------------------------------------------

#[test]
fn ggplot2_bars_stat_count_geom_bar() {
    let spec = BarSpec::bar("x");
    run(fixture!("stat_count_geom_bar", "expected.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_stat_count_group1_prop() {
    let spec = BarSpec::bar("x").group_all();
    run(fixture!("stat_count_geom_bar", "expected_group1.json"), &spec, "GeomBar");
}

// --- stat_identity + geom_col ------------------------------------------------------------------

#[test]
fn ggplot2_bars_stat_identity_geom_col() {
    let spec = BarSpec::col("x", "y").fill("g");
    run(fixture!("stat_identity_geom_col", "expected.json"), &spec, "GeomCol");
}

#[test]
fn ggplot2_bars_stat_identity_geom_col_plain() {
    // The fixture's `one <- df[df$g == "g1", ]`: keep those rows only.
    let (csv, json) = fixture!("stat_identity_geom_col", "expected_plain.json");
    let kept: String = csv
        .lines()
        .enumerate()
        .filter(|(i, l)| *i == 0 || l.split(',').nth(1) == Some("g1"))
        .map(|(_, l)| format!("{l}\n"))
        .collect();
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    let got = bar_layout(&frame_from_csv(&kept), &BarSpec::col("x", "y")).expect("builds");
    check_bars(&got, &want, "GeomCol");
}

// --- position_stack ----------------------------------------------------------------------------

#[test]
fn ggplot2_bars_position_stack() {
    let spec = BarSpec::bar("x").fill("g");
    run(fixture!("position_stack", "expected.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_position_stack_reverse() {
    let spec = BarSpec::bar("x").fill("g").position(Position::Stack { vjust: 1.0, reverse: true });
    run(fixture!("position_stack", "expected_reverse.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_position_stack_vjust() {
    let spec = BarSpec::bar("x").fill("g").position(Position::Stack { vjust: 0.5, reverse: false });
    run(fixture!("position_stack", "expected_vjust05.json"), &spec, "GeomBar");
}

// --- position_fill -----------------------------------------------------------------------------

#[test]
fn ggplot2_bars_position_fill() {
    let spec = BarSpec::bar("x").fill("g").position(Position::fill());
    run(fixture!("position_fill", "expected.json"), &spec, "GeomBar");
}

// --- position_dodge ----------------------------------------------------------------------------

#[test]
fn ggplot2_bars_position_dodge() {
    let spec = BarSpec::bar("x").fill("g").position(Position::dodge());
    run(fixture!("position_dodge", "expected.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_position_dodge_preserve_single() {
    let spec = BarSpec::bar("x")
        .fill("g")
        .position(Position::Dodge { width: None, preserve: Preserve::Single });
    run(fixture!("position_dodge", "expected_single.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_position_dodge_width() {
    let spec = BarSpec::bar("x")
        .fill("g")
        .position(Position::Dodge { width: Some(0.5), preserve: Preserve::Total });
    run(fixture!("position_dodge", "expected_width05.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_position_dodge_width_wider_than_bars() {
    // position_dodge(width = 2): bars spill outside the expanded levels, so the panel range is the
    // union of the two (stat_count.rs `discrete_axis`).
    let spec = BarSpec::bar("x")
        .fill("g")
        .position(Position::Dodge { width: Some(2.0), preserve: Preserve::Total });
    run(fixture!("position_dodge", "expected_width2.json"), &spec, "GeomBar");
}

// --- position_dodge: a group repeated at one x (review F1) -------------------------------------
// ggplot2 counts DISTINCT groups per x and ranks them densely; rows are not groups.

#[test]
fn ggplot2_bars_position_dodge_repeated_group() {
    let spec = BarSpec::col("x", "y").fill("g").position(Position::dodge());
    run(fixture!("position_dodge_dup", "expected.json"), &spec, "GeomCol");
}

#[test]
fn ggplot2_bars_position_dodge_repeated_group_preserve_single() {
    let spec = BarSpec::col("x", "y")
        .fill("g")
        .position(Position::Dodge { width: None, preserve: Preserve::Single });
    run(fixture!("position_dodge_dup", "expected_single.json"), &spec, "GeomCol");
}

// --- level order is LC_COLLATE=C (review F2) ---------------------------------------------------

#[test]
fn ggplot2_bars_level_order_is_bytewise_c_collation() {
    // x = a, B, B, c: C collation gives B < a < c (en_US would give a < B < c).
    let spec = BarSpec::bar("x");
    run(fixture!("stat_count_case", "expected.json"), &spec, "GeomBar");
}

#[test]
fn ggplot2_bars_group_order_is_bytewise_c_collation() {
    // fill levels Z < h under C; the group ids and stacking order follow.
    let spec = BarSpec::bar("x").fill("g");
    run(fixture!("stat_count_case", "expected_fill.json"), &spec, "GeomBar");
}

// --- position_stack: a zero-height bar stays in the positive stack (review F3) -----------------

#[test]
fn ggplot2_bars_position_stack_zero_height_bar() {
    let spec = BarSpec::col("x", "y").fill("g");
    run(fixture!("position_stack_zero", "expected.json"), &spec, "GeomCol");
}

// --- the PUBLIC path: GGPlot builder -> layout() ------------------------------------------------
// Every case below builds the plot through `GGPlot` / `Geom` / `Layer` / `Position` and reads
// `layout()`, then compares with the SAME fixture as the `bar_layout` test of the same name. No
// call to `bar_layout` appears here: the grammar must reach it by itself.

fn run_public(fx: (&str, &str), geom: Geom, aes: Aes, position: Option<Position>, name: &str) {
    run_public_frame(&frame_from_csv(fx.0), fx.1, geom, aes, position, name);
}

fn run_public_frame(
    df: &DataFrame,
    json: &str,
    geom: Geom,
    aes: Aes,
    position: Option<Position>,
    name: &str,
) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    let mut layer = Layer::new(geom);
    if let Some(p) = position {
        layer = layer.position(p);
    }
    let layout = GGPlot::new()
        .data(df.clone())
        .aes(aes)
        .layer(layer)
        .build()
        .expect("plot builds")
        .layout()
        .expect("layout builds");
    let bars =
        layout.layers[0].bars.as_ref().unwrap_or_else(|| panic!("{name}: no bars in layout"));
    check_bars(bars, &want, if bars.rows[0].count.is_some() { "GeomBar" } else { "GeomCol" });
    // The layout's own scales are the bar layer's scales.
    assert_eq!(layout.y, bars.y, "{name}: layout y scale");
    assert_eq!(layout.x_discrete.as_ref(), Some(&bars.x), "{name}: layout discrete x axis");
    assert_eq!(layout.x.continuous_range, bars.x.continuous_range, "{name}: layout x range");
    assert_eq!(layout.x.breaks, bars.x.breaks, "{name}: layout x breaks");
    assert_eq!(layout.x.labels, bars.x.labels, "{name}: layout x labels");
}

fn xg() -> Aes {
    Aes::new().x("x").fill("g")
}

fn xyg() -> Aes {
    Aes::new().x("x").y("y").fill("g")
}

#[test]
fn ggplot2_bars_public_geom_bar_default_width_and_stat_count() {
    // `ggplot() + geom_bar()`: width 0.9 (not the old 0.8), stat_count, position_stack.
    let fx = fixture!("stat_count_geom_bar", "expected.json");
    run_public(fx, Geom::bar(), Aes::new().x("x"), None, "geom_bar");
}

#[test]
fn ggplot2_bars_public_geom_bar_width() {
    let fx = fixture!("stat_count_geom_bar", "expected_width05.json");
    run_public(fx, Geom::bar().width(0.5), Aes::new().x("x"), None, "geom_bar(width = 0.5)");
}

#[test]
fn ggplot2_bars_public_geom_col() {
    let fx = fixture!("stat_identity_geom_col", "expected.json");
    run_public(fx, Geom::col(), xyg(), None, "geom_col");
}

#[test]
fn ggplot2_bars_public_geom_col_plain() {
    let (csv, json) = fixture!("stat_identity_geom_col", "expected_plain.json");
    let kept: String = csv
        .lines()
        .enumerate()
        .filter(|(i, l)| *i == 0 || l.split(',').nth(1) == Some("g1"))
        .map(|(_, l)| format!("{l}\n"))
        .collect();
    let aes = Aes::new().x("x").y("y");
    run_public_frame(&frame_from_csv(&kept), json, Geom::col(), aes, None, "geom_col plain");
}

#[test]
fn ggplot2_bars_public_position_stack() {
    let fx = fixture!("position_stack", "expected.json");
    run_public(fx, Geom::bar(), xg(), Some(Position::stack()), "stack");
    let fx = fixture!("position_stack", "expected_reverse.json");
    let rev = Position::Stack { vjust: 1.0, reverse: true };
    run_public(fx, Geom::bar(), xg(), Some(rev), "stack reverse");
    let fx = fixture!("position_stack", "expected_vjust05.json");
    let mid = Position::Stack { vjust: 0.5, reverse: false };
    run_public(fx, Geom::bar(), xg(), Some(mid), "stack vjust");
}

#[test]
fn ggplot2_bars_public_position_stack_is_the_default_for_bars() {
    // No `.position(..)`: the layer defaults to position_stack, as ggplot2's geom_bar() does.
    let fx = fixture!("position_stack", "expected.json");
    run_public(fx, Geom::bar(), xg(), None, "default position");
}

#[test]
fn ggplot2_bars_public_position_stack_zero_height_bar() {
    let fx = fixture!("position_stack_zero", "expected.json");
    run_public(fx, Geom::col(), xyg(), None, "stack zero");
}

#[test]
fn ggplot2_bars_public_position_fill() {
    let fx = fixture!("position_fill", "expected.json");
    run_public(fx, Geom::bar(), xg(), Some(Position::fill()), "fill");
}

#[test]
fn ggplot2_bars_public_position_dodge() {
    let fx = fixture!("position_dodge", "expected.json");
    run_public(fx, Geom::bar(), xg(), Some(Position::dodge()), "dodge");
    let fx = fixture!("position_dodge", "expected_single.json");
    let single = Position::Dodge { width: None, preserve: Preserve::Single };
    run_public(fx, Geom::bar(), xg(), Some(single), "dodge single");
    let fx = fixture!("position_dodge", "expected_width05.json");
    let w05 = Position::Dodge { width: Some(0.5), preserve: Preserve::Total };
    run_public(fx, Geom::bar(), xg(), Some(w05), "dodge width 0.5");
    let fx = fixture!("position_dodge", "expected_width2.json");
    let w2 = Position::Dodge { width: Some(2.0), preserve: Preserve::Total };
    run_public(fx, Geom::bar(), xg(), Some(w2), "dodge width 2");
}

#[test]
fn ggplot2_bars_public_position_dodge_repeated_group() {
    let fx = fixture!("position_dodge_dup", "expected.json");
    run_public(fx, Geom::col(), xyg(), Some(Position::dodge()), "dodge dup");
    let fx = fixture!("position_dodge_dup", "expected_single.json");
    let single = Position::Dodge { width: None, preserve: Preserve::Single };
    run_public(fx, Geom::col(), xyg(), Some(single), "dodge dup single");
}

#[test]
fn ggplot2_bars_public_level_and_group_order_c_collation() {
    let fx = fixture!("stat_count_case", "expected.json");
    run_public(fx, Geom::bar(), Aes::new().x("x"), None, "case");
    let fx = fixture!("stat_count_case", "expected_fill.json");
    run_public(fx, Geom::bar(), xg(), None, "case fill");
}

#[test]
fn bar_layer_over_a_numeric_empty_or_absent_x_is_refused() {
    // ggplot2 bins a numeric x by its resolution (and draws bars there); layout() does not compute
    // that, so it refuses with a typed error instead of returning the raw points.
    let mut empty = DataFrame::new();
    empty.add_column_str("x", &[]);
    empty.add_column_f32("y", &[]);
    for (df, aes, what) in [
        (frame_from_csv("x,y\n1,2\n2,3\n"), Aes::new().x("x").y("y"), "numeric x"),
        (empty, Aes::new().x("x").y("y"), "empty x"),
        (frame_from_csv("x,y\na,2\n"), Aes::new().y("y"), "no x aesthetic"),
        (frame_from_csv("x,y\na,2\n"), Aes::new().x("nope"), "absent x column"),
    ] {
        for geom in [Geom::bar(), Geom::col()] {
            let got = GGPlot::new().data(df.clone()).aes(aes.clone()).geom(geom).build();
            let err = got.expect("plot builds").layout().expect_err(what);
            assert!(matches!(err, Error::UnsupportedLayer(m) if m.contains("text x")), "{what}");
        }
    }
}

#[test]
fn bar_layer_errors_are_reported_not_swallowed() {
    let df = frame_from_csv("x,y\na,2\nb,3\n");
    let layout_of = |geom: Geom, aes: Aes| {
        GGPlot::new().data(df.clone()).aes(aes).geom(geom).build().expect("plot builds").layout()
    };
    assert!(layout_of(Geom::col(), Aes::new().x("x")).is_err(), "geom_col without y");
    assert!(layout_of(Geom::bar(), Aes::new().x("x").group("y")).is_err(), "group aesthetic");
    assert!(layout_of(Geom::bar(), Aes::new().x("x").fill("nope")).is_err(), "missing fill column");
}

// --- facets: stat and position run per panel, scales are trained across panels (review H1) ------

fn facet_f() -> Facet {
    Facet::wrap("f", 2)
}

/// Build `layers` over `df` (optionally faceted) and return the layout.
fn layout_of_layers(
    df: &DataFrame,
    aes: Aes,
    layers: Vec<Layer>,
    facet: Option<Facet>,
) -> crate::error::Result<super::BuiltLayout> {
    let mut plot = GGPlot::new().data(df.clone()).aes(aes);
    for l in layers {
        plot = plot.layer(l);
    }
    if let Some(f) = facet {
        plot = plot.facet(f);
    }
    plot.build()?.layout()
}

/// Compare a whole layout with a fixture: every layer, then the shared x and y scales.
fn check_layout(layout: &super::BuiltLayout, want: &Value, name: &str) {
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "{name}: fixture is not from ggplot2 4.0.3");
    assert_eq!(want["collate"], "C", "{name}: fixture must be generated under LC_COLLATE=C");
    let fx_layers = want["layers"].as_array().expect("layers");
    assert_eq!(layout.layers.len(), fx_layers.len(), "{name}: layer count");
    for (i, fx) in fx_layers.iter().enumerate() {
        let geom = fx["geom"].as_str().expect("geom");
        match (&layout.layers[i].bars, geom) {
            (Some(bars), "GeomBar" | "GeomCol") => check_bars_layer(bars, want, i, geom),
            (None, "GeomPoint") => {
                assert_close_slice(&format!("{name}[{i}].x"), &layout.layers[i].x, &nums(&fx["x"]));
                assert_close_slice(&format!("{name}[{i}].y"), &layout.layers[i].y, &nums(&fx["y"]));
            }
            (None, "GeomHline" | "GeomVline") => check_reference(&layout.layers[i], fx, geom, name),
            (b, g) => panic!("{name}[{i}]: layer is {g} in the fixture, bars = {}", b.is_some()),
        }
    }
    let axis = layout.x_discrete.as_ref().unwrap_or_else(|| panic!("{name}: no discrete x"));
    check_discrete(axis, &want["panel"]["x"]);
    check_continuous(&layout.y, &want["panel"]["y"]);
    assert_eq!(layout.x.continuous_range, axis.continuous_range, "{name}: layout x range");
    assert_eq!(layout.x.breaks, axis.breaks, "{name}: layout x breaks");
}

/// A `geom_hline`/`geom_vline` layer: one row per panel, its intercept, PANEL and group -1.
fn check_reference(got: &super::BuiltLayer, fx: &Value, geom: &str, name: &str) {
    let path = got.path.as_ref().unwrap_or_else(|| panic!("{name}: {geom} has no path data"));
    let r = path.reference.as_ref().unwrap_or_else(|| panic!("{name}: {geom} has no reference"));
    let field = if geom == "GeomHline" { "yintercept" } else { "xintercept" };
    assert_close_slice(&format!("{name}.{field}"), &r.intercept, &nums(&fx[field]));
    let panels: Vec<String> = path.panel.iter().map(|p| (p + 1).to_string()).collect();
    assert_eq!(panels, strs(&fx["PANEL"]), "{name}: {geom} PANEL");
    let groups: Vec<f64> = path.group.iter().map(|g| *g as f64).collect();
    assert_eq!(groups, nums(&fx["group"]), "{name}: {geom} group");
}

fn run_layers(fx: (&str, &str), aes: Aes, layers: Vec<Layer>, facet: Option<Facet>, name: &str) {
    let want: Value = serde_json::from_str(fx.1).expect("fixture is valid JSON");
    let layout = layout_of_layers(&frame_from_csv(fx.0), aes, layers, facet)
        .unwrap_or_else(|e| panic!("{name}: layout failed: {e}"));
    check_layout(&layout, &want, name);
}

fn bar_layer(geom: Geom, position: Option<Position>) -> Layer {
    let l = Layer::new(geom);
    match position {
        Some(p) => l.position(p),
        None => l,
    }
}

#[test]
fn ggplot2_bars_facet_stat_count_per_panel() {
    // geom_bar(fill) + facet_wrap: counts per panel (6 rows), shared y range (-0.1, 2.1).
    let fx = fixture!("facet_bars", "expected_bar.json");
    run_layers(fx, xg(), vec![bar_layer(Geom::bar(), None)], Some(facet_f()), "facet bar");
}

#[test]
fn ggplot2_bars_facet_position_dodge_per_panel() {
    // The dodge bar b,g1 alone in panel 1 sits at x = 2.0 (dodged per panel, preserve = total).
    let fx = fixture!("facet_bars", "expected_dodge.json");
    let l = bar_layer(Geom::col(), Some(Position::dodge()));
    run_layers(fx, xyg(), vec![l], Some(facet_f()), "facet dodge");
}

#[test]
fn ggplot2_bars_facet_position_fill_per_panel() {
    let fx = fixture!("facet_bars", "expected_fill.json");
    let l = bar_layer(Geom::bar(), Some(Position::fill()));
    run_layers(fx, xg(), vec![l], Some(facet_f()), "facet fill");
}

#[test]
fn ggplot2_bars_facet_position_stack_per_panel() {
    let fx = fixture!("facet_bars", "expected_stack.json");
    run_layers(fx, xyg(), vec![bar_layer(Geom::col(), None)], Some(facet_f()), "facet stack");
}

#[test]
fn ggplot2_bars_facet_dodge_preserve_single_is_global() {
    // preserve = "single": n is the busiest x in ANY panel, so panel q's bars are 1/3 wide.
    let fx = fixture!("facet_bars_single", "expected_single.json");
    let single = Position::Dodge { width: None, preserve: Preserve::Single };
    let l = bar_layer(Geom::col(), Some(single));
    run_layers(fx, xyg(), vec![l], Some(facet_f()), "facet dodge single");
}

#[test]
fn ggplot2_bars_facet_dodge_total_is_per_panel() {
    let fx = fixture!("facet_bars_single", "expected_total.json");
    let l = bar_layer(Geom::col(), Some(Position::dodge()));
    run_layers(fx, xyg(), vec![l], Some(facet_f()), "facet dodge total");
}

#[test]
fn ggplot2_bars_facet_group_ids_are_global() {
    // Group ids are computed before the facet split: panel q's first group is not 1.
    let fx = fixture!("facet_bars_single", "expected_bar.json");
    run_layers(fx, xg(), vec![bar_layer(Geom::bar(), None)], Some(facet_f()), "facet groups");
}

// --- scales are trained over ALL layers (review H2) --------------------------------------------

fn point_layer(aes: Aes) -> Layer {
    Layer::new(Geom::point()).aes(aes)
}

#[test]
fn ggplot2_bars_mixed_col_and_point_share_the_y_scale() {
    // geom_col + geom_point(aes(y = z)): R's y range is (-2.05, 21.05); the bars alone give less.
    let fx = fixture!("mixed_layers", "expected_col_point.json");
    let layers = vec![bar_layer(Geom::col(), None), point_layer(Aes::new().y("z"))];
    run_layers(fx, Aes::new().x("x").y("y"), layers, None, "col+point");
}

#[test]
fn ggplot2_bars_mixed_col_and_hline_extend_y() {
    let fx = fixture!("mixed_layers", "expected_col_hline.json");
    let layers = vec![bar_layer(Geom::col(), None), Layer::new(Geom::hline(30.0))];
    run_layers(fx, Aes::new().x("x").y("y"), layers, None, "col+hline");
}

#[test]
fn ggplot2_bars_mixed_point_outside_the_bar_x_range() {
    // Points at numeric x = -1 and 4.5 widen the continuous x range beyond the bars' 0.55..3.45.
    let fx = fixture!("mixed_layers", "expected_col_point_x.json");
    let layers = vec![bar_layer(Geom::col(), None), point_layer(Aes::new().x("xn").y("z"))];
    run_layers(fx, Aes::new().x("x").y("y"), layers, None, "col+point(x)");
}

/// The rows of `csv` whose column `col` is one of `keep` (header kept).
fn filter_csv(csv: &str, col: usize, keep: &[&str]) -> String {
    csv.lines()
        .enumerate()
        .filter(|(i, l)| *i == 0 || l.split(',').nth(col).is_some_and(|c| keep.contains(&c)))
        .map(|(_, l)| format!("{l}\n"))
        .collect()
}

#[test]
fn ggplot2_bars_mixed_two_bar_layers_union_their_levels() {
    // geom_bar(data = d1 [a, b]) + geom_col(data = d2 [b, c], dodge): levels a, b, c.
    let (csv, json) = fixture!("mixed_layers", "expected_two_bars.json");
    let d1 = frame_from_csv(&filter_csv(csv, 0, &["a", "b"]));
    let d2 = frame_from_csv(&filter_csv(csv, 0, &["b", "c"]));
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    let l1 = Layer::new(Geom::bar()).data(d1).aes(Aes::new().x("x"));
    let l2 = Layer::new(Geom::col()).data(d2).aes(xyg()).position(Position::dodge());
    let layout = layout_of_layers(&DataFrame::new(), Aes::new(), vec![l1, l2], None)
        .expect("two bar layers build");
    check_layout(&layout, &want, "two bars");
}

// --- typed errors for what is not drawn or not supported ---------------------------------------

#[test]
fn bar_layer_with_its_own_data_under_a_facet_is_a_typed_error() {
    let df = frame_from_csv("x,f\na,p\nb,q\n");
    let own = frame_from_csv("x,f\na,p\n");
    let layer = Layer::new(Geom::bar()).data(own);
    let err = layout_of_layers(&df, Aes::new().x("x"), vec![layer], Some(facet_f()))
        .expect_err("own data under a facet is refused");
    assert!(err.to_string().contains("own data under a facet"), "{err}");
}

#[test]
fn bars_on_a_discrete_x_are_not_drawn_yet_and_say_so() {
    let df = frame_from_csv("x,y\na,2\nb,3\n");
    let plot = GGPlot::new()
        .data(df)
        .aes(Aes::new().x("x").y("y"))
        .geom(Geom::col())
        .build()
        .expect("plot builds");
    let fb = plot.to_framebuffer().expect_err("framebuffer renderer refuses text-x bars");
    assert!(fb.to_string().contains("bars on a discrete x are not drawn yet"), "{fb}");
    let svg = plot.to_svg().expect_err("svg renderer refuses text-x bars");
    assert!(svg.to_string().contains("bars on a discrete x are not drawn yet"), "{svg}");
}

#[test]
fn bars_on_a_discrete_x_mapped_on_the_layer_are_refused_too() {
    // x only on the layer (M10: reading the plot aes alone drew these silently).
    let df = frame_from_csv("cat,y\na,2\nb,3\n");
    let layer_only = GGPlot::new()
        .data(df.clone())
        .layer(Layer::new(Geom::col()).aes(Aes::new().x("cat").y("y")))
        .build()
        .expect("plot builds");
    let numeric_plot = frame_from_csv("x,cat,y\n1,a,2\n2,b,3\n");
    let mixed = GGPlot::new()
        .data(numeric_plot)
        .aes(Aes::new().x("x").y("y"))
        .layer(Layer::new(Geom::col()).aes(Aes::new().x("cat")))
        .build()
        .expect("plot builds");
    let counted = GGPlot::new()
        .data(df)
        .layer(Layer::new(Geom::bar()).aes(Aes::new().x("cat")))
        .build()
        .expect("plot builds");
    for (name, plot) in [("layer only", layer_only), ("mixed", mixed), ("geom_bar", counted)] {
        let fb = plot.to_framebuffer().expect_err(name);
        assert!(fb.to_string().contains("bars on a discrete x are not drawn yet"), "{name}: {fb}");
        let svg = plot.to_svg().expect_err(name);
        assert!(
            svg.to_string().contains("bars on a discrete x are not drawn yet"),
            "{name}: {svg}"
        );
    }
}

// --- bars beside geom_hline / geom_vline / geom_point, and layer-level fill (bars_companions) -----
// Each fixture is `ggplot_build()` of the plot named in its test. A bar layer over a text x routes
// the whole plot to the bar layout; reference lines keep their own data (`grammar/path.rs`) and
// widen the shared scales by their intercept.

macro_rules! companions {
    ($name:literal) => {
        (
            include_str!("../../fixtures/ggplot2/bars_companions/data.csv"),
            include_str!(concat!(
                "../../fixtures/ggplot2/bars_companions/expected_",
                $name,
                ".json"
            )),
        )
    };
}

fn xy() -> Aes {
    Aes::new().x("x").y("y")
}

fn col() -> Layer {
    Layer::new(Geom::col())
}

fn hline(y: f32) -> Layer {
    Layer::new(Geom::hline(y))
}

fn vline(x: f32) -> Layer {
    Layer::new(Geom::vline(x))
}

#[test]
fn ggplot2_bars_col_and_vline_right_of_the_bars_widen_x() {
    // geom_col() + geom_vline(5): the continuous x range runs to 5, past the last bar's 3.45.
    run_layers(companions!("col_vline_right"), xy(), vec![col(), vline(5.0)], None, "vline 5");
}

#[test]
fn ggplot2_bars_col_and_vline_left_of_the_bars_widen_x() {
    run_layers(companions!("col_vline_left"), xy(), vec![col(), vline(-1.5)], None, "vline -1.5");
}

#[test]
fn ggplot2_bars_bar_with_hline_and_vline() {
    let layers = vec![Layer::new(Geom::bar()), hline(4.0), vline(0.25)];
    let aes = Aes::new().x("x");
    run_layers(companions!("bar_hline_vline"), aes, layers, None, "bar + hline + vline");
}

#[test]
fn ggplot2_bars_vline_inside_narrow_bars_leaves_x() {
    // width = 0.5 bars span 0.75..3.25; a vline at 3.25 adds nothing beyond the bar edges.
    let narrow = Layer::new(Geom::bar().stat(Stat::Identity).width(0.5));
    let fx = companions!("col_width_vline_inside");
    run_layers(fx, xy(), vec![narrow, vline(3.25)], None, "vline inside");
}

#[test]
fn ggplot2_bars_col_with_two_hlines_spans_both() {
    let layers = vec![col(), hline(-5.0), hline(7.0)];
    run_layers(companions!("col_two_hlines"), xy(), layers, None, "two hlines");
}

#[test]
fn ggplot2_bars_hline_before_the_bar_layer() {
    // Layer order does not matter to the scales; the hline is layer 0 in the fixture.
    run_layers(companions!("hline_then_col"), xy(), vec![hline(-5.0), col()], None, "hline first");
}

#[test]
fn ggplot2_bars_point_with_new_levels_extends_the_discrete_x() {
    // The point layer's text x (d, a, e, b) adds levels d and e: limits a..e, points at 4, 1, 5, 2.
    let layers = vec![col(), point_layer(Aes::new().x("xl").y("z"))];
    run_layers(companions!("col_point_new_level"), xy(), layers, None, "new levels");
}

#[test]
fn ggplot2_bars_point_with_a_numeric_x_sits_at_its_value() {
    // A numeric x on the discrete scale adds no level: each point sits at its own value (-1 .. 4.5)
    // and widens the continuous range (ScaleDiscretePosition$range_c).
    let layers = vec![col(), point_layer(Aes::new().x("xn").y("z"))];
    run_layers(companions!("col_point_numeric_x"), xy(), layers, None, "numeric x");
}

#[test]
fn ggplot2_bars_point_layer_before_the_bar_layer() {
    let point = point_layer(Aes::new().x("xl").y("z"));
    let bar = Layer::new(Geom::bar()).aes(Aes::new().x("x"));
    run_layers(companions!("point_then_bar"), Aes::new(), vec![point, bar], None, "point first");
}

#[test]
fn ggplot2_bars_col_point_and_vline() {
    let layers = vec![col(), point_layer(Aes::new().x("xl").y("z")), vline(6.0)];
    run_layers(companions!("col_point_vline"), xy(), layers, None, "col + point + vline");
}

#[test]
fn ggplot2_bars_layer_fill_under_plot_x_ranks_fill_first() {
    // aes(x) + geom_bar(aes(fill = g)) + geom_col(aes(y, fill = g), dodge): computed_mapping lists
    // the layer's fill before the plot's x, so groups are ranked by (fill, x): (g1,b)=1 .. (g2,b)=4.
    let bar = Layer::new(Geom::bar()).aes(Aes::new().fill("g"));
    let dodged = col().aes(Aes::new().y("y").fill("g")).position(Position::dodge());
    let fx = companions!("layer_fill_two_layers");
    run_layers(fx, Aes::new().x("x"), vec![bar, dodged], None, "layer fill, two layers");
}

#[test]
fn ggplot2_bars_layer_fill_dodge_ranks_fill_first() {
    let l = col().aes(Aes::new().fill("g")).position(Position::dodge());
    run_layers(companions!("layer_fill_dodge"), xy(), vec![l], None, "layer fill dodge");
}

#[test]
fn ggplot2_bars_layer_fill_stack_ranks_fill_first() {
    let l = col().aes(Aes::new().fill("g"));
    run_layers(companions!("layer_fill_stack"), xy(), vec![l], None, "layer fill stack");
}

#[test]
fn ggplot2_bars_plot_fill_with_layer_x_ranks_x_first() {
    // aes(fill = g) + geom_bar(aes(x), dodge): the layer's x comes first in computed_mapping.
    let l = Layer::new(Geom::bar()).aes(Aes::new().x("x")).position(Position::dodge());
    let fx = companions!("plot_fill_layer_x");
    run_layers(fx, Aes::new().fill("g"), vec![l], None, "plot fill, layer x");
}

#[test]
fn ggplot2_bars_layer_x_and_fill_together_rank_x_first() {
    let l = Layer::new(Geom::bar()).aes(xg()).position(Position::dodge());
    run_layers(companions!("layer_x_and_fill"), Aes::new(), vec![l], None, "layer x and fill");
}

#[test]
fn ggplot2_bars_position_fill_over_a_zero_total_keeps_zero_heights() {
    let aes = Aes::new().x("x").y("y0").fill("g");
    let l = col().position(Position::fill());
    run_layers(companions!("fill_zero_sum"), aes, vec![l], None, "fill zero sum");
}

#[test]
fn ggplot2_bars_position_fill_with_a_negative_bar() {
    let aes = Aes::new().x("x").y("ys").fill("g");
    let l = col().position(Position::fill());
    run_layers(companions!("fill_signed"), aes, vec![l], None, "fill signed");
}

#[test]
fn ggplot2_bars_facet_bar_and_hline_one_row_per_panel() {
    let layers = vec![Layer::new(Geom::bar()), hline(2.0)];
    let fx = companions!("facet_bar_hline");
    run_layers(fx, xg(), layers, Some(facet_f()), "facet bar + hline");
}

#[test]
fn ggplot2_bars_facet_dodge_and_vline() {
    let layers = vec![col().position(Position::dodge()), vline(3.75)];
    let fx = companions!("facet_col_vline");
    run_layers(fx, xyg(), layers, Some(facet_f()), "facet dodge + vline");
}

#[test]
fn ggplot2_bars_dodge_preserve_single_and_vline_at_zero() {
    let single = Position::Dodge { width: None, preserve: Preserve::Single };
    let layers = vec![col().position(single), vline(0.0)];
    run_layers(companions!("dodge_single_vline"), xyg(), layers, None, "dodge single + vline");
}

#[test]
fn ggplot2_bars_single_level_and_hline() {
    let (csv, json) = companions!("single_level_hline");
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    let df = frame_from_csv(&filter_csv(csv, 0, &["a"]));
    let layout = layout_of_layers(&df, xy(), vec![col(), hline(1.0)], None).expect("one level");
    check_layout(&layout, &want, "single level + hline");
}

#[test]
fn ggplot2_bars_col_with_a_nan_y_is_a_typed_error() {
    // ggplot2 keeps the NaN row with an NA x and NA edges (the fixture); layout() refuses it.
    let (csv, json) = companions!("col_nan_y");
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert!(want["layers"][0]["x"][1].is_null(), "R drops the x of the NaN row");
    assert!(want["layers"][0]["ymin"][1].is_null(), "R drops the edges of the NaN row");
    let aes = Aes::new().x("x").y("yn");
    let err = layout_of_layers(&frame_from_csv(csv), aes, vec![col()], None).expect_err("NaN y");
    assert!(err.to_string().contains("`yn` holds a NaN"), "{err}");
}

// --- what the bar layout refuses ------------------------------------------------------------------

fn abcd() -> DataFrame {
    frame_from_csv("x,y,z,f\na,2,1,p\nb,3,4,q\n")
}

#[test]
fn bars_under_a_non_default_coord_are_refused() {
    for coord in [
        Coord::Cartesian { xlim: None, ylim: None, flip: true },
        Coord::Cartesian { xlim: None, ylim: Some((0.0, 1.0)), flip: false },
    ] {
        let plot = GGPlot::new().data(abcd()).aes(xy()).layer(col()).coord(coord.clone());
        let err = plot.build().expect("plot builds").layout().expect_err("coord refused");
        assert!(matches!(err, Error::UnsupportedCoord(_)), "{coord:?}: {err}");
    }
}

#[test]
fn bars_beside_an_unsupported_companion_are_refused() {
    let err =
        |l: Layer| layout_of_layers(&abcd(), xy(), vec![col(), l], None).expect_err("refused");
    assert!(matches!(err(Layer::new(Geom::line())), Error::UnsupportedLayer(_)), "geom_line");
    assert!(matches!(err(Layer::new(Geom::abline(1.0, 0.0))), Error::UnsupportedLayer(_)));
    let smooth = Layer::new(Geom::point().stat(Stat::Smooth));
    assert!(matches!(err(smooth), Error::UnsupportedStat(_)), "stat_smooth point");
    let missing = point_layer(Aes::new().y("nope"));
    assert!(matches!(err(missing), Error::UnknownColumn(c) if c == "nope"), "missing y");
    let text_y = point_layer(Aes::new().y("f"));
    assert!(err(text_y).to_string().contains("`f` is text"), "text y");
}

#[test]
fn a_point_beside_bars_under_a_facet_is_refused() {
    let layers = vec![col(), point_layer(Aes::new().y("z"))];
    let err = layout_of_layers(&abcd(), xy(), layers, Some(facet_f())).expect_err("refused");
    assert!(err.to_string().contains("does not assign point panels"), "{err}");
}

#[test]
fn a_bar_stat_other_than_count_or_identity_is_refused() {
    let l = Layer::new(Geom::bar().stat(Stat::Bin { bins: 3 }));
    let err = layout_of_layers(&abcd(), xy(), vec![l], None).expect_err("stat_bin refused");
    assert!(matches!(err, Error::UnsupportedStat(_)), "{err}");
    let ok = Layer::new(Geom::bar().stat(Stat::Count));
    assert!(layout_of_layers(&abcd(), Aes::new().x("x"), vec![ok], None).is_ok(), "stat_count");
}

#[test]
fn a_short_column_is_refused_not_read_past() {
    let mut df = DataFrame::new();
    df.add_column_str("x", &["a", "b", "c"]);
    df.add_column_f32("y", &[1.0, 2.0]);
    df.add_column_str("g", &["u"]);
    let err = layout_of_layers(&df, xy(), vec![col()], None).expect_err("short y");
    assert!(err.to_string().contains("`y` has 2 values for 3 rows"), "{err}");
    let fill = Layer::new(Geom::bar()).aes(Aes::new().fill("g"));
    let err = layout_of_layers(&df, Aes::new().x("x"), vec![fill], None).expect_err("short fill");
    assert!(err.to_string().contains("`g` has 1 values for 3 rows"), "{err}");
}

#[test]
fn a_text_y_on_a_col_layer_is_refused() {
    let mut df = DataFrame::new();
    df.add_column_str("x", &["a", "b"]);
    df.add_column_str("y", &["u", "v"]);
    let err = layout_of_layers(&df, xy(), vec![col()], None).expect_err("text y");
    assert!(err.to_string().contains("column `y` has the wrong type"), "{err}");
}

#[test]
fn position_apply_matches_one_panel_and_leaves_identity_alone() {
    use super::position::{apply, apply_panels, BarRow};
    let row = |x: f64, y: f64, group: u32| BarRow {
        x,
        y,
        ymin: 0.0,
        ymax: y,
        xmin: x - 0.45,
        xmax: x + 0.45,
        width: 0.9,
        group,
        count: None,
        prop: None,
    };
    let rows = vec![row(1.0, 2.0, 1), row(1.0, 3.0, 2), row(2.0, 4.0, 1)];
    let mut same = rows.clone();
    apply(&mut same, Position::Identity);
    assert_eq!(same, rows);
    for position in [Position::dodge(), Position::Dodge { width: None, preserve: Preserve::Single }]
    {
        let mut flat = rows.clone();
        apply(&mut flat, position);
        let mut panels = vec![rows.clone()];
        apply_panels(&mut panels, position);
        assert_eq!(flat, panels[0], "{position:?}");
        // Two groups at x = 1 split its 0.9 width; the lone bar at x = 2 keeps it under "total".
        assert_close_slice(
            "dodged x",
            &flat.iter().map(|r| r.x).collect::<Vec<_>>()[..2],
            &[0.775, 1.225],
        );
    }
}

#[test]
fn a_degenerate_y_range_beside_bars_is_refused() {
    let df = frame_from_csv("x,y\na,0\nb,0\n");
    let err = layout_of_layers(&df, xy(), vec![col()], None).expect_err("all-zero bars");
    assert!(matches!(&err, Error::ScaleDomain(m) if m.contains("one value")), "{err}");
    // A reference line that gives the range its second value makes it valid again.
    assert!(layout_of_layers(&df, xy(), vec![col(), hline(1.0)], None).is_ok());
}

#[test]
fn a_non_finite_bar_or_dodge_width_is_refused() {
    let wide = Layer::new(Geom::col().width(f32::INFINITY));
    let err = layout_of_layers(&abcd(), xy(), vec![wide], None).expect_err("infinite width");
    assert!(err.to_string().contains("width must be a finite number"), "{err}");
    let nan = Position::Dodge { width: Some(f64::NAN), preserve: Preserve::Total };
    let err = layout_of_layers(&abcd(), xy(), vec![col().position(nan)], None).expect_err("NaN");
    assert!(err.to_string().contains("width must be a finite number"), "{err}");
}

// --- ignored inputs are refused, not laid out differently from ggplot2 (review of #6938) ----------

/// Rows `a,a,b,c,c` with a discrete `h` column: ggplot2 groups by every discrete mapped aesthetic,
/// so mapping `h` to colour, alpha, size or shape changes the bar count (R11, R12, R13, R34, E2).
fn grouped() -> DataFrame {
    frame_from_csv("x,y,h,n,f\na,1,u,2,p\na,2,v,10,p\nb,3,u,2,q\nc,4,u,10,q\nc,5,v,10,q\n")
}

#[test]
fn a_mapped_colour_alpha_size_shape_or_label_on_a_bar_is_refused() {
    let mapped = [
        ("colour", Aes::new().color("h")),
        ("alpha", Aes::new().alpha("h")),
        ("size", Aes::new().size("h")),
        ("shape", Aes::new().shape("h")),
        ("label", Aes::new().label("h")),
    ];
    for (what, extra) in mapped {
        let merge = |base: Aes| base.merge(&extra);
        for layer in [Layer::new(Geom::bar()), Layer::new(Geom::col())] {
            let base = if matches!(layer.geom.stat, Some(Stat::Identity)) {
                xy()
            } else {
                Aes::new().x("x")
            };
            // The mapping may sit on the plot or on the layer; both are refused.
            let on_plot =
                layout_of_layers(&grouped(), merge(base.clone()), vec![layer.clone()], None);
            let on_layer = layout_of_layers(
                &grouped(),
                base.clone(),
                vec![layer.clone().aes(merge(base))],
                None,
            );
            for err in [on_plot, on_layer] {
                let err = err.expect_err(what);
                assert!(
                    matches!(&err, Error::Rendering(m) if m.contains("data-mapped colour")),
                    "{what}: {err}"
                );
            }
        }
    }
    // Control: the same layers without the mapped aesthetic are laid out.
    assert!(layout_of_layers(&grouped(), Aes::new().x("x"), vec![Layer::new(Geom::bar())], None)
        .is_ok());
    assert!(layout_of_layers(&grouped(), xy(), vec![Layer::new(Geom::col())], None).is_ok());
}

#[test]
fn a_fixed_colour_on_a_bar_is_still_laid_out() {
    // A fixed colour changes the outline, not the groups: nothing to refuse.
    let aes = Aes::new().x("x").color_value(crate::color::Rgba::BLACK);
    assert!(layout_of_layers(&grouped(), aes, vec![Layer::new(Geom::bar())], None).is_ok());
}

// --- PositionStack skips stacking when no x repeats, fill is off and vjust is 1 (review NEW-2) ----

macro_rules! stack_skip {
    ($data:literal, $name:literal) => {
        (
            include_str!(concat!("../../fixtures/ggplot2/bars_stack_skip/", $data)),
            include_str!(concat!(
                "../../fixtures/ggplot2/bars_stack_skip/expected_",
                $name,
                ".json"
            )),
        )
    };
}

#[test]
fn ggplot2_bars_stack_skips_a_unique_x_so_a_negative_bar_keeps_its_value() {
    // R: y = -2 3 -1. Stacking would set the negative bars' y to 0.
    run_layers(stack_skip!("data.csv", "unique"), xy(), vec![col()], None, "unique x");
    // N04: a single negative row.
    run_layers(stack_skip!("one.csv", "one_row"), xy(), vec![col()], None, "one row");
    // Panels do not change it while x stays unique over the layer.
    run_layers(
        stack_skip!("data.csv", "unique_facet"),
        xy(),
        vec![col()],
        Some(Facet::wrap("f", 2)),
        "unique x, facet",
    );
}

#[test]
fn ggplot2_bars_stack_still_stacks_when_a_skip_condition_fails() {
    // vjust != 1 (U2) and fill (U3) stack even over a unique x.
    let half = col().position(Position::Stack { vjust: 0.5, reverse: false });
    run_layers(stack_skip!("data.csv", "vjust05"), xy(), vec![half], None, "vjust 0.5");
    let fill = col().position(Position::fill());
    run_layers(stack_skip!("data.csv", "fill"), xy(), vec![fill], None, "fill");
    // x repeats only across panels: the layer's x has a duplicate, so it stacks (y = 0 3 0 1).
    run_layers(
        stack_skip!("cross.csv", "cross_panel"),
        xy(),
        vec![col()],
        Some(Facet::wrap("f", 2)),
        "x repeats across panels",
    );
}

#[test]
fn every_render_target_refuses_a_position_it_would_not_apply() {
    // NEW-1: to_svg / to_framebuffer / to_terminal draw the unmoved layer unless they refuse, where
    // ggplot2 stacks, dodges or fills. Point, line, hline and numeric-x col, each with a duplicate x.
    use super::term::TerminalPlot;
    let df = frame_from_csv("x,y\n1,2\n1,-3\n2,4\n");
    let line = |p: Position| Layer::new(Geom::line()).position(p);
    let layers: Vec<(&str, Box<dyn Fn(Position) -> Layer>)> = vec![
        ("point", Box::new(|p| Layer::new(Geom::point()).position(p))),
        ("line", Box::new(line)),
        ("hline", Box::new(|p| Layer::new(Geom::hline(1.0)).position(p))),
        ("numeric-x col", Box::new(|p| Layer::new(Geom::col()).position(p))),
    ];
    for (name, make) in &layers {
        for p in [Position::stack(), Position::fill(), Position::dodge()] {
            let plot =
                GGPlot::new().data(df.clone()).aes(xy()).layer(make(p)).build().expect("plot");
            let what = format!("{name} {p:?}");
            assert!(plot.layout().is_err(), "{what}: layout");
            assert!(plot.to_svg().is_err(), "{what}: to_svg");
            assert!(plot.to_framebuffer().is_err(), "{what}: to_framebuffer");
            let term = plot.to_terminal(TerminalPlot { width: 60, height: 20, color: false });
            assert!(term.is_err(), "{what}: to_terminal");
        }
    }
    // The refusal names the position, not an unrelated cause.
    let plot = GGPlot::new()
        .data(df.clone())
        .aes(xy())
        .layer(Layer::new(Geom::point()).position(Position::stack()))
        .build()
        .expect("plot");
    let err = plot.to_svg().expect_err("svg");
    assert!(matches!(&err, Error::Rendering(m) if m.contains("position adjustment")), "{err}");
    // A default-stack bar layer is drawn only while no x repeats (ggplot2 skips the stack then).
    let numeric_col = |df: &DataFrame, p: Position| {
        GGPlot::new().data(df.clone()).aes(xy()).layer(col().position(p)).build().expect("plot")
    };
    let unique = frame_from_csv("x,y\n1,2\n2,-3\n3,4\n");
    assert!(numeric_col(&unique, Position::stack()).to_framebuffer().is_ok(), "unique x");
    assert!(numeric_col(&df, Position::stack()).to_framebuffer().is_err(), "repeated x");
    let half = Position::Stack { vjust: 0.5, reverse: false };
    assert!(numeric_col(&unique, half).to_framebuffer().is_err(), "vjust 0.5 stacks");
    // The exception draws exactly what identity draws (no stack is applied), on both targets.
    let stacked = numeric_col(&unique, Position::stack());
    let ident = numeric_col(&unique, Position::Identity);
    assert_eq!(stacked.to_svg().expect("svg"), ident.to_svg().expect("svg"), "svg = identity");
    assert_eq!(
        stacked.to_framebuffer().expect("fb").pixels(),
        ident.to_framebuffer().expect("fb").pixels(),
        "framebuffer = identity"
    );
    // Controls: identity on the same layers still renders on every target.
    for (name, make) in &layers {
        // A reference layer alone trains no axes (#6975 refuses it), so it rides on a point layer.
        let mut b = GGPlot::new().data(df.clone()).aes(xy());
        if *name == "hline" {
            b = b.layer(Layer::new(Geom::point()));
        }
        let plot = b.layer(make(Position::Identity)).build().expect("plot");
        assert!(plot.to_framebuffer().is_ok(), "{name}: identity framebuffer");
    }
}

#[test]
fn the_unique_x_test_reads_the_layers_own_data_aes_and_signed_zeros() {
    // ggplot2 stacks these (vec_duplicate_any on the layer's merged x); drawing them unmoved is wrong.
    let refused = |plot: super::BuiltGGPlot, what: &str| {
        let err = plot.to_svg().expect_err(what);
        assert!(
            matches!(&err, Error::Rendering(m) if m.contains("position adjustment")),
            "{what}: {err}"
        );
        assert!(plot.to_framebuffer().is_err(), "{what}: to_framebuffer");
    };
    // A08: 0 and -0 are one x, so the layer stacks (R: x = 0 rows stack to 0..1 and 1..3).
    let zeros = frame_from_csv("x,y\n0,1\n-0,2\n1,3\n");
    let plot = GGPlot::new().data(zeros).aes(xy()).layer(col()).build().expect("plot");
    refused(plot, "signed zeros");
    // A09: the plot data is unique but the layer's own data repeats x.
    let unique = frame_from_csv("x,y\n1,1\n2,2\n3,3\n");
    let dup = frame_from_csv("x,y\n1,1\n1,2\n3,3\n");
    let plot =
        GGPlot::new().data(unique.clone()).aes(xy()).layer(col().data(dup)).build().expect("plot");
    refused(plot, "layer data repeats x");
    // A10: the plot aes maps a unique column, the layer aes maps a repeating one.
    let both = frame_from_csv("u,d,y\n1,1,1\n2,1,2\n3,3,3\n");
    let plot = GGPlot::new()
        .data(both)
        .aes(Aes::new().x("u").y("y"))
        .layer(col().aes(Aes::new().x("d")))
        .build()
        .expect("plot");
    refused(plot, "layer aes repeats x");
    // Controls: the same shapes with unique x are drawn.
    let ok = GGPlot::new()
        .data(frame_from_csv("x,y\n0,1\n1,2\n"))
        .aes(xy())
        .layer(col())
        .build()
        .expect("plot");
    assert!(ok.to_svg().is_ok() && ok.to_framebuffer().is_ok());
}

#[test]
fn every_exact_test_a_bars_contract_names_exists() {
    // A falsifier whose `--exact` name matches no test runs zero tests and passes: it proves nothing.
    let contracts = [
        include_str!("../../../../contracts/trueno-viz/ggplot2-bars-position-dodge-v1.yaml"),
        include_str!("../../../../contracts/trueno-viz/ggplot2-bars-position-fill-v1.yaml"),
        include_str!("../../../../contracts/trueno-viz/ggplot2-bars-position-stack-v1.yaml"),
        include_str!("../../../../contracts/trueno-viz/ggplot2-bars-stat-count-geom-bar-v1.yaml"),
        include_str!(
            "../../../../contracts/trueno-viz/ggplot2-bars-stat-identity-geom-col-v1.yaml"
        ),
    ];
    let source = include_str!("ggplot2_bars_oracle.rs");
    let mut named = 0;
    for text in contracts {
        for line in text.lines().filter(|l| l.contains("--exact ")) {
            let path = line.split("--exact ").nth(1).and_then(|r| r.split_whitespace().next());
            let path = path.expect("an --exact path");
            let name = path
                .strip_prefix("grammar::ggplot2_bars_oracle::")
                .unwrap_or_else(|| panic!("{path}: not a test of this module"));
            assert!(source.contains(&format!("fn {name}(")), "{path}: no such test");
            named += 1;
        }
    }
    assert!(named >= 30, "only {named} --exact names found: the scan is blind");
}

#[test]
fn a_position_other_than_identity_on_a_non_bar_layer_is_refused() {
    let positions = [Position::stack(), Position::fill(), Position::dodge()];
    let point = |p: Position| point_layer(Aes::new().y("y")).position(p);
    for p in positions {
        // Beside bars (R17) ...
        let err =
            layout_of_layers(&grouped(), xy(), vec![col(), point(p)], None).expect_err("point");
        assert!(
            matches!(&err, Error::Rendering(m) if m.contains("not a bar layer")),
            "{p:?}: {err}"
        );
        // ... and without a bar layer (R35).
        let err = layout_of_layers(&grouped(), xy(), vec![point(p)], None).expect_err("alone");
        assert!(
            matches!(&err, Error::Rendering(m) if m.contains("not a bar layer")),
            "{p:?}: {err}"
        );
        // Reference lines are non-bar layers too (P16, P17).
        let err = layout_of_layers(&grouped(), xy(), vec![col(), hline(1.0).position(p)], None)
            .expect_err("hline");
        assert!(matches!(&err, Error::Rendering(_)), "{p:?}: {err}");
    }
    // Identity is the default and stays accepted.
    let ok = point(Position::Identity);
    assert!(layout_of_layers(&grouped(), xy(), vec![col(), ok], None).is_ok());
}

#[test]
fn a_facet_column_that_is_numeric_or_short_is_refused_under_bars() {
    // R14: facet_wrap(~n) sorts 2 before 10; the text-sorted panels would put "10" first.
    let by_n = layout_of_layers(&grouped(), xy(), vec![col()], Some(Facet::wrap("n", 2)));
    let err = by_n.expect_err("numeric facet");
    assert!(err.to_string().contains("must be text"), "{err}");
    // P09: a facet column shorter than the data would put the missing rows in a panel "NA".
    let mut df = DataFrame::new();
    df.add_column_str("x", &["a", "b", "c"]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    df.add_column_str("f", &["p", "q"]);
    let err =
        layout_of_layers(&df, xy(), vec![col()], Some(Facet::wrap("f", 2))).expect_err("short");
    assert!(err.to_string().contains("must be text"), "{err}");
    // Every facet shape is checked, and a text facet is still accepted.
    for facet in [Facet::row("n"), Facet::col("n"), Facet::grid("f", "n"), Facet::grid("n", "f")] {
        let err =
            layout_of_layers(&grouped(), xy(), vec![col()], Some(facet)).expect_err("numeric");
        assert!(err.to_string().contains("must be text"), "{err}");
    }
    assert!(layout_of_layers(&grouped(), xy(), vec![col()], Some(Facet::wrap("f", 2))).is_ok());
    assert!(layout_of_layers(&grouped(), xy(), vec![col()], Some(Facet::grid("f", "h"))).is_ok());
}

#[test]
fn geom_bar_with_a_y_mapped_is_refused_as_ggplot2_does() {
    // E1: R errors "stat_count() must only have an x or y aesthetic".
    let err = layout_of_layers(&grouped(), xy(), vec![Layer::new(Geom::bar())], None)
        .expect_err("y on geom_bar");
    assert!(err.to_string().contains("takes an x aesthetic only"), "{err}");
    let counted = Layer::new(Geom::bar().stat(Stat::Count));
    assert!(
        layout_of_layers(&grouped(), xy(), vec![counted], None).is_err(),
        "explicit stat_count"
    );
    assert!(layout_of_layers(&grouped(), xy(), vec![col()], None).is_ok(), "geom_col takes y");
    let no_y = layout_of_layers(&grouped(), Aes::new().x("x"), vec![col()], None)
        .expect_err("geom_col without y");
    assert!(no_y.to_string().contains("geom_col() needs a y aesthetic"), "{no_y}");
}

#[test]
fn a_point_companion_with_a_short_column_is_refused() {
    // P08: a point whose y holds fewer values than the frame has rows used to lose rows silently.
    let mut df = DataFrame::new();
    df.add_column_str("x", &["a", "b", "c"]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    df.add_column_f32("z", &[5.0]);
    let short_y = point_layer(Aes::new().y("z"));
    let err = layout_of_layers(&df, xy(), vec![col(), short_y], None).expect_err("short y");
    assert!(err.to_string().contains("`z` has 1 values for 3 rows"), "{err}");
    let mut df = DataFrame::new();
    df.add_column_str("x", &["a", "b", "c"]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    let mut own = DataFrame::new();
    own.add_column_str("px", &["a", "b", "c"]);
    own.add_column_f32("py", &[5.0, 6.0]);
    let layer = point_layer(Aes::new().x("px").y("py")).data(own);
    let err = layout_of_layers(&df, xy(), vec![col(), layer], None).expect_err("short own y");
    assert!(err.to_string().contains("`py` has 2 values for 3 rows"), "{err}");
}

#[test]
fn a_non_finite_stack_or_fill_vjust_is_refused() {
    for vjust in [f64::NAN, f64::INFINITY] {
        for p in
            [Position::Stack { vjust, reverse: false }, Position::Fill { vjust, reverse: false }]
        {
            let err =
                layout_of_layers(&abcd(), xy(), vec![col().position(p)], None).expect_err("vjust");
            assert!(err.to_string().contains("vjust must be a finite number"), "{p:?}: {err}");
        }
    }
    let ok = Position::Stack { vjust: 0.5, reverse: false };
    assert!(layout_of_layers(&abcd(), xy(), vec![col().position(ok)], None).is_ok());
}

#[test]
fn level_positions_scale_to_many_levels_and_match_a_linear_scan() {
    use super::stat_count::level_positions;
    let levels: Vec<String> = (0..50_000).map(|i| format!("l{i:06}")).collect();
    let values: Vec<&str> = levels.iter().rev().map(String::as_str).chain(["absent"]).collect();
    let got = level_positions(&levels, &values);
    assert_eq!(got.len(), 50_001);
    assert_eq!(got[0], 50_000.0, "last level is last");
    assert_eq!(got[49_999], 1.0);
    assert!(got[50_000].is_nan(), "an unknown value has no position");
    // A repeated level keeps its first position, as a linear scan reports.
    let repeated = vec!["b".to_string(), "a".to_string(), "b".to_string()];
    assert_eq!(level_positions(&repeated, &["b", "a"]), vec![1.0, 2.0]);
}

/// A position-scale spec with a bar layer is a typed refusal: the bar layout trains its own scales,
/// so honouring `scale_x_log10` silently would draw a different plot than R.
#[test]
fn bars_with_a_position_scale_spec_are_refused() {
    use super::scale_log10;
    let df = frame_from_csv(
        "x
a
b
",
    );
    let plot = GGPlot::new()
        .data(df)
        .aes(Aes::new().x("x"))
        .geom(Geom::bar())
        .scale_y(scale_log10())
        .build()
        .expect("plot builds");
    let got = plot.layout();
    assert!(
        matches!(&got, Err(Error::UnsupportedLayer(m)) if m.contains("geom_point only")),
        "{got:?}"
    );
}
