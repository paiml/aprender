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

use super::stat_count::{bar_layout, BarSpec, BuiltBars, DiscreteAxis};
use super::{Aes, DataFrame, GGPlot, Geom, Layer, Position, PositionScale, Preserve};

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
                    assert_eq!(f64::from(*g), *v, "fixture value {v} is not exact in f32");
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

/// Compare every layer column ggplot2 reports and aprender-viz computes.
fn check_bars(got: &BuiltBars, want: &Value, expect_geom: &str) {
    let layer = &want["layers"][0];
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
    assert!(strs(&layer["PANEL"]).iter().all(|p| p == "1"), "one panel");
    check_discrete(&got.x, &want["panel"]["x"]);
    check_continuous(&got.y, &want["panel"]["y"]);
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
fn bar_layer_over_numeric_x_keeps_the_continuous_path() {
    // A bar layer whose x is numeric is not a discrete bar layer: no `bars`, positions as before.
    let df = frame_from_csv("x,y\n1,2\n2,3\n");
    let layout = GGPlot::new()
        .data(df)
        .aes(Aes::new().x("x").y("y"))
        .geom(Geom::bar())
        .build()
        .expect("plot builds")
        .layout()
        .expect("layout builds");
    assert!(layout.layers[0].bars.is_none());
    assert!(layout.x_discrete.is_none());
    assert_eq!(layout.layers[0].x, [1.0, 2.0]);
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
