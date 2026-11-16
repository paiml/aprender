//! ggplot2 conformance oracle for `geom_area()` (PLOT-GG-001, paiml/aprender#6956, master #6909):
//! `stat_align`, `GeomArea$setup_data` and `position_stack()`. Fixtures are
//! `fixtures/ggplot2/geom_area/expected_<case>.json`, produced by ggplot2 4.0.3 under C collation
//! (`generate.R`); R is never run here.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for every position,
//! limit, range and break; labels, groups, panels, flags and counts compare exactly. `NaN` matches
//! `null` (R's `NA`) by position. Every case is built through the public builder and read back
//! with `layout()`.
//!
//! The helpers are shared with `ggplot2_geom_ribbon_oracle`.

use serde_json::Value;

use super::stat::Stat;
use super::{
    Aes, BuiltFilled, BuiltLayer, BuiltLayout, DataFrame, Facet, GGPlot, Geom, Layer, PositionScale,
};
use crate::error::Error;

pub(super) const REL_TOL: f64 = 1e-9;
pub(super) const ABS_TOL: f64 = 1e-12;

pub(super) fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

thread_local! {
    /// Largest `|got - want|` over every compared value of the current case (NaN pairs skipped).
    static MAX_ABS_ERR: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

pub(super) fn reset_max_err() {
    MAX_ABS_ERR.with(|m| m.set(0.0));
}

pub(super) fn report_max_err(row: &str, case: &str) {
    eprintln!("MAXERR {row} {case} {:e}", MAX_ABS_ERR.with(std::cell::Cell::get));
}

pub(super) fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
        if !g.is_nan() {
            MAX_ABS_ERR.with(|m| m.set(m.get().max((g - w).abs())));
        }
    }
}

/// A JSON array of numbers; `null` (R's `NA`) is `NaN`.
pub(super) fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| if e.is_null() { f64::NAN } else { e.as_f64().expect("fixture number") })
        .collect()
}

pub(super) fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_i64().expect("fixture integer"))
        .collect()
}

pub(super) fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture string").to_string())
        .collect()
}

/// Parse a CSV into a frame. A column whose every non-`NA` cell parses as a number is numeric
/// (and each value must be exact in f32); any other column is text. `NA` in a numeric column is
/// `NaN`.
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
    ($row:literal, $file:expr) => {
        frame_from_csv(include_str!(concat!("../../fixtures/ggplot2/", $row, "/", $file, ".csv")))
    };
}

pub(super) use {csv, fixture};

pub(super) fn want(json: &str, row: &str, case: &str) -> Value {
    let v: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["collation"], "C", "fixture was not made under C collation");
    assert_eq!(v["row"], row);
    assert_eq!(v["case"], case);
    v
}

pub(super) fn check_axis(name: &str, got: &PositionScale, w: &Value) {
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

/// One filled layer against R's `ggplot_build()` data: every positional column, the panel and group
/// ids, `align_padding` and `flipped_aes`. A column R lacks must be absent from ours, and the other
/// way round.
pub(super) fn check_layer(tag: &str, got: &BuiltLayer, w: &Value, geom: &str) {
    assert_eq!(w["geom"], geom, "{tag}: R geom class");
    let f: &BuiltFilled = got.filled.as_ref().expect("a filled layer resolves filled data");
    let panel: Vec<i64> = f.panel.iter().map(|v| *v as i64 + 1).collect();
    assert_eq!(panel, ints(&w["panel"]), "{tag}.panel");
    assert_eq!(f.group, ints(&w["group"]), "{tag}.group");
    assert_close_slice(&format!("{tag}.x"), &got.x, &nums(&w["x"]));
    assert_close_slice(&format!("{tag}.y"), &got.y, &nums(&w["y"]));
    assert_close_slice(&format!("{tag}.ymin"), &f.ymin, &nums(&w["ymin"]));
    assert_close_slice(&format!("{tag}.ymax"), &f.ymax, &nums(&w["ymax"]));
    for (name, col) in [("xmin", &f.xmin), ("xmax", &f.xmax)] {
        assert_eq!(col.is_some(), w.get(name).is_some(), "{tag}.{name} presence");
        if let (Some(g), Some(r)) = (col, w.get(name)) {
            assert_close_slice(&format!("{tag}.{name}"), g, &nums(r));
        }
    }
    assert_eq!(
        f.align_padding.is_some(),
        w.get("align_padding").is_some(),
        "{tag}.align_padding presence"
    );
    if let (Some(g), Some(r)) = (&f.align_padding, w.get("align_padding")) {
        let want: Vec<Option<bool>> =
            r.as_array().expect("array").iter().map(|e| e.as_i64().map(|v| v != 0)).collect();
        assert_eq!(*g, want, "{tag}.align_padding");
    }
    assert_eq!(i64::from(f.flipped_aes), ints(&w["flipped_aes"])[0], "{tag}.flipped_aes");
}

pub(super) fn build(
    data: DataFrame,
    aes: Aes,
    layers: Vec<Layer>,
    facet: Facet,
) -> crate::error::Result<BuiltLayout> {
    let mut p = GGPlot::new().data(data).aes(aes).facet(facet);
    for l in layers {
        p = p.layer(l);
    }
    p.build()?.layout()
}

pub(super) fn wrap_f() -> Facet {
    Facet::Wrap { var: "f".into(), ncol: 2 }
}

fn xyg() -> Aes {
    Aes::new().x("x").y("y").group("g")
}

fn xy() -> Aes {
    Aes::new().x("x").y("y")
}

const CASES: usize = 11;

/// Compare one area case: both scales and the single `GeomArea` layer.
fn run(case: &str, data: DataFrame, aes: Aes, facet: Facet, json: &str) {
    reset_max_err();
    let w = want(json, "geom_area", case);
    let got = build(data, aes, vec![Layer::new(Geom::area())], facet)
        .unwrap_or_else(|e| panic!("{case}: {e:?}"));
    assert_eq!(got.layers.len(), 1, "{case}: layer count");
    check_layer(case, &got.layers[0], &w["layers"][0], "GeomArea");
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
    report_max_err("geom_area", case);
}

#[test]
fn equal_groups_share_their_locations() {
    run(
        "equal",
        csv!("geom_area", concat!("data_", "equal")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "equal"),
    );
}
#[test]
fn unequal_groups_are_interpolated_onto_the_union() {
    run(
        "unequal",
        csv!("geom_area", concat!("data_", "unequal")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "unequal"),
    );
}
#[test]
fn negative_groups_stack_downward_with_r_edge_order() {
    run(
        "negative",
        csv!("geom_area", concat!("data_", "negative")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "negative"),
    );
}
#[test]
fn a_zero_crossing_adds_its_own_location() {
    run(
        "cross",
        csv!("geom_area", concat!("data_", "cross")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "cross"),
    );
}
#[test]
fn a_single_group_is_left_unaligned() {
    run(
        "single",
        csv!("geom_area", concat!("data_", "single")),
        xy(),
        Facet::None,
        fixture!("geom_area", "single"),
    );
}
#[test]
fn a_single_group_with_a_repeated_x_stacks() {
    run(
        "dupx_single",
        csv!("geom_area", concat!("data_", "dupx_single")),
        xy(),
        Facet::None,
        fixture!("geom_area", "dupx_single"),
    );
}
#[test]
fn a_repeated_x_keeps_its_first_and_last_row() {
    run(
        "dupx_groups",
        csv!("geom_area", concat!("data_", "dupx_groups")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "dupx_groups"),
    );
}
#[test]
fn alignment_runs_inside_each_panel() {
    run(
        "facet",
        csv!("geom_area", concat!("data_", "facet")),
        xyg(),
        wrap_f(),
        fixture!("geom_area", "facet"),
    );
}
#[test]
fn a_one_row_group_vanishes() {
    run(
        "onerow",
        csv!("geom_area", concat!("data_", "onerow")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "onerow"),
    );
}
#[test]
fn missing_values_are_dropped_before_alignment() {
    run(
        "na",
        csv!("geom_area", concat!("data_", "na")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "na"),
    );
}
#[test]
fn rows_are_ordered_by_panel_group_and_x() {
    run(
        "unsorted",
        csv!("geom_area", concat!("data_", "unsorted")),
        xyg(),
        Facet::None,
        fixture!("geom_area", "unsorted"),
    );
}

#[test]
fn every_fixture_case_has_a_test() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ggplot2/geom_area");
    let n = std::fs::read_dir(dir)
        .expect("fixture dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("expected_"))
        .count();
    assert_eq!(n, CASES, "a fixture was added without a test (or the other way round)");
}

// --- refusals: out-of-scope input is a typed error, never a plausible wrong answer ---------------

fn tiny() -> DataFrame {
    csv!("geom_area", "data_equal")
}

fn lay(aes: Aes, geom: Geom) -> crate::error::Result<BuiltLayout> {
    build(tiny(), aes, vec![Layer::new(geom)], Facet::None)
}

#[test]
fn a_data_mapped_fill_is_refused() {
    let r = lay(xyg().fill("g"), Geom::area());
    assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("geom_area")), "{r:?}");
}

#[test]
fn an_explicit_stat_is_refused() {
    let r = lay(xyg(), Geom::area().stat(Stat::Identity));
    assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{r:?}");
}

#[test]
fn a_missing_position_column_is_refused() {
    let r = lay(Aes::new().x("nope").y("y").group("g"), Geom::area());
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
}

#[test]
fn every_companion_but_point_is_refused() {
    for companion in [Geom::bar(), Geom::histogram(), Geom::line(), Geom::path(), Geom::step()] {
        let r = build(
            tiny(),
            xyg(),
            vec![Layer::new(Geom::area()), Layer::new(companion.clone())],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{:?}: {r:?}", companion.geom_type);
    }
}

#[test]
fn an_area_alone_with_no_finite_position_is_refused_not_answered() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[f32::NAN, f32::NAN]);
    df.add_column_f32("y", &[1.0, 2.0]);
    let r = build(df, xy(), vec![Layer::new(Geom::area())], Facet::None);
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
}

#[test]
fn an_area_does_not_make_a_point_beside_it_wrong() {
    let r = build(
        tiny(),
        xyg(),
        vec![Layer::new(Geom::area()), Layer::new(Geom::point())],
        Facet::None,
    );
    let got = r.expect("area and point share a plot");
    assert_eq!(got.layers.len(), 2);
    assert!(got.layers[0].filled.is_some() && got.layers[1].filled.is_none());
}

/// A `geom_point` beside an area gets the same refusals as beside the other marks: a missing,
/// text or infinite position column is a typed error, never an `Ok` with an empty point layer.
#[test]
fn a_bad_point_companion_beside_an_area_is_refused() {
    let area_pt = |pt: Aes| {
        build(
            tiny(),
            xyg(),
            vec![Layer::new(Geom::area()), Layer::new(Geom::point()).aes(pt)],
            Facet::None,
        )
    };
    let r = area_pt(Aes::new().x("x").y("nope"));
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    let r = area_pt(Aes::new().x("x").y("g"));
    assert!(
        matches!(r, Err(Error::UnsupportedLayer(_) | Error::Rendering(_))),
        "text column: {r:?}"
    );
    let mut df = tiny();
    let n = df.nrow();
    df.add_column_f32("inf", &vec![f32::INFINITY; n]);
    let r = build(
        df,
        xyg(),
        vec![Layer::new(Geom::area()), Layer::new(Geom::point()).aes(Aes::new().x("x").y("inf"))],
        Facet::None,
    );
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
}
