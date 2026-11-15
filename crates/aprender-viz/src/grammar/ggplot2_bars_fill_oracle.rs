//! ggplot2 conformance oracle for `position_fill()` on `geom_col()` / `geom_bar()`
//! (PLOT-GG-001, paiml/aprender#6931, master #6909).
//!
//! Fixture: `fixtures/ggplot2/position_fill/expected.json`, produced by ggplot2 4.0.3 under C
//! collation (`generate.R`); R is never run here. Each case rebuilds the same plot with
//! [`bar_layout`] and compares every bar row and both trained scales.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for positions,
//! limits and breaks; labels, groups and panels compare exactly; `NaN` matches `null`.
//!
//! This module is a sibling of the bars oracle (#6928/#6929) so the two rows do not edit one
//! file; it can fold into that file once it lands.

use serde_json::Value;

use super::{bar_layout, BarSpec, BuiltBars, DataFrame, Position};

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

const JSON: &str = include_str!("../../fixtures/ggplot2/position_fill/expected.json");
const DATA: &str = include_str!("../../fixtures/ggplot2/position_fill/data.csv");
const DATA_COUNT: &str = include_str!("../../fixtures/ggplot2/position_fill/data_count.csv");
const DATA_NEG: &str = include_str!("../../fixtures/ggplot2/position_fill/data_neg.csv");
const DATA_UNIQUE: &str = include_str!("../../fixtures/ggplot2/position_fill/data_unique.csv");
const DATA_ZERO: &str = include_str!("../../fixtures/ggplot2/position_fill/data_zero.csv");

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

thread_local! {
    /// Largest `|got - want|` over every compared value of the current case (NaN pairs skipped).
    static MAX_ABS_ERR: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
        if !g.is_nan() {
            MAX_ABS_ERR.with(|m| m.set(m.get().max((g - w).abs())));
        }
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

/// A CSV of text and numeric columns. A column is numeric when every cell parses, and then every
/// value must be exact in f32 (the frame stores f32), so the test measures the position math.
fn frame(csv: &str) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv header").split(',').collect();
    let rows: Vec<Vec<&str>> =
        lines.filter(|l| !l.trim().is_empty()).map(|l| l.split(',').collect()).collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[c].trim()).collect();
        let parsed: Option<Vec<f64>> = cells.iter().map(|s| s.parse().ok()).collect();
        match parsed {
            Some(vals) => {
                let f: Vec<f32> = vals.iter().map(|v| *v as f32).collect();
                for (a, b) in vals.iter().zip(&f) {
                    assert_eq!(f64::from(*b), *a, "fixture value {a} is not exact in f32");
                }
                df.add_column_f32(name, &f);
            }
            None => df.add_column_str(name, &cells),
        }
    }
    df
}

fn csv_for(name: &str) -> &'static str {
    match name {
        "data.csv" => DATA,
        "data_count.csv" => DATA_COUNT,
        "data_neg.csv" => DATA_NEG,
        "data_unique.csv" => DATA_UNIQUE,
        "data_zero.csv" => DATA_ZERO,
        other => panic!("unknown fixture data file {other}"),
    }
}

fn fill(vjust: f64, reverse: bool) -> Position {
    Position::Fill { vjust, reverse }
}

/// The spec each fixture case was made from (its `spec` string is in the fixture).
fn spec_for(case: &str) -> BarSpec {
    match case {
        "col_fill" | "col_fill_negative" | "col_fill_unique_x" | "col_fill_zero_total" => {
            BarSpec::col("x", "y").fill("g").position(Position::fill())
        }
        "col_fill_reverse" => BarSpec::col("x", "y").fill("g").position(fill(1.0, true)),
        "col_fill_vjust_half" => BarSpec::col("x", "y").fill("g").position(fill(0.5, false)),
        "col_fill_vjust_zero_reverse" => BarSpec::col("x", "y").fill("g").position(fill(0.0, true)),
        "col_fill_width" => BarSpec::col("x", "y").fill("g").width(0.5).position(Position::fill()),
        "bar_fill" => BarSpec::bar("x").fill("g").position(Position::fill()),
        other => panic!("no spec for fixture case {other}"),
    }
}

fn check_case(case: &Value) -> BuiltBars {
    let name = case["name"].as_str().expect("case name");
    MAX_ABS_ERR.with(|m| m.set(0.0));
    let df = frame(csv_for(case["data_file"].as_str().expect("data_file")));
    let got = bar_layout(&df, &spec_for(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let l = &case["layer"];
    let col = |f: fn(&super::BarRow) -> f64| -> Vec<f64> { got.rows.iter().map(f).collect() };
    assert_close_slice(&format!("{name}.x"), &col(|r| r.x), &nums(&l["x"]));
    assert_close_slice(&format!("{name}.y"), &col(|r| r.y), &nums(&l["y"]));
    assert_close_slice(&format!("{name}.ymin"), &col(|r| r.ymin), &nums(&l["ymin"]));
    assert_close_slice(&format!("{name}.ymax"), &col(|r| r.ymax), &nums(&l["ymax"]));
    assert_close_slice(&format!("{name}.xmin"), &col(|r| r.xmin), &nums(&l["xmin"]));
    assert_close_slice(&format!("{name}.xmax"), &col(|r| r.xmax), &nums(&l["xmax"]));
    assert_close_slice(&format!("{name}.width"), &col(|r| r.width), &nums(&l["width"]));
    let group: Vec<u32> = got.rows.iter().map(|r| r.group).collect();
    let want_group: Vec<u32> = nums(&l["group"]).iter().map(|g| *g as u32).collect();
    assert_eq!(group, want_group, "{name}.group");
    let panels: Vec<usize> = nums(&l["panel"]).iter().map(|p| *p as usize).collect();
    assert_eq!(got.panel.iter().map(|p| p + 1).collect::<Vec<_>>(), panels, "{name}.panel");
    if case["stat"] == "count" {
        let c: Vec<f64> = got.rows.iter().map(|r| r.count.expect("count stat has count")).collect();
        let p: Vec<f64> = got.rows.iter().map(|r| r.prop.expect("count stat has prop")).collect();
        assert_close_slice(&format!("{name}.count"), &c, &nums(&l["count"]));
        assert_close_slice(&format!("{name}.prop"), &p, &nums(&l["prop"]));
    } else {
        assert!(
            got.rows.iter().all(|r| r.count.is_none() && r.prop.is_none()),
            "{name}: identity stat has no count"
        );
    }
    let px = &case["panel"]["x"];
    assert_eq!(got.x.limits, strs(&px["limits"]), "{name}.x.limits");
    assert_close_slice(
        &format!("{name}.x.continuous_range"),
        &[got.x.continuous_range.0, got.x.continuous_range.1],
        &nums(&px["continuous_range"]),
    );
    assert_close_slice(&format!("{name}.x.breaks"), &got.x.breaks, &nums(&px["break_positions"]));
    assert_eq!(got.x.labels, strs(&px["labels"]), "{name}.x.labels");
    let py = &case["panel"]["y"];
    assert_close_slice(
        &format!("{name}.y.limits"),
        &[got.y.limits.0, got.y.limits.1],
        &nums(&py["limits"]),
    );
    assert_close_slice(
        &format!("{name}.y.continuous_range"),
        &[got.y.continuous_range.0, got.y.continuous_range.1],
        &nums(&py["continuous_range"]),
    );
    assert_close_slice(&format!("{name}.y.breaks"), &got.y.breaks, &nums(&py["breaks"]));
    assert_eq!(got.y.labels, strs(&py["labels"]), "{name}.y.labels");
    assert_eq!(
        got.y.n_breaks_censored as u64,
        py["n_breaks_censored"].as_u64().expect("count"),
        "{name}.y.n_breaks_censored"
    );
    assert_close_slice(
        &format!("{name}.y.minor_breaks"),
        &got.y.minor_breaks,
        &nums(&py["minor_breaks"]),
    );
    eprintln!("MAXERR position_fill {name} {:e}", MAX_ABS_ERR.with(std::cell::Cell::get));
    got
}

fn fixture() -> Value {
    let v: Value = serde_json::from_str(JSON).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["versions"]["collate"], "C", "fixture was not made under C collation");
    v
}

fn run(name: &str) -> BuiltBars {
    let v = fixture();
    let case = v["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("fixture has no case {name}"));
    check_case(case)
}

#[test]
fn fixture_cases_are_all_covered() {
    let v = fixture();
    let names: Vec<&str> = v["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["name"].as_str().expect("n"))
        .collect();
    assert_eq!(names.len(), 9, "a case was added without a test: {names:?}");
    for n in names {
        let _ = spec_for(n);
    }
}

#[test]
fn position_fill_col_default() {
    let got = run("col_fill");
    // Every stack normalises to 1: the top edge of each x is exactly 1.
    assert!(got.rows.iter().any(|r| (r.ymax - 1.0).abs() < 1e-12));
}

#[test]
fn position_fill_col_reverse() {
    run("col_fill_reverse");
}

#[test]
fn position_fill_col_vjust_half() {
    run("col_fill_vjust_half");
}

#[test]
fn position_fill_col_vjust_zero_reverse() {
    run("col_fill_vjust_zero_reverse");
}

#[test]
fn position_fill_col_width() {
    run("col_fill_width");
}

#[test]
fn position_fill_bar_count() {
    run("bar_fill");
}

#[test]
fn position_fill_negative_values() {
    run("col_fill_negative");
}

#[test]
fn position_fill_unique_x_still_normalises() {
    let got = run("col_fill_unique_x");
    assert!(got.rows.iter().all(|r| (r.y - 1.0).abs() < 1e-12));
}

#[test]
fn position_fill_zero_total_stack() {
    run("col_fill_zero_total");
}

/// The comparison can see a wrong adjustment: stacking instead of filling must fail it.
#[test]
fn stack_is_not_fill() {
    let v = fixture();
    let case = &v["cases"][0];
    let df = frame(csv_for("data.csv"));
    let spec = BarSpec::col("x", "y").fill("g").position(Position::stack());
    let got = bar_layout(&df, &spec).expect("builds");
    let want = nums(&case["layer"]["ymax"]);
    let ymax: Vec<f64> = got.rows.iter().map(|r| r.ymax).collect();
    assert!(
        ymax.iter().zip(&want).any(|(g, w)| !close(*g, *w)),
        "stack output equals the fill oracle: the oracle cannot tell them apart"
    );
}
