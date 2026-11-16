//! ggplot2 conformance for `stat_identity` + `geom_col` (PLOT-GG-001, paiml/aprender#6909, row 9).
//!
//! The fixture `fixtures/ggplot2/geom_col/expected.json` holds `ggplot_build()` output
//! from ggplot2 4.0.3 (R 4.6.2, `LC_COLLATE=C`) for four `geom_col()` specs over `data.csv`: plain,
//! stacked with a `fill`, `position = "identity"`, and `width = 0.5`. Each case is rebuilt with
//! [`bar_layout`] and compared field by field. R is never run here.
//!
//! Tolerance: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for every number (as `ggplot2_oracle`);
//! levels, labels and group ids compare exactly.

use serde_json::Value;

use super::{bar_layout, BarSpec, BuiltBars, DataFrame, Position};

const JSON: &str = include_str!("../../fixtures/ggplot2/geom_col/expected.json");
const CSV: &str = include_str!("../../fixtures/ggplot2/geom_col/data.csv");

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
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

/// Largest absolute difference seen across every comparison of the run (printed by `--nocapture`).
fn assert_close(what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: got {got:?} want {want:?}");
    let mut worst = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?}");
        worst = worst.max((g - w).abs());
    }
    worst
}

/// The data rows named by `rows` (1-based, or all rows), as a frame: x, g text; y exact in f32.
fn frame(rows: Option<Vec<usize>>) -> DataFrame {
    let lines: Vec<Vec<&str>> = CSV
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').collect())
        .collect();
    let pick: Vec<usize> = rows.unwrap_or_else(|| (1..=lines.len()).collect());
    let col = |i: usize| -> Vec<&str> { pick.iter().map(|r| lines[r - 1][i]).collect() };
    let y: Vec<f32> = col(2)
        .iter()
        .map(|c| {
            let v: f64 = c.trim().parse().expect("y is numeric");
            let f = v as f32;
            assert_eq!(f64::from(f), v, "fixture value {v} is not exact in f32");
            f
        })
        .collect();
    let mut df = DataFrame::new();
    df.add_column_str("x", &col(0));
    df.add_column_str("g", &col(1));
    df.add_column_f32("y", &y);
    df
}

fn spec_of(name: &str) -> BarSpec {
    match name {
        "col_plain" => BarSpec::col("x", "y"),
        "col_stack_fill" => BarSpec::col("x", "y").fill("g"),
        "col_identity_fill" => BarSpec::col("x", "y").fill("g").position(Position::Identity),
        "col_width" => BarSpec::col("x", "y").width(0.5),
        other => panic!("fixture case {other} has no spec in this test"),
    }
}

fn built(case: &Value) -> BuiltBars {
    let rows =
        case["rows"].as_array().map(|_| nums(&case["rows"]).iter().map(|r| *r as usize).collect());
    bar_layout(&frame(rows), &spec_of(case["name"].as_str().expect("case name")))
        .expect("bar layout resolves")
}

fn fixture() -> Value {
    let want: Value = serde_json::from_str(JSON).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "fixture was not made by the pinned ggplot2");
    assert_eq!(want["versions"]["collate"], "C", "fixture was not made under LC_COLLATE=C");
    want
}

/// Compare one case; returns the worst absolute difference.
fn compare(case: &Value) -> f64 {
    let b = built(case);
    let name = case["name"].as_str().expect("case name");
    let l = &case["layer"];
    let fields: [(&str, fn(&super::BarRow) -> f64); 6] = [
        ("x", |r| r.x),
        ("y", |r| r.y),
        ("ymin", |r| r.ymin),
        ("ymax", |r| r.ymax),
        ("xmin", |r| r.xmin),
        ("xmax", |r| r.xmax),
    ];
    let mut worst = 0.0_f64;
    for (f, get) in fields {
        let got: Vec<f64> = b.rows.iter().map(get).collect();
        worst = worst.max(assert_close(&format!("{name}.{f}"), &got, &nums(&l[f])));
    }
    let groups: Vec<f64> = b.rows.iter().map(|r| f64::from(r.group)).collect();
    assert_close(&format!("{name}.group"), &groups, &nums(&l["group"]));
    assert!(
        b.rows.iter().all(|r| r.count.is_none() && r.prop.is_none()),
        "{name}: identity has no count"
    );

    let (px, py) = (&case["panel"]["x"], &case["panel"]["y"]);
    assert_eq!(b.x.limits, strs(&px["limits"]), "{name}.x.limits");
    assert_eq!(b.x.labels, strs(&px["labels"]), "{name}.x.labels");
    assert_eq!(b.x.limits, strs(&px["breaks"]), "{name}.x.breaks");
    worst = worst.max(assert_close(
        &format!("{name}.x.breaks"),
        &b.x.breaks,
        &nums(&px["break_positions"]),
    ));
    let xr = [b.x.continuous_range.0, b.x.continuous_range.1];
    worst =
        worst.max(assert_close(&format!("{name}.x.range"), &xr, &nums(&px["continuous_range"])));
    let yl = [b.y.limits.0, b.y.limits.1];
    worst = worst.max(assert_close(&format!("{name}.y.limits"), &yl, &nums(&py["limits"])));
    let yr = [b.y.continuous_range.0, b.y.continuous_range.1];
    worst =
        worst.max(assert_close(&format!("{name}.y.range"), &yr, &nums(&py["continuous_range"])));
    worst = worst.max(assert_close(&format!("{name}.y.breaks"), &b.y.breaks, &nums(&py["breaks"])));
    assert_eq!(b.y.labels, strs(&py["labels"]), "{name}.y.labels");
    worst = worst.max(assert_close(
        &format!("{name}.y.minor_breaks"),
        &b.y.minor_breaks,
        &nums(&py["minor_breaks"]),
    ));
    assert_eq!(
        b.y.n_breaks_censored as u64,
        py["n_breaks_censored"].as_u64().expect("count"),
        "{name}.y.n_breaks_censored"
    );
    worst
}

fn case(name: &str) -> Value {
    fixture()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("fixture has no case {name}"))
        .clone()
}

#[test]
fn ggplot2_row9_col_plain() {
    println!("col_plain max abs error {:e}", compare(&case("col_plain")));
}

#[test]
fn ggplot2_row9_col_stack_fill() {
    println!("col_stack_fill max abs error {:e}", compare(&case("col_stack_fill")));
}

#[test]
fn ggplot2_row9_col_identity_fill() {
    println!("col_identity_fill max abs error {:e}", compare(&case("col_identity_fill")));
}

#[test]
fn ggplot2_row9_col_width() {
    println!("col_width max abs error {:e}", compare(&case("col_width")));
}

#[test]
fn ggplot2_row9_fixture_cases_are_all_tested() {
    let names: Vec<String> = fixture()["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["name"].as_str().expect("name").to_string())
        .collect();
    assert_eq!(names, ["col_plain", "col_stack_fill", "col_identity_fill", "col_width"]);
}
