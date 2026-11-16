//! ggplot2 conformance oracle for `position_dodge()` on `geom_col()` / `geom_bar()`
//! (PLOT-GG-001, paiml/aprender#6930, master #6909).
//!
//! Fixture: `fixtures/ggplot2/position_dodge/expected.json`, produced by ggplot2 4.0.3 under C
//! collation (`generate.R`); R is never run here. Each case rebuilds the same plot with
//! [`bar_layout`] and compares every bar row and both trained scales.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for positions,
//! limits and breaks; labels, groups and panels compare exactly; `NaN` matches `null`.
//!
//! Sibling of `ggplot2_bars_oracle` (row 9) and `ggplot2_bars_stack_oracle`.

use serde_json::Value;

use super::{bar_layout_in, BarSpec, BuiltBars, DataFrame, Position, Preserve};

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

const JSON: &str = include_str!("../../fixtures/ggplot2/position_dodge/expected.json");
const DATA: &str = include_str!("../../fixtures/ggplot2/position_dodge/data.csv");
const DATA_EQUAL: &str = include_str!("../../fixtures/ggplot2/position_dodge/data_equal.csv");
const DATA_NEG: &str = include_str!("../../fixtures/ggplot2/position_dodge/data_neg.csv");
const DATA_COUNT: &str = include_str!("../../fixtures/ggplot2/position_dodge/data_count.csv");
const DATA_ONE: &str = include_str!("../../fixtures/ggplot2/position_dodge/data_one.csv");
const DATA_FACET: &str = include_str!("../../fixtures/ggplot2/position_dodge/data_facet.csv");

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
        "data_equal.csv" => DATA_EQUAL,
        "data_neg.csv" => DATA_NEG,
        "data_count.csv" => DATA_COUNT,
        "data_one.csv" => DATA_ONE,
        "data_facet.csv" => DATA_FACET,
        other => panic!("unknown fixture data file {other}"),
    }
}

fn dodge(width: Option<f64>, preserve: Preserve, reverse: bool) -> Position {
    Position::Dodge { width, preserve, reverse }
}

fn col(position: Position) -> BarSpec {
    BarSpec::col("x", "y").fill("g").position(position)
}

/// The spec each fixture case was made from (its `spec` string is in the fixture).
fn spec_for(case: &str) -> BarSpec {
    let total = Preserve::Total;
    let single = Preserve::Single;
    match case {
        "col_dodge_total_unequal" | "col_dodge_equal" | "col_dodge_negative" => {
            col(Position::dodge())
        }
        "col_dodge_width_half" => col(dodge(Some(0.5), total, false)),
        "col_dodge_width_one" => col(dodge(Some(1.0), total, false)),
        "col_dodge_single" | "col_dodge_facet_single" => col(dodge(None, single, false)),
        "col_dodge_reverse" => col(dodge(None, total, true)),
        "col_dodge_single_reverse" => col(dodge(None, single, true)),
        "col_dodge_narrow_bars" => col(Position::dodge()).width(0.5),
        "col_dodge_no_fill" => BarSpec::col("x", "y").position(Position::dodge()),
        "bar_dodge" => BarSpec::bar("x").fill("g").position(Position::dodge()),
        "bar_dodge_single" => BarSpec::bar("x").fill("g").position(dodge(None, single, false)),
        "col_dodge_facet_total" => col(Position::dodge()),
        other => panic!("no spec for fixture case {other}"),
    }
}

/// Row indices of each panel: facet cases split on the `f` column, in level order.
fn panels_for(case: &str, df: &DataFrame, n: usize) -> Vec<Vec<usize>> {
    if !case.contains("facet") {
        return vec![(0..n).collect()];
    }
    let f = df.get("f").expect("facet column");
    ["u", "v"].iter().map(|l| (0..n).filter(|i| f[*i].as_str() == Some(*l)).collect()).collect()
}

fn check_case(case: &Value) -> BuiltBars {
    let name = case["name"].as_str().expect("case name");
    MAX_ABS_ERR.with(|m| m.set(0.0));
    let df = frame(csv_for(case["data_file"].as_str().expect("data_file")));
    let mut levels: Vec<String> = df
        .get("x")
        .expect("x column")
        .iter()
        .map(|v| v.as_str().expect("text x").to_string())
        .collect();
    levels.sort();
    levels.dedup();
    let panels = panels_for(name, &df, df.nrow());
    let got = bar_layout_in(&df, &spec_for(name), &levels, &panels)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
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
    eprintln!("MAXERR position_dodge {name} {:e}", MAX_ABS_ERR.with(std::cell::Cell::get));
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
        .map(|c| c["name"].as_str().expect("name"))
        .collect();
    assert_eq!(names.len(), 14, "a case was added without a test: {names:?}");
    for n in names {
        let _ = spec_for(n);
    }
}

#[test]
fn position_dodge_total_splits_the_slot_between_groups_present() {
    let got = run("col_dodge_total_unequal");
    // x = p has three groups: the first bar's centre is 1 + 0.9 * ((1 - 0.5) / 3 - 0.5) = 0.775.
    assert!((got.rows[0].x - 0.775).abs() < 1e-12);
    // x = s has one group: left untouched.
    assert!((got.rows[7].x - 4.0).abs() < 1e-12);
    run("col_dodge_equal");
}

#[test]
fn position_dodge_width_param() {
    run("col_dodge_width_half");
    run("col_dodge_width_one");
    run("col_dodge_narrow_bars");
}

#[test]
fn position_dodge_preserve_single() {
    run("col_dodge_single");
    run("bar_dodge_single");
}

#[test]
fn position_dodge_reverse() {
    run("col_dodge_reverse");
    run("col_dodge_single_reverse");
}

#[test]
fn position_dodge_negative_and_no_fill() {
    run("col_dodge_negative");
    // One group per x: n = 1 everywhere, rows are unchanged.
    let got = run("col_dodge_no_fill");
    assert!(got.rows.iter().all(|r| (r.xmax - r.xmin - 0.9).abs() < 1e-12));
}

#[test]
fn position_dodge_bar_count() {
    run("bar_dodge");
}

#[test]
fn position_dodge_facets_are_dodged_per_panel() {
    run("col_dodge_facet_total");
    run("col_dodge_facet_single");
}

/// The comparison can see a wrong adjustment: identity instead of dodge must fail it.
#[test]
fn identity_is_not_dodge() {
    let v = fixture();
    let case = &v["cases"][0];
    let df = frame(csv_for("data.csv"));
    let mut levels = vec!["p".to_string(), "q".into(), "r".into(), "s".into()];
    levels.sort();
    let spec = col(Position::Identity);
    let got = bar_layout_in(&df, &spec, &levels, &[(0..df.nrow()).collect()]).expect("builds");
    let want = nums(&case["layer"]["x"]);
    let x: Vec<f64> = got.rows.iter().map(|r| r.x).collect();
    assert!(
        x.iter().zip(&want).any(|(g, w)| !close(*g, *w)),
        "identity output equals the dodge oracle: the oracle cannot tell them apart"
    );
}
