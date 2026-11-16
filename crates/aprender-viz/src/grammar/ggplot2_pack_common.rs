//! Helpers shared by the segment, polygon and text oracles (PLOT-GG-001, paiml/aprender#6909).
//! Tolerance is `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12`; `NaN` matches `null` by position.

use serde_json::Value;

use super::{Aes, BuiltLayout, DataFrame, Facet, GGPlot, Layer, PositionScale};

pub(super) const REL_TOL: f64 = 1e-9;
pub(super) const ABS_TOL: f64 = 1e-12;

pub(super) fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

pub(super) fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
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

/// Parse a fixture and check it is the pinned, C-collation oracle output for `row`/`case`.
pub(super) fn want(json: &str, row: &str, case: &str) -> Value {
    let v: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["collation"], "C", "fixture was not made under C collation");
    assert_eq!(v["row"], row);
    assert_eq!(v["case"], case);
    v
}

pub(super) fn xy() -> Aes {
    Aes::new().x("x").y("y")
}

pub(super) fn wrap_f() -> Facet {
    Facet::Wrap { var: "f".into(), ncol: 2 }
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

/// The columns every layer of these rows carries: panel, group, x, y, alpha, colour.
pub(super) fn check_common(
    tag: &str,
    panel: &[usize],
    group: &[i64],
    x: &[f64],
    y: &[f64],
    alpha: Option<f64>,
    colour: [u8; 4],
    w: &Value,
) {
    let panel: Vec<i64> = panel.iter().map(|v| *v as i64 + 1).collect();
    assert_eq!(panel, ints(&w["panel"]), "{tag}.panel");
    assert_eq!(group, ints(&w["group"]), "{tag}.group");
    let n = panel.len();
    assert_close_slice(&format!("{tag}.x"), x, &nums(&w["x"]));
    assert_close_slice(&format!("{tag}.y"), y, &nums(&w["y"]));
    let want_alpha = nums(&w["alpha"]);
    assert_eq!(want_alpha.len(), n, "{tag}: fixture row count");
    let want_colour = w["colour_rgba"].as_array().expect("colour_rgba");
    assert_eq!(want_colour.len(), n, "{tag}: colour row count");
    for i in 0..n {
        assert_eq!(colour.map(f64::from).to_vec(), nums(&want_colour[i]), "{tag}.colour[{i}]");
        match alpha {
            None => assert!(want_alpha[i].is_nan(), "{tag}.alpha[{i}]: R has {}", want_alpha[i]),
            Some(v) => assert!(close(v, want_alpha[i]), "{tag}.alpha[{i}]: got {v}"),
        }
    }
}
