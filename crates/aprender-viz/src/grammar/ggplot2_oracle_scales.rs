//! ggplot2 conformance oracle, rows 2-4 (PLOT-GG-001, paiml/aprender#6909): `scale_x/y_discrete`,
//! `scale_x/y_log10`, and the continuous-scale arguments `limits`, `breaks`, `labels`, `expand`.
//!
//! Same method and tolerance as `ggplot2_oracle.rs`: each fixture under
//! `fixtures/ggplot2/<row>/expected.json` holds named cases that ggplot2 4.0.3 built from the
//! same `data.csv`; the test rebuilds each case with this crate's grammar and compares layer
//! positions, the panel range, breaks, labels, censored-break count and minor breaks.
//! Positions, limits, ranges and breaks: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12`. Labels exact.
//! On a log axis every number is in `log10` space, as `ggplot_build()` reports it.

use std::panic::AssertUnwindSafe;

use serde_json::Value;

use super::{
    scale_continuous, scale_discrete, scale_log10, Aes, DataFrame, Expansion, GGPlot, Geom,
    PositionScale,
};

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan(); // NA matches NA only
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b; // an infinity matches the same infinity only, never a finite number
    }
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
        .map(|e| match e {
            Value::Null => f64::NAN,
            // oracle.R writes +-Inf as these two strings (JSON has no infinity).
            Value::String(s) if s == "Inf" => f64::INFINITY,
            Value::String(s) if s == "-Inf" => f64::NEG_INFINITY,
            other => other.as_f64().expect("fixture array holds numbers"),
        })
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture array holds strings").to_string())
        .collect()
}

/// Parse a CSV with a header, optionally keeping only the given 1-based rows. A cell that parses as a number must be exact in f32 and becomes a
/// number; any other cell is text, and a column is text when any of its cells is.
fn frame_from_csv(csv: &str, only_rows: Option<&[usize]>) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::trim).collect::<Vec<_>>())
        .collect();
    // `rows` in the fixture are 1-based indices into the data (repeats allowed), as in R's `df[rows, ]`.
    let rows: Vec<Vec<&str>> = match only_rows {
        Some(idx) => idx.iter().map(|i| rows[*i - 1].clone()).collect(),
        None => rows,
    };
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[c]).collect();
        if cells.iter().all(|s| s.parse::<f64>().is_ok()) {
            let col: Vec<f32> = cells
                .iter()
                .map(|s| {
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

fn check_axis(name: &str, got: &PositionScale, want: &Value) {
    let kind = want["kind"].as_str().expect("axis kind");
    assert_eq!(got.levels.is_some(), kind == "discrete", "{name}.kind");
    assert_close_slice(
        &format!("{name}.continuous_range"),
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&want["continuous_range"]),
    );
    assert_eq!(got.labels, strs(&want["labels"]), "{name}.labels");
    if kind == "discrete" {
        let levels = got.levels.clone().expect("discrete axis has levels");
        assert_eq!(levels, strs(&want["limits"]), "{name}.limits (levels)");
        assert_eq!(
            got.breaks.iter().map(|b| levels[*b as usize - 1].clone()).collect::<Vec<_>>(),
            strs(&want["breaks"]),
            "{name}.breaks"
        );
        assert_close_slice(
            &format!("{name}.break_positions"),
            &got.breaks,
            &nums(&want["break_positions"]),
        );
    } else {
        assert_close_slice(
            &format!("{name}.limits"),
            &[got.limits.0, got.limits.1],
            &nums(&want["limits"]),
        );
        assert_close_slice(&format!("{name}.breaks"), &got.breaks, &nums(&want["breaks"]));
        let censored = want["n_breaks_censored"].as_u64().expect("n_breaks_censored");
        assert_eq!(got.n_breaks_censored as u64, censored, "{name}.n_breaks_censored");
        assert_close_slice(
            &format!("{name}.minor_breaks"),
            &got.minor_breaks,
            &nums(&want["minor_breaks"]),
        );
    }
}

fn run_fixture(json: &str, csv: &str, build: impl Fn(&str, DataFrame) -> GGPlot) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(
        want["versions"]["ggplot2"], "4.0.3",
        "fixture was not produced by the pinned ggplot2"
    );
    let cases = want["cases"].as_array().expect("fixture has cases");
    assert!(!cases.is_empty());
    // A case name the builder does not know must fail, never fall through to a default plot.
    let unknown = std::panic::catch_unwind(AssertUnwindSafe(|| build("", DataFrame::new())));
    assert!(unknown.is_err(), "the case builder accepted an unknown case name");
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let rows: Option<Vec<usize>> = case["rows"]
            .is_array()
            .then(|| nums(&case["rows"]).iter().map(|r| *r as usize).collect());
        let layout = build(name, frame_from_csv(csv, rows.as_deref()))
            .build()
            .expect("plot builds")
            .layout()
            .expect("layout resolves");
        assert_eq!(layout.layers.len(), 1, "{name}");
        assert_close_slice(
            &format!("{name}: layer x"),
            &layout.layers[0].x,
            &nums(&case["layer"]["x"]),
        );
        assert_close_slice(
            &format!("{name}: layer y"),
            &layout.layers[0].y,
            &nums(&case["layer"]["y"]),
        );
        check_axis(&format!("{name}: x"), &layout.x, &case["panel"]["x"]);
        check_axis(&format!("{name}: y"), &layout.y, &case["panel"]["y"]);
    }
}

fn point(df: DataFrame, x: &str, y: &str) -> GGPlot {
    GGPlot::new().data(df).aes(Aes::new().x(x).y(y)).geom(Geom::point())
}

// --- Row 2: scale_x_discrete / scale_y_discrete ------------------------------------------------

#[test]
fn ggplot2_row2_scale_discrete() {
    run_fixture(
        include_str!("../../fixtures/ggplot2/scale_discrete/expected.json"),
        include_str!("../../fixtures/ggplot2/scale_discrete/data.csv"),
        |name, df| match name {
            "x_default" | "x_single_level" => point(df, "g", "y"),
            "x_mixed_case_levels" => point(df, "m", "y"),
            "y_default" => point(df, "x", "g"),
            "x_limits_extra_level" => point(df, "g", "y")
                .scale_x(scale_discrete().limits(["mid", "low", "high", "extra"])),
            "x_breaks_labels" => point(df, "g", "y")
                .scale_x(scale_discrete().breaks(["high", "mid"]).labels(["H", "M"])),
            "x_expand" => point(df, "g", "y")
                .scale_x(scale_discrete().expand(Expansion::new((0.0, 0.1), (0.2, 1.0)))),
            "y_limits_labels" => point(df, "x", "g")
                .scale_y(scale_discrete().limits(["mid", "low", "high"]).labels(["M", "L", "H"])),
            other => panic!("unknown case {other}"),
        },
    );
}

// --- Row 3: scale_x_log10 / scale_y_log10 ------------------------------------------------------

#[test]
fn ggplot2_row3_scale_log10() {
    run_fixture(
        include_str!("../../fixtures/ggplot2/scale_log10/expected.json"),
        include_str!("../../fixtures/ggplot2/scale_log10/data.csv"),
        |name, df| match name {
            "wide_default" => point(df, "x", "y").scale_x(scale_log10()).scale_y(scale_log10()),
            "narrow_x_sci_y" => point(df, "u", "v").scale_x(scale_log10()).scale_y(scale_log10()),
            "nonpositive_x_na" => point(df, "w", "y").scale_x(scale_log10()).scale_y(scale_log10()),
            "single_point_log" | "all_equal_log" => {
                point(df, "x", "y").scale_x(scale_log10()).scale_y(scale_log10())
            }
            "limits_equal_log" => point(df, "x", "y").scale_x(scale_log10().limits(10.0, 10.0)),
            "limits_no_expand" => point(df, "x", "y")
                .scale_x(scale_log10().limits(1.0, 10000.0).expand(Expansion::none()))
                .scale_y(scale_log10()),
            "breaks_labels" => point(df, "x", "y")
                .scale_x(
                    scale_log10()
                        .breaks([1.0, 10.0, 100.0, 1000.0])
                        .labels(["one", "ten", "hundred", "thousand"]),
                )
                .scale_y(scale_log10()),
            // R: log10(0) is -Inf and censor(only.finite = TRUE) keeps it (review of #6916).
            "zero_x" => point(df, "z", "y").scale_x(scale_log10()),
            "zero_x_limits" | "zero_only_limits" => {
                point(df, "z", "y").scale_x(scale_log10().limits(1.0, 100.0))
            }
            "inf_mixed_x" => point(df, "i", "y").scale_x(scale_log10()),
            "inf_mixed_x_limits" => point(df, "i", "y").scale_x(scale_log10().limits(1.0, 100.0)),
            "tiny_zero_negative_x" | "tiny_single_x" => point(df, "t", "y").scale_x(scale_log10()),
            "single_one_log" => point(df, "x", "y").scale_x(scale_log10()),
            "all_negative_limits" => point(df, "w", "y").scale_x(scale_log10().limits(1.0, 100.0)),
            "break_at_zero" => {
                point(df, "x", "y").scale_x(scale_log10().breaks([0.0, 1.0, 10.0, 100.0]))
            }
            other => panic!("unknown case {other}"),
        },
    );
}

// --- Row 4: scale_x/y_continuous(limits, breaks, labels, expand) -------------------------------

#[test]
fn ggplot2_row4_scale_continuous_args() {
    run_fixture(
        include_str!("../../fixtures/ggplot2/scale_continuous_args/expected.json"),
        include_str!("../../fixtures/ggplot2/scale_continuous_args/data.csv"),
        |name, df| match name {
            "limits_no_expand_x" => point(df, "x", "y")
                .scale_x(scale_continuous().limits(0.0, 10.0).expand(Expansion::none())),
            "limits_no_expand_y" => point(df, "x", "y")
                .scale_y(scale_continuous().limits(-4.0, 8.0).expand(Expansion::none())),
            "breaks_labels_expansion" => point(df, "x", "y").scale_x(
                scale_continuous()
                    .breaks([2.0, 5.0, 8.0, 20.0])
                    .labels(["two", "five", "eight", "twenty"])
                    .expand(Expansion::new((0.02, 0.1), (1.0, 0.0))),
            ),
            "limits_breaks_y" => point(df, "x", "y")
                .scale_y(scale_continuous().limits(-3.0, 8.0).breaks([0.0, 4.0, 8.0])),
            "limits_reversed_x" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(10.0, 0.0))
            }
            "limits_oob_na_y" => point(df, "x", "y").scale_y(scale_continuous().limits(0.0, 4.0)),
            "limits_oob_na_x" => point(df, "x", "y").scale_x(scale_continuous().limits(2.0, 8.0)),
            "single_point" | "all_equal" => point(df, "x", "y"),
            "limits_equal_x" => point(df, "x", "y").scale_x(scale_continuous().limits(3.0, 3.0)),
            "single_point_add_expand" => point(df, "x", "y")
                .scale_x(scale_continuous().expand(Expansion::new((0.1, 0.1), (2.0, 2.0)))),
            "add_expansion_y" => {
                point(df, "x", "y").scale_y(scale_continuous().expand(Expansion::add(2.0)))
            }
            "single_zero" => point(df, "z", "y"),
            "single_zero_add_expand" => point(df, "z", "y")
                .scale_x(scale_continuous().expand(Expansion::new((0.1, 0.1), (2.0, 2.0)))),
            "limits_zero_zero" => point(df, "z", "y").scale_x(scale_continuous().limits(0.0, 0.0)),
            // +-Inf data: ignored by training, kept by the limits censor. R's plot has no scale
            // call, and ggplot2's default x scale IS scale_x_continuous(); the explicit default
            // spec is what keeps the rows here (see `no_spec_layout_drops_infinite_rows`).
            "inf_mixed_x" => point(df, "f", "y").scale_x(scale_continuous()),
            "inf_limits_x" | "inf_only_limits_x" => {
                point(df, "f", "y").scale_x(scale_continuous().limits(0.0, 3.0))
            }
            "inf_mixed_y_limits" => {
                point(df, "z", "f").scale_y(scale_continuous().limits(-1.0, 4.0))
            }
            other => panic!("unknown case {other}"),
        },
    );
}

// --- Rows 2-4 together: mixed scales on fresh data (second pass on #6916) ----------------------

/// 23 cases on one frame: discrete on either axis with and without a spec, continuous defaults on
/// wide, tiny and negative data, log10 on wide and sub-decade data (the `log_sub_breaks` path),
/// and a log axis beside a discrete or limited one, plus two ranges wider than `sqrt(f64::MAX)`.
#[test]
fn ggplot2_rows_2_to_4_mixed_scales() {
    run_fixture(
        include_str!("../../fixtures/ggplot2/scale_mixed/expected.json"),
        include_str!("../../fixtures/ggplot2/scale_mixed/data.csv"),
        |name, df| match name {
            "disc_x_nospec" => point(df, "g", "y"),
            "disc_y_nospec" => point(df, "x", "g"),
            "disc_x_default" => point(df, "g", "y").scale_x(scale_discrete()),
            "disc_limits_reversed" => {
                point(df, "g", "y").scale_x(scale_discrete().limits(["d", "c", "b", "a"]))
            }
            "disc_expand_add1" => {
                point(df, "g", "y").scale_x(scale_discrete().expand(Expansion::add(1.0)))
            }
            "disc_breaks_labels" => {
                point(df, "g", "y").scale_x(scale_discrete().breaks(["d", "a"]).labels(["D", "A"]))
            }
            "disc_y_limits_extra" => {
                point(df, "x", "g").scale_y(scale_discrete().limits(["a", "b", "c", "d", "e"]))
            }
            "disc_limits_subset" => {
                point(df, "g", "y").scale_x(scale_discrete().limits(["b", "a"]))
            }
            "cont_big_default" => point(df, "big", "y").scale_x(scale_continuous()),
            "cont_small_default" => point(df, "small", "y").scale_x(scale_continuous()),
            "cont_neg_default" => point(df, "n", "y").scale_x(scale_continuous()),
            "cont_breaks_outside" => {
                point(df, "x", "y").scale_x(scale_continuous().breaks([-100.0, 0.0, 5.0, 100.0]))
            }
            "cont_expand_asym" => point(df, "x", "y")
                .scale_y(scale_continuous().expand(Expansion::new((0.0, 0.2), (0.0, 0.0)))),
            "cont_y_labels" => point(df, "x", "y")
                .scale_y(scale_continuous().breaks([0.0, 2.5, 5.0]).labels(["lo", "mid", "hi"])),
            "cont_small_limits_none" => point(df, "x", "small")
                .scale_y(scale_continuous().limits(2e-05, 4e-05).expand(Expansion::none())),
            "log_wide" => point(df, "p", "y").scale_x(scale_log10()),
            "log_narrow" => point(df, "q", "y").scale_x(scale_log10()),
            "log_y" => point(df, "x", "p").scale_y(scale_log10()),
            "log_limits_breaks" => point(df, "p", "y")
                .scale_x(scale_log10().limits(1.0, 10000.0).breaks([1.0, 100.0, 10000.0])),
            "log_x_cont_y_limits" => point(df, "q", "x")
                .scale_x(scale_log10())
                .scale_y(scale_continuous().limits(0.0, 10.0)),
            "disc_x_log_y" => point(df, "g", "big").scale_y(scale_log10()),
            // Wider than sqrt(f64::MAX): labeling::extended returns seq(dmin, dmax, length.out = 5).
            "log_limits_huge" => point(df, "p", "y").scale_x(scale_log10().limits(1e300, 1e300)),
            "cont_limits_huge" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(-1e300, 1e300))
            }
            other => panic!("unknown case {other}"),
        },
    );
}

// --- Level order, NA rows and refusals (review of #6916) --------------------------------------

/// The oracle's level order is `LC_COLLATE=C`; the fixture says so, and the test refuses a
/// fixture generated under another locale (the order would differ and the comparison mean nothing).
#[test]
fn ggplot2_row2_oracle_is_pinned_to_c_collation() {
    let want: Value =
        serde_json::from_str(include_str!("../../fixtures/ggplot2/scale_discrete/expected.json"))
            .expect("fixture is valid JSON");
    assert_eq!(want["versions"]["collate"], "C");
    let case = want["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == "x_mixed_case_levels")
        .expect("mixed-case level case is in the fixture");
    // R's own C-locale order, not ours: upper case, then digits are before lower case.
    assert_eq!(
        strs(&case["panel"]["x"]["limits"]),
        ["10", "9", "A", "B", "Zeta", "a", "alpha", "b"]
    );
}

/// A row the scale cannot place stays in the layer as NA (ggplot2 drops it only when drawing),
/// so the layer has one entry per data row.
#[test]
fn ggplot2_row4_unplaceable_rows_stay_as_na() {
    let df =
        frame_from_csv(include_str!("../../fixtures/ggplot2/scale_continuous_args/data.csv"), None);
    let n = 8;
    let layout = point(df, "x", "y")
        .scale_x(scale_continuous().limits(2.0, 8.0))
        .build()
        .expect("plot builds")
        .layout()
        .expect("layout resolves");
    assert_eq!(layout.layers[0].x.len(), n);
    assert_eq!(layout.layers[0].y.len(), n);
    assert_eq!(layout.layers[0].x.iter().filter(|v| v.is_nan()).count(), 4);
}

/// ggplot2 errors on text under a continuous scale ("Discrete value supplied to a continuous
/// scale"); so does this.
#[test]
fn ggplot2_row4_text_in_continuous_scale_is_an_error() {
    let df = frame_from_csv(include_str!("../../fixtures/ggplot2/scale_discrete/data.csv"), None);
    let plot = point(df, "g", "y").scale_x(scale_continuous()).build().expect("plot builds");
    assert!(matches!(plot.layout(), Err(crate::error::Error::ScaleDomain(_))));
}

/// ggplot2 returns an infinite range (`Inf, -Inf`) for a log axis whose data are all non-positive,
/// which no panel can use; this crate refuses instead of returning it.
#[test]
fn ggplot2_row3_log_axis_without_positive_data_is_an_error() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[-1.0, -5.0, -2.0]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    let plot = point(df, "x", "y").scale_x(scale_log10()).build().expect("plot builds");
    assert!(matches!(plot.layout(), Err(crate::error::Error::ScaleDomain(_))));
}

/// Explicit limits entirely outside the data: every row is NA and the panel is the limits (ggplot2:
/// `limits = c(100, 200)` with data 1, 5, 20). This still trains a finite range first; the branch
/// with nothing at all to train on is `ggplot2_row3_all_negative_with_limits_needs_no_training`.
#[test]
fn ggplot2_row4_explicit_limits_need_no_finite_data() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[1.0, 5.0, 20.0]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    let layout = point(df, "x", "y")
        .scale_x(scale_continuous().limits(100.0, 200.0))
        .build()
        .expect("plot builds")
        .layout()
        .expect("layout resolves");
    assert!(layout.layers[0].x.iter().all(|v| v.is_nan()));
    assert_eq!(layout.x.limits, (100.0, 200.0));
    assert_close_slice(
        "continuous_range",
        &[layout.x.continuous_range.0, layout.x.continuous_range.1],
        &[95.0, 205.0],
    );
}

/// Nothing finite to train on (all negative on `log10`) is an error only without limits: with
/// `limits = c(1, 100)` ggplot2 gives layer `NA NA`, limits `0 2` (the `all_negative_limits`
/// fixture case), and the Err guard must not fire.
#[test]
fn ggplot2_row3_all_negative_with_limits_needs_no_training() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[-4.0, -500.0]);
    df.add_column_f32("y", &[1.0, 2.0]);
    let layout = point(df, "x", "y")
        .scale_x(scale_log10().limits(1.0, 100.0))
        .build()
        .expect("plot builds")
        .layout()
        .expect("limits make training unnecessary");
    assert!(layout.layers[0].x.iter().all(|v| v.is_nan()));
    assert_close_slice("limits", &[layout.x.limits.0, layout.x.limits.1], &[0.0, 2.0]);
}

/// R: `log10(0)` is `-Inf` (the row stays), `log10(-4)` is `NA`.
#[test]
fn ggplot2_row3_zero_is_neg_inf_negative_is_na() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[-4.0, 0.0, 1.0, 10.0]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0, 4.0]);
    let layout = point(df, "x", "y").scale_x(scale_log10()).build().expect("builds").layout();
    let x = layout.expect("layout resolves").layers[0].x.clone();
    assert!(x[0].is_nan());
    assert_eq!(x[1], f64::NEG_INFINITY);
    assert_close_slice("rest", &x[2..], &[0.0, 1.0]);
}

/// Explicit limits that are not finite after the transform are refused (a deliberate divergence):
/// ggplot2 errors for `limits = c(0, 100)` on log10 and silently substitutes the data range for a
/// negative or infinite limit.
#[test]
fn ggplot2_row3_row4_non_finite_explicit_limits_are_an_error() {
    let frame = || {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 10.0]);
        df.add_column_f32("y", &[1.0, 2.0]);
        df
    };
    for spec in [
        scale_log10().limits(0.0, 100.0),
        scale_log10().limits(-1.0, 100.0),
        scale_continuous().limits(f64::NEG_INFINITY, 3.0),
        scale_continuous().limits(0.0, f64::INFINITY),
    ] {
        let plot = point(frame(), "x", "y").scale_x(spec).build().expect("plot builds");
        assert!(matches!(plot.layout(), Err(crate::error::Error::ScaleDomain(_))));
    }
}

/// A log axis whose only data are zeros has nothing finite to train on: Err without limits
/// (ggplot2: an infinite range).
#[test]
fn ggplot2_row3_only_zero_without_limits_is_an_error() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[0.0, 0.0]);
    df.add_column_f32("y", &[1.0, 2.0]);
    let plot = point(df, "x", "y").scale_x(scale_log10()).build().expect("plot builds");
    assert!(matches!(plot.layout(), Err(crate::error::Error::ScaleDomain(_))));
}

/// `close` treats an infinity as equal only to the same infinity, so an `Inf` that should be a
/// number cannot slip through the comparison.
#[test]
fn oracle_comparison_separates_infinities() {
    assert!(close(f64::INFINITY, f64::INFINITY));
    assert!(close(f64::NEG_INFINITY, f64::NEG_INFINITY));
    assert!(!close(f64::INFINITY, f64::NEG_INFINITY));
    assert!(!close(f64::INFINITY, 1e300));
    assert!(!close(f64::NEG_INFINITY, f64::NAN));
}

// --- Renderer guard (review of #6916) ----------------------------------------------------------

fn render_both(plot: &super::BuiltGGPlot) -> (crate::error::Result<String>, bool) {
    // Neither render may panic whatever the data hold; report the SVG result and the framebuffer's.
    (plot.to_svg(), plot.to_framebuffer().is_ok())
}

/// The renderer draws from its own linear scales, so a plot with a position-scale spec is refused
/// (the same refusal for every spec) rather than drawn unscaled: a 0 on a log axis, a row outside
/// the limits and an all-negative log axis give an Err from `to_svg` and `to_framebuffer`, never a
/// panic and never NaN or Inf coordinates.
#[test]
fn renderer_refuses_scaled_plots_instead_of_drawing_them_linearly() {
    let frame = |x: &[f32]| {
        let mut df = DataFrame::new();
        df.add_column_f32("x", x);
        df.add_column_f32("y", &vec![1.0; x.len()]);
        df
    };
    let plots = [
        // 0 on a log axis (-Inf in scale space) and a row outside the explicit limits.
        point(frame(&[0.0, 1.0, 10.0, 500.0]), "x", "y")
            .scale_x(scale_log10().limits(1.0, 100.0))
            .build(),
        point(frame(&[0.0, 1.0, 10.0]), "x", "y").scale_x(scale_log10()).build(),
        point(frame(&[-4.0, -500.0]), "x", "y").scale_x(scale_log10()).build(),
        point(frame(&[1.0, 5.0, 20.0]), "x", "y")
            .scale_y(scale_continuous().limits(2.0, 3.0))
            .build(),
    ];
    for plot in plots {
        let plot = plot.expect("plot builds");
        let (svg, fb_ok) = render_both(&plot);
        assert!(matches!(svg, Err(crate::error::Error::Rendering(_))), "{svg:?}");
        assert!(!fb_ok);
    }
}

/// With no scale spec, infinite and missing values in the data must not panic or hang a render
/// and must not put `NaN`/`inf` into the SVG.
#[test]
fn renderer_survives_infinite_and_missing_data() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[f32::NEG_INFINITY, 1.0, f32::INFINITY, f32::NAN, 5.0]);
    df.add_column_f32("y", &[2.0, f32::INFINITY, 3.0, 4.0, f32::NEG_INFINITY]);
    let plot = point(df, "x", "y").build().expect("plot builds");
    let (svg, fb_ok) = render_both(&plot);
    let svg = svg.expect("renders");
    assert!(fb_ok);
    assert!(!svg.contains("NaN") && !svg.to_lowercase().contains("inf"), "non-finite in SVG");
}

// --- Scope of the scaled layout: what it refuses instead of answering wrongly ------------------

fn scope_frame() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[1.0, 10.0, 100.0]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    df.add_column_str("g", &["a", "b", "a"]);
    df
}

fn scoped(p: GGPlot) -> crate::error::Result<super::BuiltLayout> {
    p.build().expect("plot builds").layout()
}

fn log_x() -> GGPlot {
    point(scope_frame(), "x", "y").scale_x(scale_log10())
}

/// `layout()` lays out no coord transform under a position scale, so every non-default coord is
/// refused with its own message, for a spec on either axis and for a text position alone.
#[test]
fn scaled_layout_refuses_every_non_default_coord() {
    use super::Coord;
    let cases = [
        (Coord::cartesian().flip(), "coord_flip"),
        (Coord::cartesian().xlim(0.0, 1.0), "coord_cartesian limits"),
        (Coord::cartesian().ylim(0.0, 1.0), "coord_cartesian limits"),
        (Coord::polar(), "coord_polar"),
        (Coord::fixed(1.0), "coord_fixed"),
    ];
    for (coord, want) in cases {
        let plots = [
            log_x().coord(coord.clone()),
            point(scope_frame(), "x", "y").scale_y(scale_continuous()).coord(coord.clone()),
            point(scope_frame(), "g", "y").coord(coord.clone()),
        ];
        for p in plots {
            let got = scoped(p);
            assert!(
                matches!(&got, Err(crate::error::Error::UnsupportedCoord(m))
                    if m.contains(want) && m.contains("position scales")),
                "{want}: {got:?}"
            );
        }
    }
    assert!(scoped(log_x().coord(Coord::cartesian())).is_ok());
}

/// A facet would split the panels, which the scaled layout does not do.
#[test]
fn scaled_layout_refuses_every_facet() {
    use super::Facet;
    for facet in [Facet::wrap("g", 2), Facet::grid("g", "g"), Facet::row("g"), Facet::col("g")] {
        for p in [log_x().facet(facet.clone()), point(scope_frame(), "g", "y").facet(facet.clone())]
        {
            let got = scoped(p);
            assert!(
                matches!(&got, Err(crate::error::Error::UnsupportedLayer(m)) if m.contains("facet")),
                "{got:?}"
            );
        }
    }
    assert!(scoped(log_x().facet(Facet::none())).is_ok());
}

/// Every layer is checked, not only the first: a second, non-point layer is refused.
#[test]
fn scaled_layout_refuses_any_non_point_layer() {
    use super::Layer;
    for geom in [Geom::line(), Geom::bar(), Geom::text(), Geom::area(), Geom::histogram()] {
        let p = log_x().layer(Layer::new(geom.clone()));
        let got = scoped(p);
        assert!(
            matches!(&got, Err(crate::error::Error::UnsupportedLayer(m)) if m.contains("geom_point only")),
            "{:?}: {got:?}",
            geom.geom_type
        );
    }
    let two_points = log_x().layer(Layer::new(Geom::point()));
    assert_eq!(scoped(two_points).expect("two point layers resolve").layers.len(), 2);
}

/// `layout()` computes no stat; a point layer with any stat but identity is refused, on any layer.
#[test]
fn scaled_layout_refuses_every_non_identity_stat() {
    use super::{Layer, Stat};
    let stats = [
        Stat::count(),
        Stat::bin(5),
        Stat::density(),
        Stat::boxplot(),
        Stat::smooth(),
        Stat::bin2d(2, 2),
        Stat::sum(),
        Stat::mean(),
    ];
    for stat in stats {
        let p = log_x().layer(Layer::new(Geom::point().stat(stat.clone())));
        assert!(
            matches!(scoped(p), Err(crate::error::Error::UnsupportedStat(_))),
            "{stat:?} on the second layer"
        );
    }
    let identity = log_x().layer(Layer::new(Geom::point().stat(Stat::identity())));
    assert!(scoped(identity).is_ok());
}

/// A text position with no spec is ggplot2's default discrete scale: the plot takes the scaled
/// path (level positions 1..n), and both renderers refuse it rather than drawing it linearly.
#[test]
fn a_text_position_without_a_spec_is_discrete_and_not_drawn() {
    let layout = scoped(point(scope_frame(), "g", "y")).expect("discrete default resolves");
    assert_eq!(layout.layers[0].x, [1.0, 2.0, 1.0]);
    assert_eq!(layout.x.levels.as_deref(), Some(&["a".to_string(), "b".to_string()][..]));
    for p in [point(scope_frame(), "g", "y"), point(scope_frame(), "y", "g")] {
        let plot = p.build().expect("plot builds");
        let (svg, fb_ok) = render_both(&plot);
        assert!(matches!(svg, Err(crate::error::Error::Rendering(_))), "{svg:?}");
        assert!(!fb_ok);
    }
}

/// The divergence from ggplot2 this crate keeps: with NO scale spec and numeric positions,
/// `layout()` drops the rows with a non-finite position (render-total-v1), while ggplot2's
/// `ggplot_build()` keeps +-Inf rows. The explicit default spec keeps them, as R does.
#[test]
fn no_spec_layout_drops_infinite_rows() {
    let df =
        frame_from_csv(include_str!("../../fixtures/ggplot2/scale_continuous_args/data.csv"), None);
    let dropped = scoped(point(df.clone(), "f", "y")).expect("resolves");
    assert_eq!(dropped.layers[0].x.len(), 6);
    assert!(dropped.layers[0].x.iter().all(|v| v.is_finite()));
    let kept = scoped(point(df, "f", "y").scale_x(scale_continuous())).expect("resolves");
    assert_eq!(kept.layers[0].x.len(), 8);
    assert_eq!(kept.layers[0].x.iter().filter(|v| v.is_infinite()).count(), 2);
}

/// A discrete axis takes text only. ggplot2 places numbers on a continuous range beside the
/// levels, turns a missing value into an `NA` level placed last, and gives an axis with no
/// levels the panel `c(-Inf, Inf)` (all measured in ggplot2 4.0.3); none is reproduced, so each
/// is a typed refusal, never a position.
#[test]
fn a_discrete_axis_refuses_numbers_missing_values_and_no_levels() {
    use super::Layer;
    let refused = |p: GGPlot, want: &str| {
        let got = scoped(p);
        assert!(
            matches!(&got, Err(crate::error::Error::ScaleDomain(m)) if m.contains(want)),
            "{want}: {got:?}"
        );
    };
    // A second layer puts numbers on the text axis of the first.
    let mut numbers = DataFrame::new();
    numbers.add_column_f32("g", &[1.5]);
    numbers.add_column_f32("y", &[2.0]);
    refused(
        point(scope_frame(), "g", "y").layer(Layer::new(Geom::point()).data(numbers)),
        "holds the number 1.5",
    );
    // A text column shorter than the frame: its last row is missing (R: NA).
    let mut ragged = DataFrame::new();
    ragged.add_column_str("g", &["a", "b"]);
    ragged.add_column_f32("y", &[1.0, 2.0, 3.0]);
    refused(point(ragged, "g", "y"), "missing value");
    refused(point(scope_frame(), "x", "y").scale_x(scale_discrete()), "no levels");
    refused(
        point(scope_frame(), "g", "y").scale_x(scale_discrete().limits(Vec::<&str>::new())),
        "no levels",
    );
}

/// Every row of the frame is laid out: a cell missing from a short numeric column is `NaN`
/// (ggplot2 keeps the `NA` row in the layer data), and an absent position column is an error
/// (R: "object not found"), never an empty layer.
#[test]
fn every_row_is_laid_out_and_an_absent_column_is_an_error() {
    let mut ragged = DataFrame::new();
    ragged.add_column_f32("x", &[1.0, 2.0, 3.0]);
    ragged.add_column_f32("y", &[4.0, 5.0]);
    let layout = scoped(point(ragged, "x", "y").scale_x(scale_continuous())).expect("resolves");
    assert_eq!(layout.layers[0].x, [1.0, 2.0, 3.0]);
    assert_eq!(layout.layers[0].y[..2], [4.0, 5.0]);
    assert!(layout.layers[0].y[2].is_nan());
    let absent =
        scoped(point(scope_frame(), "nope", "y").scale_x(scale_continuous().limits(0.0, 1.0)));
    assert!(
        matches!(&absent, Err(crate::error::Error::UnknownColumn(m)) if m.contains("nope")),
        "{absent:?}"
    );
}

/// `expansion(mult = 0.1)` on `c(0, 10)` is `c(-1, 11)` (`ggplot2:::expand_range4`), and
/// `scales::zero_range(c(1, Inf))` is `FALSE`.
#[test]
fn expansion_mult_and_zero_range_match_r() {
    assert_eq!(Expansion::mult(0.1).apply((0.0, 10.0)), (-1.0, 11.0));
    assert!(!super::scale_spec::zero_range(1.0, f64::INFINITY));
    assert!(!super::scale_spec::zero_range(f64::NEG_INFINITY, f64::INFINITY));
}

// --- Review of #6916, pass 3: edge cases (fixture `scale_edge`) --------------------------------

/// Each case is a ggplot2 4.0.3 build over `scale_edge/data.csv`: a `NaN` limit is R's `NA` (that
/// end takes the data range), limits near `f64::MAX` keep finite breaks, duplicate discrete
/// breaks collapse, unsorted breaks get R's minor breaks, subnormal limits get R's labels, and
/// `limits = c(1, 1 + 1e-15)` is a zero-width range (`zero_range` tolerance).
#[test]
fn ggplot2_scale_edge_cases() {
    let m = f64::MAX;
    run_fixture(
        include_str!("../../fixtures/ggplot2/scale_edge/expected.json"),
        include_str!("../../fixtures/ggplot2/scale_edge/data.csv"),
        |name, df| match name {
            "limits_na_lo" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(f64::NAN, 10.0))
            }
            "limits_na_hi" => point(df, "x", "y").scale_x(scale_continuous().limits(0.0, f64::NAN)),
            "limits_na_both" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(f64::NAN, f64::NAN))
            }
            "limits_na_lo_log" => {
                point(df, "x", "y").scale_x(scale_log10().limits(f64::NAN, 100.0))
            }
            "limits_pm_max" => point(df, "x", "y").scale_x(scale_continuous().limits(-m, m)),
            "limits_zero_max" => point(df, "x", "y").scale_x(scale_continuous().limits(0.0, m)),
            "limits_pm_max_no_expand" => point(df, "x", "y")
                .scale_x(scale_continuous().limits(-m, m).expand(Expansion::none())),
            "disc_dup_limits" => {
                point(df, "g", "y").scale_x(scale_discrete().limits(["a", "a", "b"]))
            }
            "disc_dup_breaks" => point(df, "g", "y").scale_x(scale_discrete().breaks(["a", "a"])),
            "disc_dup_breaks_labels" => point(df, "g", "y")
                .scale_x(scale_discrete().breaks(["a", "a", "b"]).labels(["A", "B", "C"])),
            "breaks_unsorted" => {
                point(df, "x", "y").scale_x(scale_continuous().breaks([5.0, 1.0, 3.0]))
            }
            "limits_subnormal" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(0.0, 5e-324))
            }
            "limits_zero_range_tol" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(1.0, 1.0 + 1e-15))
            }
            other => panic!("unknown case {other}"),
        },
    );
}

/// Inputs R errors on (`breaks` and `labels` of different lengths), or answers with numbers this
/// crate declines to reproduce (negative or non-finite expansion, a `NA` limit whose substituted
/// pair is inverted, short labels on default discrete breaks): each is a typed
/// `ScaleDomain` error, never other numbers. `refusals.json` records what R did.
#[test]
fn ggplot2_scale_edge_refusals() {
    // The ones R itself errors on; for every other name R returned a range.
    const R_ERRORS: [&str; 4] = [
        "cont_labels_short",
        "cont_labels_long",
        "cont_labels_default_breaks",
        "disc_labels_short",
    ];
    let want: Value =
        serde_json::from_str(include_str!("../../fixtures/ggplot2/scale_edge/refusals.json"))
            .expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3");
    let refusals = want["refusals"].as_array().expect("refusals");
    assert_eq!(refusals.len(), 11);
    let csv = include_str!("../../fixtures/ggplot2/scale_edge/data.csv");
    for r in refusals {
        let name = r["name"].as_str().expect("name");
        let df = frame_from_csv(csv, None);
        let plot = match name {
            "cont_labels_short" => point(df, "x", "y")
                .scale_x(scale_continuous().breaks([1.0, 3.0, 5.0]).labels(["a", "b"])),
            "cont_labels_long" => point(df, "x", "y")
                .scale_x(scale_continuous().breaks([1.0, 3.0, 5.0]).labels(["a", "b", "c", "d"])),
            "cont_labels_default_breaks" => {
                point(df, "x", "y").scale_x(scale_continuous().labels(["a", "b"]))
            }
            "disc_labels_short" => point(df, "g", "y")
                .scale_x(scale_discrete().breaks(["a", "b", "c"]).labels(["A", "B"])),
            "disc_labels_default_breaks_short" => {
                point(df, "g", "y").scale_x(scale_discrete().labels(["A", "B"]))
            }
            "expand_neg_mult" => {
                point(df, "x", "y").scale_x(scale_continuous().expand(Expansion::mult(-0.6)))
            }
            "expand_neg_add" => {
                point(df, "x", "y").scale_x(scale_continuous().expand(Expansion::add(-1.0)))
            }
            "expand_nan" => point(df, "x", "y")
                .scale_x(scale_continuous().expand(Expansion::new((f64::NAN, 0.0), (0.0, 0.0)))),
            "expand_inf_add" => point(df, "x", "y")
                .scale_x(scale_continuous().expand(Expansion::add(f64::INFINITY))),
            "disc_expand_neg" => {
                point(df, "g", "y").scale_x(scale_discrete().expand(Expansion::add(-0.9)))
            }
            "limits_na_hi_below_data" => {
                point(df, "x", "y").scale_x(scale_continuous().limits(10.0, f64::NAN))
            }
            other => panic!("unknown refusal {other}"),
        };
        let got = scoped(plot);
        assert!(matches!(&got, Err(crate::error::Error::ScaleDomain(_))), "{name}: {got:?}");
        assert_eq!(r["r_error"].is_string(), R_ERRORS.contains(&name), "{name}: R outcome");
    }
}

/// A `NA` limit takes the data range, so an axis with no finite data cannot resolve it; a labels
/// vector that matches an explicit `limits` of a discrete scale resolves, one that does not is
/// refused.
#[test]
fn na_limit_needs_data_and_discrete_labels_match_limits() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[f32::INFINITY]);
    df.add_column_f32("y", &[1.0]);
    let got = scoped(point(df, "x", "y").scale_x(scale_continuous().limits(f64::NAN, 5.0)));
    assert!(matches!(&got, Err(crate::error::Error::ScaleDomain(m)) if m.contains("train")));
    let ok = scoped(
        point(scope_frame(), "g", "y")
            .scale_x(scale_discrete().limits(["a", "b", "c"]).labels(["A", "B", "C"])),
    )
    .expect("labels match the limits");
    assert_eq!(ok.x.labels, ["A", "B", "C"]);
    let bad = scoped(
        point(scope_frame(), "g", "y")
            .scale_x(scale_discrete().limits(["a", "b", "c"]).labels(["A", "B"])),
    );
    assert!(matches!(bad, Err(crate::error::Error::ScaleDomain(_))));
}

/// A text position in a LATER layer makes the plot discrete: the first layer's numbers then reach
/// a text axis and are refused. Detection must be `any` layer, not `all` of them.
#[test]
fn a_text_position_in_any_layer_selects_the_scaled_path() {
    use super::Layer;
    let mut text = DataFrame::new();
    text.add_column_str("x", &["a", "b"]);
    text.add_column_f32("y", &[1.0, 2.0]);
    let mut numbers = DataFrame::new();
    numbers.add_column_f32("x", &[1.0, 2.0]);
    numbers.add_column_f32("y", &[1.0, 2.0]);
    let got = scoped(point(numbers, "x", "y").layer(Layer::new(Geom::point()).data(text)));
    assert!(
        matches!(&got, Err(crate::error::Error::ScaleDomain(m)) if m.contains("holds the number")),
        "{got:?}"
    );
}
