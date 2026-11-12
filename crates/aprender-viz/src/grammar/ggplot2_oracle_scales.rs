//! ggplot2 conformance oracle, rows 2-4 (PLOT-GG-001, paiml/aprender#6909): `scale_x/y_discrete`,
//! `scale_x/y_log10`, and the continuous-scale arguments `limits`, `breaks`, `labels`, `expand`.
//!
//! Same method and tolerance as `ggplot2_oracle.rs`: each fixture under
//! `fixtures/ggplot2/<row>/expected.json` holds named cases that ggplot2 4.0.3 built from the
//! same `data.csv`; the test rebuilds each case with this crate's grammar and compares layer
//! positions, the panel range, breaks, labels, censored-break count and minor breaks.
//! Positions, limits, ranges and breaks: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12`. Labels exact.
//! On a log axis every number is in `log10` space, as `ggplot_build()` reports it.

use serde_json::Value;

use super::{
    scale_continuous, scale_discrete, scale_log10, Aes, DataFrame, Expansion, GGPlot, Geom,
    PositionScale,
};

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

/// Parse a CSV with a header. A cell that parses as a number must be exact in f32 and becomes a
/// number; any other cell is text, and a column is text when any of its cells is.
fn frame_from_csv(csv: &str, keep: impl Fn(&[&str]) -> bool) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::trim).collect::<Vec<_>>())
        .filter(|r| keep(r))
        .collect();
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
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let keep = |r: &[&str]| name != "x_single_level" || r[0] == "low";
        let layout = build(name, frame_from_csv(csv, keep))
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
            "add_expansion_y" => {
                point(df, "x", "y").scale_y(scale_continuous().expand(Expansion::add(2.0)))
            }
            other => panic!("unknown case {other}"),
        },
    );
}
