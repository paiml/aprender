//! ggplot2 4.0.3 oracle for the `geom_freqpoly` row (PLOT-GG-001, paiml/aprender#6963):
//! `stat_bin` with `pad = TRUE` through [`GGPlot::layout`]. Fixtures are committed
//! under `fixtures/ggplot2/geom_freqpoly/`; R is never run here. Tolerance and helpers are those
//! of `ggplot2_oracle.rs` (1e-9 relative, counts and labels exact).

use serde_json::Value;

use super::ggplot2_oracle::{assert_close_slice, check_axis, frame_from_csv, nums};
use super::{Aes, BuiltLayout, GGPlot, Geom};

const CSV: &str = include_str!("../../fixtures/ggplot2/geom_freqpoly/data.csv");
const CSV_CONSTANT: &str = include_str!("../../fixtures/ggplot2/geom_freqpoly/data_constant.csv");

fn run(csv: &str, geom: Geom) -> BuiltLayout {
    GGPlot::new()
        .data(frame_from_csv(csv))
        .aes(Aes::new().x("x"))
        .layer(super::Layer::new(geom))
        .build()
        .expect("builds")
        .layout()
        .expect("layout")
}

fn check(case: &str, json: &str, csv: &str, geom: Geom) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "{case}: not the pinned ggplot2");
    assert_eq!(want["versions"]["collate"], "C", "{case}: fixture not made under C collation");
    let got = run(csv, geom);
    let (g, w) = (&got.layers[0], &want["layers"][0]);
    assert_eq!(w["geom"], "GeomPath", "{case}");
    let b = g.bins.as_ref().expect("a frequency polygon layer carries its bins");
    assert_close_slice(&format!("{case}.count"), &b.count, &nums(&w["count"]));
    assert_close_slice(&format!("{case}.x"), &g.x, &nums(&w["x"]));
    assert_close_slice(&format!("{case}.y"), &g.y, &nums(&w["y"]));
    assert_close_slice(&format!("{case}.xmin"), &b.xmin, &nums(&w["xmin"]));
    assert_close_slice(&format!("{case}.xmax"), &b.xmax, &nums(&w["xmax"]));
    // `width` is the stat's bin width, which is `xmax - xmin`.
    let span: Vec<f64> =
        nums(&w["xmax"]).iter().zip(nums(&w["xmin"])).map(|(a, b)| a - b).collect();
    assert_close_slice(&format!("{case}.width"), &b.width, &span);
    assert_close_slice(&format!("{case}.density"), &b.density, &nums(&w["density"]));
    check_axis(&format!("{case}.x"), &got.x, &want["panel"]["x"]);
    check_axis(&format!("{case}.y"), &got.y, &want["panel"]["y"]);
}

#[test]
fn default() {
    let json = include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_default.json");
    check("default", json, CSV, Geom::gg_freqpoly());
}

#[test]
fn bins10() {
    let json = include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_bins10.json");
    check("bins10", json, CSV, Geom::gg_freqpoly().bins(10));
}

#[test]
fn binwidth2() {
    let json = include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_binwidth2.json");
    check("binwidth2", json, CSV, Geom::gg_freqpoly().binwidth(2.0));
}

#[test]
fn binwidth1_boundary0() {
    let json =
        include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_binwidth1_boundary0.json");
    check("binwidth1_boundary0", json, CSV, Geom::gg_freqpoly().binwidth(1.0).boundary(0.0));
}

#[test]
fn binwidth5_center0() {
    let json = include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_binwidth5_center0.json");
    check("binwidth5_center0", json, CSV, Geom::gg_freqpoly().binwidth(5.0).center(0.0));
}

#[test]
fn bins5_boundary2() {
    let json = include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_bins5_boundary2.json");
    check("bins5_boundary2", json, CSV, Geom::gg_freqpoly().bins(5).boundary(2.0));
}

#[test]
fn constant() {
    let json = include_str!("../../fixtures/ggplot2/geom_freqpoly/expected_constant.json");
    check("constant", json, CSV_CONSTANT, Geom::gg_freqpoly());
}

/// Missing `x` is dropped before the stat; no `x` left, or groups, is a typed refusal.
#[test]
fn refusals_and_missing_values() {
    use super::data::DataFrame;
    use crate::error::Error;
    let freq = |d: DataFrame, aes: Aes| {
        GGPlot::new()
            .data(d)
            .aes(aes)
            .layer(super::Layer::new(Geom::gg_freqpoly()))
            .build()
            .and_then(|p| p.layout())
    };
    let x = Aes::new().x("x");
    let clean = freq(DataFrame::from_xy(&[1.0, 2.0, 4.0, 8.0], &[0.0; 4]), x.clone()).expect("ok");
    let gappy = freq(DataFrame::from_xy(&[1.0, f32::NAN, 2.0, 4.0, 8.0], &[0.0; 5]), x.clone());
    assert_eq!(gappy.expect("gappy").layers[0].y, clean.layers[0].y);
    let y = &clean.layers[0].y;
    assert_eq!((y.first(), y.last()), (Some(&0.0), Some(&0.0)), "padded with zero counts");
    let none = freq(DataFrame::from_xy(&[f32::NAN, f32::NAN], &[0.0; 2]), x);
    assert!(matches!(none, Err(Error::Rendering(_))));
    let mut grouped = DataFrame::from_xy(&[1.0, 2.0, 3.0], &[0.0; 3]);
    grouped.add_column_str("h", &["a", "b", "a"]);
    let r = freq(grouped, Aes::new().x("x").color("h"));
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("grouped")));
}
