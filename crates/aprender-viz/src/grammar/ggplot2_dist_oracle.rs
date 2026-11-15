//! ggplot2 4.0.3 oracle for the distribution rows (PLOT-GG-001, paiml/aprender#6909), the
//! `geom_histogram` row first: `stat_bin` through [`GGPlot::layout`]. Fixtures are committed
//! under `fixtures/ggplot2/geom_histogram/`; R is never run here. Tolerance and helpers are those
//! of `ggplot2_oracle.rs` (1e-9 relative, counts and labels exact).

use serde_json::Value;

use super::ggplot2_oracle::{assert_close_slice, check_axis, frame_from_csv, nums};
use super::{Aes, BuiltLayout, GGPlot, Geom};

const CSV: &str = include_str!("../../fixtures/ggplot2/geom_histogram/data.csv");
const CSV_CONSTANT: &str = include_str!("../../fixtures/ggplot2/geom_histogram/data_constant.csv");

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
    assert_eq!(w["geom"], "GeomBar", "{case}");
    let b = g.bins.as_ref().expect("a histogram layer carries its bins");
    assert_close_slice(&format!("{case}.count"), &b.count, &nums(&w["count"]));
    assert_close_slice(&format!("{case}.x"), &g.x, &nums(&w["x"]));
    assert_close_slice(&format!("{case}.y"), &g.y, &nums(&w["y"]));
    assert_close_slice(&format!("{case}.xmin"), &b.xmin, &nums(&w["xmin"]));
    assert_close_slice(&format!("{case}.xmax"), &b.xmax, &nums(&w["xmax"]));
    // The layer's own `width` column is GeomBar's 0.9 default; the bin width is `xmax - xmin`.
    let span: Vec<f64> =
        nums(&w["xmax"]).iter().zip(nums(&w["xmin"])).map(|(a, b)| a - b).collect();
    assert_close_slice(&format!("{case}.width"), &b.width, &span);
    assert_close_slice(&format!("{case}.density"), &b.density, &nums(&w["density"]));
    check_axis(&format!("{case}.x"), &got.x, &want["panel"]["x"]);
    check_axis(&format!("{case}.y"), &got.y, &want["panel"]["y"]);
}

#[test]
fn default() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_default.json");
    check("default", json, CSV, Geom::gg_histogram());
}

#[test]
fn bins10() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_bins10.json");
    check("bins10", json, CSV, Geom::gg_histogram().bins(10));
}

#[test]
fn bins1() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_bins1.json");
    check("bins1", json, CSV, Geom::gg_histogram().bins(1));
}

#[test]
fn binwidth2() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_binwidth2.json");
    check("binwidth2", json, CSV, Geom::gg_histogram().binwidth(2.0));
}

#[test]
fn binwidth1_boundary0() {
    let json =
        include_str!("../../fixtures/ggplot2/geom_histogram/expected_binwidth1_boundary0.json");
    check("binwidth1_boundary0", json, CSV, Geom::gg_histogram().binwidth(1.0).boundary(0.0));
}

#[test]
fn binwidth5_center0() {
    let json =
        include_str!("../../fixtures/ggplot2/geom_histogram/expected_binwidth5_center0.json");
    check("binwidth5_center0", json, CSV, Geom::gg_histogram().binwidth(5.0).center(0.0));
}

#[test]
fn bins5_boundary2() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_bins5_boundary2.json");
    check("bins5_boundary2", json, CSV, Geom::gg_histogram().bins(5).boundary(2.0));
}

#[test]
fn constant() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_constant.json");
    check("constant", json, CSV_CONSTANT, Geom::gg_histogram());
}
