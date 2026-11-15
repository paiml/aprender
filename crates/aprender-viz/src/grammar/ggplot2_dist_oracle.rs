//! ggplot2 4.0.3 oracle for the distribution rows (PLOT-GG-001, paiml/aprender#6909), the
//! `geom_histogram` row first: `stat_bin` through [`GGPlot::layout`]. Fixtures are committed
//! under `fixtures/ggplot2/geom_histogram/`; R is never run here. Tolerance and helpers are those
//! of `ggplot2_oracle.rs` (1e-9 relative, counts and labels exact).

use serde_json::Value;

use super::ggplot2_oracle::{assert_close_slice, check_axis, frame_from_csv, nums};
use super::{Aes, BuiltLayout, DataFrame, GGPlot, Geom};

const CSV: &str = include_str!("../../fixtures/ggplot2/geom_histogram/data.csv");
const CSV_CONSTANT: &str = include_str!("../../fixtures/ggplot2/geom_histogram/data_constant.csv");

const CSV_GROUPS: &str = include_str!("../../fixtures/ggplot2/geom_histogram/data_groups.csv");

/// `data_groups.csv`: `x` is numeric, every other column a discrete label.
fn frame_groups(csv: &str) -> DataFrame {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[c].trim()).collect();
        if *name == "x" {
            let xs: Vec<f32> = cells.iter().map(|v| v.parse().expect("numeric x")).collect();
            df.add_column_f32("x", &xs);
        } else {
            df.add_column_str(name, &cells);
        }
    }
    df
}

fn run(csv: &str, geom: Geom) -> BuiltLayout {
    layout_of(frame_from_csv(csv), Aes::new().x("x"), geom)
}

fn layout_of(data: DataFrame, aes: Aes, geom: Geom) -> BuiltLayout {
    GGPlot::new()
        .data(data)
        .aes(aes)
        .layer(super::Layer::new(geom))
        .build()
        .expect("builds")
        .layout()
        .expect("layout")
}

fn check(case: &str, json: &str, csv: &str, geom: Geom) {
    check_layout(case, json, run(csv, geom));
}

/// A grouped case: `data_groups.csv` with the discrete column `group` mapped to `fill`.
fn check_grouped(case: &str, json: &str, group: &str, geom: Geom) {
    let got = layout_of(frame_groups(CSV_GROUPS), Aes::new().x("x").fill(group), geom);
    check_layout(case, json, got);
}

fn check_layout(case: &str, json: &str, got: BuiltLayout) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "{case}: not the pinned ggplot2");
    assert_eq!(want["versions"]["collate"], "C", "{case}: fixture not made under C collation");
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
    // position_stack(): the stacked bar's edges (0 and `count` for a single group).
    assert_close_slice(&format!("{case}.ymin"), &b.ymin, &nums(&w["ymin"]));
    assert_close_slice(&format!("{case}.ymax"), &b.ymax, &nums(&w["ymax"]));
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

#[test]
fn groups2() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_groups2.json");
    check_grouped("groups2", json, "g2", Geom::gg_histogram().bins(8));
}

#[test]
fn groups3_empty() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_groups3_empty.json");
    check_grouped("groups3_empty", json, "g3", Geom::gg_histogram().binwidth(2.0).boundary(0.0));
}

#[test]
fn groups3_default() {
    let json = include_str!("../../fixtures/ggplot2/geom_histogram/expected_groups3_default.json");
    check_grouped("groups3_default", json, "g3", Geom::gg_histogram());
}
