//! ggplot2 conformance oracle for `geom_polygon` (PLOT-GG-001, paiml/aprender#6909). Fixtures are
//! `fixtures/ggplot2/geom_polygon/expected_<case>.json`, produced by ggplot2 4.0.3 under C
//! collation (`generate.R`); R is never run here. Tolerance and helpers: `ggplot2_pack_common`.

use serde_json::Value;

use super::ggplot2_pack_common::{
    build, check_axis, check_common, close, frame_from_csv, nums, want, wrap_f, xy,
};
use super::{Aes, BuiltLayout, DataFrame, Facet, Geom, Layer};
use crate::color::Rgba;
use crate::error::Error;

const ROW: &str = "geom_polygon";

macro_rules! fixture {
    ($case:literal) => {
        include_str!(concat!("../../fixtures/ggplot2/geom_polygon/expected_", $case, ".json"))
    };
}

macro_rules! csv {
    ($file:literal) => {
        frame_from_csv(include_str!(concat!("../../fixtures/ggplot2/geom_polygon/", $file, ".csv")))
    };
}

fn check(got: &BuiltLayout, w: &Value) {
    let layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), 1);
    let (l, wl) = (&got.layers[0], &layers[0]);
    assert_eq!(wl["geom"], "GeomPolygon", "R geom class");
    let p = l.path.as_ref().expect("a polygon layer resolves path data");
    let c = p.aes.colour;
    check_common("polygon", &p.panel, &p.group, &l.x, &l.y, p.aes.alpha, [c.r, c.g, c.b, c.a], wl);
    let fill = p.fill.expect("a polygon has a fill");
    let want_fill = wl["fill_rgba"].as_array().expect("fill_rgba");
    for (i, f) in want_fill.iter().enumerate() {
        assert_eq!([fill.r, fill.g, fill.b, fill.a].map(f64::from).to_vec(), nums(f), "fill[{i}]");
    }
    for i in 0..p.panel.len() {
        assert!(close(p.aes.linewidth, nums(&wl["linewidth"])[i]), "linewidth[{i}]");
        assert_eq!(f64::from(p.aes.linetype), nums(&wl["linetype"])[i], "linetype[{i}]");
    }
    assert!(p.xend.is_none() && p.yend.is_none(), "a polygon has no end columns");
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet) {
    run_geom(case, json, data, aes, facet, Geom::polygon());
}

fn run_geom(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet, g: Geom) {
    let w = want(json, ROW, case);
    let got = build(data, aes, vec![Layer::new(g)], facet).expect("layout resolves");
    check(&got, &w);
}

#[test]
fn one_group_keeps_row_order() {
    run("one_group", fixture!("one_group"), csv!("data"), xy(), Facet::None);
}

#[test]
fn numeric_group_ids_rank_sorted_values() {
    run("group_num", fixture!("group_num"), csv!("data"), xy().group("gn"), Facet::None);
}

#[test]
fn text_group_ids_rank_sorted_levels() {
    run("group_text", fixture!("group_text"), csv!("data"), xy().group("g"), Facet::None);
}

#[test]
fn a_polygon_is_not_closed_at_build_time() {
    run("single", fixture!("single"), csv!("data_single"), xy(), Facet::None);
}

#[test]
fn degenerate_polygons_are_kept() {
    run(
        "degenerate",
        fixture!("degenerate"),
        csv!("data_degenerate"),
        xy().group("gn"),
        Facet::None,
    );
}

#[test]
fn negative_positions() {
    run("negative", fixture!("negative"), csv!("data_negative"), xy().group("gn"), Facet::None);
}

#[test]
fn facet_wrap_panels() {
    run("facet", fixture!("facet"), csv!("data"), xy().group("gn"), wrap_f());
}

#[test]
fn fixed_colour_and_alpha() {
    let g = Geom::polygon().aes(Aes::new().color_value(Rgba::RED).alpha_value(0.5));
    run_geom(
        "colour_alpha",
        fixture!("colour_alpha"),
        csv!("data"),
        xy().group("gn"),
        Facet::None,
        g,
    );
}

#[test]
fn na_vertices_are_kept_and_not_trained() {
    run("na", fixture!("na"), csv!("data_na"), xy().group("gn"), Facet::None);
}

#[test]
fn a_missing_column_is_an_unknown_column() {
    let res = build(
        csv!("data"),
        Aes::new().x("x").y("nope"),
        vec![Layer::new(Geom::polygon())],
        Facet::None,
    );
    assert!(matches!(res, Err(Error::UnknownColumn(_))), "got {res:?}");
}

#[test]
fn a_data_mapped_fill_is_refused() {
    let res = build(csv!("data"), xy().fill("g"), vec![Layer::new(Geom::polygon())], Facet::None);
    assert!(matches!(res, Err(Error::Rendering(_))), "got {res:?}");
}

#[test]
fn a_coord_flip_is_refused() {
    let r = super::GGPlot::new()
        .data(csv!("data"))
        .aes(xy())
        .layer(Layer::new(Geom::polygon()))
        .coord(super::coord::Coord::Cartesian { xlim: None, ylim: None, flip: true })
        .build()
        .and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "got {r:?}");
}

/// A polygon-only plot under every non-default coord is refused (no other path-family layer needed).
#[test]
fn every_non_default_coord_is_refused_for_a_lone_polygon() {
    use super::coord::Coord;
    for coord in [
        Coord::cartesian().flip(),
        Coord::cartesian().xlim(0.0, 10.0),
        Coord::cartesian().ylim(0.0, 10.0),
        Coord::polar(),
        Coord::fixed(1.0),
    ] {
        let r = super::GGPlot::new()
            .data(csv!("data"))
            .aes(xy())
            .layer(Layer::new(Geom::polygon()))
            .coord(coord)
            .build()
            .and_then(|b| b.layout());
        assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "got {r:?}");
    }
}

/// A `geom_point` beside a polygon that maps a missing column is refused, not returned empty.
#[test]
fn a_point_beside_a_polygon_with_a_missing_column_is_refused() {
    let res = build(
        csv!("data"),
        xy(),
        vec![
            Layer::new(Geom::polygon()),
            Layer::new(Geom::point()).aes(Aes::new().x("nope").y("y")),
        ],
        Facet::None,
    );
    assert!(res.is_err(), "got {res:?}");
}
