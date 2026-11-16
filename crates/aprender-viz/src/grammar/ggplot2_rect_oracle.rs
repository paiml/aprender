//! ggplot2 conformance oracle for `geom_rect` (PLOT-GG-001, paiml/aprender#6909; not a core row).
//! Every case is built through the public builder and read back with `layout()`; see
//! `ggplot2_rects_common` for the tolerance and the comparison. Fixtures:
//! `fixtures/ggplot2/geom_rect/expected_<case>.json`.

use super::ggplot2_rects_common::{
    bounds, build, build_with, check, check_refused, csv, fixture, layer, want, BLUE, RED,
};
use super::rects::RectKind;
use super::{Aes, Coord, DataFrame, Facet, Geom, Stat};
use crate::error::Error;

const ROW: &str = "geom_rect";
const G: &str = "GeomRect";

fn run(case: &str, json: &str, data: DataFrame, geom: Geom) {
    let w = want(json, ROW, case);
    let got = build(data, bounds(), vec![layer(geom)]).expect("layout resolves a rect");
    check(&got, &w, G, RectKind::Rect);
}

#[test]
fn bounds_are_the_data() {
    run("default", fixture!("geom_rect", "default"), csv!("geom_rect", "data"), Geom::rect());
}

#[test]
fn reversed_bounds_are_kept_as_given() {
    run(
        "reversed",
        fixture!("geom_rect", "reversed"),
        csv!("geom_rect", "data_reversed"),
        Geom::rect(),
    );
}

#[test]
fn a_zero_area_rect_trains_its_edges() {
    run(
        "zero_area",
        fixture!("geom_rect", "zero_area"),
        csv!("geom_rect", "data_zero_area"),
        Geom::rect(),
    );
}

#[test]
fn a_single_rect() {
    run("single", fixture!("geom_rect", "single"), csv!("geom_rect", "data_single"), Geom::rect());
}

#[test]
fn a_single_zero_area_rect_has_a_degenerate_scale() {
    let w = want(fixture!("geom_rect", "single_zero_area"), ROW, "single_zero_area");
    let r = build(csv!("geom_rect", "data_single_zero"), bounds(), vec![layer(Geom::rect())]);
    check_refused(r, &w);
}

#[test]
fn negative_bounds() {
    run(
        "negative",
        fixture!("geom_rect", "negative"),
        csv!("geom_rect", "data_negative"),
        Geom::rect(),
    );
}

#[test]
fn missing_bounds_stay_as_na_rows() {
    run("na_rows", fixture!("geom_rect", "na_rows"), csv!("geom_rect", "data_na"), Geom::rect());
}

#[test]
fn fixed_fill_colour_and_alpha() {
    let g = Geom::rect().aes(Aes::new().fill_value(RED).color_value(BLUE).alpha_value(0.5));
    run(
        "fill_colour_alpha",
        fixture!("geom_rect", "fill_colour_alpha"),
        csv!("geom_rect", "data"),
        g,
    );
}

#[test]
fn a_point_layer_shares_the_scales() {
    let w = want(fixture!("geom_rect", "with_point"), ROW, "with_point");
    let point = layer(Geom::point()).aes(Aes::new().x("xmin").y("ymin"));
    let rect = layer(Geom::rect()).aes(bounds());
    let got = build(csv!("geom_rect", "data"), Aes::new(), vec![rect, point]).expect("layout");
    check(&got, &w, G, RectKind::Rect);
}

// --- refusals: typed errors, never a guess or a panic ------------------------------------------

fn frame() -> DataFrame {
    let mut df = DataFrame::new();
    for (n, v) in [("xmin", 0.0), ("xmax", 1.0), ("ymin", 0.0), ("ymax", 2.0)] {
        df.add_column_f32(n, &[v, v + 1.0]);
    }
    df
}

#[test]
fn a_missing_bound_mapping_or_column_is_refused() {
    let r = build(
        frame(),
        Aes::new().xmin("xmin").xmax("xmax").ymin("ymin"),
        vec![layer(Geom::rect())],
    );
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("ymax")), "mapping");
    let r = build(frame(), bounds().ymax("nope"), vec![layer(Geom::rect())]);
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
}

#[test]
fn x_and_y_aesthetics_are_refused() {
    for aes in [bounds().x("xmin"), bounds().y("ymin")] {
        let r = build(frame(), aes, vec![layer(Geom::rect())]);
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("not x or y")), "x/y");
    }
}

#[test]
fn empty_and_infinite_and_nan_data_never_panic() {
    let mut empty = DataFrame::new();
    for n in ["xmin", "xmax", "ymin", "ymax"] {
        empty.add_column_f32(n, &[]);
    }
    let r = build(empty, bounds(), vec![layer(Geom::rect())]);
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
    let mut nan = DataFrame::new();
    for n in ["xmin", "xmax", "ymin", "ymax"] {
        nan.add_column_f32(n, &[f32::NAN]);
    }
    let r = build(nan, bounds(), vec![layer(Geom::rect())]);
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
    let mut inf = frame();
    inf.add_column_f32("xmax", &[f32::INFINITY, 2.0]);
    let r = build(inf, bounds(), vec![layer(Geom::rect())]);
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("infinite")), "infinite");
}

#[test]
fn mapped_aesthetics_facets_stats_and_coords_are_refused() {
    let mut df = frame();
    df.add_column_f32("z", &[1.0, 2.0]);
    df.add_column_str("f", &["a", "b"]);
    for aes in [bounds().fill("z"), bounds().color("z"), bounds().group("z")] {
        let r = build(df.clone(), aes, vec![layer(Geom::rect())]);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
    let r = build_with(df.clone(), bounds(), vec![layer(Geom::rect())], Facet::wrap("f", 2));
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("facet")), "facet");
    let r = build(df.clone(), bounds(), vec![layer(Geom::rect().stat(Stat::Count))]);
    assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{r:?}");
    let p = super::GGPlot::new().data(df).aes(bounds()).layer(layer(Geom::rect()));
    let r = p.coord(Coord::cartesian().flip()).build().and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{r:?}");
}
