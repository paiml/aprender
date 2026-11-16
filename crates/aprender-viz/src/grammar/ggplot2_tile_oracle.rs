//! ggplot2 conformance oracle for `geom_tile` (PLOT-GG-001, paiml/aprender#6909; core row 14).
//! Every case is built through the public builder and read back with `layout()`; see
//! `ggplot2_rects_common` for the tolerance and the comparison. Fixtures:
//! `fixtures/ggplot2/geom_tile/expected_<case>.json`.

use super::ggplot2_rects_common::{
    build, build_with, check, csv, fixture, layer, want, xy, BLUE, RED,
};
use super::rects::RectKind;
use super::{Aes, Coord, DataFrame, Facet, Geom, Stat};
use crate::error::Error;

const ROW: &str = "geom_tile";
const G: &str = "GeomTile";

/// Build one fixture case from its data, geom and any further layers, and compare it.
fn run(case: &str, json: &str, data: DataFrame, geom: Geom) {
    let w = want(json, ROW, case);
    let got = build(data, xy(), vec![layer(geom)]).expect("layout resolves a tile");
    check(&got, &w, G, RectKind::Tile);
}

#[test]
fn default_size_is_the_data_resolution() {
    run("default", fixture!("geom_tile", "default"), csv!("geom_tile", "data"), Geom::tile());
}

#[test]
fn width_and_height_parameters() {
    let g = Geom::tile().width(0.5).height(0.25);
    run("width_height", fixture!("geom_tile", "width_height"), csv!("geom_tile", "data"), g);
}

#[test]
fn width_only_keeps_the_resolved_height() {
    let g = Geom::tile().width(0.5);
    run("width_only", fixture!("geom_tile", "width_only"), csv!("geom_tile", "data"), g);
}

#[test]
fn height_only_keeps_the_resolved_width() {
    let g = Geom::tile().height(3.0);
    run("height_only", fixture!("geom_tile", "height_only"), csv!("geom_tile", "data"), g);
}

#[test]
fn uneven_spacing_takes_the_smallest_gap() {
    run("uneven", fixture!("geom_tile", "uneven"), csv!("geom_tile", "data_uneven"), Geom::tile());
}

#[test]
fn a_single_tile_is_one_unit_wide() {
    run("single", fixture!("geom_tile", "single"), csv!("geom_tile", "data_single"), Geom::tile());
}

#[test]
fn a_single_sized_tile() {
    let g = Geom::tile().width(4.0).height(0.5);
    run("single_sized", fixture!("geom_tile", "single_sized"), csv!("geom_tile", "data_single"), g);
}

#[test]
fn negative_positions() {
    run(
        "negative",
        fixture!("geom_tile", "negative"),
        csv!("geom_tile", "data_negative"),
        Geom::tile(),
    );
}

#[test]
fn quarter_steps_leave_the_width_column_at_one() {
    run(
        "quarter",
        fixture!("geom_tile", "quarter"),
        csv!("geom_tile", "data_quarter"),
        Geom::tile(),
    );
}

#[test]
fn fixed_fill_colour_and_alpha() {
    let g = Geom::tile().aes(Aes::new().fill_value(RED).color_value(BLUE).alpha_value(0.5));
    run(
        "fill_colour_alpha",
        fixture!("geom_tile", "fill_colour_alpha"),
        csv!("geom_tile", "data"),
        g,
    );
}

#[test]
fn missing_positions_stay_as_na_rows() {
    run("na_rows", fixture!("geom_tile", "na_rows"), csv!("geom_tile", "data_na"), Geom::tile());
}

#[test]
fn a_point_layer_shares_the_scales() {
    let w = want(fixture!("geom_tile", "with_point"), ROW, "with_point");
    let layers =
        vec![layer(Geom::tile()), layer(Geom::point()).data(csv!("geom_tile", "data_single"))];
    let got = build(csv!("geom_tile", "data_uneven"), xy(), layers).expect("layout");
    check(&got, &w, G, RectKind::Tile);
}

// --- refusals: typed errors, never a guess or a panic ------------------------------------------

fn frame(x: &[f32], y: &[f32]) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column_f32("x", x);
    df.add_column_f32("y", y);
    df
}

fn tile_plot(data: DataFrame, aes: Aes) -> crate::error::Result<super::BuiltLayout> {
    build(data, aes, vec![layer(Geom::tile())])
}

#[test]
fn empty_and_all_missing_data_are_a_scale_error_not_a_panic() {
    for (x, y) in [(vec![], vec![]), (vec![f32::NAN], vec![1.0]), (vec![1.0], vec![f32::NAN])] {
        let r = tile_plot(frame(&x, &y), xy());
        assert!(matches!(r, Err(Error::ScaleDomain(_))), "{x:?}/{y:?}: {r:?}");
    }
}

#[test]
fn a_non_finite_width_or_height_is_refused() {
    for g in [Geom::tile().width(f32::NAN), Geom::tile().height(f32::INFINITY)] {
        let r = build(frame(&[1.0, 2.0], &[1.0, 2.0]), xy(), vec![layer(g)]);
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("finite")), "finite");
    }
}

#[test]
fn an_infinite_position_is_refused() {
    let r = tile_plot(frame(&[1.0, f32::INFINITY], &[1.0, 2.0]), xy());
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("infinite")), "infinite");
}

#[test]
fn a_missing_or_text_column_is_refused() {
    let r = tile_plot(frame(&[1.0], &[1.0]), Aes::new().x("x").y("nope"));
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    let mut df = frame(&[1.0], &[1.0]);
    df.add_column_str("t", &["a"]);
    let r = tile_plot(df, Aes::new().x("x").y("t"));
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("text")), "text");
}

#[test]
fn data_mapped_aesthetics_a_group_and_a_facet_are_refused() {
    let mut df = frame(&[1.0, 2.0], &[1.0, 2.0]);
    df.add_column_f32("z", &[1.0, 2.0]);
    df.add_column_str("f", &["a", "b"]);
    for aes in [xy().fill("z"), xy().color("z"), xy().alpha("z"), xy().group("z"), xy().size("z")] {
        let r = tile_plot(df.clone(), aes);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
    let r = build_with(df, xy(), vec![layer(Geom::tile())], Facet::wrap("f", 2));
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("facet")), "facet");
}

#[test]
fn bounds_mapped_on_a_tile_or_raster_are_refused() {
    let mut df = frame(&[1.0, 2.0], &[1.0, 2.0]);
    df.add_column_f32("b", &[0.0, 1.0]);
    for aes in [xy().xmin("b"), xy().xmax("b"), xy().ymin("b"), xy().ymax("b")] {
        for g in [Geom::tile(), Geom::raster()] {
            let r = build(df.clone(), aes.clone(), vec![layer(g)]);
            assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("xmin")), "bounds");
        }
    }
}

#[test]
fn a_non_identity_stat_and_a_non_default_coord_are_refused() {
    let df = frame(&[1.0, 2.0], &[1.0, 2.0]);
    let r = build(df.clone(), xy(), vec![layer(Geom::tile().stat(Stat::Count))]);
    assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{r:?}");
    for coord in [Coord::cartesian().flip(), Coord::polar(), Coord::fixed(1.0)] {
        let p = super::GGPlot::new().data(df.clone()).aes(xy()).layer(layer(Geom::tile()));
        let r = p.coord(coord).build().and_then(|b| b.layout());
        assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{r:?}");
    }
}

#[test]
fn a_tile_beside_an_unresolved_geom_is_still_refused() {
    let df = frame(&[1.0, 2.0], &[1.0, 2.0]);
    let r = build(df, xy(), vec![layer(Geom::tile()), layer(Geom::area())]);
    assert!(matches!(r, Err(Error::UnsupportedLayer(m)) if m.starts_with("geom_area")), "area");
}

#[test]
fn the_renderer_returns_a_refused_layers_error_instead_of_training_nothing() {
    // `to_framebuffer` reaches the position scales without `layout()`. A tile with a data-mapped
    // fill (the usual heatmap) is refused by `build_layer`; training must return that error, not
    // fall back to the empty 0..1 scale.
    let mut df = frame(&[1.0, 2.0], &[1.0, 2.0]);
    df.add_column_f32("z", &[1.0, 2.0]);
    let p = super::GGPlot::new().data(df).aes(xy().fill("z")).layer(layer(Geom::tile()));
    let r = p.build().and_then(|b| b.to_framebuffer());
    assert!(matches!(r, Err(Error::Rendering(_))), "{:?}", r.map(|_| ()));
}

#[test]
fn a_tile_is_refused_by_both_renderers_not_returned_empty() {
    let p = super::GGPlot::new()
        .data(frame(&[1.0, 2.0], &[1.0, 2.0]))
        .aes(xy())
        .layer(layer(Geom::tile()))
        .build()
        .expect("builds");
    let drawn = |m: &&str| *m == super::rects::NOT_DRAWN;
    assert!(matches!(p.to_framebuffer(), Err(Error::UnsupportedLayer(m)) if drawn(&m)));
    assert!(matches!(p.to_svg(), Err(Error::UnsupportedLayer(m)) if drawn(&m)));
}
