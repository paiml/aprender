//! ggplot2 conformance oracle for `geom_raster` (PLOT-GG-001, paiml/aprender#6909; not a core row).
//! Every case is built through the public builder and read back with `layout()`; see
//! `ggplot2_rects_common` for the tolerance and the comparison. Fixtures:
//! `fixtures/ggplot2/geom_raster/expected_<case>.json`.

use super::ggplot2_rects_common::{build, build_with, check, csv, fixture, layer, want, xy, RED};
use super::rects::RectKind;
use super::{Aes, Coord, DataFrame, Facet, Geom, Stat};
use crate::error::Error;

const ROW: &str = "geom_raster";
const G: &str = "GeomRaster";

fn run(case: &str, json: &str, data: DataFrame, geom: Geom) {
    let w = want(json, ROW, case);
    let got = build(data, xy(), vec![layer(geom)]).expect("layout resolves a raster");
    check(&got, &w, G, RectKind::Raster);
}

#[test]
fn a_regular_grid() {
    run("regular", fixture!("geom_raster", "regular"), csv!("geom_raster", "data"), Geom::raster());
}

#[test]
fn hjust_and_vjust_zero() {
    let g = Geom::raster().hjust(0.0).vjust(0.0);
    run("hjust0_vjust0", fixture!("geom_raster", "hjust0_vjust0"), csv!("geom_raster", "data"), g);
}

#[test]
fn hjust_and_vjust_one() {
    let g = Geom::raster().hjust(1.0).vjust(1.0);
    run("hjust1_vjust1", fixture!("geom_raster", "hjust1_vjust1"), csv!("geom_raster", "data"), g);
}

#[test]
fn hjust_and_vjust_differ() {
    let g = Geom::raster().hjust(0.25).vjust(0.75);
    run("hjust_quarter", fixture!("geom_raster", "hjust_quarter"), csv!("geom_raster", "data"), g);
}

#[test]
fn a_single_row_has_a_unit_height() {
    run(
        "single_row",
        fixture!("geom_raster", "single_row"),
        csv!("geom_raster", "data_single_row"),
        Geom::raster(),
    );
}

#[test]
fn a_single_column_has_a_unit_width() {
    run(
        "single_column",
        fixture!("geom_raster", "single_column"),
        csv!("geom_raster", "data_single_column"),
        Geom::raster(),
    );
}

#[test]
fn a_single_cell() {
    run(
        "single_cell",
        fixture!("geom_raster", "single_cell"),
        csv!("geom_raster", "data_single_cell"),
        Geom::raster(),
    );
}

#[test]
fn uneven_x_warns_and_uses_the_smallest_step() {
    run(
        "uneven_x",
        fixture!("geom_raster", "uneven_x"),
        csv!("geom_raster", "data_uneven_x"),
        Geom::raster(),
    );
}

// NOT A BUG: ggplot2 4.0.3's `GeomRaster$setup_data` raises the same text, "Raster pixels are placed
// at uneven horizontal intervals and will be shifted", from BOTH its x and its y branch (the y
// branch was copy-pasted and never reworded). Captured on kfocus with ggplot2 4.0.3: uneven x and
// uneven y each give exactly that first line, and `expected_uneven_{x,y}.json` record it. The
// comparison is exact, so the word "horizontal" here must stay.
#[test]
fn uneven_y_warns_with_the_horizontal_text() {
    run(
        "uneven_y",
        fixture!("geom_raster", "uneven_y"),
        csv!("geom_raster", "data_uneven_y"),
        Geom::raster(),
    );
}

#[test]
fn half_steps() {
    run(
        "half_steps",
        fixture!("geom_raster", "half_steps"),
        csv!("geom_raster", "data_half"),
        Geom::raster(),
    );
}

#[test]
fn negative_unsorted_positions() {
    let g = Geom::raster().hjust(0.0).vjust(1.0);
    run(
        "negative_unsorted",
        fixture!("geom_raster", "negative_unsorted"),
        csv!("geom_raster", "data_negative"),
        g,
    );
}

#[test]
fn fixed_fill_and_alpha() {
    let g = Geom::raster().aes(Aes::new().fill_value(RED).alpha_value(0.5));
    run("fill_alpha", fixture!("geom_raster", "fill_alpha"), csv!("geom_raster", "data"), g);
}

#[test]
fn a_raster_has_no_outline_even_when_one_is_asked_for() {
    let g = Geom::raster().aes(Aes::new().color_value(RED));
    let got = build(csv!("geom_raster", "data"), xy(), vec![layer(g)]).expect("layout");
    let r = got.layers[0].rects.as_ref().expect("raster data");
    assert!(r.colour.is_none() && r.linewidth.is_none() && r.linetype.is_none());
}

// --- refusals: typed errors, never a guess or a panic ------------------------------------------

fn frame(x: &[f32], y: &[f32]) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column_f32("x", x);
    df.add_column_f32("y", y);
    df
}

#[test]
fn empty_and_all_missing_data_are_a_scale_error_not_a_panic() {
    for (x, y) in [(vec![], vec![]), (vec![f32::NAN], vec![1.0]), (vec![1.0], vec![f32::NAN])] {
        let r = build(frame(&x, &y), xy(), vec![layer(Geom::raster())]);
        assert!(matches!(r, Err(Error::ScaleDomain(_))), "{x:?}/{y:?}: {r:?}");
    }
}

#[test]
fn a_non_finite_hjust_or_vjust_is_refused() {
    for g in [Geom::raster().hjust(f32::NAN), Geom::raster().vjust(f32::INFINITY)] {
        let r = build(frame(&[1.0, 2.0], &[1.0, 2.0]), xy(), vec![layer(g)]);
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("finite")), "finite");
    }
}

#[test]
fn infinite_text_mapped_group_and_facet_are_refused() {
    let r = build(frame(&[1.0, f32::INFINITY], &[1.0, 2.0]), xy(), vec![layer(Geom::raster())]);
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("infinite")), "infinite");
    let mut df = frame(&[1.0, 2.0], &[1.0, 2.0]);
    df.add_column_f32("z", &[1.0, 2.0]);
    df.add_column_str("f", &["a", "b"]);
    for aes in [xy().fill("z"), xy().alpha("z"), xy().group("z")] {
        let r = build(df.clone(), aes, vec![layer(Geom::raster())]);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
    let r = build_with(df.clone(), xy(), vec![layer(Geom::raster())], Facet::wrap("f", 2));
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("facet")), "facet");
    let r = build(df.clone(), Aes::new().x("x").y("f"), vec![layer(Geom::raster())]);
    assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("text")), "text");
    let r = build(df.clone(), xy(), vec![layer(Geom::raster().stat(Stat::Count))]);
    assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{r:?}");
    let p = super::GGPlot::new().data(df).aes(xy()).layer(layer(Geom::raster()));
    let r = p.coord(Coord::polar()).build().and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{r:?}");
}
