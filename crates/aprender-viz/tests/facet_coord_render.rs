//! APEX-001 EV-2c: facet and coord are honoured by the renderer (paiml/aprender#3233).
//!
//! Before this suite, `Facet` and `Coord` were configuration that the render path did not read.
//! `GGPlot::facet()` stored a value that `build()` did not copy into `BuiltGGPlot`, so it was
//! discarded before rendering began; `Coord::flip()` set a field no one consulted. Both
//! type-checked, ran, and did nothing. The crate's own `test_ggplot_facet` set a facet and
//! asserted `fb.width() > 0`, which passes just as well with faceting entirely absent — so the
//! defect was not merely untested, it was *covered by a test that could not fail*.
//!
//! These tests are written so that they cannot pass in that world: each one names a value that
//! differs between "honoured" and "ignored", rather than asserting that rendering succeeded.

use trueno_viz::color::Rgba;
use trueno_viz::error::Error;
use trueno_viz::grammar::{
    apply, panels, Aes, Coord, DataFrame, Facet, GGPlot, Geom, Panel, Theme,
};

/// Four rows over three lineages, deliberately NOT in sorted order, so a test that expects
/// lexicographic panels is distinguishable from one that accepts first-appearance order.
fn fixture() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0]);
    df.add_column_f32("y", &[10.0, 20.0, 30.0, 40.0]);
    df.add_column_str("lineage", &["beta", "alpha", "beta", "gamma"]);
    df
}

/// Does every source row appear in exactly one panel?
///
/// Factored out because two tests need it: the real one, and the mutation self-check that proves
/// this predicate can actually fail.
fn partition_is_total(ps: &[Panel], n_rows: usize) -> bool {
    let mut seen = vec![0usize; n_rows];
    for p in ps {
        for &i in &p.rows {
            match seen.get_mut(i) {
                Some(c) => *c += 1,
                None => return false, // an index outside the frame
            }
        }
    }
    seen.iter().all(|&c| c == 1)
}

/// Count pixels of exactly the panel background colour inside a rectangle.
///
/// Panel background is the one colour that marks where a panel *is*: the figure background is
/// white, grid lines are white, axes are dark grey, marks are blue.
fn panel_pixels(
    fb: &trueno_viz::framebuffer::Framebuffer,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> usize {
    let want = Rgba::rgb(235, 235, 235);
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            if fb.get_pixel(x, y) == Some(want) {
                n += 1;
            }
        }
    }
    n
}

fn faceted_plot(df: DataFrame, facet: Facet) -> trueno_viz::grammar::BuiltGGPlot {
    GGPlot::new()
        .data(df)
        .aes(Aes::new().x("x").y("y"))
        .geom(Geom::point())
        .facet(facet)
        .theme(Theme::grey())
        .build()
        .expect("plot builds")
}

// --- the row's three RED criteria -------------------------------------------------------------

/// **RED (1)**: `facet_wrap(~lineage)` over three lineages renders THREE panels. Today: one.
///
/// Asserted on the rendered pixels, not on `panels()`, because the row says *renders*. With
/// `ncol = 2` the three panels occupy grid cells (0,0), (0,1) and (1,0) of an 800x600 figure, and
/// cell (1,1) is left empty. A single full-bleed panel — the old behaviour — covers all four
/// quadrants, so the emptiness of the fourth is exactly what distinguishes the two.
#[test]
fn facet_wrap_over_three_lineages_renders_three_panels() {
    let plot = faceted_plot(fixture(), Facet::wrap("lineage", 2));
    let fb = plot.to_framebuffer().expect("render succeeds");

    let q00 = panel_pixels(&fb, 0, 0, 400, 300);
    let q01 = panel_pixels(&fb, 400, 0, 800, 300);
    let q10 = panel_pixels(&fb, 0, 300, 400, 600);
    let q11 = panel_pixels(&fb, 400, 300, 800, 600);

    assert!(q00 > 1000 && q01 > 1000 && q10 > 1000, "three panels expected, got {q00} {q01} {q10}");
    assert_eq!(
        q11, 0,
        "the fourth quadrant must be empty: three levels at ncol=2 fill three cells. A non-zero \
         count here means one panel was stretched across the figure, i.e. the facet was ignored."
    );
}

/// **RED (2)**: `coord_flip` swaps the axes. Today: a no-op.
///
/// Checked by equivalence rather than by eyeballing pixels: flipping the coordinate system must
/// produce exactly the figure you would get by swapping the x and y columns and not flipping. The
/// second assertion is the one that would have failed before this row — if `flip` does nothing,
/// the flipped render equals the *unflipped* render instead.
#[test]
fn coord_flip_swaps_the_axes() {
    let mut swapped = DataFrame::new();
    swapped.add_column_f32("x", &[10.0, 20.0, 30.0, 40.0]);
    swapped.add_column_f32("y", &[1.0, 2.0, 3.0, 4.0]);

    let render = |df: DataFrame, coord: Coord| {
        GGPlot::new()
            .data(df)
            .aes(Aes::new().x("x").y("y"))
            .geom(Geom::point())
            .coord(coord)
            .build()
            .expect("plot builds")
            .to_framebuffer()
            .expect("render succeeds")
            .pixels()
            .to_vec()
    };

    let flipped = render(fixture(), Coord::cartesian().flip());
    let plain = render(fixture(), Coord::cartesian());
    let swapped_cols = render(swapped, Coord::cartesian());

    assert_eq!(
        flipped, swapped_cols,
        "flipping the coordinate system must equal swapping the two columns"
    );
    assert_ne!(
        flipped, plain,
        "the flipped render is identical to the unflipped one — `flip` is being ignored, which is \
         precisely the defect this row exists to remove"
    );
}

/// **RED (3)**: the partition is total — `Σ panel rows == input rows`, no duplicates.
#[test]
fn the_partition_is_total_over_every_facet_kind() {
    let df = fixture();
    let kinds = [
        Facet::None,
        Facet::wrap("lineage", 2),
        Facet::wrap("lineage", 1),
        Facet::row("lineage"),
        Facet::col("lineage"),
        Facet::grid("lineage", "x"),
    ];
    for facet in kinds {
        let ps = panels(&facet, &df).expect("partition succeeds");
        let total: usize = ps.iter().map(|p| p.rows.len()).sum();
        assert_eq!(total, df.nrow(), "{facet:?} moved the row count");
        assert!(partition_is_total(&ps, df.nrow()), "{facet:?} duplicated or dropped a row");
    }
}

// --- the row's stated mutation ----------------------------------------------------------------

/// **APEX-001 EV-2c's stated mutation**: "drop the partition step → the row-count test RED."
///
/// A test that asserts a property is only worth as much as that property's ability to fail. Here
/// the mutant is built explicitly — every panel receives every row, which is what dropping the
/// grouping would produce — and fed to the *same* predicate the test above uses. If
/// `partition_is_total` accepted it, the check above would be decoration.
#[test]
fn mutation_dropping_the_partition_is_caught() {
    let df = fixture();
    let real = panels(&Facet::wrap("lineage", 2), &df).expect("partition succeeds");
    assert!(partition_is_total(&real, df.nrow()), "the real partition must pass");

    let mutant: Vec<Panel> =
        real.iter().map(|p| Panel { rows: (0..df.nrow()).collect(), ..p.clone() }).collect();

    assert!(
        !partition_is_total(&mutant, df.nrow()),
        "the row-count predicate accepts a partition that gives every panel every row — it cannot \
         detect the one mutation this row names"
    );
}

// --- ordering, which decides the output bytes --------------------------------------------------

/// Panel order is a function of the level *set*, not of row order.
///
/// APEX-001 renders the same figure on two architectures and compares the bytes, so the panel
/// sequence may not depend on the order rows arrived in — nor on a `HashMap`'s iteration order.
#[test]
fn panel_order_is_lexicographic_and_not_data_order() {
    let ps = panels(&Facet::wrap("lineage", 2), &fixture()).expect("partition succeeds");
    let levels: Vec<&str> = ps.iter().map(|p| p.levels[0].as_str()).collect();
    assert_eq!(levels, vec!["alpha", "beta", "gamma"]);

    // The fixture lists beta first; a first-appearance ordering would put beta first.
    assert_ne!(levels[0], "beta", "panels follow the data's order rather than a total one");

    let mut permuted = DataFrame::new();
    permuted.add_column_f32("x", &[4.0, 3.0, 2.0, 1.0]);
    permuted.add_column_f32("y", &[40.0, 30.0, 20.0, 10.0]);
    permuted.add_column_str("lineage", &["gamma", "beta", "alpha", "beta"]);
    let ps2 = panels(&Facet::wrap("lineage", 2), &permuted).expect("partition succeeds");
    let levels2: Vec<&str> = ps2.iter().map(|p| p.levels[0].as_str()).collect();
    assert_eq!(levels, levels2, "permuting the input rows changed the panel order");

    // Placement must be a function of index, so the grid is reproducible too.
    let cells: Vec<(usize, usize)> = ps.iter().map(|p| (p.row, p.col)).collect();
    assert_eq!(cells, vec![(0, 0), (0, 1), (1, 0)]);
}

/// A facet grid is the full cross product, including combinations no row exhibits — that is what
/// makes it a grid rather than a ragged list.
#[test]
fn facet_grid_is_the_full_cross_product() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[1.0, 2.0]);
    df.add_column_f32("y", &[1.0, 2.0]);
    df.add_column_str("lineage", &["a", "b"]);
    df.add_column_str("run", &["r1", "r2"]);

    let ps = panels(&Facet::grid("lineage", "run"), &df).expect("partition succeeds");
    assert_eq!(ps.len(), 4, "2 lineages x 2 runs = 4 cells");
    assert_eq!(ps.iter().filter(|p| p.rows.is_empty()).count(), 2, "two cells are unobserved");
    assert!(partition_is_total(&ps, df.nrow()));
}

// --- refusals: the failure mode this row replaces ----------------------------------------------

/// Faceting by a column the frame lacks is an error, not an empty plot.
#[test]
fn an_unknown_facet_column_refuses() {
    let err = panels(&Facet::wrap("no_such_column", 2), &fixture()).unwrap_err();
    assert!(matches!(err, Error::UnknownColumn(_)), "got {err:?}");
}

/// `Polar` and `Fixed` have no transform yet, so they refuse instead of silently rendering
/// Cartesian output that is indistinguishable from a working polar plot.
#[test]
fn unimplemented_coords_refuse_rather_than_no_op() {
    for coord in [Coord::polar(), Coord::fixed(1.5)] {
        let err = apply(&coord, 1.0, 2.0).unwrap_err();
        assert!(matches!(err, Error::UnsupportedCoord(_)), "got {err:?}");
    }
    assert_eq!(apply(&Coord::cartesian(), 1.0, 2.0).expect("cartesian works"), (1.0, 2.0));
    assert_eq!(apply(&Coord::cartesian().flip(), 1.0, 2.0).expect("flip works"), (2.0, 1.0));
}

/// A non-numeric cell must not shift the rows after it.
///
/// `DataFrame::get_f32` used to `filter_map` missing cells away, so a single text cell in the x
/// column silently renumbered every later row: `x[i]` and `y[i]` then came from different records
/// and a panel's row indices pointed at the wrong data. Nothing reported an error. This asserts
/// the alignment directly, because it is load-bearing for every test above.
#[test]
fn a_non_numeric_cell_does_not_shift_the_rows_after_it() {
    let mut df = DataFrame::new();
    df.add_column_str("x", &["1.0", "not a number", "3.0"]);
    df.add_column_f32("y", &[10.0, 20.0, 30.0]);

    let xs = df.get_f32("x").expect("column exists");
    assert_eq!(xs.len(), df.nrow(), "the column must still have one entry per row");
    assert_eq!(xs, vec![None, None, None], "text cells read as absent, in place");

    let ys = df.get_f32("y").expect("column exists");
    assert_eq!(ys, vec![Some(10.0), Some(20.0), Some(30.0)]);

    // And the facet partition still indexes the frame, not a compacted copy.
    let ps = panels(&Facet::None, &df).expect("partition succeeds");
    assert_eq!(ps[0].rows, vec![0, 1, 2]);
}

/// Rows whose facet cell is missing or non-textual still land somewhere: the partition is total
/// for every input, so no row is quietly lost for having the wrong type.
#[test]
fn rows_with_a_missing_facet_value_get_their_own_panel() {
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[1.0, 2.0, 3.0]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    df.add_column_str("lineage", &["a", "b"]); // one short: row 2 has no value

    let ps = panels(&Facet::wrap("lineage", 2), &df).expect("partition succeeds");
    assert!(partition_is_total(&ps, df.nrow()), "the short column dropped a row");
    let levels: Vec<&str> = ps.iter().map(|p| p.levels[0].as_str()).collect();
    assert_eq!(levels, vec!["NA", "a", "b"]);
}
