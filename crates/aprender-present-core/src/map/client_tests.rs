//! Apply-level tests for the headless lens client (MAP-8 slice 1): exact render counts, the
//! sequence rules, atomic refusal and the registry checks. The aggregation oracle is in `oracle`.

use super::*;
use crate::map::{
    camera::Camera2D,
    hex::HexGrid,
    layers::{LensId, Renders},
    lens::{Lens, LensRegistry, Scale},
    urlstate::Nav,
    MapError,
};

#[path = "client_oracle.rs"]
mod oracle;

const VIEW: (f64, f64) = (800.0, 600.0);
/// Cell count of the 117 by 90 map grid.
const CELLS: u32 = 10_530;

/// A value every third cell is `None`; the others hold `c % 100` on the domain 0..100.
fn patterned() -> Vec<Option<f64>> {
    (0..CELLS)
        .map(|c| {
            if c % 3 == 0 {
                None
            } else {
                Some(f64::from(c % 100))
            }
        })
        .collect()
}

fn registry() -> LensRegistry {
    let mut reg = LensRegistry::new(CELLS);
    for id in [1, 2] {
        let lens = Lens::new(id, "L", Scale::Linear, (0.0, 100.0), patterned()).expect("lens");
        reg.add(lens).expect("register");
    }
    reg
}

fn client(active: Option<LensId>) -> Client {
    let grid = HexGrid::map_default(1.0).expect("grid");
    let cam = Camera2D::new((0.0, 0.0), 10.0, VIEW).expect("camera");
    Client::new(grid, cam, registry(), active)
        .expect("client")
        .0
}

fn r(base: u32, lens: u32, overlay: u32) -> Renders {
    Renders {
        base,
        lens,
        overlay,
    }
}

fn delta(lens: LensId, seq: u64, cells: &[(u32, Option<f64>)]) -> Delta {
    Delta::new(lens, seq, cells.to_vec()).expect("finite values")
}

fn values(c: &Client, lens: LensId) -> Vec<Option<f64>> {
    c.lenses().get(lens).expect("registered").values().to_vec()
}

#[test]
fn a_delta_to_the_active_lens_renders_exactly_one_lens_layer() {
    let mut c = client(Some(1));
    let rep = c
        .apply(&delta(1, 1, &[(4, Some(42.0)), (7, None)]))
        .expect("applies");
    assert_eq!(rep.renders, r(0, 1, 0), "no base or overlay re-render");
    assert!(rep.frame_requested);
    assert_eq!(rep.nav, Nav::Nothing, "a delta writes no history");
}

#[test]
fn a_delta_updates_exactly_the_named_cells() {
    let mut c = client(Some(1));
    let before = values(&c, 1);
    c.apply(&delta(1, 1, &[(4, Some(42.0)), (7, None)]))
        .expect("applies");
    let after = values(&c, 1);
    let changed: Vec<usize> = (0..before.len())
        .filter(|&i| before[i] != after[i])
        .collect();
    assert_eq!(changed, vec![4, 7], "exactly the two named cells");
    assert_eq!(after[4], Some(42.0));
    assert_eq!(after[7], None);
}

#[test]
fn a_delta_to_an_inactive_lens_renders_nothing_and_stores_the_values() {
    let mut c = client(Some(1));
    let rep = c.apply(&delta(2, 1, &[(5, Some(9.0))])).expect("applies");
    assert_eq!(rep.renders, r(0, 0, 0));
    assert!(!rep.frame_requested);
    assert_eq!(c.map().pending_deltas(2), 1);
    assert_eq!(values(&c, 2)[5], Some(9.0));
}

#[test]
fn stored_deltas_count_up_and_a_switch_renders_once_and_clears_them() {
    let mut c = client(Some(1));
    c.apply(&delta(2, 1, &[(5, Some(9.0))])).expect("first");
    c.apply(&delta(2, 2, &[(6, Some(8.0))])).expect("second");
    assert_eq!(c.map().pending_deltas(2), 2);
    let switched = c.switch_lens(2, 10).expect("switch");
    assert_eq!(switched.renders, r(0, 1, 0));
    assert_eq!(c.map().pending_deltas(2), 0);
}

#[test]
fn a_stale_sequence_is_refused_and_changes_nothing() {
    let mut c = client(Some(1));
    c.apply(&delta(1, 1, &[(4, Some(1.0))])).expect("seq 1");
    c.apply(&delta(1, 2, &[(4, Some(2.0))])).expect("seq 2");
    let before = c.lenses().clone();
    let hist = c.histogram(1).expect("histogram").clone();
    assert_eq!(
        c.apply(&delta(1, 2, &[(4, Some(3.0))])),
        Err(MapError::Stale { seq: 2, last: 2 }),
        "an equal sequence is stale"
    );
    assert_eq!(
        c.apply(&delta(1, 1, &[(4, Some(3.0))])),
        Err(MapError::Stale { seq: 1, last: 2 })
    );
    assert_eq!(
        c.apply(&delta(2, 0, &[(4, Some(3.0))])),
        Err(MapError::Stale { seq: 0, last: 0 })
    );
    assert_eq!(c.lenses(), &before);
    assert_eq!(c.histogram(1), Some(&hist));
    assert_eq!(c.last_seq(1), 2);
    assert_eq!(c.last_seq(2), 0);
}

#[test]
fn a_gap_is_reported_and_not_applied() {
    let mut c = client(Some(1));
    c.apply(&delta(1, 1, &[])).expect("seq 1");
    let before = c.lenses().clone();
    assert_eq!(
        c.apply(&delta(1, 3, &[(4, Some(5.0))])),
        Err(MapError::Gap {
            expected: 2,
            seq: 3
        })
    );
    assert_eq!(c.lenses(), &before);
    assert_eq!(c.last_seq(1), 1);
}

#[test]
fn the_first_delta_for_a_lens_must_carry_sequence_one() {
    let mut c = client(Some(1));
    assert_eq!(
        c.apply(&delta(1, 2, &[])),
        Err(MapError::Gap {
            expected: 1,
            seq: 2
        })
    );
    assert_eq!(c.last_seq(1), 0);
}

#[test]
fn a_delta_for_an_unregistered_lens_is_refused() {
    let mut c = client(Some(1));
    let before = c.lenses().clone();
    assert_eq!(
        c.apply(&delta(9, 1, &[(4, Some(1.0))])),
        Err(MapError::UnknownLens(9))
    );
    assert_eq!(c.lenses(), &before);
}

#[test]
fn an_out_of_range_cell_refuses_the_whole_delta() {
    let mut c = client(Some(1));
    let before = c.lenses().clone();
    let hist = c.histogram(1).expect("histogram").clone();
    let bad = delta(1, 1, &[(4, Some(42.0)), (CELLS, Some(1.0))]);
    assert_eq!(c.apply(&bad), Err(MapError::CellOutOfRange(CELLS)));
    assert_eq!(c.lenses(), &before, "cell 4 is not written either");
    assert_eq!(c.histogram(1), Some(&hist));
    assert_eq!(c.last_seq(1), 0);
}

#[test]
fn a_repeated_cell_in_one_delta_keeps_the_last_value() {
    let mut c = client(Some(1));
    let rep = c
        .apply(&delta(1, 1, &[(4, Some(1.0)), (4, Some(2.0))]))
        .expect("applies");
    assert_eq!(rep.renders, r(0, 1, 0));
    assert_eq!(values(&c, 1)[4], Some(2.0));
    let scratch =
        Histogram::of(c.lenses().get(1).expect("lens 1"), HISTOGRAM_BINS).expect("histogram");
    assert_eq!(c.histogram(1), Some(&scratch));
}

#[test]
fn histogram_total_tracks_the_cells_with_data() {
    let mut c = client(Some(1));
    assert_eq!(
        c.histogram(1).expect("histogram").total(),
        7_020,
        "10,530 cells, 3,510 of them with no data"
    );
    c.apply(&delta(1, 1, &[(4, Some(42.0)), (7, None)]))
        .expect("applies");
    assert_eq!(
        c.histogram(1).expect("histogram").total(),
        7_019,
        "cell 7 lost its value; cell 4 kept one"
    );
}

#[test]
fn a_client_refuses_an_active_lens_that_is_not_registered() {
    let grid = HexGrid::map_default(1.0).expect("grid");
    let cam = Camera2D::new((0.0, 0.0), 10.0, VIEW).expect("camera");
    let err = Client::new(grid, cam, registry(), Some(9)).err();
    assert_eq!(err, Some(MapError::UnknownLens(9)));
}

#[test]
fn a_registry_that_does_not_cover_the_grid_is_refused() {
    let grid = HexGrid::map_default(1.0).expect("grid");
    let cam = Camera2D::new((0.0, 0.0), 10.0, VIEW).expect("camera");
    let short = LensRegistry::new(CELLS - 1);
    let err = Client::new(grid, cam, short, None).err();
    assert_eq!(err, Some(MapError::Mismatch("registry")));
}
