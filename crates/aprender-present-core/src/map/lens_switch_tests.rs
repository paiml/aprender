//! Interaction-level tests for the lens switch and the brush-to-selection (MAP-7 slice 1).
//! Render counts are the scheduler's own counters; history is the URL policy's `Nav`. The switch
//! refuses a lens id the registry does not hold (MAP-8 slice 1).

use std::collections::BTreeSet;

use super::lens::{Lens, LensRegistry, Scale};
use super::urlstate::{decode, encode, Nav};
use super::{Camera2D, EntityId, HexGrid, Interaction, MapError, Renders};

const VIEW: (f64, f64) = (800.0, 600.0);
const CELLS: u32 = 10_530;

fn start() -> Interaction {
    let grid = HexGrid::map_default(1.0).expect("grid");
    let cam = Camera2D::new((0.0, 0.0), 10.0, VIEW).expect("camera");
    Interaction::new(grid, cam, None).0
}

/// Lenses 1 and 2 over every cell, all with value 1.0.
fn registry() -> LensRegistry {
    let mut reg = LensRegistry::new(CELLS);
    for id in [1, 2] {
        let values = vec![Some(1.0); CELLS as usize];
        let lens = Lens::new(id, "L", Scale::Linear, (0.0, 100.0), values).expect("lens");
        reg.add(lens).expect("register");
    }
    reg
}

fn r(base: u32, lens: u32, overlay: u32) -> Renders {
    Renders {
        base,
        lens,
        overlay,
    }
}

fn decoded_lens(map: &Interaction) -> Option<u32> {
    let hash = encode(map.url_state()).expect("state encodes");
    decode(&hash).expect("hash decodes").lens()
}

#[test]
fn lens_switch_renders_one_lens_layer_and_pushes_one_history_entry() {
    let mut map = start();
    let report = map.switch_lens(&registry(), 1, 0).expect("switch");
    assert_eq!(report.renders, r(0, 1, 0));
    let Nav::Push(hash) = &report.nav else {
        panic!("expected one pushState, got {:?}", report.nav);
    };
    assert_eq!(decode(hash).expect("hash decodes").lens(), Some(1));
}

#[test]
fn url_state_after_a_lens_switch_decodes_to_the_new_lens() {
    let mut map = start();
    let lenses = registry();
    map.switch_lens(&lenses, 1, 0).expect("switch");
    assert_eq!(map.url_state().lens(), Some(1));
    assert_eq!(decoded_lens(&map), Some(1));
    map.switch_lens(&lenses, 2, 10).expect("switch again");
    assert_eq!(decoded_lens(&map), Some(2));
}

#[test]
fn switching_to_the_current_lens_renders_nothing_and_adds_no_entry() {
    let mut map = start();
    let lenses = registry();
    map.switch_lens(&lenses, 1, 0).expect("first switch");
    let before = map.url_state().clone();
    let report = map.switch_lens(&lenses, 1, 5).expect("same lens");
    assert_eq!(report.renders, r(0, 0, 0));
    assert_eq!(report.nav, Nav::Nothing);
    assert_eq!(map.url_state(), &before);
}

#[test]
fn switching_between_two_lenses_renders_one_lens_layer() {
    let mut map = start();
    let lenses = registry();
    map.switch_lens(&lenses, 1, 0).expect("first switch");
    let report = map.switch_lens(&lenses, 2, 5).expect("second switch");
    assert_eq!(report.renders, r(0, 1, 0));
    assert!(matches!(report.nav, Nav::Push(_)));
}

#[test]
fn switching_to_an_unregistered_lens_is_refused_with_no_render_and_no_history() {
    let mut map = start();
    let lenses = registry();
    let before = map.url_state().clone();
    assert_eq!(
        map.switch_lens(&lenses, 9, 0),
        Err(MapError::UnknownLens(9))
    );
    assert_eq!(map.url_state(), &before);
    assert_eq!(map.url_state().lens(), None);
    let next = map
        .switch_lens(&lenses, 1, 0)
        .expect("registered lens still switches");
    assert_eq!(next.renders, r(0, 1, 0));
}

#[test]
fn brush_to_selection_renders_only_the_overlay_once() {
    let mut map = start();
    map.switch_lens(&registry(), 1, 0).expect("switch");
    let cells: BTreeSet<EntityId> = [3, 5, 8].into_iter().collect();
    let report = map.brush(cells.clone()).expect("brush");
    assert_eq!(report.renders, r(0, 0, 1));
    assert_eq!(report.nav, Nav::Nothing);
    assert_eq!(map.brushed(), &cells);
    assert_eq!(
        map.selected(),
        None,
        "the brush does not move the single selection"
    );
}
