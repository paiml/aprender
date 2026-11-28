//! Tests for the interaction core: hover, selection, layer toggle, keys and history writes.
//! Exact render counts throughout. The fixed grid is the 117 by 90 map with R = 1.

use super::*;
use crate::map::{
    camera::Camera2D,
    hex::HexGrid,
    layers::Renders,
    urlstate::{encode, UrlState, CAMERA_DEBOUNCE_MS},
    MAX_SCALE, MIN_SCALE,
};

const VIEW: (f64, f64) = (800.0, 600.0);

fn r(base: u32, lens: u32, overlay: u32) -> Renders {
    Renders {
        base,
        lens,
        overlay,
    }
}

fn start(scale: f64) -> (Interaction, Report) {
    let cam = Camera2D::new((0.0, 0.0), scale, VIEW).expect("camera");
    Interaction::new(HexGrid::map_default(1.0).expect("grid"), cam, Some(1))
}

/// Screen pixel at the centre of cell `id` under the current camera.
fn screen_of(it: &Interaction, id: u32) -> (f64, f64) {
    let grid = HexGrid::map_default(1.0).expect("grid");
    it.camera().screen_at(grid.centre(id).expect("centre"))
}

/// The hash the history must write for `camera`, `selection` and lens 1.
fn hash(camera: Camera2D, selection: Option<EntityId>) -> String {
    encode(&UrlState::new(Some(camera), selection, Some(1))).expect("encodable")
}

#[test]
fn new_paints_every_layer_once_and_writes_no_history() {
    let (it, rep) = start(10.0);
    assert_eq!(rep.renders, r(1, 1, 1));
    assert!(rep.frame_requested);
    assert_eq!(rep.nav, Nav::Nothing);
    assert_eq!(it.selected(), None);
    assert_eq!(it.hovered(), None);
    assert!(it.overlay_on());
}

#[test]
fn hover_renders_only_the_overlay_once_and_writes_no_history() {
    let (mut it, _) = start(10.0);
    let rep = it.hover(screen_of(&it, 5)).expect("hover");
    assert_eq!(rep.renders, r(0, 0, 1));
    assert!(rep.frame_requested);
    assert_eq!(rep.nav, Nav::Nothing);
    assert_eq!(it.hovered(), Some(5));
    assert_eq!(it.selected(), None);
    assert_eq!(it.poll_history(10_000), Ok(Nav::Nothing));
}

#[test]
fn hover_off_the_map_clears_the_hovered_cell() {
    let (mut it, _) = start(10.0);
    it.hover(screen_of(&it, 5)).expect("hover");
    let far = it.camera().screen_at((1e6, 1e6));
    let rep = it.hover(far).expect("hover off map");
    assert_eq!(rep.renders, r(0, 0, 1));
    assert_eq!(it.hovered(), None);
}

#[test]
fn click_selects_renders_the_overlay_once_and_pushes_exactly_one_entry() {
    let (mut it, _) = start(10.0);
    let cam = *it.camera();
    let rep = it.click(screen_of(&it, 5), 100).expect("click");
    assert_eq!(rep.renders, r(0, 0, 1));
    assert!(rep.frame_requested);
    assert_eq!(rep.nav, Nav::Push(hash(cam, Some(5))));
    assert_eq!(it.selected(), Some(5));
    assert_eq!(it.url_state().selection(), Some(5));
}

#[test]
fn selecting_the_same_cell_again_renders_once_and_adds_no_entry() {
    let (mut it, _) = start(10.0);
    it.click(screen_of(&it, 5), 100).expect("first click");
    let rep = it.click(screen_of(&it, 5), 200).expect("second click");
    assert_eq!(rep.renders, r(0, 0, 1));
    assert_eq!(rep.nav, Nav::Nothing);
    assert_eq!(it.selected(), Some(5));
}

#[test]
fn clicking_off_the_map_clears_the_selection_with_one_entry() {
    let (mut it, _) = start(10.0);
    it.click(screen_of(&it, 5), 100).expect("select");
    let cam = *it.camera();
    let far = it.camera().screen_at((1e6, 1e6));
    let rep = it.click(far, 200).expect("click off map");
    assert_eq!(rep.renders, r(0, 0, 1));
    assert_eq!(rep.nav, Nav::Push(hash(cam, None)));
    assert_eq!(it.selected(), None);
}

#[test]
fn layer_toggle_renders_only_the_overlay_once_and_flips_the_flag() {
    let (mut it, _) = start(10.0);
    let off = it.toggle_overlay().expect("toggle");
    assert_eq!(off.renders, r(0, 0, 1));
    assert_eq!(off.nav, Nav::Nothing);
    assert!(!it.overlay_on());
    let on = it.toggle_overlay().expect("toggle back");
    assert_eq!(on.renders, r(0, 0, 1));
    assert!(it.overlay_on());
}

#[test]
fn escape_clears_the_selection_with_one_entry_and_is_quiet_when_nothing_is_selected() {
    let (mut it, _) = start(10.0);
    it.click(screen_of(&it, 5), 100).expect("select");
    let cam = *it.camera();
    let cleared = it.key("Escape", 200).expect("escape");
    assert_eq!(cleared.renders, r(0, 0, 1));
    assert_eq!(cleared.nav, Nav::Push(hash(cam, None)));
    assert_eq!(it.selected(), None);
    let again = it.key("Escape", 300).expect("second escape");
    assert_eq!(again.renders, r(0, 0, 1));
    assert_eq!(again.nav, Nav::Nothing, "nothing selected: no new entry");
    assert_eq!(it.selected(), None);
}

#[test]
fn the_key_map_is_fixed() {
    let cases = [
        ("ArrowLeft", Some(Key::ArrowLeft)),
        ("ArrowRight", Some(Key::ArrowRight)),
        ("ArrowUp", Some(Key::ArrowUp)),
        ("ArrowDown", Some(Key::ArrowDown)),
        ("+", Some(Key::ZoomIn)),
        ("-", Some(Key::ZoomOut)),
        ("Escape", Some(Key::Escape)),
        ("=", None),
        ("arrowleft", None),
        ("a", None),
        ("", None),
    ];
    for (name, want) in cases {
        assert_eq!(Key::from_name(name), want, "binding for {name:?}");
    }
}

#[test]
fn arrow_keys_pan_by_the_step_in_the_stated_direction() {
    let (mut it, _) = start(10.0);
    // One step is PAN_STEP_PX / scale = 5 world units at scale 10. Right and up move the view
    // east and north: the centre's x grows and its y shrinks (screen y grows downward).
    let right = it.key("ArrowRight", 1).expect("right");
    assert_eq!(it.camera().centre(), (5.0, 0.0));
    assert_eq!(right.renders, r(0, 0, 0), "inside the cache margin");
    assert!(right.frame_requested, "a pan moves the view");
    it.key("ArrowLeft", 2).expect("left");
    assert_eq!(it.camera().centre(), (0.0, 0.0));
    it.key("ArrowDown", 3).expect("down");
    assert_eq!(it.camera().centre(), (0.0, 5.0));
    it.key("ArrowUp", 4).expect("up");
    assert_eq!(it.camera().centre(), (0.0, 0.0));
}

#[test]
fn arrow_presses_past_the_cache_margin_render_each_layer_once_then_go_quiet() {
    let (mut it, _) = start(1.0);
    // At scale 1 a press is 50 world units; the cache margin is 160. Presses 1 to 3 reach 150,
    // inside the margin; press 4 reaches 200 and repaints every layer once.
    for press in 1..=3 {
        let rep = it.key("ArrowRight", press).expect("press");
        assert_eq!(rep.renders, r(0, 0, 0), "press {press}");
    }
    let past = it.key("ArrowRight", 4).expect("press 4");
    assert_eq!(past.renders, r(1, 1, 1));
    let after = it.key("ArrowRight", 5).expect("press 5");
    assert_eq!(
        after.renders,
        r(0, 0, 0),
        "the new raster covers the camera"
    );
}

#[test]
fn zoom_keys_double_and_halve_the_scale_about_the_viewport_centre() {
    let (mut it, _) = start(10.0);
    let zin = it.key("+", 1).expect("zoom in");
    assert_eq!(it.camera().scale(), 20.0);
    assert_eq!(
        it.camera().centre(),
        (0.0, 0.0),
        "anchored at the viewport centre"
    );
    assert_eq!(
        zin.renders,
        r(1, 1, 1),
        "scale changed: every layer repaints once"
    );
    let zout = it.key("-", 2).expect("zoom out");
    assert_eq!(it.camera().scale(), 10.0);
    assert_eq!(it.camera().centre(), (0.0, 0.0));
    assert_eq!(zout.renders, r(1, 1, 1));
}

#[test]
fn zoom_is_clamped_at_the_scale_limits_and_then_does_nothing() {
    let (mut top, _) = start(MAX_SCALE);
    let rep = top.key("+", 1).expect("zoom in at max");
    assert_eq!(top.camera().scale(), MAX_SCALE);
    assert_eq!(rep.renders, r(0, 0, 0));
    assert!(!rep.frame_requested);
    assert_eq!(rep.nav, Nav::Nothing);

    let (mut bottom, _) = start(MIN_SCALE);
    let rep = bottom.key("-", 1).expect("zoom out at min");
    assert_eq!(bottom.camera().scale(), MIN_SCALE);
    assert_eq!(rep.renders, r(0, 0, 0));
}

#[test]
fn an_unbound_key_changes_nothing() {
    let (mut it, _) = start(10.0);
    let before = *it.camera();
    let rep = it.key("q", 1).expect("unbound");
    assert_eq!(rep.renders, r(0, 0, 0));
    assert!(!rep.frame_requested);
    assert_eq!(rep.nav, Nav::Nothing);
    assert_eq!(*it.camera(), before);
}

#[test]
fn camera_moves_are_written_as_one_debounced_replace() {
    let (mut it, _) = start(10.0);
    it.key("ArrowRight", 1_000).expect("pan");
    it.key("ArrowRight", 1_010).expect("pan again");
    let cam = *it.camera();
    assert_eq!(
        it.poll_history(1_010 + CAMERA_DEBOUNCE_MS - 1),
        Ok(Nav::Nothing)
    );
    assert_eq!(
        it.poll_history(1_010 + CAMERA_DEBOUNCE_MS),
        Ok(Nav::Replace(hash(cam, None)))
    );
    assert_eq!(
        it.poll_history(10_000),
        Ok(Nav::Nothing),
        "one replace per burst"
    );
}

#[test]
fn each_selection_change_in_a_sequence_adds_exactly_one_entry() {
    let (mut it, _) = start(10.0);
    let mut pushes = 0;
    let steps: [(Option<u32>, u64); 4] = [(Some(5), 1), (Some(6), 2), (Some(6), 3), (None, 4)];
    for (cell, now) in steps {
        let rep = match cell {
            Some(id) => it.click(screen_of(&it, id), now).expect("click"),
            None => it.key("Escape", now).expect("escape"),
        };
        if matches!(rep.nav, Nav::Push(_)) {
            pushes += 1;
        }
    }
    assert_eq!(
        pushes, 3,
        "5, then 6, then clear; the repeated 6 adds nothing"
    );
}
