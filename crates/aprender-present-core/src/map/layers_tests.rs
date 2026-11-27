//! Tests for the layer-cache scheduler. One per FALSIFY-MAP-LC id in
//! `contracts/aprender-viz-map-layer-cache-v1.yaml`, plus error paths and accessors.

use super::*;

const VIEW: (f64, f64) = (800.0, 600.0);

fn cam(scale: f64) -> Camera2D {
    Camera2D::new((0.0, 0.0), scale, VIEW).expect("valid camera")
}

fn with_lens(lens: Option<LensId>) -> Layers {
    Layers::new(cam(10.0), lens).0
}

fn step(layers: &mut Layers, event: Event) -> Outcome {
    layers.handle(event).expect("valid event")
}

fn r(base: u32, lens: u32, overlay: u32) -> Renders {
    Renders {
        base,
        lens,
        overlay,
    }
}

#[test]
fn initial_paint_renders_every_layer_once() {
    let (layers, out) = Layers::new(cam(10.0), Some(1));
    assert_eq!(out.renders, r(1, 1, 1));
    assert!(out.frame_requested);
    assert_eq!(layers.active_lens(), Some(1));
    assert!(!layers.is_moving());
}

#[test]
fn initial_paint_without_lens_skips_the_lens_layer() {
    let (_, out) = Layers::new(cam(10.0), None);
    assert_eq!(out.renders, r(1, 0, 1));
}

#[test]
fn falsify_map_lc_001_idle_frame_loop_does_no_work() {
    let mut layers = with_lens(Some(1));
    let mut frames = 0;
    let mut renders = Renders::default();
    // One second at 60 Hz with no input: every idle tick must request no frame and render nothing.
    for _ in 0..60 {
        let out = step(&mut layers, Event::Idle);
        frames += u32::from(out.frame_requested);
        renders = Renders {
            base: renders.base + out.renders.base,
            lens: renders.lens + out.renders.lens,
            overlay: renders.overlay + out.renders.overlay,
        };
    }
    assert_eq!(frames, 0, "frame callbacks over one idle second");
    assert_eq!(renders, r(0, 0, 0));
}

#[test]
fn falsify_map_lc_002_pan_inside_margin_renders_nothing_past_it_renders_once() {
    let mut layers = with_lens(Some(1));
    // 100 px and 50 px: the camera moves 10 and 5 world units, inside the 160 px margin.
    let inside = step(
        &mut layers,
        Event::Pan {
            dx_px: 100.0,
            dy_px: 50.0,
        },
    );
    assert_eq!(inside.renders, r(0, 0, 0));
    assert!(inside.frame_requested, "a pan moves the view");

    // A further 200 px takes the total to 300 px, past the margin: each layer renders once.
    let past = step(
        &mut layers,
        Event::Pan {
            dx_px: 200.0,
            dy_px: 0.0,
        },
    );
    assert_eq!(past.renders, r(1, 1, 1));
    assert!(past.frame_requested);

    // The new raster covers the camera again, so a small pan is quiet.
    let after = step(
        &mut layers,
        Event::Pan {
            dx_px: 10.0,
            dy_px: 0.0,
        },
    );
    assert_eq!(after.renders, r(0, 0, 0));
}

#[test]
fn falsify_map_lc_003_zoom_renders_nothing_while_moving_and_each_layer_once_on_settle() {
    let mut layers = with_lens(Some(1));
    let anchor = (300.0, 200.0);
    for scale in [12.0, 14.0, 16.0] {
        let out = step(&mut layers, Event::ZoomMotion { anchor, scale });
        assert_eq!(out.renders, r(0, 0, 0), "during zoom at scale {scale}");
        assert!(out.frame_requested);
        assert!(layers.is_moving());
    }
    let settle = step(&mut layers, Event::ZoomSettle);
    assert_eq!(settle.renders, r(1, 1, 1));
    assert!(!layers.is_moving());
}

#[test]
fn falsify_map_lc_003_settle_after_idle_prepaint_gives_base_zero() {
    let mut layers = with_lens(Some(1));
    let anchor = (300.0, 200.0);
    let pre = step(
        &mut layers,
        Event::IdlePrepaint {
            anchor,
            next_scale: 20.0,
        },
    );
    assert_eq!(pre.renders, r(1, 0, 0));
    assert!(!pre.frame_requested, "pre-paint is off-screen work");

    let motion = step(
        &mut layers,
        Event::ZoomMotion {
            anchor,
            scale: 20.0,
        },
    );
    assert_eq!(motion.renders, r(0, 0, 0));

    let settle = step(&mut layers, Event::ZoomSettle);
    assert_eq!(
        settle.renders.base, 0,
        "base was pre-painted for this camera"
    );
    // Lens and overlay are not pre-painted; this implementation renders them once on settle.
    assert_eq!(settle.renders.lens, 1);
    assert_eq!(settle.renders.overlay, 1);
}

#[test]
fn falsify_map_lc_004_lens_switch_and_active_delta_render_the_lens_layer_only() {
    let mut layers = with_lens(Some(1));
    assert_eq!(step(&mut layers, Event::LensSwitch(2)).renders, r(0, 1, 0));
    assert_eq!(layers.active_lens(), Some(2));
    // One message that would change many cells.
    assert_eq!(step(&mut layers, Event::LensDelta(2)).renders, r(0, 1, 0));
    // One message that changes a single cell.
    assert_eq!(step(&mut layers, Event::LensDelta(2)).renders, r(0, 1, 0));
}

#[test]
fn falsify_map_lc_005_hover_selection_and_toggle_render_the_overlay_only() {
    let mut layers = with_lens(Some(1));
    let hover = step(&mut layers, Event::Overlay(OverlayChange::Hover(Some(7))));
    assert_eq!(hover.renders, r(0, 0, 1));
    assert_eq!(layers.hovered(), Some(7));

    let select = step(&mut layers, Event::Overlay(OverlayChange::Select(Some(7))));
    assert_eq!(select.renders, r(0, 0, 1));
    assert_eq!(layers.selected(), Some(7));

    let toggle = step(&mut layers, Event::Overlay(OverlayChange::LayerToggle));
    assert_eq!(toggle.renders, r(0, 0, 1));
    assert!(!layers.overlay_on());
}

#[test]
fn falsify_map_lc_006_delta_to_inactive_lens_renders_nothing_until_selected() {
    let mut layers = with_lens(Some(1));
    let delta = step(&mut layers, Event::LensDelta(2));
    assert_eq!(delta.renders, r(0, 0, 0));
    assert!(!delta.frame_requested);
    assert_eq!(layers.pending_deltas(2), 1);

    // Selecting lens 2 renders it once and applies the stored delta.
    let switch = step(&mut layers, Event::LensSwitch(2));
    assert_eq!(switch.renders, r(0, 1, 0));
    assert_eq!(layers.pending_deltas(2), 0);
}

#[test]
fn pan_rejects_non_finite_offset_and_keeps_state() {
    let mut layers = with_lens(Some(1));
    let before = *layers.camera();
    let err = layers.handle(Event::Pan {
        dx_px: f64::NAN,
        dy_px: 0.0,
    });
    assert_eq!(err, Err(MapError::NonFinite("pan")));
    assert_eq!(*layers.camera(), before);
}

#[test]
fn zoom_motion_rejects_non_positive_scale_and_does_not_start_moving() {
    let mut layers = with_lens(Some(1));
    let err = layers.handle(Event::ZoomMotion {
        anchor: (0.0, 0.0),
        scale: 0.0,
    });
    assert_eq!(err, Err(MapError::NonPositive("scale")));
    assert!(!layers.is_moving());
}

#[test]
fn prepaint_rejects_bad_anchor_and_is_quiet_while_moving() {
    let mut layers = with_lens(Some(1));
    let err = layers.handle(Event::IdlePrepaint {
        anchor: (f64::INFINITY, 0.0),
        next_scale: 20.0,
    });
    assert_eq!(err, Err(MapError::NonFinite("anchor")));

    step(
        &mut layers,
        Event::ZoomMotion {
            anchor: (0.0, 0.0),
            scale: 12.0,
        },
    );
    let out = step(
        &mut layers,
        Event::IdlePrepaint {
            anchor: (0.0, 0.0),
            next_scale: 14.0,
        },
    );
    assert_eq!(out.renders, r(0, 0, 0));
    assert!(!out.frame_requested);
}

#[test]
fn idle_prepaint_with_covered_camera_renders_nothing() {
    let mut layers = with_lens(Some(1));
    let out = step(
        &mut layers,
        Event::IdlePrepaint {
            anchor: (400.0, 300.0),
            next_scale: 10.0,
        },
    );
    assert_eq!(out.renders, r(0, 0, 0));
}

#[test]
fn moving_defers_an_overlay_change_until_settle() {
    let mut layers = with_lens(Some(1));
    step(
        &mut layers,
        Event::ZoomMotion {
            anchor: (0.0, 0.0),
            scale: 12.0,
        },
    );
    let hover = step(&mut layers, Event::Overlay(OverlayChange::Hover(Some(3))));
    assert_eq!(hover.renders, r(0, 0, 0));
    assert_eq!(step(&mut layers, Event::ZoomSettle).renders, r(1, 1, 1));
}

#[test]
fn lens_switch_to_the_active_lens_is_a_no_op() {
    let mut layers = with_lens(Some(4));
    let out = step(&mut layers, Event::LensSwitch(4));
    assert_eq!(out.renders, r(0, 0, 0));
    assert!(!out.frame_requested);
}

#[test]
fn pending_deltas_accumulate_per_inactive_lens() {
    let mut layers = with_lens(Some(1));
    step(&mut layers, Event::LensDelta(2));
    step(&mut layers, Event::LensDelta(2));
    step(&mut layers, Event::LensDelta(3));
    assert_eq!(layers.pending_deltas(2), 2);
    assert_eq!(layers.pending_deltas(3), 1);
    assert_eq!(layers.pending_deltas(9), 0);
}

#[test]
fn lens_delta_without_active_lens_is_stored() {
    let (mut layers, _) = Layers::new(cam(10.0), None);
    let out = step(&mut layers, Event::LensDelta(5));
    assert_eq!(out.renders, r(0, 0, 0));
    assert_eq!(layers.pending_deltas(5), 1);
}

#[test]
fn hover_on_the_same_entity_still_repaints_the_overlay_once() {
    let mut layers = with_lens(Some(1));
    step(&mut layers, Event::Overlay(OverlayChange::Hover(Some(7))));
    let again = step(&mut layers, Event::Overlay(OverlayChange::Hover(Some(7))));
    assert_eq!(again.renders, r(0, 0, 1));
    let clear = step(&mut layers, Event::Overlay(OverlayChange::Hover(None)));
    assert_eq!(clear.renders, r(0, 0, 1));
    assert_eq!(layers.hovered(), None);
}

#[test]
fn pan_inside_margin_after_zoom_settle_stays_quiet() {
    let mut layers = with_lens(Some(1));
    let anchor = (400.0, 300.0);
    step(
        &mut layers,
        Event::ZoomMotion {
            anchor,
            scale: 20.0,
        },
    );
    assert_eq!(step(&mut layers, Event::ZoomSettle).renders, r(1, 1, 1));
    let out = step(
        &mut layers,
        Event::Pan {
            dx_px: 30.0,
            dy_px: -30.0,
        },
    );
    assert_eq!(out.renders, r(0, 0, 0));
}

#[test]
fn zero_pan_requests_no_frame() {
    let mut layers = with_lens(Some(1));
    let out = step(
        &mut layers,
        Event::Pan {
            dx_px: 0.0,
            dy_px: 0.0,
        },
    );
    assert_eq!(out.renders, r(0, 0, 0));
    assert!(!out.frame_requested);
}

#[test]
fn renders_total_sums_the_three_layers() {
    assert_eq!(r(1, 1, 1).total(), 3);
    assert_eq!(Renders::default().total(), 0);
}

#[test]
fn cache_margin_constant_is_160_px() {
    assert_eq!(CACHE_MARGIN_PX, 160.0);
}

#[test]
fn falsify_map_lc_002_prepaint_after_a_prepainted_settle_keeps_current_coverage() {
    let mut layers = with_lens(Some(1));
    let anchor = (300.0, 200.0);
    step(
        &mut layers,
        Event::IdlePrepaint {
            anchor,
            next_scale: 20.0,
        },
    );
    step(
        &mut layers,
        Event::ZoomMotion {
            anchor,
            scale: 20.0,
        },
    );
    step(&mut layers, Event::ZoomSettle);
    step(
        &mut layers,
        Event::IdlePrepaint {
            anchor,
            next_scale: 40.0,
        },
    );
    let pan = step(
        &mut layers,
        Event::Pan {
            dx_px: 1.0,
            dy_px: 0.0,
        },
    );
    assert_eq!(pan.renders, r(0, 0, 0), "the settled view is still cached");
}
