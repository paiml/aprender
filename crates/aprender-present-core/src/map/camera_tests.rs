//! Tests for the camera, eased zoom and fly-to. FALSIFY-MAP-CAM-001 and -002 live here;
//! -003 is in `lod.rs`, next to the tier function it checks.

use super::*;
use proptest::prelude::*;

const EPS: f64 = 1e-6;
const VIEW: (f64, f64) = (800.0, 600.0);

fn cam(centre: (f64, f64), scale: f64) -> Camera2D {
    Camera2D::new(centre, scale, VIEW).expect("valid camera")
}

/// True when `a` and `b` agree within `EPS` relative to the coordinate magnitude `mag`.
fn near(a: (f64, f64), b: (f64, f64), mag: f64) -> bool {
    let tol = EPS * (1.0 + mag);
    (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
}

fn magnitude(points: &[(f64, f64)]) -> f64 {
    points
        .iter()
        .flat_map(|p| [p.0.abs(), p.1.abs()])
        .fold(0.0, f64::max)
}

#[test]
fn new_accepts_valid_camera_and_exposes_fields() {
    let c = cam((3.0, -4.0), 2.5);
    assert_eq!(c.centre(), (3.0, -4.0));
    assert_eq!(c.scale(), 2.5);
    assert_eq!(c.viewport(), VIEW);
    assert_eq!(c.screen_centre(), (400.0, 300.0));
}

#[test]
fn new_rejects_non_finite_and_non_positive_scale() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            Camera2D::new((0.0, 0.0), bad, VIEW),
            Err(MapError::NonFinite("scale"))
        );
    }
    assert_eq!(
        Camera2D::new((0.0, 0.0), 0.0, VIEW),
        Err(MapError::NonPositive("scale"))
    );
    assert_eq!(
        Camera2D::new((0.0, 0.0), -2.0, VIEW),
        Err(MapError::NonPositive("scale"))
    );
}

#[test]
fn new_rejects_non_finite_centre_and_bad_viewport() {
    assert_eq!(
        Camera2D::new((f64::NAN, 0.0), 1.0, VIEW),
        Err(MapError::NonFinite("centre"))
    );
    assert_eq!(
        Camera2D::new((0.0, 0.0), 1.0, (f64::NAN, 1.0)),
        Err(MapError::NonFinite("viewport"))
    );
    assert_eq!(
        Camera2D::new((0.0, 0.0), 1.0, (-1.0, 1.0)),
        Err(MapError::Negative("viewport"))
    );
}

#[test]
fn screen_and_world_mappings_are_inverse_and_centre_maps_to_centre() {
    let c = cam((10.0, -20.0), 4.0);
    assert_eq!(c.world_at(c.screen_centre()), (10.0, -20.0));
    for p in [(0.0, 0.0), (800.0, 600.0), (123.5, 456.25)] {
        let back = c.screen_at(c.world_at(p));
        assert!(near(back, p, magnitude(&[p])), "round trip of {p:?}");
    }
    assert_eq!(c.screen_at((10.0, -20.0)), c.screen_centre());
}

#[test]
fn pan_moves_content_with_the_pointer() {
    let mut c = cam((0.0, 0.0), 2.0);
    let p = (100.0, 100.0);
    let before = c.world_at(p);
    c.pan(30.0, -20.0).expect("finite pan");
    // Dragging by (30, -20) px moves the centre by (-15, +10) world units.
    let after = c.world_at(p);
    let want = (before.0 - 15.0, before.1 + 10.0);
    assert!(
        near(after, want, magnitude(&[before])),
        "{after:?} vs {want:?}"
    );
}

#[test]
fn pan_rejects_non_finite_offset_and_keeps_the_camera() {
    let mut c = cam((1.0, 2.0), 3.0);
    let before = c;
    assert_eq!(c.pan(f64::NAN, 0.0), Err(MapError::NonFinite("pan")));
    assert_eq!(c.pan(0.0, f64::INFINITY), Err(MapError::NonFinite("pan")));
    assert_eq!(c, before);
}

#[test]
fn zoom_about_rejects_bad_inputs_and_leaves_camera_unchanged() {
    let mut c = cam((0.0, 0.0), 1.0);
    let before = c;
    assert_eq!(
        c.zoom_about((10.0, 10.0), 0.0),
        Err(MapError::NonPositive("scale"))
    );
    assert_eq!(
        c.zoom_about((f64::NAN, 10.0), 2.0),
        Err(MapError::NonFinite("anchor"))
    );
    assert_eq!(
        c.zoom_about((800.0, 300.0), f64::from_bits(1)),
        Err(MapError::OutOfRange("scale"))
    );
    assert_eq!(c, before);
    // Zooming out about the left edge of a camera at the domain edge pushes the centre past it.
    let mut edge = cam((1e12, 0.0), 1.0);
    assert_eq!(
        edge.zoom_about((0.0, 0.0), 0.5),
        Err(MapError::OutOfRange("centre"))
    );
}

#[test]
fn zoom_about_screen_centre_only_changes_scale() {
    let mut c = cam((5.0, 6.0), 1.0);
    c.zoom_about(c.screen_centre(), 4.0).expect("valid zoom");
    assert_eq!(c.centre(), (5.0, 6.0));
    assert_eq!(c.scale(), 4.0);
}

proptest! {
    #[test]
    fn falsify_map_cam_001_anchored_zoom_keeps_anchor_world_point(
        cx in -1e4f64..1e4, cy in -1e4f64..1e4,
        scale in 1e-2f64..1e2,
        vw in 1.0f64..2000.0, vh in 1.0f64..2000.0,
        ax in 0.0f64..1.0, ay in 0.0f64..1.0,
        new_scale in 1e-2f64..1e2,
    ) {
        let mut c = Camera2D::new((cx, cy), scale, (vw, vh)).expect("valid camera");
        // Anchors anywhere in the viewport, including far from the centre.
        let anchor = (ax * vw, ay * vh);
        let before = c.world_at(anchor);
        c.zoom_about(anchor, new_scale).expect("valid zoom");
        let after = c.world_at(anchor);
        prop_assert!(near(before, after, magnitude(&[before, after, (cx, cy)])),
            "anchor world point moved: {before:?} -> {after:?}");
        prop_assert_eq!(c.scale(), new_scale);
    }
}

#[test]
fn zoom_anim_ends_exactly_on_target_and_stops_after_finishing() {
    let mut c = cam((0.0, 0.0), 2.0);
    let anchor = (600.0, 150.0);
    let mut anim = ZoomAnim::start(&c, anchor, 8.0, 0.5).expect("valid anim");
    assert!(!anim.is_finished());
    let mut moving_steps = 0;
    while anim.step(&mut c, 0.1).expect("valid dt") {
        moving_steps += 1;
        assert!(moving_steps < 100, "animation did not finish");
    }
    assert!(anim.is_finished());
    assert_eq!(c.scale(), 8.0, "last step sets the target exactly");
    let settled = c;
    assert_eq!(anim.step(&mut c, 1.0), Ok(false));
    assert_eq!(c, settled, "a finished animation does nothing");
}

#[test]
fn zoom_anim_scale_at_starts_at_from_and_is_monotone() {
    let c = cam((0.0, 0.0), 1.0);
    let anim = ZoomAnim::start(&c, (0.0, 0.0), 16.0, 1.0).expect("valid anim");
    assert_eq!(anim.scale_at(0.0), 1.0);
    let mut prev = 1.0;
    for i in 1..=20 {
        let s = anim.scale_at(f64::from(i) / 20.0);
        assert!(s >= prev, "scale fell at step {i}");
        prev = s;
    }
    assert!((prev - 16.0).abs() < 1e-9);
}

#[test]
fn zoom_anim_start_rejects_bad_inputs() {
    let c = cam((0.0, 0.0), 1.0);
    assert_eq!(
        ZoomAnim::start(&c, (f64::NAN, 0.0), 2.0, 1.0),
        Err(MapError::NonFinite("anchor"))
    );
    assert_eq!(
        ZoomAnim::start(&c, (0.0, 0.0), 0.0, 1.0),
        Err(MapError::NonPositive("scale"))
    );
    assert_eq!(
        ZoomAnim::start(&c, (0.0, 0.0), 2.0, 0.0),
        Err(MapError::NonPositive("duration"))
    );
    assert_eq!(
        ZoomAnim::start(&c, (0.0, 0.0), 2.0, f64::NAN),
        Err(MapError::NonFinite("duration"))
    );
}

#[test]
fn zoom_anim_step_rejects_bad_dt_and_leaves_camera_unchanged() {
    let mut c = cam((0.0, 0.0), 1.0);
    let mut anim = ZoomAnim::start(&c, (10.0, 10.0), 4.0, 1.0).expect("valid anim");
    let before = c;
    assert_eq!(anim.step(&mut c, -0.1), Err(MapError::Negative("dt")));
    assert_eq!(anim.step(&mut c, f64::NAN), Err(MapError::NonFinite("dt")));
    assert_eq!(c, before);
    assert!(!anim.is_finished());
}

proptest! {
    #[test]
    fn zoom_anim_keeps_anchor_world_point_at_every_step(
        cx in -1e3f64..1e3, cy in -1e3f64..1e3,
        scale in 1e-1f64..1e1,
        ax in 0.0f64..1.0, ay in 0.0f64..1.0,
        to in 1e-1f64..1e1,
        duration in 0.05f64..2.0,
    ) {
        let mut c = Camera2D::new((cx, cy), scale, VIEW).expect("valid camera");
        let anchor = (ax * VIEW.0, ay * VIEW.1);
        let w0 = c.world_at(anchor);
        let mut anim = ZoomAnim::start(&c, anchor, to, duration).expect("valid anim");
        let mut steps = 0;
        loop {
            let more = anim.step(&mut c, 0.01).expect("valid dt");
            let w = c.world_at(anchor);
            prop_assert!(near(w0, w, magnitude(&[w0, w, (cx, cy)])),
                "anchor drifted at step {steps}: {w0:?} -> {w:?}");
            steps += 1;
            if !more {
                break;
            }
            prop_assert!(steps < 10_000);
        }
        prop_assert_eq!(c.scale(), to);
    }
}

fn fixture_start() -> Camera2D {
    cam((0.0, 0.0), 1.0)
}

fn fixture_target() -> Camera2D {
    cam((120.5, -40.25), 16.0)
}

#[test]
fn falsify_map_cam_002_fly_to_starts_at_camera_and_ends_at_target() {
    let fly = FlyTo::new(fixture_start(), fixture_target()).expect("same viewport");
    // p(0) is the start camera exactly.
    assert_eq!(fly.at(0.0), Ok(fixture_start()));
    // p(1) matches the target centre and scale within EPS.
    let end = fly.at(1.0).expect("t in range");
    assert!(near(end.centre(), (120.5, -40.25), 120.5));
    assert!((end.scale() - 16.0).abs() <= EPS * 17.0);
    // Midpoint: linear centre, geometric scale (sqrt of 16 is 4).
    let mid = fly.at(0.5).expect("t in range");
    assert!(near(mid.centre(), (60.25, -20.125), 60.25));
    assert!((mid.scale() - 4.0).abs() <= EPS * 5.0);
}

proptest! {
    #[test]
    fn falsify_map_cam_002_fly_to_random_endpoints(
        sx in -1e3f64..1e3, sy in -1e3f64..1e3, ss in 1e-2f64..1e2,
        tx in -1e3f64..1e3, ty in -1e3f64..1e3, ts in 1e-2f64..1e2,
        t in 0.0f64..1.0,
    ) {
        let start = cam((sx, sy), ss);
        let target = cam((tx, ty), ts);
        let fly = FlyTo::new(start, target).expect("same viewport");
        prop_assert_eq!(fly.at(0.0), Ok(start));
        let end = fly.at(1.0).expect("t in range");
        prop_assert!(near(end.centre(), (tx, ty), magnitude(&[(tx, ty)])));
        prop_assert!((end.scale() - ts).abs() <= EPS * (1.0 + ts));
        let mid = fly.at(t).expect("t in range");
        prop_assert!(mid.scale().is_finite() && mid.scale() > 0.0);
    }
}

#[test]
fn fly_to_clamps_t_and_rejects_non_finite_t() {
    let fly = FlyTo::new(fixture_start(), fixture_target()).expect("same viewport");
    assert_eq!(fly.at(-1.0), Ok(fixture_start()));
    assert_eq!(fly.at(2.0), Ok(fixture_target()));
    assert_eq!(fly.at(f64::NAN), Err(MapError::NonFinite("t")));
    assert_eq!(fly.start(), fixture_start());
    assert_eq!(fly.target(), fixture_target());
}

#[test]
fn fly_to_rejects_viewport_mismatch() {
    let other = Camera2D::new((0.0, 0.0), 1.0, (100.0, 100.0)).expect("valid camera");
    assert_eq!(
        FlyTo::new(fixture_start(), other),
        Err(MapError::ViewportMismatch)
    );
}

#[test]
fn values_outside_the_numeric_domain_are_rejected() {
    let view = (800.0, 600.0);
    assert!(Camera2D::new((1e308, 0.0), 1.0, view).is_err());
    assert!(Camera2D::new((0.0, 0.0), 1e-320, view).is_err());
    assert!(Camera2D::new((0.0, 0.0), 1e200, view).is_err());
    assert!(Camera2D::new((0.0, 0.0), 1.0, (1e308, 100.0)).is_err());
    let mut camera = Camera2D::new((0.0, 0.0), 1.0, view).expect("valid camera");
    let before = camera;
    assert!(camera.pan(1e308, 0.0).is_err());
    assert!(camera.zoom_about((0.0, 0.0), 1e200).is_err());
    assert_eq!(camera, before);
}

#[test]
fn a_rejected_animation_step_does_not_advance_the_animation() {
    // A camera at the edge of the domain: zooming out about a corner pushes the centre past it.
    let mut camera = Camera2D::new((1e12, 0.0), 1.0, (800.0, 600.0)).expect("valid camera");
    let mut anim = ZoomAnim::start(&camera, (0.0, 0.0), 0.5, 1.0).expect("valid animation");
    let (anim_before, camera_before) = (anim, camera);
    assert!(anim.step(&mut camera, 0.5).is_err());
    assert_eq!(anim, anim_before);
    assert_eq!(camera, camera_before);
}
