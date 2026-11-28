//! Culling for the map runtime (MAP-5): indices of world-space boxes that intersect the viewport
//! grown by a pixel margin.

use super::{camera::Camera2D, check_non_negative, check_point, MapError};

/// Axis-aligned bounding box in world units. Invariant: `min <= max` on both axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    min: (f64, f64),
    max: (f64, f64),
}

impl Bounds {
    /// Builds a box. Rejects non-finite corners and a minimum above the maximum.
    pub fn new(min: (f64, f64), max: (f64, f64)) -> Result<Self, MapError> {
        check_point(min, "bounds")?;
        check_point(max, "bounds")?;
        if min.0 > max.0 || min.1 > max.1 {
            return Err(MapError::InvertedBounds);
        }
        Ok(Self { min, max })
    }

    /// Minimum corner.
    pub fn min(&self) -> (f64, f64) {
        self.min
    }

    /// Maximum corner.
    pub fn max(&self) -> (f64, f64) {
        self.max
    }
}

/// Indices (ascending) of the items whose box intersects the viewport grown by `margin_px` on
/// every side. Rejects a non-finite or negative margin.
pub fn visible_indices(
    camera: &Camera2D,
    items: &[Bounds],
    margin_px: f64,
) -> Result<Vec<usize>, MapError> {
    check_non_negative(margin_px, "margin")?;
    let (vw, vh) = camera.viewport();
    let lo = camera.world_at((-margin_px, -margin_px));
    let hi = camera.world_at((vw + margin_px, vh + margin_px));
    Ok(items
        .iter()
        .enumerate()
        .filter(|(_, b)| b.max.0 >= lo.0 && b.min.0 <= hi.0 && b.max.1 >= lo.1 && b.min.1 <= hi.1)
        .map(|(i, _)| i)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn camera(centre: (f64, f64), scale: f64, viewport: (f64, f64)) -> Camera2D {
        Camera2D::new(centre, scale, viewport).expect("valid camera")
    }

    /// Reference implementation: project each box to screen space and test overlap with the
    /// margin-grown viewport. Shares no arithmetic with `visible_indices`.
    fn brute_force(camera: &Camera2D, items: &[Bounds], margin: f64) -> Vec<usize> {
        let (vw, vh) = camera.viewport();
        let mut out = Vec::new();
        for (i, b) in items.iter().enumerate() {
            let a = camera.screen_at(b.min());
            let z = camera.screen_at(b.max());
            let (x0, x1) = (a.0.min(z.0), a.0.max(z.0));
            let (y0, y1) = (a.1.min(z.1), a.1.max(z.1));
            if x1 >= -margin && x0 <= vw + margin && y1 >= -margin && y0 <= vh + margin {
                out.push(i);
            }
        }
        out
    }

    #[test]
    fn bounds_new_rejects_inverted_and_non_finite_corners() {
        assert_eq!(
            Bounds::new((2.0, 0.0), (1.0, 1.0)),
            Err(MapError::InvertedBounds)
        );
        assert_eq!(
            Bounds::new((0.0, 2.0), (1.0, 1.0)),
            Err(MapError::InvertedBounds)
        );
        assert_eq!(
            Bounds::new((f64::NAN, 0.0), (1.0, 1.0)),
            Err(MapError::NonFinite("bounds"))
        );
        let b = Bounds::new((0.0, 0.0), (1.0, 2.0)).expect("valid box");
        assert_eq!(b.min(), (0.0, 0.0));
        assert_eq!(b.max(), (1.0, 2.0));
    }

    #[test]
    fn visible_indices_selects_boxes_in_view_and_margin() {
        let cam = camera((0.0, 0.0), 10.0, (100.0, 100.0));
        // View spans world [-5, 5]. With a 20 px margin (2 world units) it spans [-7, 7].
        let items = [
            Bounds::new((-1.0, -1.0), (1.0, 1.0)).expect("box"), // inside
            Bounds::new((20.0, 20.0), (21.0, 21.0)).expect("box"), // far away
            Bounds::new((6.0, 0.0), (6.5, 0.5)).expect("box"),   // only in margin
        ];
        assert_eq!(visible_indices(&cam, &items, 0.0), Ok(vec![0]));
        assert_eq!(visible_indices(&cam, &items, 20.0), Ok(vec![0, 2]));
    }

    #[test]
    fn visible_indices_rejects_bad_margin_and_handles_empty_input() {
        let cam = camera((0.0, 0.0), 1.0, (10.0, 10.0));
        assert_eq!(visible_indices(&cam, &[], 0.0), Ok(vec![]));
        assert_eq!(
            visible_indices(&cam, &[], -1.0),
            Err(MapError::Negative("margin"))
        );
        assert_eq!(
            visible_indices(&cam, &[], f64::NAN),
            Err(MapError::NonFinite("margin"))
        );
    }

    proptest! {
        #[test]
        fn cull_matches_brute_force_filter(
            cx in -1e3f64..1e3, cy in -1e3f64..1e3,
            scale in 1e-2f64..1e2,
            vw in 1.0f64..1500.0, vh in 1.0f64..1500.0,
            margin in 0.0f64..400.0,
            raw in proptest::collection::vec(
                (-2e3f64..2e3, -2e3f64..2e3, 0.0f64..300.0, 0.0f64..300.0), 0..40),
        ) {
            let cam = camera((cx, cy), scale, (vw, vh));
            let items: Vec<Bounds> = raw
                .iter()
                .map(|&(x, y, w, h)| Bounds::new((x, y), (x + w, y + h)).expect("box"))
                .collect();
            let got = visible_indices(&cam, &items, margin).expect("valid margin");
            prop_assert_eq!(got, brute_force(&cam, &items, margin));
        }
    }
}
