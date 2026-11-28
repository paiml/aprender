//! Level-of-detail tier from a hex radius in screen pixels (MAP-5, contract
//! `aprender-viz-map-camera-v1`, `level_of_detail`).
//!
//! Tiers: Far for `r < 10`, Mid for `10 <= r < 22`, Near for `22 <= r < 44`, Close for `r >= 44`.

use super::{check_finite, MapError};

/// Level-of-detail boundaries in screen pixels: `[Far|Mid, Mid|Near, Near|Close]`.
pub const BOUNDARIES_PX: [f64; 3] = [10.0, 22.0, 44.0];

/// Detail tier, ordered from coarsest to finest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LodTier {
    /// Radius below 10 px.
    Far,
    /// Radius from 10 px up to 22 px.
    Mid,
    /// Radius from 22 px up to 44 px.
    Near,
    /// Radius of 44 px or more.
    Close,
}

/// Tier for a hex radius in screen pixels. Rejects non-finite and non-positive radii.
pub fn tier_for_radius(radius_px: f64) -> Result<LodTier, MapError> {
    check_finite(radius_px, "radius")?;
    if radius_px <= 0.0 {
        return Err(MapError::NonPositive("radius"));
    }
    let [b1, b2, b3] = BOUNDARIES_PX;
    Ok(if radius_px < b1 {
        LodTier::Far
    } else if radius_px < b2 {
        LodTier::Mid
    } else if radius_px < b3 {
        LodTier::Near
    } else {
        LodTier::Close
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DELTA: f64 = 1e-9;

    #[test]
    fn tier_for_radius_selects_the_stated_tier_in_each_range() {
        let cases = [
            (0.5, LodTier::Far),
            (9.999, LodTier::Far),
            (10.0, LodTier::Mid),
            (21.999, LodTier::Mid),
            (22.0, LodTier::Near),
            (43.999, LodTier::Near),
            (44.0, LodTier::Close),
            (500.0, LodTier::Close),
        ];
        for (r, want) in cases {
            assert_eq!(tier_for_radius(r), Ok(want), "radius {r}");
        }
    }

    #[test]
    fn falsify_map_cam_003_lod_boundaries_select_stated_tiers() {
        // (boundary, tier just below, tier at, tier just above), per MAP-5 row.
        let table = [
            (10.0, LodTier::Far, LodTier::Mid, LodTier::Mid),
            (22.0, LodTier::Mid, LodTier::Near, LodTier::Near),
            (44.0, LodTier::Near, LodTier::Close, LodTier::Close),
        ];
        for (b, below, at, above) in table {
            assert_eq!(tier_for_radius(b - DELTA), Ok(below), "below {b}");
            assert_eq!(tier_for_radius(b), Ok(at), "at {b}");
            assert_eq!(tier_for_radius(b + DELTA), Ok(above), "above {b}");
        }
    }

    #[test]
    fn falsify_map_cam_003_tier_is_non_decreasing_over_the_range() {
        let mut prev = LodTier::Far;
        let mut r = 0.25;
        while r <= 200.0 {
            let tier = tier_for_radius(r).expect("positive radius");
            assert!(tier >= prev, "tier fell at r={r}");
            prev = tier;
            r += 0.25;
        }
        assert_eq!(prev, LodTier::Close);
    }

    #[test]
    fn tier_for_radius_rejects_out_of_domain_radius() {
        assert_eq!(tier_for_radius(0.0), Err(MapError::NonPositive("radius")));
        assert_eq!(tier_for_radius(-3.0), Err(MapError::NonPositive("radius")));
        assert_eq!(
            tier_for_radius(f64::NAN),
            Err(MapError::NonFinite("radius"))
        );
        assert_eq!(
            tier_for_radius(f64::INFINITY),
            Err(MapError::NonFinite("radius"))
        );
    }
}
