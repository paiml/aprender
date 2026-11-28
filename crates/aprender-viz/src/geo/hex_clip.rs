//! Convex clipping, shoelace area and hexagon corners for the hex grid (MAP-3).
//!
//! Split out of `hex.rs` to keep that file under the 500-line gate. The code is moved, not changed.

/// The six corners of a hexagon with the given centre and circumradius, counter-clockwise from
/// -30 degrees.
pub(super) fn hexagon_corners(centre: (f64, f64), radius: f64) -> [(f64, f64); 6] {
    std::array::from_fn(|k| {
        let angle = (60.0 * k as f64 - 30.0).to_radians();
        (centre.0 + radius * angle.cos(), centre.1 + radius * angle.sin())
    })
}

/// Shoelace signed area of a ring. Positive for counter-clockwise rings.
///
/// The vertices are taken relative to the first one. Products of coordinates near 1e8 are near
/// 1e16, where f64 has an ulp of 2, so unshifted terms cancel to nothing. Differences of nearby
/// vertices are exact, so the area is too.
pub(super) fn signed_area(ring: &[(f64, f64)]) -> f64 {
    let Some(&(ax, ay)) = ring.first() else {
        return 0.0;
    };
    let n = ring.len();
    let rel = |i: usize| (ring[i % n].0 - ax, ring[i % n].1 - ay);
    let twice: f64 = (0..n)
        .map(|i| {
            let (x0, y0) = rel(i);
            let (x1, y1) = rel(i + 1);
            x0 * y1 - x1 * y0
        })
        .sum();
    twice / 2.0
}

/// Clip a subject ring against a convex counter-clockwise clip polygon.
pub(super) fn clip_to_convex(subject: &[(f64, f64)], clip: &[(f64, f64); 6]) -> Vec<(f64, f64)> {
    let mut poly = subject.to_vec();
    for k in 0..clip.len() {
        if poly.is_empty() {
            break;
        }
        poly = clip_half_plane(&poly, clip[k], clip[(k + 1) % clip.len()]);
    }
    poly
}

/// One Sutherland-Hodgman pass: keep the part of `input` left of the edge `a -> b`.
fn clip_half_plane(input: &[(f64, f64)], a: (f64, f64), b: (f64, f64)) -> Vec<(f64, f64)> {
    let side = |p: (f64, f64)| (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
    let n = input.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let cur = input[i];
        let prev = input[(i + n - 1) % n];
        let (sc, sp) = (side(cur), side(prev));
        if sc >= 0.0 {
            if sp < 0.0 {
                out.push(crossing(prev, cur, sp, sc));
            }
            out.push(cur);
        } else if sp >= 0.0 {
            out.push(crossing(prev, cur, sp, sc));
        }
    }
    out
}

/// The point where segment `p -> q` crosses the edge line, given the signed sides `sp`, `sq`
/// of its ends, which lie strictly on opposite sides.
fn crossing(p: (f64, f64), q: (f64, f64), sp: f64, sq: f64) -> (f64, f64) {
    let t = sp / (sp - sq);
    (p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1))
}
