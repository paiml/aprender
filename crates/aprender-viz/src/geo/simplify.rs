//! Polyline simplification: Douglas-Peucker and Visvalingam-Whyatt (`geo` feature).
//!
//! Both algorithms return a subsequence of their input: the end points are always kept,
//! no point is added or moved, and the input order is preserved. The tolerance is a
//! distance for [`douglas_peucker`] and a triangle area for [`visvalingam`].
//!
//! Invalid input is an error, never a number: a negative or non-finite tolerance, or a
//! point with a non-finite coordinate, is a [`GeoError`].

use super::projection::GeoError;

/// Simplifies a polyline with the Douglas-Peucker algorithm.
///
/// Keeps both end points, then repeatedly keeps the interior point farthest from the chord
/// between two kept points while that distance is greater than `tolerance`. The distance is
/// the perpendicular distance to the infinite line through the chord; when the chord has zero
/// length it is the distance to that single point. The search uses an explicit stack, so no
/// recursion depth grows with the input.
///
/// A `tolerance` of `0.0` returns the input unchanged, collinear points included. Fewer than
/// three points are returned unchanged.
///
/// # Errors
///
/// Returns [`GeoError::InvalidParameter`] for a negative tolerance, and
/// [`GeoError::NonFiniteInput`] for a non-finite tolerance or point coordinate.
pub fn douglas_peucker(points: &[(f64, f64)], tolerance: f64) -> Result<Vec<(f64, f64)>, GeoError> {
    validate(points, tolerance)?;
    if tolerance == 0.0 || points.len() < 3 {
        return Ok(points.to_vec());
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut chords = vec![(0, points.len() - 1)];
    while let Some((first, last)) = chords.pop() {
        if let Some(split) = farthest_beyond(points, first, last, tolerance) {
            keep[split] = true;
            chords.push((first, split));
            chords.push((split, last));
        }
    }
    Ok(select(points, &keep))
}

/// Simplifies a polyline with the Visvalingam-Whyatt algorithm.
///
/// The effective area of an interior point is the area of the triangle it forms with its two
/// current neighbours. While the smallest such area is strictly less than `tolerance`, that
/// point is removed and the areas of its neighbours are recomputed. Ties go to the earlier
/// point. The search rescans the polyline after each removal, so the cost is quadratic in the
/// number of points.
///
/// A `tolerance` of `0.0` removes nothing, so the input is returned unchanged, collinear
/// points included (their triangles have area zero, which is not below zero). Fewer than
/// three points are returned unchanged.
///
/// # Errors
///
/// Returns [`GeoError::InvalidParameter`] for a negative tolerance, and
/// [`GeoError::NonFiniteInput`] for a non-finite tolerance or point coordinate.
pub fn visvalingam(points: &[(f64, f64)], tolerance: f64) -> Result<Vec<(f64, f64)>, GeoError> {
    validate(points, tolerance)?;
    if points.len() < 3 {
        return Ok(points.to_vec());
    }
    let mut kept: Vec<usize> = (0..points.len()).collect();
    while let Some((pos, area)) = smallest_interior(points, &kept) {
        if area < tolerance {
            kept.remove(pos);
        } else {
            break;
        }
    }
    Ok(kept.iter().map(|&i| points[i]).collect())
}

/// Rejects a non-finite or negative tolerance and any non-finite point coordinate.
fn validate(points: &[(f64, f64)], tolerance: f64) -> Result<(), GeoError> {
    if !tolerance.is_finite() {
        return Err(GeoError::NonFiniteInput);
    }
    if tolerance < 0.0 {
        return Err(GeoError::InvalidParameter { name: "tolerance", value: tolerance });
    }
    if points.iter().all(|p| p.0.is_finite() && p.1.is_finite()) {
        Ok(())
    } else {
        Err(GeoError::NonFiniteInput)
    }
}

/// Index of the interior point of `points[first..=last]` farthest from the chord, when that
/// distance is greater than `tolerance`. Ties go to the earlier point.
fn farthest_beyond(
    points: &[(f64, f64)],
    first: usize,
    last: usize,
    tolerance: f64,
) -> Option<usize> {
    let (a, b) = (points[first], points[last]);
    let mut best: Option<(usize, f64)> = None;
    for (i, &p) in points.iter().enumerate().take(last).skip(first + 1) {
        let d = line_distance(p, a, b);
        if d > tolerance && improves(best, d) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// Position within `kept` of the interior point with the smallest effective triangle area,
/// with that area. `None` when `kept` has fewer than three entries.
fn smallest_interior(points: &[(f64, f64)], kept: &[usize]) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for pos in 1..kept.len().saturating_sub(1) {
        let area = triangle_area(points[kept[pos - 1]], points[kept[pos]], points[kept[pos + 1]]);
        if improves_min(best, area) {
            best = Some((pos, area));
        }
    }
    best
}

/// True when `candidate` should replace the current farthest point: no current best, or a
/// strictly larger distance.
fn improves(best: Option<(usize, f64)>, candidate: f64) -> bool {
    match best {
        None => true,
        Some((_, current)) => candidate > current,
    }
}

/// True when `candidate` should replace the current smallest area: no current best, or a
/// strictly smaller area.
fn improves_min(best: Option<(usize, f64)>, candidate: f64) -> bool {
    match best {
        None => true,
        Some((_, current)) => candidate < current,
    }
}

/// Perpendicular distance from `p` to the infinite line through `a` and `b`, or the distance
/// to `a` when `a` and `b` coincide.
fn line_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 0.0 {
        return distance(p, a);
    }
    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / len
}

/// Euclidean distance between two points.
fn distance(p: (f64, f64), q: (f64, f64)) -> f64 {
    let (dx, dy) = (p.0 - q.0, p.1 - q.1);
    (dx * dx + dy * dy).sqrt()
}

/// Area of the triangle `a`, `b`, `c`.
fn triangle_area(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs() / 2.0
}

/// The points whose `keep` flag is set, in input order.
fn select(points: &[(f64, f64)], keep: &[bool]) -> Vec<(f64, f64)> {
    points.iter().zip(keep.iter()).filter_map(|(&p, &k)| k.then_some(p)).collect()
}
