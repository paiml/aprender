//! Static 2D spatial indexes for the `geo` feature: point-in-polygon picking and nearest point.
//!
//! Both indexes are uniform grids built once from the input. A polygon is a list of rings:
//! ring 0 is the exterior and the others are holes, and containment is the even-odd rule over
//! all rings. A point on any edge or vertex of a polygon counts as inside. When several
//! polygons contain a point, [`PolygonIndex::pick`] returns the lowest id. Nearest ties return
//! the lowest id as well.
//!
//! Invalid input is an [`IndexError`], never a panic: a non-finite coordinate (at build or
//! query time) or a ring with fewer than three vertices. An index over no items answers `None`.
//! Degenerate extents (all points identical, zero width) use a single cell on that axis.

use thiserror::Error;

/// Upper bound on grid cells per axis (the grid holds at most this squared).
pub const MAX_GRID_AXIS: usize = 256;

/// A polygon as rings: ring 0 is the exterior, the rest are holes.
pub type Rings = Vec<Vec<(f64, f64)>>;

/// Why an index could not be built or queried.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum IndexError {
    /// A coordinate is NaN or infinite.
    #[error("{what} is not finite")]
    NonFinite {
        /// Which input.
        what: &'static str,
    },
    /// A ring has fewer than three vertices.
    #[error("polygon {polygon} ring {ring} has {vertices} vertices; a ring needs at least 3")]
    RingTooShort {
        /// Polygon id (its index in the input).
        polygon: usize,
        /// Ring index within the polygon.
        ring: usize,
        /// Vertex count.
        vertices: usize,
    },
}

/// Point-in-polygon index over a fixed set of polygons.
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonIndex {
    polygons: Vec<Rings>,
    layout: Option<Layout>,
}

impl PolygonIndex {
    /// Build the index. Polygon ids are positions in `polygons`.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] for a NaN or infinite coordinate, and
    /// [`IndexError::RingTooShort`] for a ring with fewer than three vertices.
    pub fn build(polygons: &[Rings]) -> Result<Self, IndexError> {
        let mut items = Vec::new();
        for (id, rings) in polygons.iter().enumerate() {
            validate_rings(id, rings)?;
            if let Some(b) = Bbox::of_points(rings.iter().flatten().copied()) {
                items.push((id, b));
            }
        }
        Ok(Self { polygons: polygons.to_vec(), layout: Layout::new(&items, polygons.len()) })
    }

    /// Number of polygons in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.polygons.len()
    }

    /// True when the index holds no polygons.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.polygons.is_empty()
    }

    /// Lowest id of a polygon containing `p`, or `None`.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] when `p` has a NaN or infinite coordinate.
    pub fn pick(&self, p: (f64, f64)) -> Result<Option<usize>, IndexError> {
        check_finite(p, "query point")?;
        let Some(layout) = &self.layout else {
            return Ok(None);
        };
        if !layout.extent.contains(p) {
            return Ok(None);
        }
        // Cell lists hold ids in ascending order, so the first hit is the lowest id.
        let hit = layout.cells[layout.grid.slot_of(p)]
            .iter()
            .copied()
            .find(|&id| rings_contain(&self.polygons[id], p));
        Ok(hit)
    }
}

/// Nearest-point index over a fixed set of points.
#[derive(Debug, Clone, PartialEq)]
pub struct PointIndex {
    points: Vec<(f64, f64)>,
    layout: Option<Layout>,
    /// Largest absolute coordinate, used to size the rounding slack of the ring bound.
    scale: f64,
}

impl PointIndex {
    /// Build the index. Point ids are positions in `points`.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] for a NaN or infinite coordinate.
    pub fn build(points: &[(f64, f64)]) -> Result<Self, IndexError> {
        check_all_finite(points, "point coordinate")?;
        let items: Vec<(usize, Bbox)> = points
            .iter()
            .enumerate()
            .filter_map(|(id, &p)| Bbox::of_points(std::iter::once(p)).map(|b| (id, b)))
            .collect();
        let scale = points.iter().flat_map(|p| [p.0.abs(), p.1.abs()]).fold(0.0, f64::max);
        Ok(Self { points: points.to_vec(), layout: Layout::new(&items, points.len()), scale })
    }

    /// Number of points in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True when the index holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Id of the point nearest to `q` (lowest id on ties), or `None` when empty.
    ///
    /// Cells are visited in rings of growing Chebyshev radius around the query cell. The search
    /// stops once the lower bound on every unvisited cell exceeds the best distance found.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] when `q` has a NaN or infinite coordinate.
    pub fn nearest(&self, q: (f64, f64)) -> Result<Option<usize>, IndexError> {
        check_finite(q, "query point")?;
        let Some(layout) = &self.layout else {
            return Ok(None);
        };
        let slack = 1e-9 * (1.0 + self.scale.max(q.0.abs()).max(q.1.abs()));
        let centre = layout.grid.cell_of(q);
        let mut best: Option<(f64, usize)> = None;
        let mut radius = 0;
        while !exhausted(&layout.grid, q, centre, radius, best, slack) {
            for slot in layout.grid.ring_slots(centre, radius) {
                for &id in &layout.cells[slot] {
                    improve(&mut best, (dist2(self.points[id], q), id));
                }
            }
            radius += 1;
        }
        Ok(best.map(|(_, id)| id))
    }
}

/// Axis-aligned bounding box of finite coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bbox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Bbox {
    /// Box around a set of points, or `None` when the set is empty.
    fn of_points(points: impl Iterator<Item = (f64, f64)>) -> Option<Self> {
        points.fold(None, |acc, (x, y)| {
            Some(match acc {
                None => Self { min_x: x, min_y: y, max_x: x, max_y: y },
                Some(b) => b.grow(x, y),
            })
        })
    }

    /// This box extended to include `(x, y)`.
    fn grow(self, x: f64, y: f64) -> Self {
        Self {
            min_x: self.min_x.min(x),
            min_y: self.min_y.min(y),
            max_x: self.max_x.max(x),
            max_y: self.max_y.max(y),
        }
    }

    /// True when `p` lies in the closed box.
    fn contains(&self, p: (f64, f64)) -> bool {
        p.0 >= self.min_x && p.0 <= self.max_x && p.1 >= self.min_y && p.1 <= self.max_y
    }
}

/// Uniform grid over an extent: `dims` cells of size `cell`, with `origin` at the low corner.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Grid {
    origin: (f64, f64),
    cell: (f64, f64),
    inv: (f64, f64),
    dims: (usize, usize),
}

impl Grid {
    /// Grid for `extent` sized for about `target` items.
    fn new(extent: Bbox, target: usize) -> Self {
        let side = ((target as f64).sqrt().ceil() as usize).clamp(1, MAX_GRID_AXIS);
        let (nx, wx, ix) = axis(extent.min_x, extent.max_x, side);
        let (ny, wy, iy) = axis(extent.min_y, extent.max_y, side);
        Self { origin: (extent.min_x, extent.min_y), cell: (wx, wy), inv: (ix, iy), dims: (nx, ny) }
    }

    /// Column of `x`, clamped into the grid.
    fn col(&self, x: f64) -> usize {
        index_of(x, self.origin.0, self.inv.0, self.dims.0)
    }

    /// Row of `y`, clamped into the grid.
    fn row(&self, y: f64) -> usize {
        index_of(y, self.origin.1, self.inv.1, self.dims.1)
    }

    /// Flat slot of the cell at column `i`, row `j`.
    fn slot(&self, i: usize, j: usize) -> usize {
        j * self.dims.0 + i
    }

    /// Flat slot of the cell containing `p`.
    fn slot_of(&self, p: (f64, f64)) -> usize {
        self.slot(self.col(p.0), self.row(p.1))
    }

    /// `(column, row)` of the cell containing `p`.
    fn cell_of(&self, p: (f64, f64)) -> (usize, usize) {
        (self.col(p.0), self.row(p.1))
    }

    /// Inclusive cell range `(c0, c1, r0, r1)` covered by a box.
    fn range(&self, b: Bbox) -> (usize, usize, usize, usize) {
        (self.col(b.min_x), self.col(b.max_x), self.row(b.min_y), self.row(b.max_y))
    }

    /// Slots of the cells at Chebyshev distance exactly `radius` from `centre`.
    fn ring_slots(&self, centre: (usize, usize), radius: usize) -> Vec<usize> {
        let (nx, ny) = (self.dims.0 as isize, self.dims.1 as isize);
        let (ci, cj, r) = (centre.0 as isize, centre.1 as isize, radius as isize);
        let (ilo, ihi) = ((ci - r).max(0), (ci + r).min(nx - 1));
        let mut out = Vec::new();
        for j in (cj - r).max(0)..=(cj + r).min(ny - 1) {
            if (j - cj).abs() == r {
                out.extend((ilo..=ihi).map(|i| self.slot(i as usize, j as usize)));
            } else {
                for i in [ci - r, ci + r] {
                    if (0..nx).contains(&i) {
                        out.push(self.slot(i as usize, j as usize));
                    }
                }
            }
        }
        out
    }

    /// Lower bound on the distance from `q` to any cell outside the block of cells within
    /// `radius` of `centre`, or `None` when no such cell exists.
    fn outside_bound(&self, q: (f64, f64), centre: (usize, usize), radius: usize) -> Option<f64> {
        let (nx, ny) = self.dims;
        let (w, h) = self.cell;
        let (cx, cy) = centre;
        let left =
            (cx > radius).then(|| (q.0 - (self.origin.0 + (cx - radius) as f64 * w)).max(0.0));
        let right = (cx + radius + 1 < nx)
            .then(|| (self.origin.0 + (cx + radius + 1) as f64 * w - q.0).max(0.0));
        let down =
            (cy > radius).then(|| (q.1 - (self.origin.1 + (cy - radius) as f64 * h)).max(0.0));
        let up = (cy + radius + 1 < ny)
            .then(|| (self.origin.1 + (cy + radius + 1) as f64 * h - q.1).max(0.0));
        [left, right, down, up].into_iter().flatten().reduce(f64::min)
    }
}

/// Grid plus the per-cell id lists, built from `(id, box)` items in ascending id order.
#[derive(Debug, Clone, PartialEq)]
struct Layout {
    extent: Bbox,
    grid: Grid,
    cells: Vec<Vec<usize>>,
}

impl Layout {
    /// Layout over `items`, or `None` when there are no items.
    fn new(items: &[(usize, Bbox)], target: usize) -> Option<Self> {
        let extent = items
            .iter()
            .map(|&(_, b)| b)
            .reduce(|a, b| a.grow(b.min_x, b.min_y).grow(b.max_x, b.max_y))?;
        let grid = Grid::new(extent, target);
        let mut cells = vec![Vec::new(); grid.dims.0 * grid.dims.1];
        for &(id, b) in items {
            let (c0, c1, r0, r1) = grid.range(b);
            for j in r0..=r1 {
                for i in c0..=c1 {
                    cells[grid.slot(i, j)].push(id);
                }
            }
        }
        Some(Self { extent, grid, cells })
    }
}

/// Per-axis grid parameters: `(cells, cell width, reciprocal width)`. A degenerate or
/// subnormal span gives one cell.
fn axis(lo: f64, hi: f64, side: usize) -> (usize, f64, f64) {
    let w = (hi - lo) / side as f64;
    if w.is_normal() {
        (side, w, 1.0 / w)
    } else {
        (1, 0.0, 0.0)
    }
}

/// Clamped, monotone cell index of `v`. A NaN product (zero offset on a single-cell axis) is 0.
fn index_of(v: f64, origin: f64, inv: f64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let t = (v - origin) * inv;
    if t.is_nan() {
        0
    } else {
        t.clamp(0.0, (n - 1) as f64) as usize
    }
}

/// True when the search can stop before visiting ring `radius`: no cell outside the rings
/// already visited exists, or every such cell is farther than the best point found, beyond
/// `slack`.
fn exhausted(
    grid: &Grid,
    q: (f64, f64),
    centre: (usize, usize),
    radius: usize,
    best: Option<(f64, usize)>,
    slack: f64,
) -> bool {
    if radius == 0 {
        return false;
    }
    match (grid.outside_bound(q, centre, radius - 1), best) {
        (None, _) => true,
        (Some(lb), Some((d2, _))) => lb > d2.sqrt() + slack,
        (Some(_), None) => false,
    }
}

/// Replace `best` when `cand` is strictly nearer, or equally near with a lower id.
fn improve(best: &mut Option<(f64, usize)>, cand: (f64, usize)) {
    let better = match *best {
        None => true,
        Some((d2, id)) => cand.0 < d2 || (cand.0 == d2 && cand.1 < id),
    };
    if better {
        *best = Some(cand);
    }
}

/// Squared distance from `point` to `q`.
fn dist2(point: (f64, f64), q: (f64, f64)) -> f64 {
    let dx = point.0 - q.0;
    let dy = point.1 - q.1;
    dx * dx + dy * dy
}

/// Typed error for a non-finite query or input point.
fn check_finite(p: (f64, f64), what: &'static str) -> Result<(), IndexError> {
    if p.0.is_finite() && p.1.is_finite() {
        Ok(())
    } else {
        Err(IndexError::NonFinite { what })
    }
}

/// Typed error for the first non-finite point in `points`.
fn check_all_finite(points: &[(f64, f64)], what: &'static str) -> Result<(), IndexError> {
    points.iter().try_for_each(|&p| check_finite(p, what))
}

/// Ring-count and finiteness checks for polygon `polygon`.
fn validate_rings(polygon: usize, rings: &Rings) -> Result<(), IndexError> {
    for (ring, pts) in rings.iter().enumerate() {
        if pts.len() < 3 {
            return Err(IndexError::RingTooShort { polygon, ring, vertices: pts.len() });
        }
        check_all_finite(pts, "polygon coordinate")?;
    }
    Ok(())
}

/// True when `p` is on an edge or vertex of any ring, or inside by the even-odd rule.
fn rings_contain(rings: &[Vec<(f64, f64)>], p: (f64, f64)) -> bool {
    if rings.iter().any(|ring| on_boundary(ring, p)) {
        return true;
    }
    let crossings: usize = rings.iter().map(|ring| crossings_right(ring, p)).sum();
    crossings % 2 == 1
}

/// True when `p` lies on one of the closed edges of `ring`.
fn on_boundary(ring: &[(f64, f64)], p: (f64, f64)) -> bool {
    let n = ring.len();
    (0..n).any(|i| on_segment(ring[i], ring[(i + 1) % n], p))
}

/// True when `p` lies on the closed segment `a`-`b` (exact cross product and bounds).
fn on_segment(a: (f64, f64), b: (f64, f64), p: (f64, f64)) -> bool {
    let cross = (b.0 - a.0) * (p.1 - a.1) - (p.0 - a.0) * (b.1 - a.1);
    cross == 0.0
        && p.0 >= a.0.min(b.0)
        && p.0 <= a.0.max(b.0)
        && p.1 >= a.1.min(b.1)
        && p.1 <= a.1.max(b.1)
}

/// Number of edges of `ring` crossed by the ray from `p` towards +x (half-open edge rule).
fn crossings_right(ring: &[(f64, f64)], p: (f64, f64)) -> usize {
    let n = ring.len();
    (0..n)
        .filter(|&i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            if (a.1 > p.1) == (b.1 > p.1) {
                return false;
            }
            let x_at = a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0);
            p.0 < x_at
        })
        .count()
}
