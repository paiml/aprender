//! Hexagonal grid for the geo kernel (MAP-3): axial cells, pixel-to-cell picking,
//! polygon-to-cell rasterisation by exact overlap area, and area-weighted spread.
//!
//! Pointy-top hexagons of circumradius `R` (centre to corner). Axial cell `(q, r)` has centre
//! `x = ox + R sqrt3 (q + r/2)`, `y = oy + 1.5 R r`. Cube coordinates are `(q, -q - r, r)`, and
//! [`HexLayout::pick`] rounds them with cube rounding, so it runs in constant time and
//! deterministically: a point on a shared edge goes to the same one of its two cells on every
//! call.
//!
//! Invalid input is an error, never a number: a bad radius, a non-finite input, a zero-area
//! polygon, a polygon that reaches outside the window, or more candidate cells than
//! [`MAX_CANDIDATE_CELLS`] is a [`HexError`].

/// Largest number of candidate cells one rasterisation may visit.
pub const MAX_CANDIDATE_CELLS: u128 = 4_000_000;

/// Largest number of cells an [`AxialWindow`] may hold (2^24, about 16.8 million cells, or
/// 128 MiB of `f64` in a [`CellColumn`]). Larger windows are a [`HexError::InvalidWindow`].
pub const MAX_WINDOW_CELLS: u128 = 1 << 24;

/// Largest `|value|` that [`spread`] accepts. The MAP-3 epsilon is stated for this range.
pub const MAX_SPREAD_VALUE: f64 = 1e12;

/// `sqrt(3)`, as a literal so that it is a constant.
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Coordinates at or above `2^52` cannot be cube-rounded exactly; picks beyond it are errors.
const MAX_COORD: f64 = 4_503_599_627_370_496.0;

/// Overlap below this fraction of the polygon area is treated as no overlap.
const NEGLIGIBLE_REL: f64 = 1e-12;

/// Extra cells of candidate range on each side. A hexagon's axial q or r varies by at most
/// 2/3 from its centre, so two cells cover the rounding at the bounding box.
const CANDIDATE_MARGIN: i64 = 2;

/// The six axial neighbour offsets, in counter-clockwise order.
const AXIAL_OFFSETS: [(i64, i64); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

/// An axial hex cell `(q, r)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Cell {
    /// Axial column.
    pub q: i64,
    /// Axial row.
    pub r: i64,
}

/// Errors from the hex grid.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum HexError {
    /// The radius is not finite or not positive.
    #[error("hex radius must be finite and > 0, got {0}")]
    InvalidRadius(f64),
    /// An input is NaN or infinite.
    #[error("{what} is not finite")]
    NonFinite {
        /// Which input.
        what: &'static str,
    },
    /// A point is too far from the origin to pick exactly.
    #[error("{what} is outside the exact integer range of cell coordinates")]
    OutOfRange {
        /// Which input.
        what: &'static str,
    },
    /// A ring has fewer than three vertices.
    #[error("ring {ring} has {vertices} vertices; a ring needs at least 3")]
    RingTooShort {
        /// Ring index.
        ring: usize,
        /// Vertex count.
        vertices: usize,
    },
    /// A ring vertex is NaN or infinite.
    #[error("ring {ring} vertex {vertex} is not finite")]
    NonFiniteVertex {
        /// Ring index.
        ring: usize,
        /// Vertex index.
        vertex: usize,
    },
    /// The polygon has no positive area, or has no rings.
    #[error("polygon has no positive area")]
    ZeroArea,
    /// The polygon reaches a cell outside the axial window.
    #[error("cell ({q}, {r}) is reached by the polygon but lies outside the window")]
    OutsideWindow {
        /// Axial q of the cell.
        q: i64,
        /// Axial r of the cell.
        r: i64,
    },
    /// The window bounds are reversed or the window is too large to index.
    #[error("window q {q_min}..={q_max}, r {r_min}..={r_max} is empty or too large")]
    InvalidWindow {
        /// Minimum q.
        q_min: i64,
        /// Maximum q.
        q_max: i64,
        /// Minimum r.
        r_min: i64,
        /// Maximum r.
        r_max: i64,
    },
    /// Rasterisation would visit more candidate cells than allowed.
    #[error("rasterisation would visit {candidates} candidate cells; the limit is 4000000")]
    TooManyCells {
        /// Candidate count.
        candidates: u128,
    },
    /// The value is outside `|value| <= 1e12`.
    #[error("value {value} is outside |value| <= 1e12")]
    ValueOutOfRange {
        /// The value.
        value: f64,
    },
    /// An area is NaN, infinite or negative.
    #[error("area {index} is not finite and non-negative")]
    InvalidArea {
        /// Index into the area list.
        index: usize,
    },
}

/// Pointy-top hex layout: the centre of cell `(0, 0)` is `origin`, and the circumradius is
/// `radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HexLayout {
    origin: (f64, f64),
    radius: f64,
}

impl HexLayout {
    /// A validated layout. The radius must be finite and positive, and the origin finite.
    pub fn new(origin: (f64, f64), radius: f64) -> Result<Self, HexError> {
        if !(origin.0.is_finite() && origin.1.is_finite()) {
            return Err(HexError::NonFinite { what: "origin" });
        }
        if !(radius.is_finite() && radius > 0.0) {
            return Err(HexError::InvalidRadius(radius));
        }
        Ok(Self { origin, radius })
    }

    /// Pixel position of the centre of cell `(0, 0)`.
    pub fn origin(&self) -> (f64, f64) {
        self.origin
    }

    /// Circumradius: centre to corner.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Pixel centre of a cell.
    pub fn center(&self, c: Cell) -> (f64, f64) {
        let q = c.q as f64;
        let r = c.r as f64;
        (self.origin.0 + self.radius * SQRT3 * (q + r / 2.0), self.origin.1 + self.radius * 1.5 * r)
    }

    /// The cell containing a pixel point, by cube rounding. Constant time.
    pub fn pick(&self, p: (f64, f64)) -> Result<Cell, HexError> {
        if !(p.0.is_finite() && p.1.is_finite()) {
            return Err(HexError::NonFinite { what: "point" });
        }
        let dx = p.0 - self.origin.0;
        let dy = p.1 - self.origin.1;
        let qf = (SQRT3 / 3.0 * dx - dy / 3.0) / self.radius;
        let rf = (2.0 / 3.0 * dy) / self.radius;
        if !(qf.is_finite() && rf.is_finite() && qf.abs() < MAX_COORD && rf.abs() < MAX_COORD) {
            return Err(HexError::OutOfRange { what: "point" });
        }
        let (q, r) = cube_round(qf, -qf - rf, rf);
        Ok(Cell { q, r })
    }

    /// The six corners of a cell, counter-clockwise from -30 degrees.
    pub fn corners(&self, c: Cell) -> [(f64, f64); 6] {
        hexagon_corners(self.center(c), self.radius)
    }

    /// The six corners of a cell, relative to `anchor`. The centre is formed from the difference
    /// of the origin and the anchor first, so the corners stay exact near the anchor even when
    /// both are large.
    fn corners_about(&self, c: Cell, anchor: (f64, f64)) -> [(f64, f64); 6] {
        let (q, r) = (c.q as f64, c.r as f64);
        let centre = (
            (self.origin.0 - anchor.0) + self.radius * SQRT3 * (q + r / 2.0),
            (self.origin.1 - anchor.1) + self.radius * 1.5 * r,
        );
        hexagon_corners(centre, self.radius)
    }

    /// Area of one hexagon: `3 sqrt3 / 2 * R^2`.
    pub fn cell_area(&self) -> f64 {
        1.5 * SQRT3 * self.radius * self.radius
    }
}

/// The six axial neighbours of a cell.
pub fn neighbours(c: Cell) -> [Cell; 6] {
    AXIAL_OFFSETS.map(|(dq, dr)| Cell { q: c.q + dq, r: c.r + dr })
}

/// A finite inclusive axial window `q_min..=q_max` by `r_min..=r_max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxialWindow {
    q_min: i64,
    q_max: i64,
    r_min: i64,
    r_max: i64,
}

impl AxialWindow {
    /// A validated window. Bounds must be ordered, and the cell count must not exceed
    /// [`MAX_WINDOW_CELLS`].
    pub fn new(q_min: i64, q_max: i64, r_min: i64, r_max: i64) -> Result<Self, HexError> {
        let window = Self { q_min, q_max, r_min, r_max };
        let invalid = HexError::InvalidWindow { q_min, q_max, r_min, r_max };
        if q_min > q_max || r_min > r_max {
            return Err(invalid);
        }
        match window.width().checked_mul(window.height()) {
            Some(n) if n <= MAX_WINDOW_CELLS && n <= usize::MAX as u128 => Ok(window),
            _ => Err(invalid),
        }
    }

    fn width(&self) -> u128 {
        (self.q_max as i128 - self.q_min as i128 + 1) as u128
    }

    fn height(&self) -> u128 {
        (self.r_max as i128 - self.r_min as i128 + 1) as u128
    }

    /// Whether a cell lies in the window.
    pub fn contains(&self, c: Cell) -> bool {
        (self.q_min..=self.q_max).contains(&c.q) && (self.r_min..=self.r_max).contains(&c.r)
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        (self.width() * self.height()) as usize
    }

    /// Whether the window has no cells. A validated window never does.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major index of a cell, or `None` outside the window.
    pub fn index_of(&self, c: Cell) -> Option<usize> {
        if !self.contains(c) {
            return None;
        }
        let col = (c.q as i128 - self.q_min as i128) as usize;
        let row = (c.r as i128 - self.r_min as i128) as usize;
        Some(row * self.width() as usize + col)
    }

    /// Every cell, in index order.
    pub fn cells(&self) -> impl Iterator<Item = Cell> + '_ {
        (self.r_min..=self.r_max)
            .flat_map(move |r| (self.q_min..=self.q_max).map(move |q| Cell { q, r }))
    }
}

/// One `f64` per cell of a window, indexed by [`AxialWindow::index_of`].
#[derive(Debug, Clone, PartialEq)]
pub struct CellColumn {
    window: AxialWindow,
    values: Vec<f64>,
}

impl CellColumn {
    /// An all-zero column over a window.
    pub fn zeros(window: AxialWindow) -> Self {
        let values = vec![0.0; window.len()];
        Self { window, values }
    }

    /// The window this column is indexed by.
    pub fn window(&self) -> &AxialWindow {
        &self.window
    }

    /// Raw values in index order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Value at a cell, or `None` outside the window.
    pub fn get(&self, c: Cell) -> Option<f64> {
        self.window.index_of(c).map(|i| self.values[i])
    }

    /// Add to a cell. A non-finite value or a cell outside the window is an error, and a refused
    /// add leaves the column unchanged.
    pub fn add(&mut self, c: Cell, v: f64) -> Result<(), HexError> {
        if !v.is_finite() {
            return Err(HexError::NonFinite { what: "cell value" });
        }
        let i = self.window.index_of(c).ok_or(HexError::OutsideWindow { q: c.q, r: c.r })?;
        self.values[i] += v;
        Ok(())
    }
}

/// Area of a polygon given as rings. The first ring is the exterior and the rest are holes,
/// each inside the exterior. Ring orientation does not matter.
pub fn polygon_area(rings: &[Vec<(f64, f64)>]) -> Result<f64, HexError> {
    validate_rings(rings)?;
    let area = exterior_minus_holes(rings, |ring| signed_area(ring).abs());
    if !area.is_finite() {
        return Err(HexError::NonFinite { what: "polygon area" });
    }
    if area <= 0.0 {
        return Err(HexError::ZeroArea);
    }
    Ok(area)
}

/// Exact overlap area of a polygon with each hex cell it reaches, as `(cell, area)`.
///
/// Each cell's area is the area of the polygon intersected with that hexagon. Each ring is
/// clipped against the convex hexagon with Sutherland-Hodgman, and the holes subtract. Only
/// cells with positive overlap are returned, in row-major order. A reached cell outside
/// `window` is an error.
pub fn rasterise(
    layout: &HexLayout,
    window: &AxialWindow,
    rings: &[Vec<(f64, f64)>],
) -> Result<Vec<(Cell, f64)>, HexError> {
    let total = polygon_area(rings)?;
    let (q_lo, q_hi, r_lo, r_hi) = candidate_range(layout, rings)?;
    let width = (q_hi as i128 - q_lo as i128 + 1) as u128;
    let height = (r_hi as i128 - r_lo as i128 + 1) as u128;
    let candidates = width * height;
    if candidates > MAX_CANDIDATE_CELLS {
        return Err(HexError::TooManyCells { candidates });
    }
    let threshold = NEGLIGIBLE_REL * total;
    // Clip in coordinates local to the first exterior vertex. Absolute coordinates far from the
    // origin lose their rounding error to the clipping; differences from a nearby anchor do not.
    let anchor = rings.first().and_then(|ring| ring.first()).copied().ok_or(HexError::ZeroArea)?;
    let local = translate_rings(rings, anchor);
    let mut out = Vec::new();
    for r in r_lo..=r_hi {
        for q in q_lo..=q_hi {
            let c = Cell { q, r };
            let a = cell_overlap(layout, c, anchor, &local);
            if a <= threshold {
                continue;
            }
            if !window.contains(c) {
                return Err(HexError::OutsideWindow { q, r });
            }
            out.push((c, a));
        }
    }
    Ok(out)
}

/// Area-weighted spread `v_c = value * a_c / A`, with `A` the sum of the areas.
///
/// The sum of the returned values is `value` within relative 1e-9. `|value|` must not exceed
/// [`MAX_SPREAD_VALUE`]. Areas must be finite and non-negative, with a positive sum.
pub fn spread(value: f64, areas: &[(Cell, f64)]) -> Result<Vec<(Cell, f64)>, HexError> {
    if !value.is_finite() {
        return Err(HexError::NonFinite { what: "value" });
    }
    if value.abs() > MAX_SPREAD_VALUE {
        return Err(HexError::ValueOutOfRange { value });
    }
    if let Some(index) = areas.iter().position(|(_, a)| !(a.is_finite() && *a >= 0.0)) {
        return Err(HexError::InvalidArea { index });
    }
    let total: f64 = areas.iter().map(|(_, a)| a).sum();
    if !total.is_finite() {
        return Err(HexError::NonFinite { what: "area sum" });
    }
    if total <= 0.0 {
        return Err(HexError::ZeroArea);
    }
    Ok(areas.iter().map(|&(c, a)| (c, value * a / total)).collect())
}

/// Reject empty input, too-short rings and non-finite vertices.
fn validate_rings(rings: &[Vec<(f64, f64)>]) -> Result<(), HexError> {
    if rings.is_empty() {
        return Err(HexError::ZeroArea);
    }
    for (ring, pts) in rings.iter().enumerate() {
        if pts.len() < 3 {
            return Err(HexError::RingTooShort { ring, vertices: pts.len() });
        }
        if let Some(vertex) = pts.iter().position(|&(x, y)| !(x.is_finite() && y.is_finite())) {
            return Err(HexError::NonFiniteVertex { ring, vertex });
        }
    }
    Ok(())
}

/// Sum of `measure` over the rings: the exterior (first ring) adds, each hole subtracts.
fn exterior_minus_holes(rings: &[Vec<(f64, f64)>], measure: impl Fn(&[(f64, f64)]) -> f64) -> f64 {
    rings
        .iter()
        .enumerate()
        .map(|(i, ring)| {
            let m = measure(ring);
            if i == 0 {
                m
            } else {
                -m
            }
        })
        .sum()
}

/// The rings shifted so that `anchor` is the origin.
fn translate_rings(rings: &[Vec<(f64, f64)>], anchor: (f64, f64)) -> Vec<Vec<(f64, f64)>> {
    rings
        .iter()
        .map(|ring| ring.iter().map(|&(x, y)| (x - anchor.0, y - anchor.1)).collect())
        .collect()
}

/// Overlap area of the polygon with one hexagon. `local` is the polygon translated by `anchor`,
/// and the hexagon is built in the same frame.
fn cell_overlap(layout: &HexLayout, c: Cell, anchor: (f64, f64), local: &[Vec<(f64, f64)>]) -> f64 {
    let hex = layout.corners_about(c, anchor);
    exterior_minus_holes(local, |ring| signed_area(&clip_to_convex(ring, &hex)).abs())
}

/// The six corners of a hexagon with the given centre and circumradius, counter-clockwise from
/// -30 degrees.
fn hexagon_corners(centre: (f64, f64), radius: f64) -> [(f64, f64); 6] {
    std::array::from_fn(|k| {
        let angle = (60.0 * k as f64 - 30.0).to_radians();
        (centre.0 + radius * angle.cos(), centre.1 + radius * angle.sin())
    })
}

/// Candidate axial range `(q_lo, q_hi, r_lo, r_hi)` that covers every cell a polygon can
/// reach: the cells of the four bounding-box corners, widened by [`CANDIDATE_MARGIN`].
fn candidate_range(
    layout: &HexLayout,
    rings: &[Vec<(f64, f64)>],
) -> Result<(i64, i64, i64, i64), HexError> {
    let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
    let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in rings.iter().flatten() {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let mut lo = (i64::MAX, i64::MAX);
    let mut hi = (i64::MIN, i64::MIN);
    for corner in [(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let c = layout.pick(corner)?;
        lo = (lo.0.min(c.q), lo.1.min(c.r));
        hi = (hi.0.max(c.q), hi.1.max(c.r));
    }
    Ok((
        lo.0 - CANDIDATE_MARGIN,
        hi.0 + CANDIDATE_MARGIN,
        lo.1 - CANDIDATE_MARGIN,
        hi.1 + CANDIDATE_MARGIN,
    ))
}

/// Shoelace signed area of a ring. Positive for counter-clockwise rings.
///
/// The vertices are taken relative to the first one. Products of coordinates near 1e8 are near
/// 1e16, where f64 has an ulp of 2, so unshifted terms cancel to nothing. Differences of nearby
/// vertices are exact, so the area is too.
fn signed_area(ring: &[(f64, f64)]) -> f64 {
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
fn clip_to_convex(subject: &[(f64, f64)], clip: &[(f64, f64); 6]) -> Vec<(f64, f64)> {
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

/// Cube rounding of fractional cube coordinates `(x, y, z)` with `x + y + z = 0`. Returns the
/// axial `(q, r)`. The component with the largest rounding error is reset, so the result stays
/// on the lattice. Ties resolve in a fixed order, so the result is deterministic.
fn cube_round(x: f64, y: f64, z: f64) -> (i64, i64) {
    let (mut rx, ry, mut rz) = (x.round(), y.round(), z.round());
    let (dx, dy, dz) = ((rx - x).abs(), (ry - y).abs(), (rz - z).abs());
    if dx > dy && dx > dz {
        rx = -ry - rz;
    } else if dy <= dz {
        rz = -rx - ry;
    }
    (rx as i64, rz as i64)
}
