//! Static 2D spatial indexes for the `geo` feature: point-in-polygon picking and nearest point.
//!
//! Both indexes are uniform grids built once from the input. A polygon is a list of rings:
//! ring 0 is the exterior and the others are holes, and containment is the even-odd rule over
//! all rings. A point on any edge or vertex of a polygon counts as inside. When several
//! polygons contain a point, [`PolygonIndex::pick`] returns the lowest id. Nearest ties return
//! the lowest id as well.
//!
//! Invalid input is an [`IndexError`], never a panic: a non-finite coordinate (at build or
//! query time) or a ring with fewer than three vertices.

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
    count: usize,
}

impl PolygonIndex {
    /// Build the index. Polygon ids are positions in `polygons`.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] for a NaN or infinite coordinate, and
    /// [`IndexError::RingTooShort`] for a ring with fewer than three vertices.
    pub fn build(polygons: &[Rings]) -> Result<Self, IndexError> {
        Ok(Self { count: polygons.len() })
    }

    /// Number of polygons in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// True when the index holds no polygons.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Lowest id of a polygon containing `p`, or `None`.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] when `p` has a NaN or infinite coordinate.
    pub fn pick(&self, p: (f64, f64)) -> Result<Option<usize>, IndexError> {
        let _ = p;
        Ok(None)
    }
}

/// Nearest-point index over a fixed set of points.
#[derive(Debug, Clone, PartialEq)]
pub struct PointIndex {
    count: usize,
}

impl PointIndex {
    /// Build the index. Point ids are positions in `points`.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] for a NaN or infinite coordinate.
    pub fn build(points: &[(f64, f64)]) -> Result<Self, IndexError> {
        Ok(Self { count: points.len() })
    }

    /// Number of points in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// True when the index holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Id of the point nearest to `q` (lowest id on ties), or `None` when empty.
    ///
    /// # Errors
    /// [`IndexError::NonFinite`] when `q` has a NaN or infinite coordinate.
    pub fn nearest(&self, q: (f64, f64)) -> Result<Option<usize>, IndexError> {
        let _ = q;
        Ok(None)
    }
}

/// True when `p` is on an edge or vertex of any ring, or inside by the even-odd rule.
///
/// Exposed to the tests as the per-polygon predicate that the brute-force oracle scans with.
#[must_use]
pub(crate) fn rings_contain(rings: &[Vec<(f64, f64)>], p: (f64, f64)) -> bool {
    let _ = (rings, p);
    todo!("MAP-3 phase 2: rings_contain is a RED stub")
}
