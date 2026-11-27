//! Hexagonal grid for the geo kernel (MAP-3). RED stub: signatures only.

/// Largest number of candidate cells one rasterisation may visit.
pub const MAX_CANDIDATE_CELLS: u128 = 4_000_000;

/// Largest `|value|` that [`spread`] accepts.
pub const MAX_SPREAD_VALUE: f64 = 1e12;

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
    /// The polygon has no positive area.
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

/// Pointy-top hex layout: centre of cell `(0, 0)` at `origin`, circumradius `radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HexLayout {
    origin: (f64, f64),
    radius: f64,
}

impl HexLayout {
    /// Validated layout.
    pub fn new(origin: (f64, f64), radius: f64) -> Result<Self, HexError> {
        let _ = (origin, radius);
        unimplemented!("MAP-3 RED stub")
    }
    /// Pixel centre of a cell.
    pub fn center(&self, c: Cell) -> (f64, f64) {
        let _ = c;
        unimplemented!("MAP-3 RED stub")
    }
    /// Cell containing a pixel point, by cube rounding.
    pub fn pick(&self, p: (f64, f64)) -> Result<Cell, HexError> {
        let _ = p;
        unimplemented!("MAP-3 RED stub")
    }
    /// The six corners of a cell, counter-clockwise.
    pub fn corners(&self, c: Cell) -> [(f64, f64); 6] {
        let _ = c;
        unimplemented!("MAP-3 RED stub")
    }
    /// Area of one hexagon.
    pub fn cell_area(&self) -> f64 {
        unimplemented!("MAP-3 RED stub")
    }
}

/// The six axial neighbours of a cell.
pub fn neighbours(c: Cell) -> [Cell; 6] {
    let _ = c;
    unimplemented!("MAP-3 RED stub")
}

/// A finite inclusive axial window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxialWindow {
    q_min: i64,
    q_max: i64,
    r_min: i64,
    r_max: i64,
}

impl AxialWindow {
    /// Validated window.
    pub fn new(q_min: i64, q_max: i64, r_min: i64, r_max: i64) -> Result<Self, HexError> {
        let _ = (q_min, q_max, r_min, r_max);
        unimplemented!("MAP-3 RED stub")
    }
    /// Whether a cell lies in the window.
    pub fn contains(&self, c: Cell) -> bool {
        let _ = c;
        unimplemented!("MAP-3 RED stub")
    }
    /// Number of cells.
    pub fn len(&self) -> usize {
        unimplemented!("MAP-3 RED stub")
    }
    /// Whether the window has no cells (never true for a validated window).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Row-major index of a cell, or `None` outside the window.
    pub fn index_of(&self, c: Cell) -> Option<usize> {
        let _ = c;
        unimplemented!("MAP-3 RED stub")
    }
    /// Every cell, in index order.
    pub fn cells(&self) -> impl Iterator<Item = Cell> + '_ {
        let _ = self;
        std::iter::empty()
    }
}

/// One `f64` per cell of a window.
#[derive(Debug, Clone, PartialEq)]
pub struct CellColumn {
    window: AxialWindow,
    values: Vec<f64>,
}

impl CellColumn {
    /// All-zero column.
    pub fn zeros(window: AxialWindow) -> Self {
        let _ = window;
        unimplemented!("MAP-3 RED stub")
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
        let _ = c;
        unimplemented!("MAP-3 RED stub")
    }
    /// Add to a cell; a cell outside the window is an error.
    pub fn add(&mut self, c: Cell, v: f64) -> Result<(), HexError> {
        let _ = (c, v);
        unimplemented!("MAP-3 RED stub")
    }
}

/// Signed-area sum of rings: first ring is the exterior, the rest are holes.
pub fn polygon_area(rings: &[Vec<(f64, f64)>]) -> Result<f64, HexError> {
    let _ = rings;
    unimplemented!("MAP-3 RED stub")
}

/// Exact overlap area of a polygon with each hex cell it reaches, as `(cell, area)`.
pub fn rasterise(
    layout: &HexLayout,
    window: &AxialWindow,
    rings: &[Vec<(f64, f64)>],
) -> Result<Vec<(Cell, f64)>, HexError> {
    let _ = (layout, window, rings);
    unimplemented!("MAP-3 RED stub")
}

/// Area-weighted spread `v_c = value * a_c / A`, with `A` the sum of the areas.
pub fn spread(value: f64, areas: &[(Cell, f64)]) -> Result<Vec<(Cell, f64)>, HexError> {
    let _ = (value, areas);
    unimplemented!("MAP-3 RED stub")
}
