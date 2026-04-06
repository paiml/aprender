//! SVG Grid Protocol Engine
//!
//! Cell-based 16x9 grid layout for 1080p video-optimized SVG generation.
//! Provides provable non-overlap via occupied-set tracking and a manifest
//! that documents all allocations as an XML comment.
//!
//! # Grid Geometry
//!
//! - 16 columns x 9 rows = 144 cells
//! - Each cell is 120x120 pixels
//! - 10px cell padding shrinks each allocation's render bounds
//! - 20px internal padding further shrinks content zones
//! - Canvas: 1920x1080 (16:9)

use std::collections::HashSet;
use std::fmt;

// ── Constants ──────────────────────────────────────────────────────────────────

/// Number of grid columns
pub const GRID_COLS: u8 = 16;

/// Number of grid rows
pub const GRID_ROWS: u8 = 9;

/// Size of each grid cell in pixels
pub const CELL_SIZE: f32 = 120.0;

/// Padding between cell edge and render bounds
pub const CELL_PADDING: f32 = 10.0;

/// Padding between render bounds and content zone
pub const INTERNAL_PADDING: f32 = 20.0;

/// Minimum gap between stroked/filtered boxes
pub const MIN_BLOCK_GAP: f32 = 20.0;

/// Total number of cells in the grid
pub const TOTAL_CELLS: usize = (GRID_COLS as usize) * (GRID_ROWS as usize);

/// Canvas width in pixels (16 * 120)
pub const CANVAS_WIDTH: f32 = 1920.0;

/// Canvas height in pixels (9 * 120)
pub const CANVAS_HEIGHT: f32 = 1080.0;

// ── PixelBounds ────────────────────────────────────────────────────────────────

/// Axis-aligned rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelBounds {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl PixelBounds {
    /// Create new pixel bounds.
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Right edge x coordinate.
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge y coordinate.
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Center x coordinate.
    pub fn cx(&self) -> f32 {
        self.x + self.w / 2.0
    }

    /// Center y coordinate.
    pub fn cy(&self) -> f32 {
        self.y + self.h / 2.0
    }
}

impl fmt::Display for PixelBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}x{})", self.x, self.y, self.w, self.h)
    }
}

// ── GridSpan ───────────────────────────────────────────────────────────────────

/// A rectangular span of grid cells defined by top-left (c1, r1) and
/// bottom-right (c2, r2) inclusive.
///
/// Coordinates are 0-based: columns 0..15, rows 0..8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpan {
    /// Left column (inclusive, 0-based)
    pub c1: u8,
    /// Top row (inclusive, 0-based)
    pub r1: u8,
    /// Right column (inclusive, 0-based)
    pub c2: u8,
    /// Bottom row (inclusive, 0-based)
    pub r2: u8,
}

impl GridSpan {
    /// Create a new grid span. `c2 >= c1` and `r2 >= r1` required.
    pub fn new(c1: u8, r1: u8, c2: u8, r2: u8) -> Self {
        Self { c1, r1, c2, r2 }
    }

    /// Raw pixel bounds: cell edges with no padding.
    pub fn pixel_bounds(&self) -> PixelBounds {
        let x = self.c1 as f32 * CELL_SIZE;
        let y = self.r1 as f32 * CELL_SIZE;
        let w = (self.c2 - self.c1 + 1) as f32 * CELL_SIZE;
        let h = (self.r2 - self.r1 + 1) as f32 * CELL_SIZE;
        PixelBounds::new(x, y, w, h)
    }

    /// Render bounds: raw bounds inset by CELL_PADDING (10px) on each side.
    pub fn render_bounds(&self) -> PixelBounds {
        let raw = self.pixel_bounds();
        PixelBounds::new(
            raw.x + CELL_PADDING,
            raw.y + CELL_PADDING,
            raw.w - 2.0 * CELL_PADDING,
            raw.h - 2.0 * CELL_PADDING,
        )
    }

    /// Content zone: render bounds inset by INTERNAL_PADDING (20px).
    pub fn content_zone(&self) -> PixelBounds {
        let rb = self.render_bounds();
        PixelBounds::new(
            rb.x + INTERNAL_PADDING,
            rb.y + INTERNAL_PADDING,
            rb.w - 2.0 * INTERNAL_PADDING,
            rb.h - 2.0 * INTERNAL_PADDING,
        )
    }

    /// Enumerate all (column, row) cells in this span.
    pub fn cells(&self) -> Vec<(u8, u8)> {
        let mut out = Vec::with_capacity(self.cell_count());
        for r in self.r1..=self.r2 {
            for c in self.c1..=self.c2 {
                out.push((c, r));
            }
        }
        out
    }

    /// Number of cells in this span.
    pub fn cell_count(&self) -> usize {
        (self.c2 - self.c1 + 1) as usize * (self.r2 - self.r1 + 1) as usize
    }

    /// Check if the span is within grid bounds.
    fn is_in_bounds(&self) -> bool {
        self.c1 <= self.c2 && self.r1 <= self.r2 && self.c2 < GRID_COLS && self.r2 < GRID_ROWS
    }
}

impl fmt::Display for GridSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})..({},{})", self.c1, self.r1, self.c2, self.r2)
    }
}

// ── GridError ──────────────────────────────────────────────────────────────────

/// Errors from grid allocation.
#[derive(Debug, Clone)]
pub enum GridError {
    /// A cell in the requested span is already occupied.
    CellOccupied { col: u8, row: u8, existing_name: String },
    /// The span extends outside the 16x9 grid.
    OutOfBounds { span: GridSpan },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CellOccupied { col, row, existing_name } => {
                write!(f, "Cell ({}, {}) already occupied by '{}'", col, row, existing_name)
            }
            Self::OutOfBounds { span } => {
                write!(f, "Span {} is outside the {}x{} grid", span, GRID_COLS, GRID_ROWS)
            }
        }
    }
}

impl std::error::Error for GridError {}

// ── Allocation (internal) ──────────────────────────────────────────────────────

/// A recorded allocation in the grid.
#[derive(Debug, Clone)]
struct Allocation {
    name: String,
    span: GridSpan,
    step: usize,
}

// ── GridProtocol ───────────────────────────────────────────────────────────────

/// Occupied-set engine for provable non-overlap cell allocation.
#[derive(Debug)]
pub struct GridProtocol {
    /// Set of occupied (col, row) cells.
    occupied: HashSet<(u8, u8)>,
    /// Name lookup for occupied cells (for error messages).
    cell_owner: std::collections::HashMap<(u8, u8), String>,
    /// Ordered allocation log.
    allocations: Vec<Allocation>,
}

impl GridProtocol {
    /// Create a new empty grid protocol.
    pub fn new() -> Self {
        Self {
            occupied: HashSet::new(),
            cell_owner: std::collections::HashMap::new(),
            allocations: Vec::new(),
        }
    }

    /// Allocate a named region. Returns the render bounds on success.
    pub fn allocate(&mut self, name: &str, span: GridSpan) -> Result<PixelBounds, GridError> {
        if !span.is_in_bounds() {
            return Err(GridError::OutOfBounds { span });
        }

        // Check every cell for conflicts
        for (c, r) in span.cells() {
            if self.occupied.contains(&(c, r)) {
                let existing = self.cell_owner.get(&(c, r)).cloned().unwrap_or_default();
                return Err(GridError::CellOccupied { col: c, row: r, existing_name: existing });
            }
        }

        // Mark cells occupied
        let step = self.allocations.len();
        for (c, r) in span.cells() {
            self.occupied.insert((c, r));
            self.cell_owner.insert((c, r), name.to_string());
        }

        self.allocations.push(Allocation { name: name.to_string(), span, step });

        Ok(span.render_bounds())
    }

    /// Dry-run check: would this span succeed?
    pub fn try_allocate(&self, span: &GridSpan) -> bool {
        if !span.is_in_bounds() {
            return false;
        }
        span.cells().iter().all(|cell| !self.occupied.contains(cell))
    }

    /// Number of occupied cells.
    pub fn cells_used(&self) -> usize {
        self.occupied.len()
    }

    /// Number of free cells.
    pub fn cells_free(&self) -> usize {
        TOTAL_CELLS - self.occupied.len()
    }

    /// Produce an XML comment manifest of all allocations.
    pub fn manifest(&self) -> String {
        let mut out = String::from("<!-- GRID PROTOCOL MANIFEST\n");
        out.push_str(&format!(
            "  Canvas: {}x{} | Grid: {}x{} | Cell: {}px\n",
            CANVAS_WIDTH, CANVAS_HEIGHT, GRID_COLS, GRID_ROWS, CELL_SIZE
        ));
        out.push_str(&format!(
            "  Cells used: {} / {} ({:.0}%)\n",
            self.cells_used(),
            TOTAL_CELLS,
            self.cells_used() as f32 / TOTAL_CELLS as f32 * 100.0
        ));
        out.push_str("  Allocations:\n");
        for alloc in &self.allocations {
            let rb = alloc.span.render_bounds();
            out.push_str(&format!(
                "    [{}] \"{}\" span={} render=({},{},{}x{})\n",
                alloc.step, alloc.name, alloc.span, rb.x, rb.y, rb.w, rb.h,
            ));
        }
        out.push_str("-->");
        out
    }
}

impl Default for GridProtocol {
    fn default() -> Self {
        Self::new()
    }
}

// ── LayoutTemplate ─────────────────────────────────────────────────────────────

/// Pre-built layout templates for common slide types (A-G).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutTemplate {
    /// A: Title slide — full-width title bar + centered subtitle
    TitleSlide,
    /// B: Two-column — left and right halves
    TwoColumn,
    /// C: Dashboard — header + 2x2 quadrants
    Dashboard,
    /// D: Code walkthrough — code left, notes right
    CodeWalkthrough,
    /// E: Diagram — header + full-width diagram area
    Diagram,
    /// F: Key concepts — header + 3-column cards
    KeyConcepts,
    /// G: Reflection & readings — header + two sections
    ReflectionReadings,
}

impl LayoutTemplate {
    /// Return the named allocations for this template.
    pub fn allocations(&self) -> Vec<(&'static str, GridSpan)> {
        match self {
            Self::TitleSlide => vec![
                ("title", GridSpan::new(1, 2, 14, 4)),
                ("subtitle", GridSpan::new(2, 5, 13, 6)),
            ],
            Self::TwoColumn => vec![
                ("header", GridSpan::new(0, 0, 15, 1)),
                ("left", GridSpan::new(0, 2, 7, 8)),
                ("right", GridSpan::new(8, 2, 15, 8)),
            ],
            Self::Dashboard => vec![
                ("header", GridSpan::new(0, 0, 15, 1)),
                ("top_left", GridSpan::new(0, 2, 7, 4)),
                ("top_right", GridSpan::new(8, 2, 15, 4)),
                ("bottom_left", GridSpan::new(0, 5, 7, 8)),
                ("bottom_right", GridSpan::new(8, 5, 15, 8)),
            ],
            Self::CodeWalkthrough => vec![
                ("header", GridSpan::new(0, 0, 15, 1)),
                ("code", GridSpan::new(0, 2, 9, 8)),
                ("notes", GridSpan::new(10, 2, 15, 8)),
            ],
            Self::Diagram => vec![
                ("header", GridSpan::new(0, 0, 15, 1)),
                ("diagram", GridSpan::new(0, 2, 15, 8)),
            ],
            Self::KeyConcepts => vec![
                ("header", GridSpan::new(0, 0, 15, 1)),
                ("card_left", GridSpan::new(0, 2, 4, 8)),
                ("card_center", GridSpan::new(5, 2, 10, 8)),
                ("card_right", GridSpan::new(11, 2, 15, 8)),
            ],
            Self::ReflectionReadings => vec![
                ("header", GridSpan::new(0, 0, 15, 1)),
                ("reflection", GridSpan::new(0, 2, 15, 5)),
                ("readings", GridSpan::new(0, 6, 15, 8)),
            ],
        }
    }

    /// Allocate all regions for this template into the given protocol.
    pub fn apply(
        &self,
        protocol: &mut GridProtocol,
    ) -> Result<Vec<(&'static str, PixelBounds)>, GridError> {
        let mut results = Vec::new();
        for (name, span) in self.allocations() {
            let bounds = protocol.allocate(name, span)?;
            results.push((name, bounds));
        }
        Ok(results)
    }
}

impl fmt::Display for LayoutTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TitleSlide => write!(f, "A: Title Slide"),
            Self::TwoColumn => write!(f, "B: Two Column"),
            Self::Dashboard => write!(f, "C: Dashboard"),
            Self::CodeWalkthrough => write!(f, "D: Code Walkthrough"),
            Self::Diagram => write!(f, "E: Diagram"),
            Self::KeyConcepts => write!(f, "F: Key Concepts"),
            Self::ReflectionReadings => write!(f, "G: Reflection & Readings"),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(GRID_COLS, 16);
        assert_eq!(GRID_ROWS, 9);
        assert_eq!(CELL_SIZE, 120.0);
        assert_eq!(TOTAL_CELLS, 144);
        assert_eq!(CANVAS_WIDTH, 1920.0);
        assert_eq!(CANVAS_HEIGHT, 1080.0);
    }

    #[test]
    fn test_pixel_bounds() {
        let pb = PixelBounds::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(pb.right(), 110.0);
        assert_eq!(pb.bottom(), 70.0);
        assert_eq!(pb.cx(), 60.0);
        assert_eq!(pb.cy(), 45.0);
    }

    #[test]
    fn test_pixel_bounds_display() {
        let pb = PixelBounds::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(format!("{}", pb), "(10, 20, 100x50)");
    }

    #[test]
    fn test_grid_span_pixel_bounds() {
        // Single cell at (0, 0)
        let span = GridSpan::new(0, 0, 0, 0);
        let pb = span.pixel_bounds();
        assert_eq!(pb.x, 0.0);
        assert_eq!(pb.y, 0.0);
        assert_eq!(pb.w, 120.0);
        assert_eq!(pb.h, 120.0);

        // 2x2 block at (1, 1)
        let span = GridSpan::new(1, 1, 2, 2);
        let pb = span.pixel_bounds();
        assert_eq!(pb.x, 120.0);
        assert_eq!(pb.y, 120.0);
        assert_eq!(pb.w, 240.0);
        assert_eq!(pb.h, 240.0);
    }

    #[test]
    fn test_grid_span_render_bounds() {
        let span = GridSpan::new(0, 0, 0, 0);
        let rb = span.render_bounds();
        assert_eq!(rb.x, 10.0);
        assert_eq!(rb.y, 10.0);
        assert_eq!(rb.w, 100.0);
        assert_eq!(rb.h, 100.0);
    }

    #[test]
    fn test_grid_span_content_zone() {
        let span = GridSpan::new(0, 0, 0, 0);
        let cz = span.content_zone();
        assert_eq!(cz.x, 30.0);
        assert_eq!(cz.y, 30.0);
        assert_eq!(cz.w, 60.0);
        assert_eq!(cz.h, 60.0);
    }

    #[test]
    fn test_grid_span_cells() {
        let span = GridSpan::new(1, 2, 2, 3);
        let cells = span.cells();
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&(1, 2)));
        assert!(cells.contains(&(2, 2)));
        assert!(cells.contains(&(1, 3)));
        assert!(cells.contains(&(2, 3)));
    }

    #[test]
    fn test_grid_span_cell_count() {
        assert_eq!(GridSpan::new(0, 0, 0, 0).cell_count(), 1);
        assert_eq!(GridSpan::new(0, 0, 1, 1).cell_count(), 4);
        assert_eq!(GridSpan::new(0, 0, 15, 8).cell_count(), 144);
    }

    #[test]
    fn test_grid_span_display() {
        let span = GridSpan::new(1, 2, 3, 4);
        assert_eq!(format!("{}", span), "(1,2)..(3,4)");
    }

    #[test]
    fn test_grid_protocol_allocate() {
        let mut gp = GridProtocol::new();
        let result = gp.allocate("header", GridSpan::new(0, 0, 15, 1));
        assert!(result.is_ok());
        assert_eq!(gp.cells_used(), 32); // 16 * 2
        assert_eq!(gp.cells_free(), 144 - 32);
    }

    #[test]
    fn test_grid_protocol_overlap_rejected() {
        let mut gp = GridProtocol::new();
        gp.allocate("header", GridSpan::new(0, 0, 15, 1)).expect("unexpected failure");

        let result = gp.allocate("overlap", GridSpan::new(5, 0, 10, 2));
        assert!(result.is_err());
        match result.unwrap_err() {
            GridError::CellOccupied { col, row, existing_name } => {
                assert!((5..=10).contains(&col));
                assert_eq!(row, 0);
                assert_eq!(existing_name, "header");
            }
            other => panic!("Expected CellOccupied, got: {}", other),
        }
    }

    #[test]
    fn test_grid_protocol_out_of_bounds() {
        let mut gp = GridProtocol::new();
        let result = gp.allocate("oob", GridSpan::new(0, 0, 16, 0));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GridError::OutOfBounds { .. }));
    }

    #[test]
    fn test_grid_protocol_try_allocate() {
        let mut gp = GridProtocol::new();
        let span = GridSpan::new(0, 0, 3, 3);
        assert!(gp.try_allocate(&span));

        gp.allocate("block", span).expect("unexpected failure");
        assert!(!gp.try_allocate(&span));

        // Adjacent span should be free
        assert!(gp.try_allocate(&GridSpan::new(4, 0, 7, 3)));

        // Out of bounds
        assert!(!gp.try_allocate(&GridSpan::new(0, 0, 16, 0)));
    }

    #[test]
    fn test_grid_protocol_manifest() {
        let mut gp = GridProtocol::new();
        gp.allocate("header", GridSpan::new(0, 0, 15, 1)).expect("unexpected failure");
        gp.allocate("body", GridSpan::new(0, 2, 15, 8)).expect("unexpected failure");

        let manifest = gp.manifest();
        assert!(manifest.contains("GRID PROTOCOL MANIFEST"));
        assert!(manifest.contains("\"header\""));
        assert!(manifest.contains("\"body\""));
        assert!(manifest.contains("Cells used: 144"));
    }

    #[test]
    fn test_grid_protocol_default() {
        let gp = GridProtocol::default();
        assert_eq!(gp.cells_used(), 0);
        assert_eq!(gp.cells_free(), 144);
    }

    #[test]
    fn test_grid_error_display() {
        let err = GridError::CellOccupied { col: 5, row: 3, existing_name: "header".to_string() };
        let msg = format!("{}", err);
        assert!(msg.contains("(5, 3)"));
        assert!(msg.contains("header"));

        let err = GridError::OutOfBounds { span: GridSpan::new(0, 0, 16, 0) };
        let msg = format!("{}", err);
        assert!(msg.contains("outside"));
    }

    #[test]
    fn test_no_cell_in_two_allocations() {
        let mut gp = GridProtocol::new();
        gp.allocate("a", GridSpan::new(0, 0, 7, 4)).expect("unexpected failure");
        gp.allocate("b", GridSpan::new(8, 0, 15, 4)).expect("unexpected failure");
        gp.allocate("c", GridSpan::new(0, 5, 15, 8)).expect("unexpected failure");

        // Full grid is occupied
        assert_eq!(gp.cells_used(), 144);
        assert_eq!(gp.cells_free(), 0);
    }

    // ── Layout Template tests ──────────────────────────────────────────────

    #[test]
    fn test_layout_template_title_slide() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::TitleSlide.apply(&mut gp);
        assert!(result.is_ok());
        let allocs = result.expect("operation failed");
        assert_eq!(allocs.len(), 2);
        assert_eq!(allocs[0].0, "title");
        assert_eq!(allocs[1].0, "subtitle");
    }

    #[test]
    fn test_layout_template_two_column() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::TwoColumn.apply(&mut gp);
        assert!(result.is_ok());
        let allocs = result.expect("operation failed");
        assert_eq!(allocs.len(), 3);
    }

    #[test]
    fn test_layout_template_dashboard() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::Dashboard.apply(&mut gp);
        assert!(result.is_ok());
        let allocs = result.expect("operation failed");
        assert_eq!(allocs.len(), 5);
    }

    #[test]
    fn test_layout_template_code_walkthrough() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::CodeWalkthrough.apply(&mut gp);
        assert!(result.is_ok());
        let allocs = result.expect("operation failed");
        assert_eq!(allocs.len(), 3);
        assert_eq!(allocs[1].0, "code");
        assert_eq!(allocs[2].0, "notes");
    }

    #[test]
    fn test_layout_template_diagram() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::Diagram.apply(&mut gp);
        assert!(result.is_ok());
        let allocs = result.expect("operation failed");
        assert_eq!(allocs.len(), 2);
        assert_eq!(allocs[0].0, "header");
        assert_eq!(allocs[1].0, "diagram");
    }

    #[test]
    fn test_layout_template_key_concepts() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::KeyConcepts.apply(&mut gp);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation failed").len(), 4);
    }

    #[test]
    fn test_layout_template_reflection_readings() {
        let mut gp = GridProtocol::new();
        let result = LayoutTemplate::ReflectionReadings.apply(&mut gp);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation failed").len(), 3);
    }

    #[test]
    fn test_layout_template_no_overlaps() {
        // Every template must produce non-overlapping allocations
        let templates = [
            LayoutTemplate::TitleSlide,
            LayoutTemplate::TwoColumn,
            LayoutTemplate::Dashboard,
            LayoutTemplate::CodeWalkthrough,
            LayoutTemplate::Diagram,
            LayoutTemplate::KeyConcepts,
            LayoutTemplate::ReflectionReadings,
        ];

        for template in &templates {
            let mut gp = GridProtocol::new();
            let result = template.apply(&mut gp);
            assert!(
                result.is_ok(),
                "Template {} has overlapping allocations: {:?}",
                template,
                result.unwrap_err()
            );
        }
    }

    #[test]
    fn test_layout_template_display() {
        assert_eq!(format!("{}", LayoutTemplate::TitleSlide), "A: Title Slide");
        assert_eq!(format!("{}", LayoutTemplate::TwoColumn), "B: Two Column");
        assert_eq!(format!("{}", LayoutTemplate::Dashboard), "C: Dashboard");
        assert_eq!(format!("{}", LayoutTemplate::CodeWalkthrough), "D: Code Walkthrough");
        assert_eq!(format!("{}", LayoutTemplate::Diagram), "E: Diagram");
        assert_eq!(format!("{}", LayoutTemplate::KeyConcepts), "F: Key Concepts");
        assert_eq!(format!("{}", LayoutTemplate::ReflectionReadings), "G: Reflection & Readings");
    }

    #[test]
    fn test_canvas_dimensions_match_grid() {
        assert_eq!(CANVAS_WIDTH, GRID_COLS as f32 * CELL_SIZE);
        assert_eq!(CANVAS_HEIGHT, GRID_ROWS as f32 * CELL_SIZE);
    }

    #[test]
    fn test_full_grid_span() {
        let full = GridSpan::new(0, 0, 15, 8);
        assert_eq!(full.cell_count(), 144);
        let pb = full.pixel_bounds();
        assert_eq!(pb.w, CANVAS_WIDTH);
        assert_eq!(pb.h, CANVAS_HEIGHT);
    }

    #[test]
    fn test_render_bounds_shrink() {
        let span = GridSpan::new(0, 0, 1, 1);
        let raw = span.pixel_bounds();
        let render = span.render_bounds();
        assert_eq!(render.x, raw.x + CELL_PADDING);
        assert_eq!(render.y, raw.y + CELL_PADDING);
        assert_eq!(render.w, raw.w - 2.0 * CELL_PADDING);
        assert_eq!(render.h, raw.h - 2.0 * CELL_PADDING);
    }

    #[test]
    fn test_content_zone_shrink() {
        let span = GridSpan::new(0, 0, 3, 3);
        let render = span.render_bounds();
        let content = span.content_zone();
        assert_eq!(content.x, render.x + INTERNAL_PADDING);
        assert_eq!(content.y, render.y + INTERNAL_PADDING);
        assert_eq!(content.w, render.w - 2.0 * INTERNAL_PADDING);
        assert_eq!(content.h, render.h - 2.0 * INTERNAL_PADDING);
    }

    #[test]
    fn test_grid_protocol_sequential_allocations() {
        let mut gp = GridProtocol::new();

        // Tile the grid with 4x3 blocks (4 across, 3 down)
        for row_block in 0..3u8 {
            for col_block in 0..4u8 {
                let name = format!("block_{}_{}", col_block, row_block);
                let c1 = col_block * 4;
                let r1 = row_block * 3;
                let result = gp.allocate(&name, GridSpan::new(c1, r1, c1 + 3, r1 + 2));
                assert!(result.is_ok(), "Failed to allocate {}: {:?}", name, result);
            }
        }
        assert_eq!(gp.cells_used(), 144);
    }
}
