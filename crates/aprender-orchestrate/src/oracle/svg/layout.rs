//! Layout Engine
//!
//! Grid-based layout with collision detection for diagram elements.

use super::shapes::{Point, Rect, Size};
use std::collections::HashMap;

/// Material Design 3 grid size (8px)
pub const GRID_SIZE: f32 = 8.0;

/// Standard viewport for diagrams
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    /// Width in pixels
    pub width: f32,
    /// Height in pixels
    pub height: f32,
    /// Padding from edges
    pub padding: f32,
}

impl Viewport {
    /// Create a new viewport
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            padding: GRID_SIZE * 3.0, // 24px default padding
        }
    }

    /// Standard 16:9 presentation viewport (1920x1080)
    pub fn presentation() -> Self {
        Self::new(1920.0, 1080.0)
    }

    /// Standard 4:3 document viewport (800x600)
    pub fn document() -> Self {
        Self::new(800.0, 600.0)
    }

    /// Square viewport
    pub fn square(size: f32) -> Self {
        Self::new(size, size)
    }

    /// Set padding
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// Get the usable content area
    pub fn content_area(&self) -> Rect {
        Rect::new(
            self.padding,
            self.padding,
            self.width - 2.0 * self.padding,
            self.height - 2.0 * self.padding,
        )
    }

    /// Get the center point
    pub fn center(&self) -> Point {
        Point::new(self.width / 2.0, self.height / 2.0)
    }

    /// Generate SVG viewBox attribute
    pub fn view_box(&self) -> String {
        format!("0 0 {} {}", self.width, self.height)
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self::presentation()
    }
}

/// Layout rectangle with ID for tracking
#[derive(Debug, Clone)]
pub struct LayoutRect {
    /// Unique ID
    pub id: String,
    /// Rectangle bounds
    pub rect: Rect,
    /// Layer (higher = on top)
    pub layer: i32,
}

impl LayoutRect {
    /// Create a new layout rect
    pub fn new(id: &str, rect: Rect) -> Self {
        Self { id: id.to_string(), rect, layer: 0 }
    }

    /// Set the layer
    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    /// Get the bounding box
    pub fn bounds(&self) -> &Rect {
        &self.rect
    }

    /// Check if this overlaps with another layout rect
    pub fn overlaps(&self, other: &LayoutRect) -> bool {
        self.rect.intersects(&other.rect)
    }
}

/// Simple layout engine with collision detection
#[derive(Debug)]
pub struct LayoutEngine {
    /// All placed elements
    pub elements: HashMap<String, LayoutRect>,
    /// Viewport
    viewport: Viewport,
    /// Grid size for snapping
    grid_size: f32,
}

impl LayoutEngine {
    /// Create a new layout engine
    pub fn new(viewport: Viewport) -> Self {
        Self { elements: HashMap::new(), viewport, grid_size: GRID_SIZE }
    }

    /// Set custom grid size
    pub fn with_grid_size(mut self, size: f32) -> Self {
        self.grid_size = size;
        self
    }

    /// Snap a value to the grid
    pub fn snap_to_grid(&self, value: f32) -> f32 {
        (value / self.grid_size).round() * self.grid_size
    }

    /// Snap a point to the grid
    pub fn snap_point(&self, point: Point) -> Point {
        Point::new(self.snap_to_grid(point.x), self.snap_to_grid(point.y))
    }

    /// Snap a rectangle to the grid
    pub fn snap_rect(&self, rect: &Rect) -> Rect {
        Rect::new(
            self.snap_to_grid(rect.position.x),
            self.snap_to_grid(rect.position.y),
            self.snap_to_grid(rect.size.width),
            self.snap_to_grid(rect.size.height),
        )
        .with_radius(rect.corner_radius)
    }

    /// Add an element to the layout
    pub fn add(&mut self, id: &str, rect: Rect) -> bool {
        let snapped = self.snap_rect(&rect);
        let layout_rect = LayoutRect::new(id, snapped);

        // Check for collisions
        if self.has_collision(&layout_rect) {
            return false;
        }

        self.elements.insert(id.to_string(), layout_rect);
        true
    }

    /// Add element with layer
    pub fn add_with_layer(&mut self, id: &str, rect: Rect, layer: i32) -> bool {
        let snapped = self.snap_rect(&rect);
        let layout_rect = LayoutRect::new(id, snapped).with_layer(layer);

        // Check for collisions on the same layer
        if self.has_collision_on_layer(&layout_rect, layer) {
            return false;
        }

        self.elements.insert(id.to_string(), layout_rect);
        true
    }

    /// Check if a rect would collide with existing elements
    pub fn has_collision(&self, new_rect: &LayoutRect) -> bool {
        for existing in self.elements.values() {
            if existing.id != new_rect.id && existing.overlaps(new_rect) {
                return true;
            }
        }
        false
    }

    /// Check for collision on a specific layer
    pub fn has_collision_on_layer(&self, new_rect: &LayoutRect, layer: i32) -> bool {
        for existing in self.elements.values() {
            if existing.id != new_rect.id && existing.layer == layer && existing.overlaps(new_rect)
            {
                return true;
            }
        }
        false
    }

    /// Get all elements that would overlap with a rect
    pub fn find_collisions(&self, rect: &Rect) -> Vec<&LayoutRect> {
        let test_rect = LayoutRect::new("_test", rect.clone());
        self.elements.values().filter(|e| e.overlaps(&test_rect)).collect()
    }

    /// Remove an element
    pub fn remove(&mut self, id: &str) -> Option<LayoutRect> {
        self.elements.remove(id)
    }

    /// Get an element by ID
    pub fn get(&self, id: &str) -> Option<&LayoutRect> {
        self.elements.get(id)
    }

    /// Get all elements
    pub fn all_elements(&self) -> impl Iterator<Item = &LayoutRect> {
        self.elements.values()
    }

    /// Get elements sorted by layer (back to front)
    pub fn elements_by_layer(&self) -> Vec<&LayoutRect> {
        let mut elements: Vec<_> = self.elements.values().collect();
        elements.sort_by_key(|e| e.layer);
        elements
    }

    /// Check if a point is inside any element
    pub fn element_at(&self, point: &Point) -> Option<&LayoutRect> {
        // Return topmost element (highest layer)
        let mut candidates: Vec<_> =
            self.elements.values().filter(|e| e.rect.contains(point)).collect();
        candidates.sort_by_key(|e| -e.layer);
        candidates.first().copied()
    }

    /// Find a free position for a rect with given size
    pub fn find_free_position(&self, size: Size, start: Point) -> Option<Point> {
        let content = self.viewport.content_area();
        let max_x = content.right() - size.width;
        let max_y = content.bottom() - size.height;

        // Try positions in a spiral pattern from start
        let mut x = self.snap_to_grid(start.x.max(content.position.x));
        let mut y = self.snap_to_grid(start.y.max(content.position.y));

        while y <= max_y {
            while x <= max_x {
                let test_rect = Rect::new(x, y, size.width, size.height);
                let layout_rect = LayoutRect::new("_test", test_rect);

                if !self.has_collision(&layout_rect) {
                    return Some(Point::new(x, y));
                }

                x += self.grid_size;
            }
            x = self.snap_to_grid(content.position.x);
            y += self.grid_size;
        }

        None
    }

    /// Check if a rect is within the viewport content area
    pub fn is_within_bounds(&self, rect: &Rect) -> bool {
        let content = self.viewport.content_area();
        rect.position.x >= content.position.x
            && rect.position.y >= content.position.y
            && rect.right() <= content.right()
            && rect.bottom() <= content.bottom()
    }

    /// Get layout validation errors
    pub fn validate(&self) -> Vec<LayoutError> {
        let mut errors = Vec::new();

        // Check for overlaps
        let elements: Vec<_> = self.elements.values().collect();
        for i in 0..elements.len() {
            for j in (i + 1)..elements.len() {
                if elements[i].layer == elements[j].layer && elements[i].overlaps(elements[j]) {
                    errors.push(LayoutError::Overlap {
                        id1: elements[i].id.clone(),
                        id2: elements[j].id.clone(),
                    });
                }
            }
        }

        // Check for out-of-bounds
        for element in &elements {
            if !self.is_within_bounds(&element.rect) {
                errors.push(LayoutError::OutOfBounds { id: element.id.clone() });
            }
        }

        // Check grid alignment
        for element in &elements {
            let rect = &element.rect;
            if rect.position.x % self.grid_size != 0.0 || rect.position.y % self.grid_size != 0.0 {
                errors.push(LayoutError::NotAligned { id: element.id.clone() });
            }
        }

        errors
    }

    /// Get the viewport
    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.elements.clear();
    }

    /// Get element count
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

impl Default for LayoutEngine {
    fn default() -> Self {
        Self::new(Viewport::default())
    }
}

/// Layout validation error
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutError {
    /// Two elements overlap
    Overlap { id1: String, id2: String },
    /// Element is outside viewport
    OutOfBounds { id: String },
    /// Element is not grid-aligned
    NotAligned { id: String },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overlap { id1, id2 } => write!(f, "Elements '{}' and '{}' overlap", id1, id2),
            Self::OutOfBounds { id } => write!(f, "Element '{}' is outside viewport", id),
            Self::NotAligned { id } => write!(f, "Element '{}' is not grid-aligned", id),
        }
    }
}

/// Auto-layout algorithms
pub mod auto_layout {
    use super::*;

    /// Arrange elements in a horizontal row
    pub fn row(elements: &[(&str, Size)], start: Point, spacing: f32) -> Vec<(String, Rect)> {
        let mut x = start.x;
        let mut result = Vec::new();

        for (id, size) in elements {
            result.push(((*id).to_string(), Rect::new(x, start.y, size.width, size.height)));
            x += size.width + spacing;
        }

        result
    }

    /// Arrange elements in a vertical column
    pub fn column(elements: &[(&str, Size)], start: Point, spacing: f32) -> Vec<(String, Rect)> {
        let mut y = start.y;
        let mut result = Vec::new();

        for (id, size) in elements {
            result.push(((*id).to_string(), Rect::new(start.x, y, size.width, size.height)));
            y += size.height + spacing;
        }

        result
    }

    /// Arrange elements in a grid
    pub fn grid(
        elements: &[(&str, Size)],
        start: Point,
        columns: usize,
        h_spacing: f32,
        v_spacing: f32,
    ) -> Vec<(String, Rect)> {
        let mut result = Vec::new();
        let mut x = start.x;
        let mut y = start.y;
        let mut row_height: f32 = 0.0;

        for (i, (id, size)) in elements.iter().enumerate() {
            if i > 0 && i % columns == 0 {
                // New row
                x = start.x;
                y += row_height + v_spacing;
                row_height = 0.0;
            }

            result.push(((*id).to_string(), Rect::new(x, y, size.width, size.height)));
            x += size.width + h_spacing;
            row_height = row_height.max(size.height);
        }

        result
    }

    /// Center elements horizontally within a viewport
    pub fn center_horizontal(
        elements: &[(String, Rect)],
        viewport: &Viewport,
    ) -> Vec<(String, Rect)> {
        if elements.is_empty() {
            return vec![];
        }

        // Calculate total width
        let min_x = elements.iter().map(|(_, r)| r.position.x).fold(f32::INFINITY, f32::min);
        let max_x = elements.iter().map(|(_, r)| r.right()).fold(f32::NEG_INFINITY, f32::max);
        let total_width = max_x - min_x;

        let center_offset = (viewport.width - total_width) / 2.0 - min_x;

        elements
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    Rect::new(
                        r.position.x + center_offset,
                        r.position.y,
                        r.size.width,
                        r.size.height,
                    ),
                )
            })
            .collect()
    }

    /// Center elements vertically within a viewport
    pub fn center_vertical(
        elements: &[(String, Rect)],
        viewport: &Viewport,
    ) -> Vec<(String, Rect)> {
        if elements.is_empty() {
            return vec![];
        }

        // Calculate total height
        let min_y = elements.iter().map(|(_, r)| r.position.y).fold(f32::INFINITY, f32::min);
        let max_y = elements.iter().map(|(_, r)| r.bottom()).fold(f32::NEG_INFINITY, f32::max);
        let total_height = max_y - min_y;

        let center_offset = (viewport.height - total_height) / 2.0 - min_y;

        elements
            .iter()
            .map(|(id, r)| {
                (
                    id.clone(),
                    Rect::new(
                        r.position.x,
                        r.position.y + center_offset,
                        r.size.width,
                        r.size.height,
                    ),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewport_creation() {
        let vp = Viewport::new(800.0, 600.0);
        assert_eq!(vp.width, 800.0);
        assert_eq!(vp.height, 600.0);
    }

    #[test]
    fn test_viewport_center() {
        let vp = Viewport::new(100.0, 100.0);
        let center = vp.center();
        assert_eq!(center.x, 50.0);
        assert_eq!(center.y, 50.0);
    }

    #[test]
    fn test_viewport_content_area() {
        let vp = Viewport::new(100.0, 100.0).with_padding(10.0);
        let content = vp.content_area();
        assert_eq!(content.position.x, 10.0);
        assert_eq!(content.position.y, 10.0);
        assert_eq!(content.size.width, 80.0);
        assert_eq!(content.size.height, 80.0);
    }

    #[test]
    fn test_layout_engine_add() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));

        assert!(engine.add("rect1", Rect::new(0.0, 0.0, 50.0, 50.0)));
        assert!(engine.add("rect2", Rect::new(60.0, 0.0, 50.0, 50.0)));

        // Overlapping rect should fail
        assert!(!engine.add("rect3", Rect::new(25.0, 25.0, 50.0, 50.0)));
    }

    #[test]
    fn test_layout_engine_snap() {
        let engine = LayoutEngine::new(Viewport::default());

        assert_eq!(engine.snap_to_grid(13.0), 16.0);
        assert_eq!(engine.snap_to_grid(12.0), 16.0);
        assert_eq!(engine.snap_to_grid(11.0), 8.0);
    }

    #[test]
    fn test_layout_engine_collision() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));

        engine.add("rect1", Rect::new(0.0, 0.0, 50.0, 50.0));

        let collisions = engine.find_collisions(&Rect::new(25.0, 25.0, 50.0, 50.0));
        assert_eq!(collisions.len(), 1);
        assert_eq!(collisions[0].id, "rect1");
    }

    #[test]
    fn test_layout_engine_layers() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));

        // Same position but different layers should work
        assert!(engine.add_with_layer("rect1", Rect::new(0.0, 0.0, 50.0, 50.0), 0));
        assert!(engine.add_with_layer("rect2", Rect::new(0.0, 0.0, 50.0, 50.0), 1));

        // Same layer should fail
        assert!(!engine.add_with_layer("rect3", Rect::new(0.0, 0.0, 50.0, 50.0), 0));
    }

    #[test]
    fn test_layout_engine_validate() {
        let mut engine = LayoutEngine::new(Viewport::new(100.0, 100.0).with_padding(0.0));

        engine
            .elements
            .insert("rect1".to_string(), LayoutRect::new("rect1", Rect::new(0.0, 0.0, 50.0, 50.0)));
        engine.elements.insert(
            "rect2".to_string(),
            LayoutRect::new("rect2", Rect::new(200.0, 0.0, 50.0, 50.0)), // Out of bounds
        );

        let errors = engine.validate();
        assert!(errors
            .iter()
            .any(|e| matches!(e, LayoutError::OutOfBounds { id } if id == "rect2")));
    }

    #[test]
    fn test_auto_layout_row() {
        let elements = vec![
            ("a", Size::new(50.0, 30.0)),
            ("b", Size::new(60.0, 30.0)),
            ("c", Size::new(40.0, 30.0)),
        ];

        let layout = auto_layout::row(&elements, Point::new(10.0, 10.0), 5.0);

        assert_eq!(layout[0].1.position.x, 10.0);
        assert_eq!(layout[1].1.position.x, 65.0); // 10 + 50 + 5
        assert_eq!(layout[2].1.position.x, 130.0); // 65 + 60 + 5
    }

    #[test]
    fn test_auto_layout_column() {
        let elements = vec![("a", Size::new(50.0, 30.0)), ("b", Size::new(50.0, 40.0))];

        let layout = auto_layout::column(&elements, Point::new(10.0, 10.0), 5.0);

        assert_eq!(layout[0].1.position.y, 10.0);
        assert_eq!(layout[1].1.position.y, 45.0); // 10 + 30 + 5
    }

    #[test]
    fn test_auto_layout_grid() {
        let elements = vec![
            ("a", Size::new(50.0, 30.0)),
            ("b", Size::new(50.0, 30.0)),
            ("c", Size::new(50.0, 30.0)),
            ("d", Size::new(50.0, 30.0)),
        ];

        let layout = auto_layout::grid(&elements, Point::new(0.0, 0.0), 2, 10.0, 10.0);

        assert_eq!(layout[0].1.position.x, 0.0);
        assert_eq!(layout[0].1.position.y, 0.0);
        assert_eq!(layout[1].1.position.x, 60.0); // 0 + 50 + 10
        assert_eq!(layout[1].1.position.y, 0.0);
        assert_eq!(layout[2].1.position.x, 0.0);
        assert_eq!(layout[2].1.position.y, 40.0); // 0 + 30 + 10
    }

    #[test]
    fn test_viewport_presentation() {
        let vp = Viewport::presentation();
        assert_eq!(vp.width, 1920.0);
        assert_eq!(vp.height, 1080.0);
    }

    #[test]
    fn test_viewport_document() {
        let vp = Viewport::document();
        assert_eq!(vp.width, 800.0);
        assert_eq!(vp.height, 600.0);
    }

    #[test]
    fn test_viewport_square() {
        let vp = Viewport::square(500.0);
        assert_eq!(vp.width, 500.0);
        assert_eq!(vp.height, 500.0);
    }

    #[test]
    fn test_viewport_view_box() {
        let vp = Viewport::new(100.0, 200.0);
        assert_eq!(vp.view_box(), "0 0 100 200");
    }

    #[test]
    fn test_viewport_default() {
        let vp = Viewport::default();
        assert_eq!(vp.width, 1920.0);
        assert_eq!(vp.height, 1080.0);
    }

    #[test]
    fn test_layout_rect_new() {
        let rect = LayoutRect::new("test", Rect::new(10.0, 20.0, 30.0, 40.0));
        assert_eq!(rect.id, "test");
        assert_eq!(rect.layer, 0);
    }

    #[test]
    fn test_layout_rect_with_layer() {
        let rect = LayoutRect::new("test", Rect::new(0.0, 0.0, 10.0, 10.0)).with_layer(5);
        assert_eq!(rect.layer, 5);
    }

    #[test]
    fn test_layout_rect_bounds() {
        let rect = LayoutRect::new("test", Rect::new(10.0, 20.0, 30.0, 40.0));
        let bounds = rect.bounds();
        assert_eq!(bounds.position.x, 10.0);
        assert_eq!(bounds.position.y, 20.0);
    }

    #[test]
    fn test_layout_rect_overlaps() {
        let rect1 = LayoutRect::new("r1", Rect::new(0.0, 0.0, 50.0, 50.0));
        let rect2 = LayoutRect::new("r2", Rect::new(25.0, 25.0, 50.0, 50.0));
        let rect3 = LayoutRect::new("r3", Rect::new(100.0, 100.0, 50.0, 50.0));
        assert!(rect1.overlaps(&rect2));
        assert!(!rect1.overlaps(&rect3));
    }

    #[test]
    fn test_layout_engine_get() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        engine.add("rect1", Rect::new(0.0, 0.0, 50.0, 50.0));

        assert!(engine.get("rect1").is_some());
        assert!(engine.get("nonexistent").is_none());
    }

    #[test]
    fn test_layout_engine_remove() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        engine.add("rect1", Rect::new(0.0, 0.0, 50.0, 50.0));

        let removed = engine.remove("rect1");
        assert!(removed.is_some());
        assert!(engine.get("rect1").is_none());
    }

    #[test]
    fn test_layout_engine_clear() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        engine.add("rect1", Rect::new(0.0, 0.0, 50.0, 50.0));
        engine.add("rect2", Rect::new(60.0, 0.0, 50.0, 50.0));

        engine.clear();
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
    }

    #[test]
    fn test_layout_engine_len_is_empty() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);

        engine.add("rect1", Rect::new(0.0, 0.0, 50.0, 50.0));
        assert!(!engine.is_empty());
        assert_eq!(engine.len(), 1);
    }

    #[test]
    fn test_layout_engine_viewport() {
        let vp = Viewport::new(123.0, 456.0);
        let engine = LayoutEngine::new(vp);
        assert_eq!(engine.viewport().width, 123.0);
        assert_eq!(engine.viewport().height, 456.0);
    }

    #[test]
    fn test_layout_engine_snap_point() {
        let engine = LayoutEngine::new(Viewport::default());
        let point = engine.snap_point(Point::new(13.0, 27.0));
        assert_eq!(point.x, 16.0);
        assert_eq!(point.y, 24.0);
    }

    #[test]
    fn test_layout_engine_elements_by_layer() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        engine.add_with_layer("back", Rect::new(0.0, 0.0, 50.0, 50.0), 0);
        engine.add_with_layer("front", Rect::new(60.0, 0.0, 50.0, 50.0), 2);
        engine.add_with_layer("middle", Rect::new(120.0, 0.0, 50.0, 50.0), 1);

        let elements = engine.elements_by_layer();
        assert_eq!(elements[0].layer, 0);
        assert_eq!(elements[1].layer, 1);
        assert_eq!(elements[2].layer, 2);
    }

    #[test]
    fn test_layout_engine_element_at() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        engine.add_with_layer("back", Rect::new(0.0, 0.0, 100.0, 100.0), 0);
        engine.add_with_layer("front", Rect::new(0.0, 0.0, 50.0, 50.0), 1);

        // Should return topmost element
        let element = engine.element_at(&Point::new(25.0, 25.0));
        assert!(element.is_some());
        assert_eq!(element.expect("unexpected failure").id, "front");

        // Point outside all elements
        let outside = engine.element_at(&Point::new(150.0, 150.0));
        assert!(outside.is_none());
    }

    #[test]
    fn test_layout_engine_find_free_position() {
        let mut engine = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));
        engine.add("block", Rect::new(0.0, 0.0, 80.0, 80.0));

        let free_pos = engine.find_free_position(Size::new(50.0, 50.0), Point::new(0.0, 0.0));
        assert!(free_pos.is_some());
        let pos = free_pos.expect("unexpected failure");
        // Should find a position that doesn't overlap
        assert!(pos.x >= 80.0 || pos.y >= 80.0);
    }

    #[test]
    fn test_layout_engine_is_within_bounds() {
        let engine = LayoutEngine::new(Viewport::new(100.0, 100.0).with_padding(10.0));

        // Within bounds
        let rect_in = Rect::new(10.0, 10.0, 50.0, 50.0);
        assert!(engine.is_within_bounds(&rect_in));

        // Outside bounds
        let rect_out = Rect::new(95.0, 95.0, 50.0, 50.0);
        assert!(!engine.is_within_bounds(&rect_out));
    }

    #[test]
    fn test_layout_engine_default() {
        let engine = LayoutEngine::default();
        assert_eq!(engine.viewport().width, 1920.0);
        assert!(engine.is_empty());
    }

    #[test]
    fn test_layout_error_display() {
        let overlap = LayoutError::Overlap { id1: "a".to_string(), id2: "b".to_string() };
        assert!(overlap.to_string().contains("overlap"));

        let oob = LayoutError::OutOfBounds { id: "c".to_string() };
        assert!(oob.to_string().contains("outside viewport"));

        let aligned = LayoutError::NotAligned { id: "d".to_string() };
        assert!(aligned.to_string().contains("not grid-aligned"));
    }

    #[test]
    fn test_auto_layout_center_horizontal_empty() {
        let result = auto_layout::center_horizontal(&[], &Viewport::new(100.0, 100.0));
        assert!(result.is_empty());
    }

    #[test]
    fn test_auto_layout_center_horizontal() {
        let elements = vec![
            ("a".to_string(), Rect::new(0.0, 10.0, 20.0, 20.0)),
            ("b".to_string(), Rect::new(30.0, 10.0, 20.0, 20.0)),
        ];
        let vp = Viewport::new(100.0, 100.0);
        let centered = auto_layout::center_horizontal(&elements, &vp);

        // Total width is 50 (from 0 to 50), centered in 100 should be at 25
        assert_eq!(centered[0].1.position.x, 25.0);
        assert_eq!(centered[1].1.position.x, 55.0);
    }

    #[test]
    fn test_auto_layout_center_vertical_empty() {
        let result = auto_layout::center_vertical(&[], &Viewport::new(100.0, 100.0));
        assert!(result.is_empty());
    }

    #[test]
    fn test_auto_layout_center_vertical() {
        let elements = vec![
            ("a".to_string(), Rect::new(10.0, 0.0, 20.0, 20.0)),
            ("b".to_string(), Rect::new(10.0, 30.0, 20.0, 20.0)),
        ];
        let vp = Viewport::new(100.0, 100.0);
        let centered = auto_layout::center_vertical(&elements, &vp);

        // Total height is 50 (from 0 to 50), centered in 100 should be at 25
        assert_eq!(centered[0].1.position.y, 25.0);
        assert_eq!(centered[1].1.position.y, 55.0);
    }

    #[test]
    fn test_layout_engine_with_grid_size() {
        let engine = LayoutEngine::new(Viewport::default()).with_grid_size(16.0);
        assert_eq!(engine.snap_to_grid(10.0), 16.0);
        assert_eq!(engine.snap_to_grid(24.0), 32.0);
    }

    #[test]
    fn test_layout_engine_snap_rect() {
        let engine = LayoutEngine::new(Viewport::default());
        let rect = Rect::new(13.0, 27.0, 45.0, 67.0).with_radius(5.0);
        let snapped = engine.snap_rect(&rect);
        assert_eq!(snapped.position.x, 16.0);
        assert_eq!(snapped.position.y, 24.0);
        assert_eq!(snapped.size.width, 48.0);
        assert_eq!(snapped.size.height, 64.0);
        assert_eq!(snapped.corner_radius, 5.0); // Preserved
    }

    #[test]
    fn test_layout_validate_not_aligned() {
        let mut engine = LayoutEngine::new(Viewport::new(100.0, 100.0).with_padding(0.0));
        // Force insert unaligned element
        engine.elements.insert(
            "unaligned".to_string(),
            LayoutRect::new("unaligned", Rect::new(3.0, 5.0, 10.0, 10.0)),
        );

        let errors = engine.validate();
        assert!(errors.iter().any(|e| matches!(e, LayoutError::NotAligned { .. })));
    }
}
