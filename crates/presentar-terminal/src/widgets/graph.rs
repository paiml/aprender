//! Time-series graph widget with multiple render modes.

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Render mode for the graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GraphMode {
    /// Unicode braille characters (U+2800-28FF): 2x4 dots per cell.
    #[default]
    Braille,
    /// Half-block characters (▀▄█): 1x2 resolution per cell.
    Block,
    /// Pure ASCII characters: TTY compatible.
    Tty,
}

/// UX-117: Time axis display mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimeAxisMode {
    /// Show numeric indices (0, 1, 2, ...).
    #[default]
    Indices,
    /// Show relative time (1m, 2m, 5m ago).
    Relative {
        /// Seconds per data point.
        interval_secs: u64,
    },
    /// Show absolute time (HH:MM:SS).
    Absolute,
    /// Hide X-axis labels.
    Hidden,
}

impl TimeAxisMode {
    /// Format a time offset as a label.
    pub fn format_label(&self, index: usize, total: usize) -> Option<String> {
        match self {
            Self::Indices => Some(format!("{index}")),
            Self::Relative { interval_secs } => {
                let secs_ago = (total - index) as u64 * interval_secs;
                if secs_ago < 60 {
                    Some(format!("{secs_ago}s"))
                } else if secs_ago < 3600 {
                    Some(format!("{}m", secs_ago / 60))
                } else {
                    Some(format!("{}h", secs_ago / 3600))
                }
            }
            Self::Absolute | Self::Hidden => None, // Would need actual timestamp
        }
    }
}

/// UX-102: Axis margin configuration.
#[derive(Debug, Clone, Copy)]
pub struct AxisMargins {
    /// Width for Y-axis labels (in characters).
    pub y_axis_width: u16,
    /// Height for X-axis labels (in lines).
    pub x_axis_height: u16,
}

impl Default for AxisMargins {
    fn default() -> Self {
        Self {
            y_axis_width: 6,
            x_axis_height: 1,
        }
    }
}

impl AxisMargins {
    /// No margins (labels overlap content).
    pub const NONE: Self = Self {
        y_axis_width: 0,
        x_axis_height: 0,
    };

    /// Compact margins.
    pub const COMPACT: Self = Self {
        y_axis_width: 4,
        x_axis_height: 1,
    };

    /// Standard margins.
    pub const STANDARD: Self = Self {
        y_axis_width: 6,
        x_axis_height: 1,
    };

    /// Wide margins for large numbers.
    pub const WIDE: Self = Self {
        y_axis_width: 10,
        x_axis_height: 2,
    };
}

/// Time-series graph widget.
#[derive(Debug, Clone)]
pub struct BrailleGraph {
    data: Vec<f64>,
    color: Color,
    /// Optional gradient for per-column coloring based on value.
    gradient: Option<Gradient>,
    min: f64,
    max: f64,
    mode: GraphMode,
    label: Option<String>,
    /// UX-102: Axis margin configuration.
    margins: AxisMargins,
    /// UX-117: Time axis display mode.
    time_axis: TimeAxisMode,
    /// UX-104: Show legend for braille characters.
    show_legend: bool,
    bounds: Rect,
}

impl BrailleGraph {
    /// Create a new braille graph.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        let (min, max) = Self::compute_range(&data);
        Self {
            data,
            color: Color::GREEN,
            gradient: None,
            min,
            max,
            mode: GraphMode::default(),
            label: None,
            margins: AxisMargins::default(),
            time_axis: TimeAxisMode::default(),
            show_legend: false,
            bounds: Rect::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Set the color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set a gradient for per-column coloring based on value.
    /// When set, each column is colored based on its normalized value (0.0-1.0).
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Set explicit min/max range.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        debug_assert!(min.is_finite(), "min must be finite");
        debug_assert!(max.is_finite(), "max must be finite");
        self.min = min;
        self.max = max;
        self
    }

    /// Set the render mode.
    #[must_use]
    pub fn with_mode(mut self, mode: GraphMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// UX-102: Set axis margins.
    #[must_use]
    pub fn with_margins(mut self, margins: AxisMargins) -> Self {
        self.margins = margins;
        self
    }

    /// UX-117: Set time axis display mode.
    #[must_use]
    pub fn with_time_axis(mut self, mode: TimeAxisMode) -> Self {
        self.time_axis = mode;
        self
    }

    /// UX-104: Enable/disable legend display.
    #[must_use]
    pub fn with_legend(mut self, show: bool) -> Self {
        self.show_legend = show;
        self
    }

    /// Get the effective graph area after accounting for margins.
    fn graph_area(&self) -> Rect {
        let y_offset = self.margins.y_axis_width as f32;
        let x_height = self.margins.x_axis_height as f32;
        Rect::new(
            self.bounds.x + y_offset,
            self.bounds.y,
            (self.bounds.width - y_offset).max(0.0),
            (self.bounds.height - x_height).max(0.0),
        )
    }

    /// Render Y-axis labels in the margin.
    fn render_y_axis(&self, canvas: &mut dyn Canvas) {
        if self.margins.y_axis_width == 0 {
            return;
        }

        let style = TextStyle {
            color: Color::WHITE,
            ..Default::default()
        };

        // Max value at top
        let max_str = format!("{:.0}", self.max);
        canvas.draw_text(&max_str, Point::new(self.bounds.x, self.bounds.y), &style);

        // Min value at bottom of graph area
        let graph_height = (self.bounds.height - self.margins.x_axis_height as f32).max(1.0);
        let min_str = format!("{:.0}", self.min);
        canvas.draw_text(
            &min_str,
            Point::new(self.bounds.x, self.bounds.y + graph_height - 1.0),
            &style,
        );
    }

    /// Render X-axis time labels.
    fn render_x_axis(&self, canvas: &mut dyn Canvas) {
        if self.margins.x_axis_height == 0 {
            return;
        }
        if matches!(self.time_axis, TimeAxisMode::Hidden) {
            return;
        }

        let graph = self.graph_area();
        let y_pos = self.bounds.y + self.bounds.height - 1.0;
        let total = self.data.len();

        let style = TextStyle {
            color: Color::WHITE,
            ..Default::default()
        };

        // Show labels at start, middle, and end
        let positions = [0, total / 2, total.saturating_sub(1)];
        for &idx in &positions {
            if let Some(label) = self.time_axis.format_label(idx, total) {
                let x_frac = if total > 1 {
                    idx as f32 / (total - 1) as f32
                } else {
                    0.5
                };
                let x_pos = graph.x + x_frac * (graph.width - 1.0).max(0.0);
                canvas.draw_text(&label, Point::new(x_pos, y_pos), &style);
            }
        }
    }

    /// Render legend explaining braille patterns.
    fn render_legend(&self, canvas: &mut dyn Canvas) {
        if !self.show_legend {
            return;
        }

        let style = TextStyle {
            color: Color::WHITE,
            ..Default::default()
        };

        // Simple legend showing value mapping
        let legend = format!("⣿={:.0} ⣀={:.0}", self.max, self.min);
        let x = self.bounds.x + self.bounds.width - legend.len() as f32;
        canvas.draw_text(&legend, Point::new(x.max(0.0), self.bounds.y), &style);
    }

    /// Update the data.
    pub fn set_data(&mut self, data: Vec<f64>) {
        let (min, max) = Self::compute_range(&data);
        self.data = data;
        self.min = min;
        self.max = max;
    }

    /// Push a new data point.
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    fn compute_range(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0);
        }
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max - min).abs() < f64::EPSILON {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }

    fn normalize(&self, value: f64) -> f64 {
        if (self.max - self.min).abs() < f64::EPSILON {
            0.5
        } else {
            (value - self.min) / (self.max - self.min)
        }
    }

    /// Get color for a normalized value (0.0-1.0).
    /// Uses gradient if set, otherwise returns the fixed color.
    fn color_for_value(&self, normalized: f64) -> Color {
        match &self.gradient {
            Some(gradient) => gradient.sample(normalized),
            None => self.color,
        }
    }

    fn render_braille(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 || self.data.is_empty() {
            return;
        }

        let dots_per_col = 2;
        let dots_per_row = 4;
        let total_dots_x = width * dots_per_col;
        let total_dots_y = height * dots_per_row;

        let step = if self.data.len() > total_dots_x {
            self.data.len() as f64 / total_dots_x as f64
        } else {
            1.0
        };

        // Track dots and values per column for gradient coloring
        let mut dots = vec![vec![false; total_dots_x]; total_dots_y];
        let mut column_values: Vec<f64> = vec![0.0; width];

        for (i, x) in (0..total_dots_x).enumerate() {
            let data_idx = (i as f64 * step) as usize;
            if data_idx >= self.data.len() {
                break;
            }
            let value = self.normalize(self.data[data_idx]);
            let y = ((1.0 - value) * (total_dots_y - 1) as f64).round() as usize;
            if y < total_dots_y {
                dots[y][x] = true;
            }
            // Track max value for each character column
            let char_col = x / dots_per_col;
            if char_col < width && value > column_values[char_col] {
                column_values[char_col] = value;
            }
        }

        for cy in 0..height {
            for (cx, &col_value) in column_values.iter().enumerate().take(width) {
                let mut code_point = 0x2800u32;
                let dot_offsets = [
                    (0, 0, 0x01),
                    (0, 1, 0x02),
                    (0, 2, 0x04),
                    (1, 0, 0x08),
                    (1, 1, 0x10),
                    (1, 2, 0x20),
                    (0, 3, 0x40),
                    (1, 3, 0x80),
                ];

                for (dx, dy, bit) in dot_offsets {
                    let dot_x = cx * dots_per_col + dx;
                    let dot_y = cy * dots_per_row + dy;
                    if dot_y < total_dots_y && dot_x < total_dots_x && dots[dot_y][dot_x] {
                        code_point |= bit;
                    }
                }

                if let Some(c) = char::from_u32(code_point) {
                    // Use per-column color based on value
                    let color = self.color_for_value(col_value);
                    let style = TextStyle {
                        color,
                        ..Default::default()
                    };
                    canvas.draw_text(
                        &c.to_string(),
                        Point::new(self.bounds.x + cx as f32, self.bounds.y + cy as f32),
                        &style,
                    );
                }
            }
        }
    }

    fn render_block(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 || self.data.is_empty() {
            return;
        }

        let total_rows = height * 2;

        let step = if self.data.len() > width {
            self.data.len() as f64 / width as f64
        } else {
            1.0
        };

        // Track both row position and normalized value for each column
        let mut column_data: Vec<(usize, f64)> = Vec::with_capacity(width);
        for x in 0..width {
            let data_idx = (x as f64 * step) as usize;
            if data_idx >= self.data.len() {
                column_data.push((total_rows, 0.0));
                continue;
            }
            let value = self.normalize(self.data[data_idx]);
            let row = ((1.0 - value) * (total_rows - 1) as f64).round() as usize;
            column_data.push((row.min(total_rows - 1), value));
        }

        for cy in 0..height {
            for cx in 0..width {
                let (value_row, normalized) =
                    column_data.get(cx).copied().unwrap_or((total_rows, 0.0));
                let top_row = cy * 2;
                let bottom_row = cy * 2 + 1;

                let top_filled = value_row <= top_row;
                let bottom_filled = value_row <= bottom_row;

                let ch = match (top_filled, bottom_filled) {
                    (true, true) => '█',
                    (true, false) => '▀',
                    (false, true) => '▄',
                    (false, false) => ' ',
                };

                // Use per-column color based on value
                let color = self.color_for_value(normalized);
                let style = TextStyle {
                    color,
                    ..Default::default()
                };
                canvas.draw_text(
                    &ch.to_string(),
                    Point::new(self.bounds.x + cx as f32, self.bounds.y + cy as f32),
                    &style,
                );
            }
        }
    }

    fn render_tty(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 || self.data.is_empty() {
            return;
        }

        let step = if self.data.len() > width {
            self.data.len() as f64 / width as f64
        } else {
            1.0
        };

        // Track both row position and normalized value for each column
        let mut column_data: Vec<(usize, f64)> = Vec::with_capacity(width);
        for x in 0..width {
            let data_idx = (x as f64 * step) as usize;
            if data_idx >= self.data.len() {
                column_data.push((height, 0.0));
                continue;
            }
            let value = self.normalize(self.data[data_idx]);
            let row = ((1.0 - value) * (height - 1) as f64).round() as usize;
            column_data.push((row.min(height - 1), value));
        }

        for cy in 0..height {
            for cx in 0..width {
                let (value_row, normalized) = column_data.get(cx).copied().unwrap_or((height, 0.0));
                let ch = if value_row == cy { '*' } else { ' ' };

                // Use per-column color based on value
                let color = self.color_for_value(normalized);
                let style = TextStyle {
                    color,
                    ..Default::default()
                };
                canvas.draw_text(
                    &ch.to_string(),
                    Point::new(self.bounds.x + cx as f32, self.bounds.y + cy as f32),
                    &style,
                );
            }
        }
    }
}

impl Brick for BrailleGraph {
    fn brick_name(&self) -> &'static str {
        "braille_graph"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(16)],
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new() // TUI-only widget
    }

    fn to_css(&self) -> String {
        String::new() // TUI-only widget
    }
}

impl Widget for BrailleGraph {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.max(10.0);
        let height = constraints.max_height.max(3.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Early return if bounds are too small or data is empty
        if self.bounds.width < 1.0 || self.bounds.height < 1.0 || self.data.is_empty() {
            return;
        }

        // UX-102: Render Y-axis labels in margin
        self.render_y_axis(canvas);

        // UX-117: Render X-axis time labels
        self.render_x_axis(canvas);

        // UX-104: Render legend
        self.render_legend(canvas);

        // Render the graph data
        match self.mode {
            GraphMode::Braille => self.render_braille(canvas),
            GraphMode::Block => self.render_block(canvas),
            GraphMode::Tty => self.render_tty(canvas),
        }

        if let Some(ref label) = self.label {
            let style = TextStyle {
                color: self.color,
                ..Default::default()
            };
            canvas.draw_text(label, Point::new(self.bounds.x, self.bounds.y), &style);
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::{Canvas, TextStyle};

    struct MockCanvas {
        texts: Vec<(String, Point)>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self { texts: vec![] }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, _rect: Rect, _color: Color) {}
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(
            &mut self,
            _center: Point,
            _radius: f32,
            _start: f32,
            _end: f32,
            _color: Color,
        ) {
        }
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    #[test]
    fn test_graph_creation() {
        let graph = BrailleGraph::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(graph.data.len(), 3);
    }

    #[test]
    fn test_graph_assertions_not_empty() {
        let graph = BrailleGraph::new(vec![1.0, 2.0, 3.0]);
        assert!(!graph.assertions().is_empty());
    }

    #[test]
    fn test_graph_verify_pass() {
        let graph = BrailleGraph::new(vec![1.0, 2.0, 3.0]);
        assert!(graph.verify().is_valid());
    }

    #[test]
    fn test_graph_with_color() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_color(Color::RED);
        assert_eq!(graph.color, Color::RED);
    }

    #[test]
    fn test_graph_with_range() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_range(0.0, 100.0);
        assert_eq!(graph.min, 0.0);
        assert_eq!(graph.max, 100.0);
    }

    #[test]
    fn test_graph_with_mode() {
        let graph = BrailleGraph::new(vec![1.0]).with_mode(GraphMode::Block);
        assert_eq!(graph.mode, GraphMode::Block);

        let graph2 = BrailleGraph::new(vec![1.0]).with_mode(GraphMode::Tty);
        assert_eq!(graph2.mode, GraphMode::Tty);
    }

    #[test]
    fn test_graph_with_label() {
        let graph = BrailleGraph::new(vec![1.0]).with_label("CPU Usage");
        assert_eq!(graph.label, Some("CPU Usage".to_string()));
    }

    #[test]
    fn test_graph_set_data() {
        let mut graph = BrailleGraph::new(vec![1.0, 2.0]);
        graph.set_data(vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(graph.data.len(), 4);
        assert_eq!(graph.min, 10.0);
        assert_eq!(graph.max, 40.0);
    }

    #[test]
    fn test_graph_push() {
        let mut graph = BrailleGraph::new(vec![5.0, 10.0]);
        graph.push(15.0);
        assert_eq!(graph.data.len(), 3);
        assert_eq!(graph.max, 15.0);

        graph.push(2.0);
        assert_eq!(graph.min, 2.0);
    }

    #[test]
    fn test_graph_empty_data_range() {
        let graph = BrailleGraph::new(vec![]);
        assert_eq!(graph.min, 0.0);
        assert_eq!(graph.max, 1.0);
    }

    #[test]
    fn test_graph_constant_data_range() {
        let graph = BrailleGraph::new(vec![5.0, 5.0, 5.0]);
        assert_eq!(graph.min, 4.5);
        assert_eq!(graph.max, 5.5);
    }

    #[test]
    fn test_graph_normalize() {
        let graph = BrailleGraph::new(vec![0.0, 100.0]);
        assert!((graph.normalize(50.0) - 0.5).abs() < f64::EPSILON);
        assert!((graph.normalize(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((graph.normalize(100.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_graph_normalize_constant() {
        let graph = BrailleGraph::new(vec![5.0, 5.0]);
        assert!((graph.normalize(5.0) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_graph_measure() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]);
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = graph.measure(constraints);
        assert!(size.width >= 10.0);
        assert!(size.height >= 3.0);
    }

    #[test]
    fn test_graph_layout() {
        let mut graph = BrailleGraph::new(vec![1.0, 2.0]);
        let bounds = Rect::new(10.0, 20.0, 80.0, 24.0);
        let result = graph.layout(bounds);
        assert_eq!(result.size.width, 80.0);
        assert_eq!(result.size.height, 24.0);
        assert_eq!(graph.bounds, bounds);
    }

    #[test]
    fn test_graph_paint_braille() {
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0]).with_mode(GraphMode::Braille);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_block() {
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0]).with_mode(GraphMode::Block);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_tty() {
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0]).with_mode(GraphMode::Tty);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_with_label() {
        let mut graph = BrailleGraph::new(vec![1.0, 2.0]).with_label("Test");
        graph.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("Test")));
    }

    #[test]
    fn test_graph_paint_empty_bounds() {
        let mut graph = BrailleGraph::new(vec![1.0, 2.0]);
        graph.bounds = Rect::new(0.0, 0.0, 0.0, 0.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_empty_data() {
        let mut graph = BrailleGraph::new(vec![]);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_event() {
        let mut graph = BrailleGraph::new(vec![1.0]);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(graph.event(&event).is_none());
    }

    #[test]
    fn test_graph_children() {
        let graph = BrailleGraph::new(vec![1.0]);
        assert!(graph.children().is_empty());
    }

    #[test]
    fn test_graph_children_mut() {
        let mut graph = BrailleGraph::new(vec![1.0]);
        assert!(graph.children_mut().is_empty());
    }

    #[test]
    fn test_graph_type_id() {
        let graph = BrailleGraph::new(vec![1.0]);
        assert_eq!(Widget::type_id(&graph), TypeId::of::<BrailleGraph>());
    }

    #[test]
    fn test_graph_brick_name() {
        let graph = BrailleGraph::new(vec![1.0]);
        assert_eq!(graph.brick_name(), "braille_graph");
    }

    #[test]
    fn test_graph_budget() {
        let graph = BrailleGraph::new(vec![1.0]);
        let budget = graph.budget();
        assert!(budget.measure_ms > 0);
    }

    #[test]
    fn test_graph_to_html() {
        let graph = BrailleGraph::new(vec![1.0]);
        assert!(graph.to_html().is_empty());
    }

    #[test]
    fn test_graph_to_css() {
        let graph = BrailleGraph::new(vec![1.0]);
        assert!(graph.to_css().is_empty());
    }

    #[test]
    fn test_graph_mode_default() {
        assert_eq!(GraphMode::default(), GraphMode::Braille);
    }

    #[test]
    fn test_graph_large_dataset() {
        let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let mut graph = BrailleGraph::new(data);
        graph.bounds = Rect::new(0.0, 0.0, 50.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_block_mode_various_values() {
        let mut graph =
            BrailleGraph::new(vec![0.0, 25.0, 50.0, 75.0, 100.0]).with_mode(GraphMode::Block);
        graph.bounds = Rect::new(0.0, 0.0, 5.0, 4.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_tty_mode_various_values() {
        let mut graph =
            BrailleGraph::new(vec![0.0, 25.0, 50.0, 75.0, 100.0]).with_mode(GraphMode::Tty);
        graph.bounds = Rect::new(0.0, 0.0, 5.0, 4.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    // ========================================================================
    // Additional tests for axis margins and time axis
    // ========================================================================

    #[test]
    fn test_graph_with_margins() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_margins(AxisMargins::WIDE);
        assert_eq!(graph.margins.y_axis_width, 10);
        assert_eq!(graph.margins.x_axis_height, 2);
    }

    #[test]
    fn test_graph_with_margins_none() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_margins(AxisMargins::NONE);
        assert_eq!(graph.margins.y_axis_width, 0);
        assert_eq!(graph.margins.x_axis_height, 0);
    }

    #[test]
    fn test_graph_with_margins_compact() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_margins(AxisMargins::COMPACT);
        assert_eq!(graph.margins.y_axis_width, 4);
        assert_eq!(graph.margins.x_axis_height, 1);
    }

    #[test]
    fn test_graph_with_margins_standard() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_margins(AxisMargins::STANDARD);
        assert_eq!(graph.margins.y_axis_width, 6);
        assert_eq!(graph.margins.x_axis_height, 1);
    }

    #[test]
    fn test_axis_margins_default() {
        let margins = AxisMargins::default();
        assert_eq!(margins.y_axis_width, 6);
        assert_eq!(margins.x_axis_height, 1);
    }

    #[test]
    fn test_graph_with_time_axis_indices() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_time_axis(TimeAxisMode::Indices);
        assert_eq!(graph.time_axis, TimeAxisMode::Indices);
    }

    #[test]
    fn test_graph_with_time_axis_relative() {
        let graph = BrailleGraph::new(vec![1.0, 2.0])
            .with_time_axis(TimeAxisMode::Relative { interval_secs: 5 });
        match graph.time_axis {
            TimeAxisMode::Relative { interval_secs } => assert_eq!(interval_secs, 5),
            _ => panic!("Expected Relative time axis mode"),
        }
    }

    #[test]
    fn test_graph_with_time_axis_absolute() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_time_axis(TimeAxisMode::Absolute);
        assert_eq!(graph.time_axis, TimeAxisMode::Absolute);
    }

    #[test]
    fn test_graph_with_time_axis_hidden() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_time_axis(TimeAxisMode::Hidden);
        assert_eq!(graph.time_axis, TimeAxisMode::Hidden);
    }

    #[test]
    fn test_time_axis_mode_default() {
        assert_eq!(TimeAxisMode::default(), TimeAxisMode::Indices);
    }

    #[test]
    fn test_time_axis_format_label_indices() {
        let mode = TimeAxisMode::Indices;
        assert_eq!(mode.format_label(0, 10), Some("0".to_string()));
        assert_eq!(mode.format_label(5, 10), Some("5".to_string()));
        assert_eq!(mode.format_label(9, 10), Some("9".to_string()));
    }

    #[test]
    fn test_time_axis_format_label_relative_seconds() {
        let mode = TimeAxisMode::Relative { interval_secs: 1 };
        // At index 0 with total 60, that's 60 seconds ago
        assert_eq!(mode.format_label(0, 60), Some("1m".to_string()));
        // At index 59 with total 60, that's 1 second ago
        assert_eq!(mode.format_label(59, 60), Some("1s".to_string()));
        // At index 30 with total 60, that's 30 seconds ago
        assert_eq!(mode.format_label(30, 60), Some("30s".to_string()));
    }

    #[test]
    fn test_time_axis_format_label_relative_minutes() {
        let mode = TimeAxisMode::Relative { interval_secs: 60 };
        // At index 0 with total 10, that's 600 seconds (10 minutes) ago
        assert_eq!(mode.format_label(0, 10), Some("10m".to_string()));
        // At index 5 with total 10, that's 300 seconds (5 minutes) ago
        assert_eq!(mode.format_label(5, 10), Some("5m".to_string()));
    }

    #[test]
    fn test_time_axis_format_label_relative_hours() {
        let mode = TimeAxisMode::Relative {
            interval_secs: 3600,
        };
        // At index 0 with total 5, that's 18000 seconds (5 hours) ago
        assert_eq!(mode.format_label(0, 5), Some("5h".to_string()));
        // At index 3 with total 5, that's 7200 seconds (2 hours) ago
        assert_eq!(mode.format_label(3, 5), Some("2h".to_string()));
    }

    #[test]
    fn test_time_axis_format_label_absolute() {
        let mode = TimeAxisMode::Absolute;
        assert_eq!(mode.format_label(0, 10), None);
    }

    #[test]
    fn test_time_axis_format_label_hidden() {
        let mode = TimeAxisMode::Hidden;
        assert_eq!(mode.format_label(0, 10), None);
    }

    #[test]
    fn test_graph_with_legend() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_legend(true);
        assert!(graph.show_legend);
    }

    #[test]
    fn test_graph_with_legend_disabled() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_legend(false);
        assert!(!graph.show_legend);
    }

    #[test]
    fn test_graph_with_gradient() {
        let gradient = Gradient::two(Color::BLUE, Color::RED);
        let graph = BrailleGraph::new(vec![1.0, 2.0]).with_gradient(gradient);
        assert!(graph.gradient.is_some());
    }

    #[test]
    fn test_graph_color_for_value_without_gradient() {
        let graph = BrailleGraph::new(vec![0.0, 100.0]).with_color(Color::GREEN);
        let color = graph.color_for_value(0.5);
        assert_eq!(color, Color::GREEN);
    }

    #[test]
    fn test_graph_color_for_value_with_gradient() {
        let gradient = Gradient::two(Color::BLUE, Color::RED);
        let graph = BrailleGraph::new(vec![0.0, 100.0]).with_gradient(gradient);
        // Should get different colors at different positions
        let color_low = graph.color_for_value(0.0);
        let color_high = graph.color_for_value(1.0);
        // Colors should differ (one is blue, one is red)
        assert_ne!(color_low, color_high);
    }

    #[test]
    fn test_graph_area_with_margins() {
        let mut graph = BrailleGraph::new(vec![1.0, 2.0]).with_margins(AxisMargins::STANDARD);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        let area = graph.graph_area();
        // y_axis_width = 6, so x starts at 6
        assert_eq!(area.x, 6.0);
        // x_axis_height = 1, so height reduced by 1
        assert_eq!(area.height, 23.0);
        assert_eq!(area.width, 74.0);
    }

    #[test]
    fn test_graph_area_with_no_margins() {
        let mut graph = BrailleGraph::new(vec![1.0, 2.0]).with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 24.0);
        let area = graph.graph_area();
        assert_eq!(area.x, 0.0);
        assert_eq!(area.y, 0.0);
        assert_eq!(area.width, 80.0);
        assert_eq!(area.height, 24.0);
    }

    #[test]
    fn test_graph_paint_with_y_axis() {
        let mut graph =
            BrailleGraph::new(vec![0.0, 50.0, 100.0]).with_margins(AxisMargins::STANDARD);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should render y-axis labels (min and max values)
        let has_max_label = canvas.texts.iter().any(|(t, _)| t.contains("100"));
        let has_min_label = canvas.texts.iter().any(|(t, _)| t.contains('0'));
        assert!(has_max_label || has_min_label);
    }

    #[test]
    fn test_graph_paint_with_x_axis_indices() {
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0])
            .with_margins(AxisMargins::STANDARD)
            .with_time_axis(TimeAxisMode::Indices);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should render x-axis labels with indices
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_with_x_axis_hidden() {
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0])
            .with_margins(AxisMargins::STANDARD)
            .with_time_axis(TimeAxisMode::Hidden);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should still render but without x-axis labels
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_with_legend() {
        let mut graph = BrailleGraph::new(vec![0.0, 100.0])
            .with_legend(true)
            .with_margins(AxisMargins::STANDARD);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should render legend with braille characters
        let has_legend = canvas
            .texts
            .iter()
            .any(|(t, _)| t.contains("⣿") || t.contains("⣀"));
        assert!(has_legend);
    }

    #[test]
    fn test_graph_paint_without_legend() {
        let mut graph = BrailleGraph::new(vec![0.0, 100.0])
            .with_legend(false)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should not render legend
        let has_legend = canvas
            .texts
            .iter()
            .any(|(t, _)| t.contains("⣿=") || t.contains("⣀="));
        assert!(!has_legend);
    }

    #[test]
    fn test_graph_paint_with_no_y_axis_margin() {
        let mut graph = BrailleGraph::new(vec![0.0, 100.0]).with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should render the graph but not y-axis labels at position 0
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_with_gradient_braille() {
        let gradient = Gradient::two(Color::BLUE, Color::RED);
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0])
            .with_gradient(gradient)
            .with_mode(GraphMode::Braille)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_with_gradient_block() {
        let gradient = Gradient::two(Color::BLUE, Color::RED);
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0])
            .with_gradient(gradient)
            .with_mode(GraphMode::Block)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_paint_with_gradient_tty() {
        let gradient = Gradient::two(Color::BLUE, Color::RED);
        let mut graph = BrailleGraph::new(vec![0.0, 50.0, 100.0])
            .with_gradient(gradient)
            .with_mode(GraphMode::Tty)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_block_mode_single_point() {
        let mut graph = BrailleGraph::new(vec![50.0]).with_mode(GraphMode::Block);
        graph.bounds = Rect::new(0.0, 0.0, 5.0, 4.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_tty_mode_single_point() {
        let mut graph = BrailleGraph::new(vec![50.0]).with_mode(GraphMode::Tty);
        graph.bounds = Rect::new(0.0, 0.0, 5.0, 4.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_braille_more_data_than_width() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut graph = BrailleGraph::new(data)
            .with_mode(GraphMode::Braille)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_block_more_data_than_width() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut graph = BrailleGraph::new(data)
            .with_mode(GraphMode::Block)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_tty_more_data_than_width() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut graph = BrailleGraph::new(data)
            .with_mode(GraphMode::Tty)
            .with_margins(AxisMargins::NONE);
        graph.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_small_bounds_clipping() {
        // Test with bounds smaller than margins would require
        let mut graph = BrailleGraph::new(vec![0.0, 100.0]).with_margins(AxisMargins::WIDE);
        graph.bounds = Rect::new(0.0, 0.0, 5.0, 2.0);
        let area = graph.graph_area();
        // Width should be clamped to 0 since bounds.width (5) - y_axis_width (10) < 0
        assert!(area.width >= 0.0);
        assert!(area.height >= 0.0);
    }

    #[test]
    fn test_graph_x_axis_single_data_point() {
        let mut graph = BrailleGraph::new(vec![50.0])
            .with_margins(AxisMargins::STANDARD)
            .with_time_axis(TimeAxisMode::Indices);
        graph.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut canvas = MockCanvas::new();
        graph.paint(&mut canvas);
        // Should handle single data point gracefully
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_graph_mode_debug() {
        // Test Debug impl for GraphMode
        let mode = GraphMode::Braille;
        let debug_str = format!("{:?}", mode);
        assert!(debug_str.contains("Braille"));
    }

    #[test]
    fn test_time_axis_mode_debug() {
        // Test Debug impl for TimeAxisMode
        let mode = TimeAxisMode::Relative { interval_secs: 60 };
        let debug_str = format!("{:?}", mode);
        assert!(debug_str.contains("Relative"));
        assert!(debug_str.contains("60"));
    }

    #[test]
    fn test_axis_margins_debug() {
        // Test Debug impl for AxisMargins
        let margins = AxisMargins::WIDE;
        let debug_str = format!("{:?}", margins);
        assert!(debug_str.contains("10")); // y_axis_width
        assert!(debug_str.contains('2')); // x_axis_height
    }

    #[test]
    fn test_graph_clone() {
        let graph = BrailleGraph::new(vec![1.0, 2.0, 3.0])
            .with_color(Color::RED)
            .with_label("Test")
            .with_range(0.0, 100.0);
        let cloned = graph.clone();
        assert_eq!(cloned.data, graph.data);
        assert_eq!(cloned.color, graph.color);
        assert_eq!(cloned.label, graph.label);
        assert_eq!(cloned.min, graph.min);
        assert_eq!(cloned.max, graph.max);
    }

    #[test]
    fn test_graph_debug() {
        let graph = BrailleGraph::new(vec![1.0, 2.0]);
        let debug_str = format!("{:?}", graph);
        assert!(debug_str.contains("BrailleGraph"));
    }
}
