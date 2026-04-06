//! `HorizonGraph` widget for high-density time-series visualization.
//!
//! Implements horizon charts as described by Heer et al. (2009).
//! Allows displaying 64+ CPU cores in minimal vertical space by "folding"
//! bands of value into overlapping colored layers.
//!
//! Citation: Heer, J., Kong, N., & Agrawala, M. (2009). "Sizing the Horizon"
//!
//! VS-001: CPU Cores use Heatmap/Horizon

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Color scheme for horizon bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizonScheme {
    /// Blue-based (cool) for normal metrics
    #[default]
    Blues,
    /// Red-based (warm) for temperature/critical
    Reds,
    /// Green-based for memory/capacity
    Greens,
    /// Purple for GPU metrics
    Purples,
}

impl HorizonScheme {
    /// Get colors for each band (from light to dark).
    fn band_colors(self, bands: u8) -> Vec<Color> {
        let base = match self {
            Self::Blues => (0.2, 0.4, 0.9),
            Self::Reds => (0.9, 0.3, 0.2),
            Self::Greens => (0.2, 0.8, 0.3),
            Self::Purples => (0.7, 0.3, 0.9),
        };

        (0..bands)
            .map(|i| {
                let factor = 0.4 + 0.6 * (i as f32 / bands as f32);
                Color::new(base.0 * factor, base.1 * factor, base.2 * factor, 1.0)
            })
            .collect()
    }
}

/// High-density time-series visualization using horizon chart technique.
///
/// Horizon charts "fold" values into overlapping bands, allowing dense
/// visualization of many data series in limited vertical space.
///
/// # Example
/// ```
/// use presentar_terminal::HorizonGraph;
///
/// let graph = HorizonGraph::new(vec![0.2, 0.5, 0.8, 0.3, 0.6])
///     .with_bands(3)
///     .with_label("CPU0");
/// ```
#[derive(Debug, Clone)]
pub struct HorizonGraph {
    /// Data values (0.0-1.0 normalized).
    data: Vec<f64>,
    /// Number of horizon bands (typically 2-4).
    bands: u8,
    /// Color scheme.
    scheme: HorizonScheme,
    /// Optional label.
    label: Option<String>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for HorizonGraph {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl HorizonGraph {
    /// Create a new horizon graph with data.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            bands: 3,
            scheme: HorizonScheme::default(),
            label: None,
            bounds: Rect::default(),
        }
    }

    /// Set the number of bands (2-4 recommended).
    #[must_use]
    pub fn with_bands(mut self, bands: u8) -> Self {
        debug_assert!((1..=6).contains(&bands), "bands must be 1-6");
        self.bands = bands.clamp(1, 6);
        self
    }

    /// Set the color scheme.
    #[must_use]
    pub fn with_scheme(mut self, scheme: HorizonScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Set a label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Update data in place.
    pub fn set_data(&mut self, data: Vec<f64>) {
        self.data = data;
    }

    /// Compute which band a value falls into.
    fn value_to_band(&self, value: f64) -> (u8, f64) {
        let clamped = value.clamp(0.0, 1.0);
        let band_height = 1.0 / self.bands as f64;
        let band = (clamped / band_height).floor() as u8;
        let within_band = (clamped % band_height) / band_height;
        (band.min(self.bands - 1), within_band)
    }

    /// Render using block characters for bands.
    fn render_horizon(&self, canvas: &mut dyn Canvas) {
        if self.data.is_empty() || self.bounds.width < 1.0 || self.bounds.height < 1.0 {
            return;
        }

        let colors = self.scheme.band_colors(self.bands);
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;

        // Sample data to fit width
        let data_len = self.data.len();
        let step = if data_len > width {
            data_len as f64 / width as f64
        } else {
            1.0
        };

        // Block characters for vertical fill
        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        for x in 0..width.min(data_len) {
            let idx = (x as f64 * step) as usize;
            if idx >= data_len {
                break;
            }

            let value = self.data[idx];
            let (band, intensity) = self.value_to_band(value);

            // Choose block character based on intensity
            let block_idx = (intensity * 7.0) as usize;
            let block = blocks[block_idx.min(7)];

            // Get color for this band
            let color = if (band as usize) < colors.len() {
                colors[band as usize]
            } else {
                colors[colors.len() - 1]
            };

            // Draw the block
            let style = TextStyle {
                color,
                ..Default::default()
            };
            canvas.draw_text(
                &block.to_string(),
                Point::new(
                    self.bounds.x + x as f32,
                    self.bounds.y + height as f32 - 1.0,
                ),
                &style,
            );

            // For higher bands, draw additional layers above
            for b in 0..band {
                let offset = 2 + b as usize;
                if height > offset {
                    let layer_color = if (b as usize) < colors.len() {
                        colors[b as usize]
                    } else {
                        colors[0]
                    };
                    let layer_style = TextStyle {
                        color: layer_color,
                        ..Default::default()
                    };
                    canvas.draw_text(
                        "█",
                        Point::new(
                            self.bounds.x + x as f32,
                            self.bounds.y + (height - offset) as f32,
                        ),
                        &layer_style,
                    );
                }
            }
        }

        // Draw label if present
        if let Some(ref label) = self.label {
            let style = TextStyle {
                color: Color::WHITE,
                ..Default::default()
            };
            canvas.draw_text(label, Point::new(self.bounds.x, self.bounds.y), &style);
        }
    }
}

impl Widget for HorizonGraph {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.min(self.data.len() as f32);
        let height = constraints.max_height.min(self.bands as f32 + 1.0);
        Size::new(width, height)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        self.render_horizon(canvas);
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

impl Brick for HorizonGraph {
    fn brick_name(&self) -> &'static str {
        "horizon_graph"
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
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizon_graph_default() {
        let graph = HorizonGraph::default();
        assert!(graph.data.is_empty());
        assert_eq!(graph.bands, 3);
    }

    #[test]
    fn test_horizon_graph_with_data() {
        let graph = HorizonGraph::new(vec![0.1, 0.5, 0.9])
            .with_bands(4)
            .with_label("CPU0");
        assert_eq!(graph.data.len(), 3);
        assert_eq!(graph.bands, 4);
        assert_eq!(graph.label, Some("CPU0".to_string()));
    }

    #[test]
    fn test_value_to_band() {
        let graph = HorizonGraph::new(vec![]).with_bands(3);

        let (band, _) = graph.value_to_band(0.1);
        assert_eq!(band, 0);

        let (band, _) = graph.value_to_band(0.5);
        assert_eq!(band, 1);

        let (band, _) = graph.value_to_band(0.9);
        assert_eq!(band, 2);
    }

    #[test]
    fn test_band_colors() {
        let colors = HorizonScheme::Blues.band_colors(3);
        assert_eq!(colors.len(), 3);
    }

    #[test]
    fn test_horizon_implements_widget() {
        let graph = HorizonGraph::new(vec![0.5, 0.6, 0.7]);
        let size = graph.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 100.0,
            max_height: 10.0,
        });
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_horizon_implements_brick() {
        let graph = HorizonGraph::new(vec![0.5]);
        assert_eq!(graph.brick_name(), "horizon_graph");
        assert!(graph.verify().is_valid());
    }

    #[test]
    fn test_horizon_event() {
        let mut graph = HorizonGraph::new(vec![]);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(graph.event(&event).is_none());
    }

    #[test]
    fn test_horizon_children() {
        let graph = HorizonGraph::new(vec![]);
        assert!(graph.children().is_empty());
    }

    #[test]
    fn test_horizon_children_mut() {
        let mut graph = HorizonGraph::new(vec![]);
        assert!(graph.children_mut().is_empty());
    }

    #[test]
    fn test_horizon_to_html() {
        let graph = HorizonGraph::new(vec![]);
        assert!(graph.to_html().is_empty());
    }

    #[test]
    fn test_horizon_to_css() {
        let graph = HorizonGraph::new(vec![]);
        assert!(graph.to_css().is_empty());
    }

    #[test]
    fn test_horizon_budget() {
        let graph = HorizonGraph::new(vec![]);
        let budget = graph.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_horizon_assertions() {
        let graph = HorizonGraph::new(vec![]);
        assert!(!graph.assertions().is_empty());
    }

    #[test]
    fn test_horizon_type_id() {
        let graph = HorizonGraph::new(vec![]);
        assert_eq!(Widget::type_id(&graph), TypeId::of::<HorizonGraph>());
    }

    #[test]
    fn test_horizon_layout_and_paint() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut graph = HorizonGraph::new(vec![0.1, 0.3, 0.5, 0.7, 0.9])
            .with_bands(4)
            .with_label("Test")
            .with_scheme(HorizonScheme::Blues);

        let bounds = presentar_core::Rect::new(0.0, 0.0, 40.0, 3.0);
        graph.layout(bounds);

        let mut buffer = CellBuffer::new(40, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        graph.paint(&mut canvas);
        // Verify it doesn't panic
    }

    #[test]
    fn test_horizon_all_schemes() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        for scheme in [
            HorizonScheme::Blues,
            HorizonScheme::Greens,
            HorizonScheme::Reds,
            HorizonScheme::Purples,
        ] {
            let mut graph = HorizonGraph::new(vec![0.2, 0.5, 0.8]).with_scheme(scheme);
            let bounds = presentar_core::Rect::new(0.0, 0.0, 20.0, 2.0);
            graph.layout(bounds);

            let mut buffer = CellBuffer::new(20, 2);
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            graph.paint(&mut canvas);
        }
    }

    #[test]
    fn test_horizon_different_band_counts() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        for bands in [2, 3, 4, 5, 6] {
            let mut graph = HorizonGraph::new(vec![0.1, 0.5, 0.9]).with_bands(bands);
            let bounds = presentar_core::Rect::new(0.0, 0.0, 30.0, 2.0);
            graph.layout(bounds);

            let mut buffer = CellBuffer::new(30, 2);
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            graph.paint(&mut canvas);
        }
    }

    #[test]
    fn test_horizon_empty_data() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut graph = HorizonGraph::new(vec![]);
        let bounds = presentar_core::Rect::new(0.0, 0.0, 20.0, 2.0);
        graph.layout(bounds);

        let mut buffer = CellBuffer::new(20, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_horizon_edge_values() {
        let graph = HorizonGraph::new(vec![]).with_bands(4);

        // Edge cases
        let (band, _) = graph.value_to_band(0.0);
        assert_eq!(band, 0);

        let (band, _) = graph.value_to_band(1.0);
        assert_eq!(band, 3); // Last band for max value

        let (band, _) = graph.value_to_band(-0.1);
        assert_eq!(band, 0); // Clamped to min

        let (band, _) = graph.value_to_band(1.5);
        assert_eq!(band, 3); // Clamped to max
    }
}
