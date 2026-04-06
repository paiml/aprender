//! Geometry types for Grammar of Graphics.
//!
//! Defines visual representations of data.

use super::aes::Aes;
use super::stat::Stat;

/// Shape types for point geometries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointShape {
    /// Filled circle.
    #[default]
    Circle,
    /// Filled square.
    Square,
    /// Filled triangle.
    Triangle,
    /// Diamond shape.
    Diamond,
    /// Cross (+).
    Cross,
    /// X shape.
    X,
}

/// Geometry type specification.
#[derive(Debug, Clone)]
pub enum GeomType {
    /// Points.
    Point {
        /// Point shape.
        shape: PointShape,
    },
    /// Lines connecting points.
    Line {
        /// Line width.
        width: f32,
    },
    /// Area under a line.
    Area {
        /// Fill alpha.
        alpha: f32,
    },
    /// Bars.
    Bar {
        /// Bar width (0-1 fraction of available space).
        width: f32,
    },
    /// Histogram bars.
    Histogram {
        /// Number of bins.
        bins: usize,
    },
    /// Box plot.
    Boxplot,
    /// Violin plot.
    Violin,
    /// Tile/rectangle (for heatmaps).
    Tile,
    /// Text labels.
    Text,
    /// Horizontal line.
    Hline {
        /// Y intercept.
        yintercept: f32,
    },
    /// Vertical line.
    Vline {
        /// X intercept.
        xintercept: f32,
    },
    /// Smooth curve (loess/lm).
    Smooth {
        /// Smoothing method.
        method: SmoothMethod,
    },
}

/// Smoothing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmoothMethod {
    /// Local polynomial regression (LOESS).
    #[default]
    Loess,
    /// Linear regression.
    Linear,
}

/// A geometry layer.
#[derive(Debug, Clone)]
pub struct Geom {
    /// The geometry type.
    pub geom_type: GeomType,
    /// Layer-specific aesthetics.
    pub aes: Option<Aes>,
    /// Statistical transformation.
    pub stat: Option<Stat>,
}

impl Geom {
    /// Create a point geometry.
    #[must_use]
    pub fn point() -> Self {
        Self { geom_type: GeomType::Point { shape: PointShape::Circle }, aes: None, stat: None }
    }

    /// Create a line geometry.
    #[must_use]
    pub fn line() -> Self {
        Self { geom_type: GeomType::Line { width: 1.0 }, aes: None, stat: None }
    }

    /// Create an area geometry.
    #[must_use]
    pub fn area() -> Self {
        Self { geom_type: GeomType::Area { alpha: 0.3 }, aes: None, stat: None }
    }

    /// Create a bar geometry.
    #[must_use]
    pub fn bar() -> Self {
        Self { geom_type: GeomType::Bar { width: 0.8 }, aes: None, stat: Some(Stat::Count) }
    }

    /// Create a histogram geometry.
    #[must_use]
    pub fn histogram() -> Self {
        Self {
            geom_type: GeomType::Histogram { bins: 30 },
            aes: None,
            stat: Some(Stat::Bin { bins: 30 }),
        }
    }

    /// Create a box plot geometry.
    #[must_use]
    pub fn boxplot() -> Self {
        Self { geom_type: GeomType::Boxplot, aes: None, stat: Some(Stat::Boxplot) }
    }

    /// Create a violin plot geometry.
    #[must_use]
    pub fn violin() -> Self {
        Self { geom_type: GeomType::Violin, aes: None, stat: Some(Stat::Density) }
    }

    /// Create a tile geometry (for heatmaps).
    #[must_use]
    pub fn tile() -> Self {
        Self { geom_type: GeomType::Tile, aes: None, stat: None }
    }

    /// Create a text geometry.
    #[must_use]
    pub fn text() -> Self {
        Self { geom_type: GeomType::Text, aes: None, stat: None }
    }

    /// Create a horizontal line.
    #[must_use]
    pub fn hline(yintercept: f32) -> Self {
        Self { geom_type: GeomType::Hline { yintercept }, aes: None, stat: None }
    }

    /// Create a vertical line.
    #[must_use]
    pub fn vline(xintercept: f32) -> Self {
        Self { geom_type: GeomType::Vline { xintercept }, aes: None, stat: None }
    }

    /// Create a smooth line.
    #[must_use]
    pub fn smooth() -> Self {
        Self {
            geom_type: GeomType::Smooth { method: SmoothMethod::Loess },
            aes: None,
            stat: Some(Stat::Smooth),
        }
    }

    /// Set the point shape.
    #[must_use]
    pub fn shape(mut self, shape: PointShape) -> Self {
        if let GeomType::Point { shape: ref mut s } = self.geom_type {
            *s = shape;
        }
        self
    }

    /// Set the line/bar width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        match &mut self.geom_type {
            GeomType::Line { width: ref mut w } | GeomType::Bar { width: ref mut w } => *w = width,
            _ => {}
        }
        self
    }

    /// Set the number of bins.
    #[must_use]
    pub fn bins(mut self, bins: usize) -> Self {
        if let GeomType::Histogram { bins: ref mut b } = self.geom_type {
            *b = bins;
            self.stat = Some(Stat::Bin { bins });
        }
        self
    }

    /// Set the area alpha.
    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        if let GeomType::Area { alpha: ref mut a } = self.geom_type {
            *a = alpha.clamp(0.0, 1.0);
        }
        self
    }

    /// Add layer-specific aesthetics.
    #[must_use]
    pub fn aes(mut self, aes: Aes) -> Self {
        self.aes = Some(aes);
        self
    }

    /// Set statistical transformation.
    #[must_use]
    pub fn stat(mut self, stat: Stat) -> Self {
        self.stat = Some(stat);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geom_point() {
        let g = Geom::point().shape(PointShape::Square);
        match g.geom_type {
            GeomType::Point { shape } => assert_eq!(shape, PointShape::Square),
            _ => panic!("Expected point geom"),
        }
    }

    #[test]
    fn test_geom_line_width() {
        let g = Geom::line().width(2.5);
        match g.geom_type {
            GeomType::Line { width } => assert!((width - 2.5).abs() < 0.01),
            _ => panic!("Expected line geom"),
        }
    }

    #[test]
    fn test_geom_histogram_bins() {
        let g = Geom::histogram().bins(50);
        match g.geom_type {
            GeomType::Histogram { bins } => assert_eq!(bins, 50),
            _ => panic!("Expected histogram geom"),
        }
    }

    #[test]
    fn test_geom_with_aes() {
        let g = Geom::point().aes(Aes::new().color("category"));
        assert!(g.aes.is_some());
        assert_eq!(g.aes.expect("value should be present").color, Some("category".to_string()));
    }

    #[test]
    fn test_geom_area() {
        let g = Geom::area();
        match g.geom_type {
            GeomType::Area { alpha } => assert!((alpha - 0.3).abs() < 0.01),
            _ => panic!("Expected area geom"),
        }
    }

    #[test]
    fn test_geom_area_alpha() {
        let g = Geom::area().alpha(0.7);
        match g.geom_type {
            GeomType::Area { alpha } => assert!((alpha - 0.7).abs() < 0.01),
            _ => panic!("Expected area geom"),
        }
    }

    #[test]
    fn test_geom_area_alpha_clamp() {
        // Test clamping at bounds
        let g1 = Geom::area().alpha(1.5);
        let g2 = Geom::area().alpha(-0.5);
        match g1.geom_type {
            GeomType::Area { alpha } => assert!((alpha - 1.0).abs() < 0.01),
            _ => panic!("Expected area geom"),
        }
        match g2.geom_type {
            GeomType::Area { alpha } => assert!(alpha.abs() < 0.01),
            _ => panic!("Expected area geom"),
        }
    }

    #[test]
    fn test_geom_bar() {
        let g = Geom::bar();
        match g.geom_type {
            GeomType::Bar { width } => assert!((width - 0.8).abs() < 0.01),
            _ => panic!("Expected bar geom"),
        }
        assert!(g.stat.is_some());
    }

    #[test]
    fn test_geom_bar_width() {
        let g = Geom::bar().width(0.5);
        match g.geom_type {
            GeomType::Bar { width } => assert!((width - 0.5).abs() < 0.01),
            _ => panic!("Expected bar geom"),
        }
    }

    #[test]
    fn test_geom_boxplot() {
        let g = Geom::boxplot();
        assert!(matches!(g.geom_type, GeomType::Boxplot));
        assert!(g.stat.is_some());
    }

    #[test]
    fn test_geom_violin() {
        let g = Geom::violin();
        assert!(matches!(g.geom_type, GeomType::Violin));
        assert!(g.stat.is_some());
    }

    #[test]
    fn test_geom_tile() {
        let g = Geom::tile();
        assert!(matches!(g.geom_type, GeomType::Tile));
        assert!(g.stat.is_none());
    }

    #[test]
    fn test_geom_text() {
        let g = Geom::text();
        assert!(matches!(g.geom_type, GeomType::Text));
    }

    #[test]
    fn test_geom_hline() {
        let g = Geom::hline(5.0);
        match g.geom_type {
            GeomType::Hline { yintercept } => assert!((yintercept - 5.0).abs() < 0.01),
            _ => panic!("Expected hline geom"),
        }
    }

    #[test]
    fn test_geom_vline() {
        let g = Geom::vline(3.0);
        match g.geom_type {
            GeomType::Vline { xintercept } => assert!((xintercept - 3.0).abs() < 0.01),
            _ => panic!("Expected vline geom"),
        }
    }

    #[test]
    fn test_geom_smooth() {
        let g = Geom::smooth();
        match g.geom_type {
            GeomType::Smooth { method } => assert_eq!(method, SmoothMethod::Loess),
            _ => panic!("Expected smooth geom"),
        }
        assert!(g.stat.is_some());
    }

    #[test]
    fn test_geom_with_stat() {
        let g = Geom::point().stat(Stat::identity());
        assert!(g.stat.is_some());
    }

    #[test]
    fn test_point_shapes() {
        // Test all point shapes
        let shapes = [
            PointShape::Circle,
            PointShape::Square,
            PointShape::Triangle,
            PointShape::Diamond,
            PointShape::Cross,
            PointShape::X,
        ];
        for shape in shapes {
            let g = Geom::point().shape(shape);
            match g.geom_type {
                GeomType::Point { shape: s } => assert_eq!(s, shape),
                _ => panic!("Expected point geom"),
            }
        }
    }

    #[test]
    fn test_point_shape_default() {
        assert_eq!(PointShape::default(), PointShape::Circle);
    }

    #[test]
    fn test_smooth_method_default() {
        assert_eq!(SmoothMethod::default(), SmoothMethod::Loess);
    }

    #[test]
    fn test_shape_on_non_point() {
        // shape() on non-point geom should be no-op
        let g = Geom::line().shape(PointShape::Square);
        assert!(matches!(g.geom_type, GeomType::Line { .. }));
    }

    #[test]
    fn test_width_on_non_line_bar() {
        // width() on non-line/bar geom should be no-op
        let g = Geom::point().width(5.0);
        assert!(matches!(g.geom_type, GeomType::Point { .. }));
    }

    #[test]
    fn test_bins_on_non_histogram() {
        // bins() on non-histogram geom should be no-op
        let g = Geom::point().bins(100);
        assert!(matches!(g.geom_type, GeomType::Point { .. }));
    }

    #[test]
    fn test_alpha_on_non_area() {
        // alpha() on non-area geom should be no-op
        let g = Geom::point().alpha(0.5);
        assert!(matches!(g.geom_type, GeomType::Point { .. }));
    }

    #[test]
    fn test_geom_debug() {
        let geoms = vec![
            Geom::point(),
            Geom::line(),
            Geom::area(),
            Geom::bar(),
            Geom::histogram(),
            Geom::boxplot(),
            Geom::violin(),
            Geom::tile(),
            Geom::text(),
            Geom::hline(0.0),
            Geom::vline(0.0),
            Geom::smooth(),
        ];
        for g in geoms {
            let _ = format!("{g:?}");
        }
    }

    #[test]
    fn test_geom_clone() {
        let g1 = Geom::histogram().bins(25);
        let g2 = g1.clone();
        match g2.geom_type {
            GeomType::Histogram { bins } => assert_eq!(bins, 25),
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_smooth_methods_debug() {
        let methods = [SmoothMethod::Loess, SmoothMethod::Linear];
        for m in methods {
            let debug = format!("{m:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_geom_type_debug() {
        let types = vec![
            GeomType::Point { shape: PointShape::Circle },
            GeomType::Line { width: 1.0 },
            GeomType::Area { alpha: 0.5 },
            GeomType::Bar { width: 0.8 },
            GeomType::Histogram { bins: 30 },
            GeomType::Boxplot,
            GeomType::Violin,
            GeomType::Tile,
            GeomType::Text,
            GeomType::Hline { yintercept: 0.0 },
            GeomType::Vline { xintercept: 0.0 },
            GeomType::Smooth { method: SmoothMethod::Loess },
            GeomType::Smooth { method: SmoothMethod::Linear },
        ];
        for t in types {
            let debug = format!("{t:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_point_shape_clone_eq() {
        let shapes = [
            PointShape::Circle,
            PointShape::Square,
            PointShape::Triangle,
            PointShape::Diamond,
            PointShape::Cross,
            PointShape::X,
        ];
        for shape in shapes {
            let cloned = shape;
            assert_eq!(shape, cloned);
        }
    }
}
