//! Theme system for Grammar of Graphics.
//!
//! Controls the non-data visual appearance of plots.

use crate::color::Rgba;

/// Theme specification.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Background color.
    pub background: Rgba,
    /// Panel background color.
    pub panel_background: Rgba,
    /// Grid line color.
    pub grid_color: Rgba,
    /// Axis line color.
    pub axis_color: Rgba,
    /// Text color.
    pub text_color: Rgba,
    /// Show grid lines.
    pub show_grid: bool,
    /// Show axis lines.
    pub show_axis: bool,
    /// Show panel border.
    pub show_panel_border: bool,
    /// Grid line width.
    pub grid_width: f32,
    /// Axis line width.
    pub axis_width: f32,
    /// Margin around the plot.
    pub margin: u32,
}

impl Default for Theme {
    fn default() -> Self {
        Self::grey()
    }
}

impl Theme {
    /// Grey theme (ggplot2 default-like).
    #[must_use]
    pub fn grey() -> Self {
        Self {
            background: Rgba::WHITE,
            panel_background: Rgba::rgb(235, 235, 235),
            grid_color: Rgba::WHITE,
            axis_color: Rgba::rgb(50, 50, 50),
            text_color: Rgba::rgb(50, 50, 50),
            show_grid: true,
            show_axis: true,
            show_panel_border: false,
            grid_width: 1.0,
            axis_width: 1.0,
            margin: 40,
        }
    }

    /// Minimal theme with white background.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            background: Rgba::WHITE,
            panel_background: Rgba::WHITE,
            grid_color: Rgba::rgb(220, 220, 220),
            axis_color: Rgba::rgb(100, 100, 100),
            text_color: Rgba::BLACK,
            show_grid: true,
            show_axis: true,
            show_panel_border: false,
            grid_width: 0.5,
            axis_width: 0.5,
            margin: 40,
        }
    }

    /// Black and white theme.
    #[must_use]
    pub fn bw() -> Self {
        Self {
            background: Rgba::WHITE,
            panel_background: Rgba::WHITE,
            grid_color: Rgba::rgb(200, 200, 200),
            axis_color: Rgba::BLACK,
            text_color: Rgba::BLACK,
            show_grid: true,
            show_axis: true,
            show_panel_border: true,
            grid_width: 0.5,
            axis_width: 1.0,
            margin: 40,
        }
    }

    /// Classic theme with no grid.
    #[must_use]
    pub fn classic() -> Self {
        Self {
            background: Rgba::WHITE,
            panel_background: Rgba::WHITE,
            grid_color: Rgba::WHITE,
            axis_color: Rgba::BLACK,
            text_color: Rgba::BLACK,
            show_grid: false,
            show_axis: true,
            show_panel_border: false,
            grid_width: 0.0,
            axis_width: 1.0,
            margin: 40,
        }
    }

    /// Dark theme.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            background: Rgba::rgb(30, 30, 30),
            panel_background: Rgba::rgb(40, 40, 40),
            grid_color: Rgba::rgb(60, 60, 60),
            axis_color: Rgba::rgb(180, 180, 180),
            text_color: Rgba::rgb(220, 220, 220),
            show_grid: true,
            show_axis: true,
            show_panel_border: false,
            grid_width: 0.5,
            axis_width: 0.5,
            margin: 40,
        }
    }

    /// Void theme (nothing but data).
    #[must_use]
    pub fn void() -> Self {
        Self {
            background: Rgba::WHITE,
            panel_background: Rgba::WHITE,
            grid_color: Rgba::WHITE,
            axis_color: Rgba::WHITE,
            text_color: Rgba::WHITE,
            show_grid: false,
            show_axis: false,
            show_panel_border: false,
            grid_width: 0.0,
            axis_width: 0.0,
            margin: 10,
        }
    }

    /// Set background color.
    #[must_use]
    pub fn background(mut self, color: Rgba) -> Self {
        self.background = color;
        self
    }

    /// Set panel background color.
    #[must_use]
    pub fn panel_background(mut self, color: Rgba) -> Self {
        self.panel_background = color;
        self
    }

    /// Set grid color.
    #[must_use]
    pub fn grid_color(mut self, color: Rgba) -> Self {
        self.grid_color = color;
        self
    }

    /// Set margin.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Enable or disable grid lines.
    #[must_use]
    pub fn grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Enable or disable axis lines.
    #[must_use]
    pub fn axis(mut self, show: bool) -> Self {
        self.show_axis = show;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_grey() {
        let t = Theme::grey();
        assert!(t.show_grid);
        assert!(t.show_axis);
    }

    #[test]
    fn test_theme_dark() {
        let t = Theme::dark();
        assert_eq!(t.background.r, 30);
    }

    #[test]
    fn test_theme_customization() {
        let t = Theme::minimal().background(Rgba::rgb(250, 250, 250)).margin(50).grid(false);

        assert_eq!(t.margin, 50);
        assert!(!t.show_grid);
    }

    #[test]
    fn test_theme_minimal() {
        let t = Theme::minimal();
        assert_eq!(t.panel_background, Rgba::WHITE);
        assert!(t.show_grid);
        assert!((t.grid_width - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_theme_bw() {
        let t = Theme::bw();
        assert!(t.show_panel_border);
        assert_eq!(t.axis_color, Rgba::BLACK);
        assert_eq!(t.text_color, Rgba::BLACK);
    }

    #[test]
    fn test_theme_classic() {
        let t = Theme::classic();
        assert!(!t.show_grid);
        assert!(t.show_axis);
        assert!((t.grid_width).abs() < 0.01);
    }

    #[test]
    fn test_theme_void() {
        let t = Theme::void();
        assert!(!t.show_grid);
        assert!(!t.show_axis);
        assert!(!t.show_panel_border);
        assert_eq!(t.margin, 10);
    }

    #[test]
    fn test_theme_default() {
        let t = Theme::default();
        // Default is grey
        assert_eq!(t.background, Rgba::WHITE);
        assert!(t.show_grid);
    }

    #[test]
    fn test_theme_panel_background() {
        let t = Theme::minimal().panel_background(Rgba::rgb(240, 240, 240));
        assert_eq!(t.panel_background, Rgba::rgb(240, 240, 240));
    }

    #[test]
    fn test_theme_grid_color() {
        let t = Theme::minimal().grid_color(Rgba::rgb(200, 200, 200));
        assert_eq!(t.grid_color, Rgba::rgb(200, 200, 200));
    }

    #[test]
    fn test_theme_axis() {
        let t = Theme::minimal().axis(false);
        assert!(!t.show_axis);
    }

    #[test]
    fn test_theme_debug_clone() {
        let t1 = Theme::dark();
        let t2 = t1.clone();
        assert_eq!(t1.background.r, t2.background.r);
        let _ = format!("{t2:?}");
    }

    #[test]
    fn test_all_themes_valid() {
        // Verify all theme constructors produce valid themes
        let themes = [
            Theme::grey(),
            Theme::minimal(),
            Theme::bw(),
            Theme::classic(),
            Theme::dark(),
            Theme::void(),
        ];
        for t in themes {
            assert!(t.margin > 0 || t.margin == 10); // void has 10
            assert!(t.grid_width >= 0.0);
            assert!(t.axis_width >= 0.0);
        }
    }
}
