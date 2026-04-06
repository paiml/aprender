//! Resource bar widget for visualizing planned vs actual resource usage.
//!
//! Horizontal bar charts that show how actual usage compares to planned budgets.
//! Useful for tracking GPU hours, training time, or compute costs.

use crate::color::Rgba;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use crate::geometry::Rect;

/// A horizontal bar showing planned vs actual resource usage.
///
/// The bar displays two overlapping regions:
/// - Background bar showing the planned budget
/// - Foreground bar showing actual usage
///
/// Colors indicate whether usage is under or over budget.
#[derive(Debug, Clone)]
pub struct ResourceBar {
    /// Label for the resource (e.g., "GPU Hours", "Training Time").
    label: String,
    /// Planned/budgeted value.
    planned: f64,
    /// Actual value.
    actual: f64,
    /// Unit of measurement (e.g., "hours", "GB", "$").
    unit: String,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Color for under-budget usage.
    under_budget_color: Rgba,
    /// Color for over-budget usage.
    over_budget_color: Rgba,
    /// Background color for the planned bar.
    background_color: Rgba,
}

impl Default for ResourceBar {
    fn default() -> Self {
        Self {
            label: String::new(),
            planned: 1.0,
            actual: 0.0,
            unit: String::new(),
            width: 200,
            height: 20,
            under_budget_color: Rgba::rgb(76, 175, 80), // Material Green
            over_budget_color: Rgba::rgb(244, 67, 54),  // Material Red
            background_color: Rgba::rgb(224, 224, 224), // Light Gray
        }
    }
}

impl ResourceBar {
    /// Create a new resource bar.
    ///
    /// # Arguments
    ///
    /// * `label` - Display label for the resource
    /// * `planned` - Planned/budgeted value
    /// * `actual` - Actual usage value
    /// * `unit` - Unit of measurement
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        planned: f64,
        actual: f64,
        unit: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            planned: planned.max(0.0),
            actual: actual.max(0.0),
            unit: unit.into(),
            ..Self::default()
        }
    }

    /// Set the bar dimensions.
    #[must_use]
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(20);
        self.height = height.max(8);
        self
    }

    /// Set the color for under-budget usage.
    #[must_use]
    pub fn under_budget_color(mut self, color: Rgba) -> Self {
        self.under_budget_color = color;
        self
    }

    /// Set the color for over-budget usage.
    #[must_use]
    pub fn over_budget_color(mut self, color: Rgba) -> Self {
        self.over_budget_color = color;
        self
    }

    /// Set the background color.
    #[must_use]
    pub fn background_color(mut self, color: Rgba) -> Self {
        self.background_color = color;
        self
    }

    /// Calculate the actual/planned ratio as a percentage (0.0 to infinity).
    #[must_use]
    pub fn percentage(&self) -> f64 {
        if self.planned <= 0.0 {
            if self.actual <= 0.0 {
                return 0.0;
            }
            return f64::INFINITY;
        }
        (self.actual / self.planned) * 100.0
    }

    /// Check if actual usage exceeds the planned budget.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.actual > self.planned
    }

    /// Get the label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the planned value.
    #[must_use]
    pub fn planned(&self) -> f64 {
        self.planned
    }

    /// Get the actual value.
    #[must_use]
    pub fn actual(&self) -> f64 {
        self.actual
    }

    /// Get the unit.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Get the current fill color based on budget status.
    #[must_use]
    pub fn fill_color(&self) -> Rgba {
        if self.is_over_budget() {
            self.over_budget_color
        } else {
            self.under_budget_color
        }
    }

    /// Render the resource bar to a framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        // Draw background (planned bar - full width)
        let bg_rect = Rect::new(0.0, 0.0, self.width as f32, self.height as f32);
        fill_rect(fb, &bg_rect, self.background_color);

        // Calculate actual bar width
        let max_value = self.planned.max(self.actual);
        if max_value <= 0.0 {
            return Ok(());
        }

        let actual_width = ((self.actual / max_value) * f64::from(self.width)) as f32;
        let actual_width = actual_width.min(self.width as f32);

        // Draw actual usage bar
        let fill_color = self.fill_color();
        let actual_rect = Rect::new(0.0, 0.0, actual_width, self.height as f32);
        fill_rect(fb, &actual_rect, fill_color);

        // If over budget, draw planned marker line
        if self.is_over_budget() {
            let planned_x = ((self.planned / max_value) * f64::from(self.width)) as u32;
            let planned_x = planned_x.min(self.width - 1);
            for y in 0..self.height {
                fb.set_pixel(planned_x, y, Rgba::BLACK);
            }
        }

        Ok(())
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::TRANSPARENT);
        self.render(&mut fb)?;
        Ok(fb)
    }
}

/// Fill a rectangle with a solid color.
fn fill_rect(fb: &mut Framebuffer, rect: &Rect, color: Rgba) {
    let x_start = rect.x.max(0.0) as u32;
    let y_start = rect.y.max(0.0) as u32;
    let x_end = (rect.x + rect.width).min(fb.width() as f32) as u32;
    let y_end = (rect.y + rect.height).min(fb.height() as f32) as u32;

    for y in y_start..y_end {
        for x in x_start..x_end {
            fb.set_pixel(x, y, color);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_bar_percentage() {
        let bar = ResourceBar::new("GPU Hours", 100.0, 50.0, "hours");
        assert!((bar.percentage() - 50.0).abs() < f64::EPSILON);

        let bar = ResourceBar::new("GPU Hours", 100.0, 100.0, "hours");
        assert!((bar.percentage() - 100.0).abs() < f64::EPSILON);

        let bar = ResourceBar::new("GPU Hours", 100.0, 150.0, "hours");
        assert!((bar.percentage() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_bar_percentage_edge_cases() {
        // Zero planned, zero actual
        let bar = ResourceBar::new("GPU Hours", 0.0, 0.0, "hours");
        assert!((bar.percentage() - 0.0).abs() < f64::EPSILON);

        // Zero planned, non-zero actual
        let bar = ResourceBar::new("GPU Hours", 0.0, 50.0, "hours");
        assert!(bar.percentage().is_infinite());
    }

    #[test]
    fn test_resource_bar_over_budget() {
        let under = ResourceBar::new("Time", 10.0, 5.0, "hours");
        assert!(!under.is_over_budget());

        let at = ResourceBar::new("Time", 10.0, 10.0, "hours");
        assert!(!at.is_over_budget());

        let over = ResourceBar::new("Time", 10.0, 15.0, "hours");
        assert!(over.is_over_budget());
    }

    #[test]
    fn test_resource_bar_render() {
        let bar = ResourceBar::new("GPU Hours", 100.0, 75.0, "hours").dimensions(200, 20);

        let fb = bar.to_framebuffer();
        assert!(fb.is_ok());

        let fb = fb.expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 20);
    }

    #[test]
    fn test_resource_bar_render_over_budget() {
        let bar = ResourceBar::new("GPU Hours", 100.0, 150.0, "hours").dimensions(200, 20);

        let fb = bar.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_resource_bar_colors() {
        let bar = ResourceBar::new("Test", 100.0, 50.0, "units");
        assert_eq!(bar.fill_color(), bar.under_budget_color);

        let bar = ResourceBar::new("Test", 100.0, 150.0, "units");
        assert_eq!(bar.fill_color(), bar.over_budget_color);
    }

    #[test]
    fn test_resource_bar_accessors() {
        let bar = ResourceBar::new("GPU Hours", 100.0, 75.0, "hours");
        assert_eq!(bar.label(), "GPU Hours");
        assert!((bar.planned() - 100.0).abs() < f64::EPSILON);
        assert!((bar.actual() - 75.0).abs() < f64::EPSILON);
        assert_eq!(bar.unit(), "hours");
    }

    #[test]
    fn test_resource_bar_default() {
        let bar = ResourceBar::default();
        assert_eq!(bar.label(), "");
        assert_eq!(bar.unit(), "");
    }

    #[test]
    fn test_resource_bar_clone_debug() {
        let bar = ResourceBar::new("Test", 10.0, 5.0, "units");
        let cloned = bar.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("ResourceBar"));
    }

    #[test]
    fn test_resource_bar_custom_colors() {
        let bar = ResourceBar::new("Test", 100.0, 50.0, "units")
            .under_budget_color(Rgba::BLUE)
            .over_budget_color(Rgba::RED)
            .background_color(Rgba::WHITE);

        let fb = bar.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_resource_bar_dimensions_minimum() {
        // Test dimension clamping to minimums
        let bar = ResourceBar::new("Test", 10.0, 5.0, "units").dimensions(1, 1);
        let fb = bar.to_framebuffer();
        assert!(fb.is_ok());
    }
}
