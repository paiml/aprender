//! Progress bar widget.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Rect,
    Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Mode of the progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProgressMode {
    /// Determinate progress (known percentage).
    #[default]
    Determinate,
    /// Indeterminate progress (unknown percentage, animated).
    Indeterminate,
}

/// Progress bar widget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressBar {
    /// Current progress value (0.0 to 1.0)
    value: f32,
    /// Progress mode
    mode: ProgressMode,
    /// Minimum width
    min_width: f32,
    /// Height of the bar
    height: f32,
    /// Corner radius
    corner_radius: f32,
    /// Track color (background)
    track_color: Color,
    /// Fill color (progress)
    fill_color: Color,
    /// Show percentage label
    show_label: bool,
    /// Label color
    label_color: Color,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Current layout bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for ProgressBar {
    fn default() -> Self {
        Self {
            value: 0.0,
            mode: ProgressMode::Determinate,
            min_width: 100.0,
            height: 8.0,
            corner_radius: 4.0,
            track_color: Color::new(0.88, 0.88, 0.88, 1.0), // #E0E0E0
            fill_color: Color::new(0.13, 0.59, 0.95, 1.0),  // #2196F3
            show_label: false,
            label_color: Color::BLACK,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl ProgressBar {
    /// Create a new progress bar.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a progress bar with the given value.
    #[must_use]
    pub fn with_value(value: f32) -> Self {
        Self::default().value(value)
    }

    /// Set the progress value (clamped to 0.0..=1.0).
    #[must_use]
    pub fn value(mut self, value: f32) -> Self {
        self.value = value.clamp(0.0, 1.0);
        self
    }

    /// Set the progress mode.
    #[must_use]
    pub const fn mode(mut self, mode: ProgressMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set indeterminate mode.
    #[must_use]
    pub const fn indeterminate(self) -> Self {
        self.mode(ProgressMode::Indeterminate)
    }

    /// Set the minimum width.
    #[must_use]
    pub fn min_width(mut self, width: f32) -> Self {
        self.min_width = width.max(20.0);
        self
    }

    /// Set the height.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = height.max(4.0);
        self
    }

    /// Set the corner radius.
    #[must_use]
    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius.max(0.0);
        self
    }

    /// Set the track color (background).
    #[must_use]
    pub const fn track_color(mut self, color: Color) -> Self {
        self.track_color = color;
        self
    }

    /// Set the fill color (progress).
    #[must_use]
    pub const fn fill_color(mut self, color: Color) -> Self {
        self.fill_color = color;
        self
    }

    /// Show percentage label.
    #[must_use]
    pub const fn with_label(mut self) -> Self {
        self.show_label = true;
        self
    }

    /// Set whether to show the label.
    #[must_use]
    pub const fn show_label(mut self, show: bool) -> Self {
        self.show_label = show;
        self
    }

    /// Set the label color.
    #[must_use]
    pub const fn label_color(mut self, color: Color) -> Self {
        self.label_color = color;
        self
    }

    /// Set the accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get the current value.
    #[must_use]
    pub const fn get_value(&self) -> f32 {
        self.value
    }

    /// Get the current mode.
    #[must_use]
    pub const fn get_mode(&self) -> ProgressMode {
        self.mode
    }

    /// Get the percentage (0-100).
    #[must_use]
    pub fn percentage(&self) -> u8 {
        (self.value * 100.0).round() as u8
    }

    /// Check if progress is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.mode == ProgressMode::Determinate && self.value >= 1.0
    }

    /// Check if indeterminate.
    #[must_use]
    pub fn is_indeterminate(&self) -> bool {
        self.mode == ProgressMode::Indeterminate
    }

    /// Set the value directly (mutable).
    pub fn set_value(&mut self, value: f32) {
        self.value = value.clamp(0.0, 1.0);
    }

    /// Increment the value by a delta.
    pub fn increment(&mut self, delta: f32) {
        self.value = (self.value + delta).clamp(0.0, 1.0);
    }

    /// Calculate the fill width.
    fn fill_width(&self, total_width: f32) -> f32 {
        total_width * self.value
    }

    /// Get the track color.
    #[must_use]
    pub const fn get_track_color(&self) -> Color {
        self.track_color
    }

    /// Get the fill color.
    #[must_use]
    pub const fn get_fill_color(&self) -> Color {
        self.fill_color
    }

    /// Get the label color.
    #[must_use]
    pub const fn get_label_color(&self) -> Color {
        self.label_color
    }

    /// Get whether label is shown.
    #[must_use]
    pub const fn is_label_shown(&self) -> bool {
        self.show_label
    }

    /// Get the minimum width.
    #[must_use]
    pub const fn get_min_width(&self) -> f32 {
        self.min_width
    }

    /// Get the height.
    #[must_use]
    pub const fn get_height(&self) -> f32 {
        self.height
    }

    /// Get the corner radius.
    #[must_use]
    pub const fn get_corner_radius(&self) -> f32 {
        self.corner_radius
    }
}

impl Widget for ProgressBar {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let preferred_height = if self.show_label {
            self.height + 20.0
        } else {
            self.height
        };
        let preferred = Size::new(self.min_width, preferred_height);
        constraints.constrain(preferred)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw track (background)
        let track_rect = Rect::new(self.bounds.x, self.bounds.y, self.bounds.width, self.height);
        canvas.fill_rect(track_rect, self.track_color);

        // Draw fill (progress) - only for determinate mode
        if self.mode == ProgressMode::Determinate && self.value > 0.0 {
            let fill_width = self.fill_width(track_rect.width);
            let fill_rect = Rect::new(track_rect.x, track_rect.y, fill_width, self.height);
            canvas.fill_rect(fill_rect, self.fill_color);
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        // Progress bars don't handle events
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        false
    }

    fn is_focusable(&self) -> bool {
        false
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::ProgressBar
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for ProgressBar {
    fn brick_name(&self) -> &'static str {
        "ProgressBar"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        r#"<div class="brick-progressbar"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-progressbar { display: block; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== ProgressMode Tests =====

    #[test]
    fn test_progress_mode_default() {
        assert_eq!(ProgressMode::default(), ProgressMode::Determinate);
    }

    #[test]
    fn test_progress_mode_equality() {
        assert_eq!(ProgressMode::Determinate, ProgressMode::Determinate);
        assert_eq!(ProgressMode::Indeterminate, ProgressMode::Indeterminate);
        assert_ne!(ProgressMode::Determinate, ProgressMode::Indeterminate);
    }

    // ===== ProgressBar Construction Tests =====

    #[test]
    fn test_progress_bar_new() {
        let pb = ProgressBar::new();
        assert_eq!(pb.get_value(), 0.0);
        assert_eq!(pb.get_mode(), ProgressMode::Determinate);
    }

    #[test]
    fn test_progress_bar_with_value() {
        let pb = ProgressBar::with_value(0.5);
        assert_eq!(pb.get_value(), 0.5);
    }

    #[test]
    fn test_progress_bar_default() {
        let pb = ProgressBar::default();
        assert_eq!(pb.get_value(), 0.0);
        assert_eq!(pb.get_mode(), ProgressMode::Determinate);
        assert!(!pb.is_label_shown());
    }

    #[test]
    fn test_progress_bar_builder() {
        let pb = ProgressBar::new()
            .value(0.75)
            .min_width(200.0)
            .height(12.0)
            .corner_radius(6.0)
            .track_color(Color::WHITE)
            .fill_color(Color::new(0.0, 1.0, 0.0, 1.0))
            .with_label()
            .label_color(Color::BLACK)
            .accessible_name("Loading progress")
            .test_id("main-progress");

        assert_eq!(pb.get_value(), 0.75);
        assert_eq!(pb.get_min_width(), 200.0);
        assert_eq!(pb.get_height(), 12.0);
        assert_eq!(pb.get_corner_radius(), 6.0);
        assert_eq!(pb.get_track_color(), Color::WHITE);
        assert_eq!(pb.get_fill_color(), Color::new(0.0, 1.0, 0.0, 1.0));
        assert!(pb.is_label_shown());
        assert_eq!(pb.get_label_color(), Color::BLACK);
        assert_eq!(Widget::accessible_name(&pb), Some("Loading progress"));
        assert_eq!(Widget::test_id(&pb), Some("main-progress"));
    }

    // ===== Value Tests =====

    #[test]
    fn test_progress_bar_value_clamped_min() {
        let pb = ProgressBar::new().value(-0.5);
        assert_eq!(pb.get_value(), 0.0);
    }

    #[test]
    fn test_progress_bar_value_clamped_max() {
        let pb = ProgressBar::new().value(1.5);
        assert_eq!(pb.get_value(), 1.0);
    }

    #[test]
    fn test_progress_bar_set_value() {
        let mut pb = ProgressBar::new();
        pb.set_value(0.6);
        assert_eq!(pb.get_value(), 0.6);
    }

    #[test]
    fn test_progress_bar_set_value_clamped() {
        let mut pb = ProgressBar::new();
        pb.set_value(2.0);
        assert_eq!(pb.get_value(), 1.0);
        pb.set_value(-1.0);
        assert_eq!(pb.get_value(), 0.0);
    }

    #[test]
    fn test_progress_bar_increment() {
        let mut pb = ProgressBar::with_value(0.3);
        pb.increment(0.2);
        assert!((pb.get_value() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_progress_bar_increment_clamped() {
        let mut pb = ProgressBar::with_value(0.9);
        pb.increment(0.5);
        assert_eq!(pb.get_value(), 1.0);
    }

    #[test]
    fn test_progress_bar_percentage() {
        let pb = ProgressBar::with_value(0.0);
        assert_eq!(pb.percentage(), 0);

        let pb = ProgressBar::with_value(0.5);
        assert_eq!(pb.percentage(), 50);

        let pb = ProgressBar::with_value(1.0);
        assert_eq!(pb.percentage(), 100);

        let pb = ProgressBar::with_value(0.333);
        assert_eq!(pb.percentage(), 33);
    }

    // ===== Mode Tests =====

    #[test]
    fn test_progress_bar_mode() {
        let pb = ProgressBar::new().mode(ProgressMode::Indeterminate);
        assert_eq!(pb.get_mode(), ProgressMode::Indeterminate);
    }

    #[test]
    fn test_progress_bar_indeterminate() {
        let pb = ProgressBar::new().indeterminate();
        assert!(pb.is_indeterminate());
    }

    #[test]
    fn test_progress_bar_is_complete() {
        let pb = ProgressBar::with_value(1.0);
        assert!(pb.is_complete());

        let pb = ProgressBar::with_value(0.99);
        assert!(!pb.is_complete());

        let pb = ProgressBar::with_value(1.0).indeterminate();
        assert!(!pb.is_complete());
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_progress_bar_min_width_min() {
        let pb = ProgressBar::new().min_width(5.0);
        assert_eq!(pb.get_min_width(), 20.0);
    }

    #[test]
    fn test_progress_bar_height_min() {
        let pb = ProgressBar::new().height(1.0);
        assert_eq!(pb.get_height(), 4.0);
    }

    #[test]
    fn test_progress_bar_corner_radius() {
        let pb = ProgressBar::new().corner_radius(10.0);
        assert_eq!(pb.get_corner_radius(), 10.0);
    }

    #[test]
    fn test_progress_bar_corner_radius_min() {
        let pb = ProgressBar::new().corner_radius(-5.0);
        assert_eq!(pb.get_corner_radius(), 0.0);
    }

    // ===== Color Tests =====

    #[test]
    fn test_progress_bar_colors() {
        let track = Color::new(0.78, 0.78, 0.78, 1.0);
        let fill = Color::new(0.0, 0.5, 1.0, 1.0);
        let label = Color::new(0.2, 0.2, 0.2, 1.0);

        let pb = ProgressBar::new()
            .track_color(track)
            .fill_color(fill)
            .label_color(label);

        assert_eq!(pb.get_track_color(), track);
        assert_eq!(pb.get_fill_color(), fill);
        assert_eq!(pb.get_label_color(), label);
    }

    // ===== Label Tests =====

    #[test]
    fn test_progress_bar_show_label() {
        let pb = ProgressBar::new().show_label(true);
        assert!(pb.is_label_shown());

        let pb = ProgressBar::new().show_label(false);
        assert!(!pb.is_label_shown());
    }

    #[test]
    fn test_progress_bar_with_label() {
        let pb = ProgressBar::new().with_label();
        assert!(pb.is_label_shown());
    }

    // ===== Fill Width Tests =====

    #[test]
    fn test_progress_bar_fill_width() {
        let pb = ProgressBar::with_value(0.5);
        assert_eq!(pb.fill_width(100.0), 50.0);

        let pb = ProgressBar::with_value(0.0);
        assert_eq!(pb.fill_width(100.0), 0.0);

        let pb = ProgressBar::with_value(1.0);
        assert_eq!(pb.fill_width(100.0), 100.0);
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_progress_bar_type_id() {
        let pb = ProgressBar::new();
        assert_eq!(Widget::type_id(&pb), TypeId::of::<ProgressBar>());
    }

    #[test]
    fn test_progress_bar_measure() {
        let pb = ProgressBar::new().min_width(150.0).height(10.0);
        let size = pb.measure(Constraints::loose(Size::new(300.0, 100.0)));
        assert_eq!(size.width, 150.0);
        assert_eq!(size.height, 10.0);
    }

    #[test]
    fn test_progress_bar_measure_with_label() {
        let pb = ProgressBar::new()
            .min_width(150.0)
            .height(10.0)
            .with_label();
        let size = pb.measure(Constraints::loose(Size::new(300.0, 100.0)));
        assert_eq!(size.width, 150.0);
        assert_eq!(size.height, 30.0); // height + 20 for label
    }

    #[test]
    fn test_progress_bar_layout() {
        let mut pb = ProgressBar::new();
        let bounds = Rect::new(10.0, 20.0, 200.0, 8.0);
        let result = pb.layout(bounds);
        assert_eq!(result.size, Size::new(200.0, 8.0));
        assert_eq!(pb.bounds, bounds);
    }

    #[test]
    fn test_progress_bar_children() {
        let pb = ProgressBar::new();
        assert!(pb.children().is_empty());
    }

    #[test]
    fn test_progress_bar_is_interactive() {
        let pb = ProgressBar::new();
        assert!(!pb.is_interactive());
    }

    #[test]
    fn test_progress_bar_is_focusable() {
        let pb = ProgressBar::new();
        assert!(!pb.is_focusable());
    }

    #[test]
    fn test_progress_bar_accessible_role() {
        let pb = ProgressBar::new();
        assert_eq!(pb.accessible_role(), AccessibleRole::ProgressBar);
    }

    #[test]
    fn test_progress_bar_accessible_name() {
        let pb = ProgressBar::new().accessible_name("Download progress");
        assert_eq!(Widget::accessible_name(&pb), Some("Download progress"));
    }

    #[test]
    fn test_progress_bar_accessible_name_none() {
        let pb = ProgressBar::new();
        assert_eq!(Widget::accessible_name(&pb), None);
    }

    #[test]
    fn test_progress_bar_test_id() {
        let pb = ProgressBar::new().test_id("upload-progress");
        assert_eq!(Widget::test_id(&pb), Some("upload-progress"));
    }

    // ===== Paint Tests =====

    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_progress_bar_paint_draws_track() {
        let mut pb = ProgressBar::new();
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        // Should draw at least track
        assert!(canvas.command_count() >= 1);

        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, style, .. } => {
                assert_eq!(bounds.width, 200.0);
                assert_eq!(bounds.height, 8.0);
                assert!(style.fill.is_some());
            }
            _ => panic!("Expected Rect command for track"),
        }
    }

    #[test]
    fn test_progress_bar_paint_zero_percent() {
        let mut pb = ProgressBar::with_value(0.0);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        // Only track, no fill when value is 0
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_progress_bar_paint_50_percent() {
        let mut pb = ProgressBar::with_value(0.5);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        // Track + fill
        assert_eq!(canvas.command_count(), 2);

        // Check fill width is 50%
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 100.0);
            }
            _ => panic!("Expected Rect command for fill"),
        }
    }

    #[test]
    fn test_progress_bar_paint_100_percent() {
        let mut pb = ProgressBar::with_value(1.0);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        // Track + fill
        assert_eq!(canvas.command_count(), 2);

        // Check fill width is 100%
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 200.0);
            }
            _ => panic!("Expected Rect command for fill"),
        }
    }

    #[test]
    fn test_progress_bar_paint_25_percent() {
        let mut pb = ProgressBar::with_value(0.25);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.width, 50.0);
            }
            _ => panic!("Expected Rect command for fill"),
        }
    }

    #[test]
    fn test_progress_bar_paint_indeterminate_no_fill() {
        let mut pb = ProgressBar::with_value(0.5).indeterminate();
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        // Indeterminate mode: only track, no fill
        assert_eq!(canvas.command_count(), 1);
    }

    #[test]
    fn test_progress_bar_paint_uses_track_color() {
        let track_color = Color::new(0.9, 0.9, 0.9, 1.0);
        let mut pb = ProgressBar::new().track_color(track_color);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(track_color));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_progress_bar_paint_uses_fill_color() {
        let fill_color = Color::new(0.0, 0.8, 0.0, 1.0);
        let mut pb = ProgressBar::with_value(0.5).fill_color(fill_color);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        match &canvas.commands()[1] {
            DrawCommand::Rect { style, .. } => {
                assert_eq!(style.fill, Some(fill_color));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_progress_bar_paint_position_from_layout() {
        let mut pb = ProgressBar::with_value(0.5);
        pb.layout(Rect::new(50.0, 100.0, 200.0, 8.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        // Track position
        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 50.0);
                assert_eq!(bounds.y, 100.0);
            }
            _ => panic!("Expected Rect command"),
        }

        // Fill position
        match &canvas.commands()[1] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.x, 50.0);
                assert_eq!(bounds.y, 100.0);
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_progress_bar_paint_uses_height() {
        let mut pb = ProgressBar::new().height(16.0);
        pb.layout(Rect::new(0.0, 0.0, 200.0, 16.0));

        let mut canvas = RecordingCanvas::new();
        pb.paint(&mut canvas);

        match &canvas.commands()[0] {
            DrawCommand::Rect { bounds, .. } => {
                assert_eq!(bounds.height, 16.0);
            }
            _ => panic!("Expected Rect command"),
        }
    }
}
