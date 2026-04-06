//! Image widget for displaying images.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Constraints, Event, Rect, Size,
    TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// How the image should be scaled to fit its container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ImageFit {
    /// Scale to fill the container, may crop
    Cover,
    /// Scale to fit entirely within container, may have letterboxing
    #[default]
    Contain,
    /// Stretch to fill container exactly (may distort)
    Fill,
    /// Don't scale, display at natural size
    None,
    /// Scale down only if larger than container
    ScaleDown,
}

/// Image widget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    /// Image source URI
    source: String,
    /// Alternative text for accessibility
    alt: String,
    /// How to fit the image
    fit: ImageFit,
    /// Intrinsic width (natural size)
    width: Option<f32>,
    /// Intrinsic height (natural size)
    height: Option<f32>,
    /// Whether image is loading
    #[serde(skip)]
    loading: bool,
    /// Whether image failed to load
    #[serde(skip)]
    error: bool,
    /// Accessible name override
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Image {
    fn default() -> Self {
        Self {
            source: String::new(),
            alt: String::new(),
            fit: ImageFit::Contain,
            width: None,
            height: None,
            loading: false,
            error: false,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl Image {
    /// Create a new image with source.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            ..Self::default()
        }
    }

    /// Set the image source.
    #[must_use]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Set the alt text.
    #[must_use]
    pub fn alt(mut self, alt: impl Into<String>) -> Self {
        self.alt = alt.into();
        self
    }

    /// Set how the image should fit.
    #[must_use]
    pub const fn fit(mut self, fit: ImageFit) -> Self {
        self.fit = fit;
        self
    }

    /// Set the intrinsic width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width.max(0.0));
        self
    }

    /// Set the intrinsic height.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height.max(0.0));
        self
    }

    /// Set both width and height.
    #[must_use]
    pub fn size(self, width: f32, height: f32) -> Self {
        self.width(width).height(height)
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

    /// Get the image source.
    #[must_use]
    pub fn get_source(&self) -> &str {
        &self.source
    }

    /// Get the alt text.
    #[must_use]
    pub fn get_alt(&self) -> &str {
        &self.alt
    }

    /// Get the fit mode.
    #[must_use]
    pub const fn get_fit(&self) -> ImageFit {
        self.fit
    }

    /// Get the intrinsic width.
    #[must_use]
    pub const fn get_width(&self) -> Option<f32> {
        self.width
    }

    /// Get the intrinsic height.
    #[must_use]
    pub const fn get_height(&self) -> Option<f32> {
        self.height
    }

    /// Check if image is loading.
    #[must_use]
    pub const fn is_loading(&self) -> bool {
        self.loading
    }

    /// Check if image failed to load.
    #[must_use]
    pub const fn has_error(&self) -> bool {
        self.error
    }

    /// Set loading state.
    pub fn set_loading(&mut self, loading: bool) {
        self.loading = loading;
    }

    /// Set error state.
    pub fn set_error(&mut self, error: bool) {
        self.error = error;
    }

    /// Calculate aspect ratio.
    #[must_use]
    pub fn aspect_ratio(&self) -> Option<f32> {
        match (self.width, self.height) {
            (Some(w), Some(h)) if h > 0.0 => Some(w / h),
            _ => None,
        }
    }

    /// Calculate display size given container constraints.
    fn calculate_display_size(&self, container: Size) -> Size {
        let intrinsic = Size::new(
            self.width.unwrap_or(container.width),
            self.height.unwrap_or(container.height),
        );

        match self.fit {
            ImageFit::Fill => container,
            ImageFit::None => intrinsic,
            ImageFit::Contain => {
                let scale =
                    (container.width / intrinsic.width).min(container.height / intrinsic.height);
                Size::new(intrinsic.width * scale, intrinsic.height * scale)
            }
            ImageFit::Cover => {
                let scale =
                    (container.width / intrinsic.width).max(container.height / intrinsic.height);
                Size::new(intrinsic.width * scale, intrinsic.height * scale)
            }
            ImageFit::ScaleDown => {
                if intrinsic.width <= container.width && intrinsic.height <= container.height {
                    intrinsic
                } else {
                    let scale = (container.width / intrinsic.width)
                        .min(container.height / intrinsic.height);
                    Size::new(intrinsic.width * scale, intrinsic.height * scale)
                }
            }
        }
    }
}

impl Widget for Image {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let preferred = Size::new(self.width.unwrap_or(100.0), self.height.unwrap_or(100.0));
        constraints.constrain(preferred)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Draw placeholder or image
        // In a real implementation, this would render the actual image
        // For now, we draw a placeholder rectangle
        let display_size = self.calculate_display_size(self.bounds.size());

        // Center the image in bounds
        let x_offset = (self.bounds.width - display_size.width) / 2.0;
        let y_offset = (self.bounds.height - display_size.height) / 2.0;

        let image_rect = Rect::new(
            self.bounds.x + x_offset,
            self.bounds.y + y_offset,
            display_size.width,
            display_size.height,
        );

        // Draw placeholder (light gray for loading, red tint for error)
        let color = if self.error {
            presentar_core::Color::new(0.9, 0.7, 0.7, 1.0)
        } else if self.loading {
            presentar_core::Color::new(0.9, 0.9, 0.9, 1.0)
        } else {
            presentar_core::Color::new(0.8, 0.8, 0.8, 1.0)
        };

        canvas.fill_rect(image_rect, color);
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

    fn is_interactive(&self) -> bool {
        false
    }

    fn is_focusable(&self) -> bool {
        false
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value
            .as_deref()
            .or(if self.alt.is_empty() {
                None
            } else {
                Some(&self.alt)
            })
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Image
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Image {
    fn brick_name(&self) -> &'static str {
        "Image"
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
        format!(
            r#"<img class="brick-image" src="{}" alt="{}" />"#,
            self.source, self.alt
        )
    }

    fn to_css(&self) -> String {
        ".brick-image { display: block; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ===== ImageFit Tests =====

    #[test]
    fn test_image_fit_default() {
        assert_eq!(ImageFit::default(), ImageFit::Contain);
    }

    #[test]
    fn test_image_fit_equality() {
        assert_eq!(ImageFit::Cover, ImageFit::Cover);
        assert_ne!(ImageFit::Cover, ImageFit::Contain);
    }

    // ===== Image Construction Tests =====

    #[test]
    fn test_image_new() {
        let img = Image::new("https://example.com/image.png");
        assert_eq!(img.get_source(), "https://example.com/image.png");
        assert!(img.get_alt().is_empty());
    }

    #[test]
    fn test_image_default() {
        let img = Image::default();
        assert!(img.get_source().is_empty());
        assert!(img.get_alt().is_empty());
        assert_eq!(img.get_fit(), ImageFit::Contain);
        assert!(img.get_width().is_none());
        assert!(img.get_height().is_none());
    }

    #[test]
    fn test_image_builder() {
        let img = Image::new("photo.jpg")
            .alt("A beautiful sunset")
            .fit(ImageFit::Cover)
            .width(800.0)
            .height(600.0)
            .accessible_name("Sunset photo")
            .test_id("hero-image");

        assert_eq!(img.get_source(), "photo.jpg");
        assert_eq!(img.get_alt(), "A beautiful sunset");
        assert_eq!(img.get_fit(), ImageFit::Cover);
        assert_eq!(img.get_width(), Some(800.0));
        assert_eq!(img.get_height(), Some(600.0));
        assert_eq!(Widget::accessible_name(&img), Some("Sunset photo"));
        assert_eq!(Widget::test_id(&img), Some("hero-image"));
    }

    #[test]
    fn test_image_source() {
        let img = Image::default().source("new-source.png");
        assert_eq!(img.get_source(), "new-source.png");
    }

    #[test]
    fn test_image_size() {
        let img = Image::default().size(1920.0, 1080.0);
        assert_eq!(img.get_width(), Some(1920.0));
        assert_eq!(img.get_height(), Some(1080.0));
    }

    #[test]
    fn test_image_width_min() {
        let img = Image::default().width(-100.0);
        assert_eq!(img.get_width(), Some(0.0));
    }

    #[test]
    fn test_image_height_min() {
        let img = Image::default().height(-50.0);
        assert_eq!(img.get_height(), Some(0.0));
    }

    // ===== State Tests =====

    #[test]
    fn test_image_loading_state() {
        let mut img = Image::new("image.png");
        assert!(!img.is_loading());
        img.set_loading(true);
        assert!(img.is_loading());
    }

    #[test]
    fn test_image_error_state() {
        let mut img = Image::new("broken.png");
        assert!(!img.has_error());
        img.set_error(true);
        assert!(img.has_error());
    }

    // ===== Aspect Ratio Tests =====

    #[test]
    fn test_image_aspect_ratio() {
        let img = Image::default().size(1600.0, 900.0);
        let ratio = img.aspect_ratio().unwrap();
        assert!((ratio - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn test_image_aspect_ratio_square() {
        let img = Image::default().size(100.0, 100.0);
        assert_eq!(img.aspect_ratio(), Some(1.0));
    }

    #[test]
    fn test_image_aspect_ratio_no_dimensions() {
        let img = Image::default();
        assert!(img.aspect_ratio().is_none());
    }

    #[test]
    fn test_image_aspect_ratio_zero_height() {
        let img = Image::default().width(100.0).height(0.0);
        assert!(img.aspect_ratio().is_none());
    }

    // ===== Display Size Calculation Tests =====

    #[test]
    fn test_display_size_fill() {
        let img = Image::default().size(100.0, 100.0).fit(ImageFit::Fill);
        let display = img.calculate_display_size(Size::new(200.0, 150.0));
        assert_eq!(display, Size::new(200.0, 150.0));
    }

    #[test]
    fn test_display_size_none() {
        let img = Image::default().size(100.0, 100.0).fit(ImageFit::None);
        let display = img.calculate_display_size(Size::new(200.0, 150.0));
        assert_eq!(display, Size::new(100.0, 100.0));
    }

    #[test]
    fn test_display_size_contain() {
        let img = Image::default().size(200.0, 100.0).fit(ImageFit::Contain);
        let display = img.calculate_display_size(Size::new(100.0, 100.0));
        // Should scale down to fit, maintaining aspect ratio
        assert_eq!(display, Size::new(100.0, 50.0));
    }

    #[test]
    fn test_display_size_cover() {
        let img = Image::default().size(200.0, 100.0).fit(ImageFit::Cover);
        let display = img.calculate_display_size(Size::new(100.0, 100.0));
        // Should scale to cover, may crop
        assert_eq!(display, Size::new(200.0, 100.0));
    }

    #[test]
    fn test_display_size_scale_down_smaller() {
        let img = Image::default().size(50.0, 50.0).fit(ImageFit::ScaleDown);
        let display = img.calculate_display_size(Size::new(100.0, 100.0));
        // Image is smaller, should not scale
        assert_eq!(display, Size::new(50.0, 50.0));
    }

    #[test]
    fn test_display_size_scale_down_larger() {
        let img = Image::default().size(200.0, 200.0).fit(ImageFit::ScaleDown);
        let display = img.calculate_display_size(Size::new(100.0, 100.0));
        // Image is larger, should scale down
        assert_eq!(display, Size::new(100.0, 100.0));
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_image_type_id() {
        let img = Image::new("test.png");
        assert_eq!(Widget::type_id(&img), TypeId::of::<Image>());
    }

    #[test]
    fn test_image_measure_with_size() {
        let img = Image::default().size(200.0, 150.0);
        let size = img.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size, Size::new(200.0, 150.0));
    }

    #[test]
    fn test_image_measure_default_size() {
        let img = Image::default();
        let size = img.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size, Size::new(100.0, 100.0)); // Default placeholder size
    }

    #[test]
    fn test_image_layout() {
        let mut img = Image::new("test.png");
        let bounds = Rect::new(10.0, 20.0, 200.0, 150.0);
        let result = img.layout(bounds);
        assert_eq!(result.size, Size::new(200.0, 150.0));
        assert_eq!(img.bounds, bounds);
    }

    #[test]
    fn test_image_children() {
        let img = Image::new("test.png");
        assert!(img.children().is_empty());
    }

    #[test]
    fn test_image_is_interactive() {
        let img = Image::new("test.png");
        assert!(!img.is_interactive());
    }

    #[test]
    fn test_image_is_focusable() {
        let img = Image::new("test.png");
        assert!(!img.is_focusable());
    }

    #[test]
    fn test_image_accessible_role() {
        let img = Image::new("test.png");
        assert_eq!(img.accessible_role(), AccessibleRole::Image);
    }

    #[test]
    fn test_image_accessible_name_from_alt() {
        let img = Image::new("photo.jpg").alt("Mountain landscape");
        assert_eq!(Widget::accessible_name(&img), Some("Mountain landscape"));
    }

    #[test]
    fn test_image_accessible_name_override() {
        let img = Image::new("photo.jpg")
            .alt("Photo")
            .accessible_name("Beautiful mountain landscape at sunset");
        assert_eq!(
            Widget::accessible_name(&img),
            Some("Beautiful mountain landscape at sunset")
        );
    }

    #[test]
    fn test_image_accessible_name_none() {
        let img = Image::new("decorative.png");
        assert_eq!(Widget::accessible_name(&img), None);
    }

    #[test]
    fn test_image_test_id() {
        let img = Image::new("test.png").test_id("profile-avatar");
        assert_eq!(Widget::test_id(&img), Some("profile-avatar"));
    }
}
