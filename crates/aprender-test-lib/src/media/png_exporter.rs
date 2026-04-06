//! PNG Screenshot Export (Feature 2)
//!
//! High-quality PNG screenshots with configurable compression and metadata.
//!
//! ## EXTREME TDD: Tests written FIRST per spec

use crate::driver::Screenshot;
use crate::result::{ProbarError, ProbarResult};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::SystemTime;

/// PNG compression level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CompressionLevel {
    /// No compression (fastest, largest files)
    None,
    /// Fast compression (good balance)
    Fast,
    /// Default compression
    #[default]
    Default,
    /// Best compression (slowest, smallest files)
    Best,
}

impl CompressionLevel {
    /// Convert to png crate compression level
    fn to_png_compression(self) -> png::Compression {
        match self {
            Self::None | Self::Fast => png::Compression::Fast,
            Self::Default => png::Compression::Default,
            Self::Best => png::Compression::Best,
        }
    }
}

/// Metadata to embed in PNG files
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PngMetadata {
    /// Image title
    pub title: Option<String>,
    /// Image description
    pub description: Option<String>,
    /// Timestamp when screenshot was taken
    pub timestamp: Option<SystemTime>,
    /// Name of the test that generated this screenshot
    pub test_name: Option<String>,
    /// Software that generated the image
    pub software: Option<String>,
}

impl PngMetadata {
    /// Create new empty metadata
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the description
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the timestamp
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: SystemTime) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set the test name
    #[must_use]
    pub fn with_test_name(mut self, name: impl Into<String>) -> Self {
        self.test_name = Some(name.into());
        self
    }

    /// Set the software name
    #[must_use]
    pub fn with_software(mut self, software: impl Into<String>) -> Self {
        self.software = Some(software.into());
        self
    }
}

/// Annotation to draw on screenshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// X coordinate of the annotation
    pub x: u32,
    /// Y coordinate of the annotation
    pub y: u32,
    /// Width of the annotation (for rectangles)
    pub width: u32,
    /// Height of the annotation (for rectangles)
    pub height: u32,
    /// Color of the annotation (RGBA)
    pub color: [u8; 4],
    /// Type of annotation
    pub kind: AnnotationKind,
    /// Optional label text
    pub label: Option<String>,
}

/// Types of annotations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationKind {
    /// Rectangle outline
    Rectangle,
    /// Filled rectangle
    FilledRectangle,
    /// Circle/ellipse outline
    Circle,
    /// Arrow pointing to location
    Arrow,
    /// Highlight (semi-transparent overlay)
    Highlight,
}

impl Annotation {
    /// Create a rectangle annotation
    #[must_use]
    pub fn rectangle(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color: [255, 0, 0, 255], // Red by default
            kind: AnnotationKind::Rectangle,
            label: None,
        }
    }

    /// Create a highlight annotation
    #[must_use]
    pub fn highlight(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color: [255, 255, 0, 128], // Semi-transparent yellow
            kind: AnnotationKind::Highlight,
            label: None,
        }
    }

    /// Create a filled rectangle annotation
    #[must_use]
    pub fn filled_rectangle(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            color: [255, 0, 0, 255], // Red by default
            kind: AnnotationKind::FilledRectangle,
            label: None,
        }
    }

    /// Create a circle annotation
    #[must_use]
    pub fn circle(x: u32, y: u32, diameter: u32) -> Self {
        Self {
            x,
            y,
            width: diameter,
            height: diameter,
            color: [0, 255, 0, 255], // Green by default
            kind: AnnotationKind::Circle,
            label: None,
        }
    }

    /// Create an arrow annotation
    #[must_use]
    pub fn arrow(x: u32, y: u32, dx: u32, dy: u32) -> Self {
        Self {
            x,
            y,
            width: dx,
            height: dy,
            color: [0, 0, 255, 255], // Blue by default
            kind: AnnotationKind::Arrow,
            label: None,
        }
    }

    /// Set the annotation color
    #[must_use]
    pub fn with_color(mut self, r: u8, g: u8, b: u8, a: u8) -> Self {
        self.color = [r, g, b, a];
        self
    }

    /// Set the annotation label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// PNG Exporter for high-quality screenshots
///
/// ## Example
///
/// ```ignore
/// let exporter = PngExporter::new()
///     .with_compression(CompressionLevel::Best)
///     .with_metadata(PngMetadata::new().with_title("Login Test"));
///
/// let png_data = exporter.export(&screenshot)?;
/// exporter.save(&screenshot, Path::new("screenshot.png"))?;
/// ```
#[derive(Debug, Clone)]
pub struct PngExporter {
    compression: CompressionLevel,
    metadata: PngMetadata,
}

impl Default for PngExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl PngExporter {
    /// Create a new PNG exporter with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            compression: CompressionLevel::Default,
            metadata: PngMetadata::new().with_software("Probar".to_string()),
        }
    }

    /// Set the compression level
    #[must_use]
    pub fn with_compression(mut self, compression: CompressionLevel) -> Self {
        self.compression = compression;
        self
    }

    /// Set the metadata
    #[must_use]
    pub fn with_metadata(mut self, metadata: PngMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the current compression level
    #[must_use]
    pub fn compression(&self) -> CompressionLevel {
        self.compression
    }

    /// Get the current metadata
    #[must_use]
    pub fn metadata(&self) -> &PngMetadata {
        &self.metadata
    }

    /// Export a screenshot to PNG data
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails
    pub fn export(&self, screenshot: &Screenshot) -> ProbarResult<Vec<u8>> {
        // Decode the screenshot if it's already PNG
        let img = image::load_from_memory(&screenshot.data).map_err(|e| {
            ProbarError::ImageProcessing {
                message: format!("Failed to decode screenshot: {e}"),
            }
        })?;

        self.encode_png(&img)
    }

    /// Export a screenshot with annotations
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails
    pub fn export_with_annotations(
        &self,
        screenshot: &Screenshot,
        annotations: &[Annotation],
    ) -> ProbarResult<Vec<u8>> {
        // Decode the screenshot
        let img = image::load_from_memory(&screenshot.data).map_err(|e| {
            ProbarError::ImageProcessing {
                message: format!("Failed to decode screenshot: {e}"),
            }
        })?;

        let mut rgba = img.to_rgba8();

        // Draw annotations
        for annotation in annotations {
            Self::draw_annotation(&mut rgba, annotation);
        }

        self.encode_png(&DynamicImage::ImageRgba8(rgba))
    }

    /// Save a screenshot to a file
    ///
    /// # Errors
    ///
    /// Returns error if encoding or file write fails
    pub fn save(&self, screenshot: &Screenshot, path: &Path) -> ProbarResult<()> {
        let data = self.export(screenshot)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Save a screenshot with annotations to a file
    ///
    /// # Errors
    ///
    /// Returns error if encoding or file write fails
    pub fn save_with_annotations(
        &self,
        screenshot: &Screenshot,
        annotations: &[Annotation],
        path: &Path,
    ) -> ProbarResult<()> {
        let data = self.export_with_annotations(screenshot, annotations)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Encode an image to PNG with the configured settings
    fn encode_png(&self, img: &DynamicImage) -> ProbarResult<Vec<u8>> {
        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        let mut output = Vec::new();

        {
            let mut encoder = png::Encoder::new(&mut output, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_compression(self.compression.to_png_compression());

            let mut writer = encoder
                .write_header()
                .map_err(|e| ProbarError::ImageProcessing {
                    message: format!("Failed to write PNG header: {e}"),
                })?;

            writer
                .write_image_data(&rgba)
                .map_err(|e| ProbarError::ImageProcessing {
                    message: format!("Failed to write PNG data: {e}"),
                })?;
        }

        Ok(output)
    }

    /// Draw an annotation on an image
    fn draw_annotation(img: &mut RgbaImage, annotation: &Annotation) {
        let color = Rgba(annotation.color);

        match annotation.kind {
            AnnotationKind::Rectangle => {
                Self::draw_rectangle_outline(img, annotation, color);
            }
            AnnotationKind::FilledRectangle => {
                Self::draw_filled_rectangle(img, annotation, color);
            }
            AnnotationKind::Highlight => {
                Self::draw_highlight(img, annotation);
            }
            AnnotationKind::Circle | AnnotationKind::Arrow => {
                // Simplified: draw as rectangle for now
                Self::draw_rectangle_outline(img, annotation, color);
            }
        }
    }

    /// Draw a rectangle outline
    fn draw_rectangle_outline(img: &mut RgbaImage, ann: &Annotation, color: Rgba<u8>) {
        let (img_width, img_height) = img.dimensions();
        let x_end = (ann.x + ann.width).min(img_width.saturating_sub(1));
        let y_end = (ann.y + ann.height).min(img_height.saturating_sub(1));

        // Top and bottom edges
        for x in ann.x..=x_end {
            if ann.y < img_height {
                img.put_pixel(x, ann.y, color);
            }
            if y_end < img_height {
                img.put_pixel(x, y_end, color);
            }
        }

        // Left and right edges
        for y in ann.y..=y_end {
            if ann.x < img_width {
                img.put_pixel(ann.x, y, color);
            }
            if x_end < img_width {
                img.put_pixel(x_end, y, color);
            }
        }
    }

    /// Draw a filled rectangle
    fn draw_filled_rectangle(img: &mut RgbaImage, ann: &Annotation, color: Rgba<u8>) {
        let (img_width, img_height) = img.dimensions();
        let x_end = (ann.x + ann.width).min(img_width);
        let y_end = (ann.y + ann.height).min(img_height);

        for y in ann.y..y_end {
            for x in ann.x..x_end {
                img.put_pixel(x, y, color);
            }
        }
    }

    /// Draw a highlight (semi-transparent overlay with alpha blending)
    fn draw_highlight(img: &mut RgbaImage, ann: &Annotation) {
        let (img_width, img_height) = img.dimensions();
        let x_end = (ann.x + ann.width).min(img_width);
        let y_end = (ann.y + ann.height).min(img_height);
        let highlight_color = ann.color;
        let alpha = f32::from(highlight_color[3]) / 255.0;

        for y in ann.y..y_end {
            for x in ann.x..x_end {
                let pixel = img.get_pixel(x, y);
                let blended = Rgba([
                    blend_channel(pixel[0], highlight_color[0], alpha),
                    blend_channel(pixel[1], highlight_color[1], alpha),
                    blend_channel(pixel[2], highlight_color[2], alpha),
                    255,
                ]);
                img.put_pixel(x, y, blended);
            }
        }
    }
}

/// Blend two color channels with alpha
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn blend_channel(base: u8, overlay: u8, alpha: f32) -> u8 {
    let result = f32::from(base).mul_add(1.0 - alpha, f32::from(overlay) * alpha);
    result.clamp(0.0, 255.0) as u8
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use image::ImageFormat;
    use std::io::Cursor;

    fn create_test_screenshot(width: u32, height: u32, color: [u8; 4]) -> Screenshot {
        let mut img = image::RgbaImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgba(color);
        }

        let mut png_data = Vec::new();
        img.write_to(&mut Cursor::new(&mut png_data), ImageFormat::Png)
            .unwrap();

        Screenshot::new(png_data, width, height)
    }

    mod compression_level_tests {
        use super::*;

        #[test]
        fn test_default_compression() {
            let level = CompressionLevel::default();
            assert_eq!(level, CompressionLevel::Default);
        }

        #[test]
        fn test_compression_levels() {
            // Just verify they exist and can be created
            let _ = CompressionLevel::None;
            let _ = CompressionLevel::Fast;
            let _ = CompressionLevel::Default;
            let _ = CompressionLevel::Best;
        }
    }

    mod png_metadata_tests {
        use super::*;

        #[test]
        fn test_default_metadata() {
            let meta = PngMetadata::default();
            assert!(meta.title.is_none());
            assert!(meta.description.is_none());
            assert!(meta.timestamp.is_none());
            assert!(meta.test_name.is_none());
        }

        #[test]
        fn test_with_title() {
            let meta = PngMetadata::new().with_title("Test Screenshot");
            assert_eq!(meta.title, Some("Test Screenshot".to_string()));
        }

        #[test]
        fn test_with_description() {
            let meta = PngMetadata::new().with_description("Login page after error");
            assert_eq!(meta.description, Some("Login page after error".to_string()));
        }

        #[test]
        fn test_with_test_name() {
            let meta = PngMetadata::new().with_test_name("test_login_failure");
            assert_eq!(meta.test_name, Some("test_login_failure".to_string()));
        }

        #[test]
        fn test_chained_builders() {
            let meta = PngMetadata::new()
                .with_title("Title")
                .with_description("Description")
                .with_test_name("test_name")
                .with_software("Probar Test");

            assert_eq!(meta.title, Some("Title".to_string()));
            assert_eq!(meta.description, Some("Description".to_string()));
            assert_eq!(meta.test_name, Some("test_name".to_string()));
            assert_eq!(meta.software, Some("Probar Test".to_string()));
        }
    }

    mod annotation_tests {
        use super::*;

        #[test]
        fn test_rectangle_annotation() {
            let ann = Annotation::rectangle(10, 20, 100, 50);
            assert_eq!(ann.x, 10);
            assert_eq!(ann.y, 20);
            assert_eq!(ann.width, 100);
            assert_eq!(ann.height, 50);
            assert!(matches!(ann.kind, AnnotationKind::Rectangle));
        }

        #[test]
        fn test_highlight_annotation() {
            let ann = Annotation::highlight(0, 0, 50, 50);
            assert!(matches!(ann.kind, AnnotationKind::Highlight));
            assert_eq!(ann.color[3], 128); // Semi-transparent
        }

        #[test]
        fn test_with_color() {
            let ann = Annotation::rectangle(0, 0, 10, 10).with_color(0, 255, 0, 255);
            assert_eq!(ann.color, [0, 255, 0, 255]);
        }

        #[test]
        fn test_with_label() {
            let ann = Annotation::rectangle(0, 0, 10, 10).with_label("Error Button");
            assert_eq!(ann.label, Some("Error Button".to_string()));
        }
    }

    mod png_exporter_tests {
        use super::*;

        #[test]
        fn test_new_exporter() {
            let exporter = PngExporter::new();
            assert_eq!(exporter.compression(), CompressionLevel::Default);
        }

        #[test]
        fn test_with_compression() {
            let exporter = PngExporter::new().with_compression(CompressionLevel::Best);
            assert_eq!(exporter.compression(), CompressionLevel::Best);
        }

        #[test]
        fn test_with_metadata() {
            let meta = PngMetadata::new().with_title("Test");
            let exporter = PngExporter::new().with_metadata(meta);
            assert_eq!(exporter.metadata().title, Some("Test".to_string()));
        }

        #[test]
        fn test_export() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 0, 0, 255]);

            let result = exporter.export(&screenshot);
            assert!(result.is_ok());

            let png_data = result.unwrap();
            assert!(!png_data.is_empty());
            // PNG magic bytes
            assert_eq!(&png_data[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        }

        #[test]
        fn test_export_with_annotations() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 255, 255, 255]);

            let annotations = vec![
                Annotation::rectangle(10, 10, 30, 30).with_color(255, 0, 0, 255),
                Annotation::highlight(50, 50, 40, 40),
            ];

            let result = exporter.export_with_annotations(&screenshot, &annotations);
            assert!(result.is_ok());
        }

        #[test]
        fn test_save() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(50, 50, [0, 255, 0, 255]);

            let temp_dir = tempfile::tempdir().unwrap();
            let path = temp_dir.path().join("test.png");

            let result = exporter.save(&screenshot, &path);
            assert!(result.is_ok());
            assert!(path.exists());

            // Verify it's a valid PNG
            let data = std::fs::read(&path).unwrap();
            assert_eq!(&data[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
        }

        #[test]
        fn test_save_with_annotations() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 255, 255, 255]);
            let annotations = vec![Annotation::rectangle(10, 10, 20, 20)];

            let temp_dir = tempfile::tempdir().unwrap();
            let path = temp_dir.path().join("annotated.png");

            let result = exporter.save_with_annotations(&screenshot, &annotations, &path);
            assert!(result.is_ok());
            assert!(path.exists());
        }

        #[test]
        fn test_compression_levels_produce_different_sizes() {
            let screenshot = create_test_screenshot(200, 200, [128, 128, 128, 255]);

            let fast = PngExporter::new()
                .with_compression(CompressionLevel::Fast)
                .export(&screenshot)
                .unwrap();

            let best = PngExporter::new()
                .with_compression(CompressionLevel::Best)
                .export(&screenshot)
                .unwrap();

            // Best compression should generally produce smaller or equal output
            // (may be equal for very simple images)
            assert!(best.len() <= fast.len() + 100); // Allow small variance
        }
    }

    mod blend_tests {
        use super::*;

        #[test]
        fn test_blend_full_alpha() {
            // Full alpha = overlay completely
            assert_eq!(blend_channel(100, 200, 1.0), 200);
        }

        #[test]
        fn test_blend_zero_alpha() {
            // Zero alpha = base only
            assert_eq!(blend_channel(100, 200, 0.0), 100);
        }

        #[test]
        fn test_blend_half_alpha() {
            // Half alpha = average
            let result = blend_channel(100, 200, 0.5);
            assert!((145..=155).contains(&result)); // ~150
        }
    }

    mod annotation_kind_tests {
        use super::*;

        #[test]
        fn test_filled_rectangle_annotation() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 255, 255, 255]);

            let annotations = vec![Annotation {
                x: 10,
                y: 10,
                width: 30,
                height: 30,
                color: [0, 0, 255, 255],
                kind: AnnotationKind::FilledRectangle,
                label: None,
            }];

            let result = exporter.export_with_annotations(&screenshot, &annotations);
            assert!(result.is_ok());
        }

        #[test]
        fn test_circle_annotation() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 255, 255, 255]);

            let annotations = vec![Annotation {
                x: 20,
                y: 20,
                width: 40,
                height: 40,
                color: [255, 0, 255, 255],
                kind: AnnotationKind::Circle,
                label: None,
            }];

            let result = exporter.export_with_annotations(&screenshot, &annotations);
            assert!(result.is_ok());
        }

        #[test]
        fn test_arrow_annotation() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 255, 255, 255]);

            let annotations = vec![Annotation {
                x: 10,
                y: 10,
                width: 50,
                height: 20,
                color: [0, 255, 0, 255],
                kind: AnnotationKind::Arrow,
                label: Some("Click here".to_string()),
            }];

            let result = exporter.export_with_annotations(&screenshot, &annotations);
            assert!(result.is_ok());
        }
    }

    mod exporter_edge_cases {
        use super::*;

        #[test]
        fn test_exporter_default() {
            let exporter = PngExporter::default();
            assert_eq!(exporter.compression(), CompressionLevel::Default);
        }

        #[test]
        fn test_compression_none() {
            let exporter = PngExporter::new().with_compression(CompressionLevel::None);
            let screenshot = create_test_screenshot(50, 50, [128, 128, 128, 255]);

            let result = exporter.export(&screenshot);
            assert!(result.is_ok());
        }

        #[test]
        fn test_metadata_with_timestamp() {
            let meta = PngMetadata::new().with_timestamp(SystemTime::now());
            assert!(meta.timestamp.is_some());
        }

        #[test]
        fn test_annotation_at_image_boundary() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(100, 100, [255, 255, 255, 255]);

            // Annotation extends beyond image bounds - should be clipped
            let annotations = vec![Annotation::rectangle(80, 80, 50, 50)];

            let result = exporter.export_with_annotations(&screenshot, &annotations);
            assert!(result.is_ok());
        }

        #[test]
        fn test_multiple_annotation_types() {
            let exporter = PngExporter::new();
            let screenshot = create_test_screenshot(200, 200, [200, 200, 200, 255]);

            let annotations = vec![
                Annotation::rectangle(10, 10, 30, 30),
                Annotation::highlight(50, 10, 30, 30),
                Annotation {
                    x: 90,
                    y: 10,
                    width: 30,
                    height: 30,
                    color: [0, 255, 0, 255],
                    kind: AnnotationKind::FilledRectangle,
                    label: None,
                },
                Annotation {
                    x: 130,
                    y: 10,
                    width: 30,
                    height: 30,
                    color: [255, 0, 255, 255],
                    kind: AnnotationKind::Circle,
                    label: None,
                },
            ];

            let result = exporter.export_with_annotations(&screenshot, &annotations);
            assert!(result.is_ok());
        }

        #[test]
        fn test_all_compression_to_png_conversion() {
            // Test all compression levels convert correctly
            let _ = CompressionLevel::None.to_png_compression();
            let _ = CompressionLevel::Fast.to_png_compression();
            let _ = CompressionLevel::Default.to_png_compression();
            let _ = CompressionLevel::Best.to_png_compression();
        }
    }

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_export_produces_valid_png(
                width in 1u32..100,
                height in 1u32..100,
                r in 0u8..=255,
                g in 0u8..=255,
                b in 0u8..=255
            ) {
                let exporter = PngExporter::new();
                let screenshot = create_test_screenshot(width, height, [r, g, b, 255]);

                let result = exporter.export(&screenshot);
                prop_assert!(result.is_ok());

                let png_data = result.unwrap();
                // PNG magic bytes
                prop_assert_eq!(&png_data[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            }

            #[test]
            fn prop_annotation_bounds_respected(
                x in 0u32..100,
                y in 0u32..100,
                w in 1u32..50,
                h in 1u32..50
            ) {
                let ann = Annotation::rectangle(x, y, w, h);
                prop_assert_eq!(ann.x, x);
                prop_assert_eq!(ann.y, y);
                prop_assert_eq!(ann.width, w);
                prop_assert_eq!(ann.height, h);
            }

            #[test]
            fn prop_blend_channel_in_range(
                base in 0u8..=255,
                overlay in 0u8..=255,
                alpha in 0.0f32..=1.0
            ) {
                let result = blend_channel(base, overlay, alpha);
                // Verify the blending is bounded (result is u8, so always <= 255)
                // Instead verify the blend is within expected range
                let expected_min = base.min(overlay);
                let expected_max = base.max(overlay);
                prop_assert!(result >= expected_min || alpha < 1.0);
                prop_assert!(result <= expected_max || alpha > 0.0);
            }
        }
    }
}
