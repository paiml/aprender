//! Banner PNG generation for Coursera readings
//!
//! Generates a 1200x400 SVG banner with title and concept bubbles,
//! then rasterizes to PNG via resvg + tiny-skia (feature-gated).

use super::key_concepts;
use super::types::{BannerConfig, TranscriptInput};
use crate::oracle::svg::builder::SvgBuilder;
use crate::oracle::svg::palette::{Color, MaterialPalette};

/// Generate a banner SVG string from a config.
pub fn generate_banner_svg(config: &BannerConfig) -> String {
    let palette = MaterialPalette::light();
    let width = config.width as f32;
    let height = config.height as f32;

    let mut builder =
        SvgBuilder::new().size(width, height).transparent().title(&config.course_title);

    // Gradient background band (subtle)
    builder = builder.rect_styled(
        "bg-band",
        0.0,
        0.0,
        width,
        height,
        palette.primary_container,
        None,
        0.0,
    );

    // Title text
    let title_y = height * 0.35;
    builder = builder.heading(width / 2.0, title_y, &config.course_title);

    // Concept bubbles: arrange horizontally
    let concepts = &config.concepts;
    let max_bubbles = concepts.len().min(6);
    if max_bubbles > 0 {
        let bubble_y = height * 0.7;
        let total_width = width * 0.8;
        let spacing = total_width / max_bubbles as f32;
        let start_x = (width - total_width) / 2.0 + spacing / 2.0;

        let bubble_colors = [
            palette.primary,
            palette.secondary,
            palette.tertiary,
            Color::rgb(0, 150, 136), // Teal
            Color::rgb(255, 109, 0), // Deep Orange
            Color::rgb(41, 98, 255), // Blue
        ];

        for (i, concept) in concepts.iter().take(max_bubbles).enumerate() {
            let cx = start_x + i as f32 * spacing;
            let color = bubble_colors[i % bubble_colors.len()];

            // Bubble background
            let bubble_w = spacing * 0.85;
            let bubble_h = 36.0;
            builder = builder.rect_styled(
                &format!("bubble-{i}"),
                cx - bubble_w / 2.0,
                bubble_y - bubble_h / 2.0,
                bubble_w,
                bubble_h,
                color.lighten(0.7),
                Some((color, 1.5)),
                18.0,
            );

            // Bubble label
            builder = builder.label(cx, bubble_y + 5.0, concept);
        }
    }

    builder.build()
}

/// Generate a banner config from a transcript by extracting concepts.
pub fn banner_config_from_transcript(
    transcript: &TranscriptInput,
    course_title: &str,
) -> BannerConfig {
    let reading = key_concepts::generate_key_concepts(transcript);
    let concepts: Vec<String> = reading.concepts.iter().take(6).map(|c| c.term.clone()).collect();

    BannerConfig { course_title: course_title.to_string(), concepts, ..Default::default() }
}

/// Rasterize an SVG string to PNG bytes.
///
/// Requires the `coursera-assets` feature (resvg + tiny-skia).
#[cfg(feature = "coursera-assets")]
pub fn svg_to_png(svg: &str, width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    use anyhow::Context;

    let tree = resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default())
        .context("Failed to parse SVG")?;

    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(width, height).context("Failed to create pixmap")?;

    resvg::render(&tree, resvg::usvg::Transform::default(), &mut pixmap.as_mut());

    pixmap.encode_png().context("Failed to encode PNG")
}

/// Stub for when resvg is not available.
#[cfg(not(feature = "coursera-assets"))]
pub fn svg_to_png(_svg: &str, _width: u32, _height: u32) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!(
        "PNG rasterization requires the 'coursera-assets' feature.\n\
         Build with: cargo build --features coursera-assets\n\
         SVG output is always available without this feature."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_banner_svg_basic() {
        let config = BannerConfig {
            course_title: "MLOps Fundamentals".to_string(),
            concepts: vec!["MLOps".to_string(), "CI/CD".to_string(), "Docker".to_string()],
            width: 1200,
            height: 400,
        };
        let svg = generate_banner_svg(&config);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("MLOps Fundamentals"));
        assert!(svg.contains("MLOps"));
        assert!(svg.contains("CI/CD"));
        assert!(svg.contains("Docker"));
    }

    #[test]
    fn test_generate_banner_svg_empty_concepts() {
        let config = BannerConfig {
            course_title: "Empty Course".to_string(),
            concepts: vec![],
            ..Default::default()
        };
        let svg = generate_banner_svg(&config);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Empty Course"));
    }

    #[test]
    fn test_generate_banner_svg_max_concepts() {
        let config = BannerConfig {
            course_title: "Full Course".to_string(),
            concepts: (0..10).map(|i| format!("Concept{i}")).collect(),
            ..Default::default()
        };
        let svg = generate_banner_svg(&config);
        assert!(svg.contains("Concept0"));
        assert!(svg.contains("Concept5"));
        // Should be limited to 6 bubbles
        assert!(!svg.contains("bubble-6"));
    }

    #[test]
    fn test_banner_config_from_transcript() {
        let t = TranscriptInput {
            text: "GPU acceleration speeds up ML inference. GPU computing enables parallel processing. \
                   API endpoints serve predictions. API calls handle requests."
                .to_string(),
            language: "en".to_string(),
            segments: vec![],
            source_path: "test.txt".to_string(),
        };
        let config = banner_config_from_transcript(&t, "ML Course");
        assert_eq!(config.course_title, "ML Course");
        assert!(config.concepts.len() <= 6);
    }

    #[test]
    fn test_banner_svg_no_background_rect() {
        let config = BannerConfig {
            course_title: "Test".to_string(),
            concepts: vec![],
            ..Default::default()
        };
        let svg = generate_banner_svg(&config);
        // Should use transparent mode (no 100% background rect)
        assert!(!svg.contains("width=\"100%\" height=\"100%\""));
    }

    #[cfg(not(feature = "coursera-assets"))]
    #[test]
    fn test_svg_to_png_without_feature() {
        let result = svg_to_png("<svg></svg>", 100, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("coursera-assets"));
    }
}
