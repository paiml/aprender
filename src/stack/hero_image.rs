//! Hero Image Detection and Validation
//!
//! Detects and validates hero images in PAIML stack repositories.
//! Supports SVG, PNG, JPG, and WebP formats.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Supported image formats for hero images
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    Png,
    Jpg,
    WebP,
    Svg,
}

impl ImageFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpg),
            "webp" => Some(Self::WebP),
            "svg" => Some(Self::Svg),
            _ => None,
        }
    }

    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpg => "jpg",
            Self::WebP => "webp",
            Self::Svg => "svg",
        }
    }
}

/// Hero image detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeroImageResult {
    /// Whether a hero image was found
    pub present: bool,
    /// Path to the hero image (if found)
    pub path: Option<PathBuf>,
    /// Detected format
    pub format: Option<ImageFormat>,
    /// Image dimensions (width, height) if detectable
    pub dimensions: Option<(u32, u32)>,
    /// File size in bytes
    pub file_size: Option<u64>,
    /// Whether the image passes validation
    pub valid: bool,
    /// Issues found during validation
    pub issues: Vec<String>,
}

impl HeroImageResult {
    /// Create result for missing hero image
    pub fn missing() -> Self {
        Self {
            present: false,
            path: None,
            format: None,
            dimensions: None,
            file_size: None,
            valid: false,
            issues: vec!["No hero image found".to_string()],
        }
    }

    /// Create result for found hero image
    pub fn found(path: PathBuf, format: ImageFormat) -> Self {
        Self {
            present: true,
            path: Some(path),
            format: Some(format),
            dimensions: None,
            file_size: None,
            valid: true,
            issues: vec![],
        }
    }

    /// Detect hero image in repository
    pub fn detect(repo_path: &Path) -> Self {
        // Priority 1: Check docs/hero.* files (SVG preferred)
        for ext in &["svg", "png", "jpg", "jpeg", "webp"] {
            let hero_path = repo_path.join(format!("docs/hero.{}", ext));
            if hero_path.exists() {
                if let Some(format) = ImageFormat::from_extension(ext) {
                    return Self::validate_at_path(&hero_path, format);
                }
            }
        }

        // Priority 2: Check assets/hero.* files (SVG preferred)
        for ext in &["svg", "png", "jpg", "jpeg", "webp"] {
            let hero_path = repo_path.join(format!("assets/hero.{}", ext));
            if hero_path.exists() {
                if let Some(format) = ImageFormat::from_extension(ext) {
                    return Self::validate_at_path(&hero_path, format);
                }
            }
        }

        // Priority 3: Parse README.md for first image
        let readme_path = repo_path.join("README.md");
        if readme_path.exists() {
            if let Some(img_ref) = Self::extract_first_image_from_readme(&readme_path) {
                let full_path = repo_path.join(&img_ref);
                if full_path.exists() {
                    if let Some(ext) = full_path.extension().and_then(|e| e.to_str()) {
                        if let Some(format) = ImageFormat::from_extension(ext) {
                            return Self::validate_at_path(&full_path, format);
                        }
                    }
                }
            }
        }

        Self::missing()
    }

    /// Validate image at path
    fn validate_at_path(path: &Path, format: ImageFormat) -> Self {
        let mut result = Self::found(path.to_path_buf(), format);
        let mut issues = Vec::new();

        // Check file size
        if let Ok(metadata) = std::fs::metadata(path) {
            let size = metadata.len();
            result.file_size = Some(size);

            // Max 2MB
            if size > 2 * 1024 * 1024 {
                issues.push(format!("Image too large: {} bytes (max 2MB)", size));
            }
        }

        // Note: Dimension checking would require image crate
        // For now, we skip dimension validation

        if !issues.is_empty() {
            result.valid = false;
            result.issues = issues;
        }

        result
    }

    /// Extract first image reference from README.md
    /// Replaced regex-lite with string parsing (DEP-REDUCE)
    fn extract_first_image_from_readme(readme_path: &Path) -> Option<String> {
        let content = std::fs::read_to_string(readme_path).ok()?;

        // Match markdown image syntax: ![alt](path)
        if let Some(img_path) = Self::extract_markdown_image(&content) {
            if !img_path.starts_with("http://") && !img_path.starts_with("https://") {
                return Some(img_path);
            }
        }

        // Match HTML img syntax: <img src="path"
        if let Some(img_path) = Self::extract_html_image(&content) {
            if !img_path.starts_with("http://") && !img_path.starts_with("https://") {
                return Some(img_path);
            }
        }

        None
    }

    /// Extract image path from markdown syntax ![alt](path)
    fn extract_markdown_image(content: &str) -> Option<String> {
        // Find ![
        let start = content.find("![")?;
        let after_bracket = &content[start + 2..];
        // Find ]( after ![
        let close_bracket = after_bracket.find("](")?;
        let after_paren = &after_bracket[close_bracket + 2..];
        // Find closing )
        let close_paren = after_paren.find(')')?;
        Some(after_paren[..close_paren].to_string())
    }

    /// Extract image path from HTML syntax <img src="path">
    fn extract_html_image(content: &str) -> Option<String> {
        // Find <img
        let img_start = content.find("<img")?;
        let after_img = &content[img_start..];
        // Find closing >
        let tag_end = after_img.find('>')?;
        let img_tag = &after_img[..tag_end];

        // Find src=" or src='
        for quote in ['"', '\''] {
            let src_pattern = format!("src={}", quote);
            if let Some(src_pos) = img_tag.find(&src_pattern) {
                let after_src = &img_tag[src_pos + src_pattern.len()..];
                if let Some(end_quote) = after_src.find(quote) {
                    return Some(after_src[..end_quote].to_string());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_from_extension() {
        assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::from_extension("PNG"), Some(ImageFormat::Png));
        assert_eq!(ImageFormat::from_extension("jpg"), Some(ImageFormat::Jpg));
        assert_eq!(ImageFormat::from_extension("jpeg"), Some(ImageFormat::Jpg));
        assert_eq!(ImageFormat::from_extension("svg"), Some(ImageFormat::Svg));
        assert_eq!(ImageFormat::from_extension("webp"), Some(ImageFormat::WebP));
        assert_eq!(ImageFormat::from_extension("gif"), None);
    }

    #[test]
    fn test_hero_image_missing() {
        let result = HeroImageResult::missing();
        assert!(!result.present);
        assert!(!result.valid);
        assert!(result.path.is_none());
    }

    #[test]
    fn test_hero_image_found() {
        let result = HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png);
        assert!(result.present);
        assert!(result.valid);
        assert_eq!(result.format, Some(ImageFormat::Png));
    }

    #[test]
    fn test_extract_markdown_image() {
        let content = "# Title\n![Hero](docs/hero.svg)\nMore text";
        let path = HeroImageResult::extract_markdown_image(content);
        assert_eq!(path, Some("docs/hero.svg".to_string()));
    }

    #[test]
    fn test_extract_html_image() {
        let content = r#"<img src="docs/hero.svg" alt="hero">"#;
        let path = HeroImageResult::extract_html_image(content);
        assert_eq!(path, Some("docs/hero.svg".to_string()));
    }

    #[test]
    fn test_extract_html_image_single_quotes() {
        let content = r#"<img src='docs/hero.svg' alt='hero'>"#;
        let path = HeroImageResult::extract_html_image(content);
        assert_eq!(path, Some("docs/hero.svg".to_string()));
    }

    #[test]
    fn test_image_format_extension() {
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Jpg.extension(), "jpg");
        assert_eq!(ImageFormat::WebP.extension(), "webp");
        assert_eq!(ImageFormat::Svg.extension(), "svg");
    }

    #[test]
    fn test_hero_image_result_fields() {
        let mut result = HeroImageResult::found(PathBuf::from("test.png"), ImageFormat::Png);
        result.dimensions = Some((800, 600));
        result.file_size = Some(1024);
        assert_eq!(result.dimensions, Some((800, 600)));
        assert_eq!(result.file_size, Some(1024));
    }

    #[test]
    fn test_hero_image_detect_nonexistent_repo() {
        let result = HeroImageResult::detect(Path::new("/nonexistent/path"));
        assert!(!result.present);
        assert!(!result.valid);
    }

    #[test]
    fn test_extract_markdown_image_no_image() {
        let content = "# Just text\nNo images here";
        let path = HeroImageResult::extract_markdown_image(content);
        assert!(path.is_none());
    }

    #[test]
    fn test_extract_html_image_no_image() {
        let content = "Just text, no img tags";
        let path = HeroImageResult::extract_html_image(content);
        assert!(path.is_none());
    }

    #[test]
    fn test_extract_html_image_no_src() {
        let content = r#"<img alt="test">"#;
        let path = HeroImageResult::extract_html_image(content);
        assert!(path.is_none());
    }

    #[test]
    fn test_image_format_equality() {
        assert_eq!(ImageFormat::Png, ImageFormat::Png);
        assert_ne!(ImageFormat::Png, ImageFormat::Jpg);
    }

    #[test]
    fn test_hero_image_result_missing_issues() {
        let result = HeroImageResult::missing();
        assert!(!result.issues.is_empty());
        assert!(result.issues[0].contains("No hero image"));
    }

    #[test]
    fn test_hero_image_found_no_issues() {
        let result = HeroImageResult::found(PathBuf::from("test.svg"), ImageFormat::Svg);
        assert!(result.issues.is_empty());
    }
}
