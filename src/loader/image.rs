//! Image loader with OCR via tesseract.
//!
//! Extracts text from `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` images
//! by shelling out to the `tesseract` CLI. Feature-gated behind `ocr`.
//!
//! # Requirements
//!
//! - `tesseract` must be installed and on `$PATH`
//! - English language data: `tesseract-ocr-eng` package

use crate::{Document, Result};
use std::path::Path;
use std::process::Command;

use super::DocumentLoader;

/// Loads image files by extracting text via Tesseract OCR.
///
/// Supports PNG, JPEG, TIFF, and BMP formats.
#[derive(Debug, Clone, Copy)]
pub struct ImageLoader;

impl DocumentLoader for ImageLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["png", "jpg", "jpeg", "tiff", "bmp"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        let output = Command::new("tesseract")
            .arg(path.as_os_str())
            .arg("stdout")
            .arg("--psm")
            .arg("6") // Assume a single uniform block of text
            .output()
            .map_err(|e| {
                crate::Error::InvalidConfig(format!(
                    "Failed to run tesseract (is it installed?): {e}"
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::Error::InvalidConfig(format!(
                "Tesseract failed on {}: {}",
                path.display(),
                stderr.trim()
            )));
        }

        let content = String::from_utf8_lossy(&output.stdout).trim().to_string();

        let title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled").to_string();

        // Extract frame number from filename if it matches pattern like "frame_00123.png"
        let mut doc = Document::new(content).with_title(title).with_source(path.to_string_lossy());

        // If the filename contains timing info (e.g., "frame_00120s.png"), extract it
        if let Some(secs) = extract_timestamp_from_filename(path) {
            doc.metadata.insert(
                "frame_time_secs".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(secs)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
        }

        Ok(doc)
    }
}

/// Try to extract a timestamp from a filename like "frame_00120s.png" or "frame_120.5.png".
fn extract_timestamp_from_filename(path: &Path) -> Option<f64> {
    let stem = path.file_stem()?.to_str()?;
    // Pattern: frame_NNNs or frame_NNN.Ns
    if let Some(rest) = stem.strip_prefix("frame_") {
        let numeric = rest.trim_end_matches('s');
        numeric.parse::<f64>().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_loader_extensions() {
        let loader = ImageLoader;
        let exts = loader.supported_extensions();
        assert!(exts.contains(&"png"));
        assert!(exts.contains(&"jpg"));
        assert!(exts.contains(&"jpeg"));
        assert!(exts.contains(&"tiff"));
        assert!(exts.contains(&"bmp"));
    }

    #[test]
    fn test_image_loader_can_load() {
        let loader = ImageLoader;
        assert!(loader.can_load(Path::new("slide.png")));
        assert!(loader.can_load(Path::new("photo.JPG")));
        assert!(!loader.can_load(Path::new("video.mp4")));
        assert!(!loader.can_load(Path::new("notes.txt")));
    }

    #[test]
    fn test_extract_timestamp_from_filename() {
        assert_eq!(extract_timestamp_from_filename(Path::new("frame_120s.png")), Some(120.0));
        assert_eq!(extract_timestamp_from_filename(Path::new("frame_45.5s.png")), Some(45.5));
        assert_eq!(extract_timestamp_from_filename(Path::new("frame_0.png")), Some(0.0));
        assert_eq!(extract_timestamp_from_filename(Path::new("slide_01.png")), None);
        assert_eq!(extract_timestamp_from_filename(Path::new("photo.jpg")), None);
    }

    #[test]
    fn test_image_loader_missing_file() {
        let loader = ImageLoader;
        let result = loader.load(Path::new("/nonexistent/image.png"));
        // Should fail (tesseract not found or file not found)
        assert!(result.is_err());
    }
}
