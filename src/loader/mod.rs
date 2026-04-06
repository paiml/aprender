//! Document loading abstraction for pluggable file format support.
//!
//! The [`DocumentLoader`] trait decouples file format handling from the
//! RAG pipeline. Built-in loaders handle text (`.txt`, `.md`) and
//! subtitle (`.srt`, `.vtt`) formats. Third parties can implement
//! `DocumentLoader` for any format.
//!
//! The [`LoaderRegistry`] dispatches loading to the appropriate loader
//! based on file extension, with support for sidecar subtitle files
//! adjacent to media files.
//!
//! # Example
//!
//! ```rust
//! use trueno_rag::loader::LoaderRegistry;
//! use std::path::Path;
//!
//! let registry = LoaderRegistry::new();
//! let extensions = registry.supported_extensions();
//! assert!(extensions.contains(&"txt"));
//! assert!(extensions.contains(&"srt"));
//! ```

#[cfg(feature = "ocr")]
mod image;
mod subtitle;
mod text;
#[cfg(feature = "transcription")]
pub mod transcription;

#[cfg(feature = "ocr")]
pub use image::ImageLoader;
pub use subtitle::SubtitleLoader;
pub use text::TextLoader;
#[cfg(feature = "transcription")]
pub use transcription::TranscriptionLoader;

use crate::{Document, Error, Result};
use std::path::Path;

/// Abstraction for loading files of any format into Documents.
///
/// Implementors handle format detection, parsing, and conversion
/// to the standard `Document` representation. A loader may support
/// multiple file extensions.
pub trait DocumentLoader: Send + Sync {
    /// File extensions this loader handles (lowercase, without dot).
    fn supported_extensions(&self) -> Vec<&str>;

    /// Returns true if this loader can handle the given path.
    ///
    /// Default implementation checks the file extension against
    /// [`supported_extensions()`](DocumentLoader::supported_extensions).
    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                let lower = ext.to_lowercase();
                self.supported_extensions().iter().any(|s| *s == lower)
            })
            .unwrap_or(false)
    }

    /// Load a file and produce a Document.
    ///
    /// The returned Document should have:
    /// - `content`: The extracted text
    /// - `source`: The file path
    /// - `title`: Derived from filename or embedded metadata
    /// - `metadata`: Format-specific fields
    fn load(&self, path: &Path) -> Result<Document>;
}

/// Registry that dispatches file loading to the appropriate [`DocumentLoader`].
///
/// Comes pre-loaded with [`TextLoader`] and [`SubtitleLoader`].
/// Register additional loaders with [`register`](LoaderRegistry::register).
pub struct LoaderRegistry {
    loaders: Vec<Box<dyn DocumentLoader>>,
}

impl LoaderRegistry {
    /// Create a registry with default loaders (text and subtitle).
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self { loaders: Vec::new() };
        registry.register(Box::new(TextLoader));
        registry.register(Box::new(SubtitleLoader));
        #[cfg(feature = "ocr")]
        registry.register(Box::new(ImageLoader));
        registry
    }

    /// Register a custom loader.
    pub fn register(&mut self, loader: Box<dyn DocumentLoader>) {
        self.loaders.push(loader);
    }

    /// Find the first loader that can handle the given path.
    #[must_use]
    pub fn loader_for(&self, path: &Path) -> Option<&dyn DocumentLoader> {
        self.loaders.iter().find(|l| l.can_load(path)).map(|l| l.as_ref())
    }

    /// Load a document, selecting the appropriate loader automatically.
    pub fn load(&self, path: &Path) -> Result<Document> {
        let loader = self.loader_for(path).ok_or_else(|| {
            Error::InvalidInput(format!("No loader registered for: {}", path.display()))
        })?;
        loader.load(path)
    }

    /// Check if a sidecar subtitle file exists for a media file.
    ///
    /// Returns the sidecar path if found (prefers `.srt` over `.vtt`).
    #[must_use]
    pub fn find_sidecar(media_path: &Path) -> Option<std::path::PathBuf> {
        for ext in &["srt", "vtt"] {
            let sidecar = media_path.with_extension(ext);
            if sidecar.exists() {
                return Some(sidecar);
            }
        }
        None
    }

    /// All supported extensions across all registered loaders.
    #[must_use]
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.loaders.iter().flat_map(|l| l.supported_extensions()).collect()
    }
}

impl Default for LoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LoaderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoaderRegistry")
            .field("loader_count", &self.loaders.len())
            .field("extensions", &self.supported_extensions())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_default_loaders() {
        let registry = LoaderRegistry::new();
        let exts = registry.supported_extensions();
        assert!(exts.contains(&"txt"));
        assert!(exts.contains(&"md"));
        assert!(exts.contains(&"srt"));
        assert!(exts.contains(&"vtt"));
    }

    #[test]
    fn test_registry_loader_for_txt() {
        let registry = LoaderRegistry::new();
        assert!(registry.loader_for(Path::new("file.txt")).is_some());
        assert!(registry.loader_for(Path::new("file.TXT")).is_some());
    }

    #[test]
    fn test_registry_loader_for_srt() {
        let registry = LoaderRegistry::new();
        assert!(registry.loader_for(Path::new("file.srt")).is_some());
    }

    #[test]
    fn test_registry_no_loader_for_unknown() {
        let registry = LoaderRegistry::new();
        assert!(registry.loader_for(Path::new("file.xyz")).is_none());
    }

    #[test]
    fn test_registry_load_missing_file() {
        let registry = LoaderRegistry::new();
        let result = registry.load(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_load_unsupported_format() {
        let registry = LoaderRegistry::new();
        let result = registry.load(Path::new("file.mp4"));
        assert!(result.is_err());
    }

    #[test]
    fn test_find_sidecar_none() {
        // No sidecar for a file in /tmp that doesn't exist
        assert!(
            LoaderRegistry::find_sidecar(Path::new("/tmp/nonexistent_video_12345.mp4")).is_none()
        );
    }

    #[test]
    fn test_registry_custom_loader() {
        struct DummyLoader;
        impl DocumentLoader for DummyLoader {
            fn supported_extensions(&self) -> Vec<&str> {
                vec!["xyz"]
            }
            fn load(&self, path: &Path) -> Result<Document> {
                Ok(Document::new("dummy").with_source(path.to_string_lossy()))
            }
        }

        let mut registry = LoaderRegistry::new();
        registry.register(Box::new(DummyLoader));
        assert!(registry.loader_for(Path::new("test.xyz")).is_some());
    }

    #[test]
    fn test_registry_debug() {
        let registry = LoaderRegistry::new();
        let debug = format!("{registry:?}");
        assert!(debug.contains("LoaderRegistry"));
        assert!(debug.contains("loader_count"));
    }

    #[test]
    fn test_registry_default() {
        let registry = LoaderRegistry::default();
        assert!(!registry.supported_extensions().is_empty());
    }

    #[test]
    fn test_find_sidecar_srt_preferred() {
        // Create temp files to test sidecar detection
        let dir = std::env::temp_dir().join("trueno_rag_test_sidecar");
        let _ = std::fs::create_dir_all(&dir);
        let video = dir.join("lecture.mp4");
        let srt = dir.join("lecture.srt");
        let vtt = dir.join("lecture.vtt");
        std::fs::write(&video, b"").unwrap();
        std::fs::write(&srt, b"").unwrap();
        std::fs::write(&vtt, b"").unwrap();

        let found = LoaderRegistry::find_sidecar(&video);
        assert!(found.is_some());
        // SRT is preferred over VTT
        assert_eq!(found.unwrap().extension().unwrap(), "srt");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_can_load_no_extension() {
        let loader = TextLoader;
        assert!(!loader.can_load(Path::new("Makefile")));
    }

    #[test]
    fn test_load_real_txt_file() {
        let dir = std::env::temp_dir().join("trueno_rag_test_load_txt");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        std::fs::write(&file, "Hello from test file.").unwrap();

        let registry = LoaderRegistry::new();
        let doc = registry.load(&file).unwrap();
        assert_eq!(doc.content, "Hello from test file.");
        assert!(doc.title.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_real_srt_file() {
        let dir = std::env::temp_dir().join("trueno_rag_test_load_srt");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.srt");
        std::fs::write(&file, "1\n00:00:01,000 --> 00:00:04,500\nHello from subtitle.\n").unwrap();

        let registry = LoaderRegistry::new();
        let doc = registry.load(&file).unwrap();
        assert!(doc.content.contains("Hello from subtitle"));
        assert!(doc.metadata.contains_key("subtitle_cues"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
