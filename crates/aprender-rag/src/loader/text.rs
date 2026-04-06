//! Text file loader for `.txt` and `.md` files.

use crate::{Document, Result};
use std::path::Path;

use super::DocumentLoader;

/// Loads plain text and Markdown files.
///
/// Supports `.txt` and `.md` extensions.
#[derive(Debug, Clone, Copy)]
pub struct TextLoader;

impl DocumentLoader for TextLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["txt", "md"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        let content = std::fs::read_to_string(path).map_err(crate::Error::Io)?;
        let title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled").to_string();

        Ok(Document::new(content).with_title(title).with_source(path.to_string_lossy()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_loader_extensions() {
        let loader = TextLoader;
        let exts = loader.supported_extensions();
        assert!(exts.contains(&"txt"));
        assert!(exts.contains(&"md"));
    }

    #[test]
    fn test_text_loader_can_load() {
        let loader = TextLoader;
        assert!(loader.can_load(Path::new("readme.txt")));
        assert!(loader.can_load(Path::new("README.MD")));
        assert!(!loader.can_load(Path::new("video.mp4")));
        assert!(!loader.can_load(Path::new("Makefile")));
    }

    #[test]
    fn test_text_loader_load() {
        let dir = std::env::temp_dir().join("trueno_rag_test_text_loader");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("hello.txt");
        std::fs::write(&file, "Hello, world!").unwrap();

        let loader = TextLoader;
        let doc = loader.load(&file).unwrap();
        assert_eq!(doc.content, "Hello, world!");
        assert_eq!(doc.title.as_deref(), Some("hello"));
        assert!(doc.source.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_loader_missing_file() {
        let loader = TextLoader;
        let result = loader.load(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_text_loader_title_from_stem() {
        let dir = std::env::temp_dir().join("trueno_rag_test_text_title");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("my-document.md");
        std::fs::write(&file, "# Header").unwrap();

        let loader = TextLoader;
        let doc = loader.load(&file).unwrap();
        assert_eq!(doc.title.as_deref(), Some("my-document"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
