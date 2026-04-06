//! Subtitle file loader for `.srt` and `.vtt` files.

use crate::media::parse_subtitles;
use crate::{Document, Result};
use std::collections::HashMap;
use std::path::Path;

use super::DocumentLoader;

/// Loads subtitle files and produces Documents with timestamp metadata.
///
/// Supports `.srt` (SubRip) and `.vtt` (WebVTT) formats.
/// The document content is the concatenated plain text of all cues.
/// Subtitle cue data is stored in `metadata["subtitle_cues"]` for
/// downstream timestamp-aware chunking.
#[derive(Debug, Clone, Copy)]
pub struct SubtitleLoader;

impl DocumentLoader for SubtitleLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["srt", "vtt"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        let raw = std::fs::read_to_string(path).map_err(crate::Error::Io)?;
        let track = parse_subtitles(&raw)?;

        let title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled").to_string();

        let mut metadata = HashMap::new();
        metadata.insert("duration_secs".into(), serde_json::json!(track.duration_secs()));
        metadata.insert("format".into(), serde_json::json!(track.format.to_string()));
        metadata.insert("cue_count".into(), serde_json::json!(track.cues.len()));
        metadata.insert(
            "subtitle_cues".into(),
            serde_json::to_value(&track.cues).map_err(crate::Error::Serialization)?,
        );

        let mut doc = Document::new(track.to_plain_text())
            .with_title(title)
            .with_source(path.to_string_lossy());
        doc.metadata = metadata;
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_loader_extensions() {
        let loader = SubtitleLoader;
        let exts = loader.supported_extensions();
        assert!(exts.contains(&"srt"));
        assert!(exts.contains(&"vtt"));
    }

    #[test]
    fn test_subtitle_loader_can_load() {
        let loader = SubtitleLoader;
        assert!(loader.can_load(Path::new("lecture.srt")));
        assert!(loader.can_load(Path::new("captions.VTT")));
        assert!(!loader.can_load(Path::new("file.txt")));
    }

    #[test]
    fn test_subtitle_loader_load_srt() {
        let dir = std::env::temp_dir().join("trueno_rag_test_sub_loader_srt");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("lecture.srt");
        std::fs::write(
            &file,
            "\
1
00:00:01,000 --> 00:00:04,500
First cue text.

2
00:00:05,000 --> 00:00:09,200
Second cue text.
",
        )
        .unwrap();

        let loader = SubtitleLoader;
        let doc = loader.load(&file).unwrap();

        assert!(doc.content.contains("First cue text."));
        assert!(doc.content.contains("Second cue text."));
        assert_eq!(doc.title.as_deref(), Some("lecture"));
        assert!(doc.metadata.contains_key("duration_secs"));
        assert!(doc.metadata.contains_key("subtitle_cues"));
        assert_eq!(doc.metadata["cue_count"], serde_json::json!(2));
        assert_eq!(doc.metadata["format"], serde_json::json!("srt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_subtitle_loader_load_vtt() {
        let dir = std::env::temp_dir().join("trueno_rag_test_sub_loader_vtt");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("captions.vtt");
        std::fs::write(
            &file,
            "\
WEBVTT

00:00:01.000 --> 00:00:04.500
VTT cue text.
",
        )
        .unwrap();

        let loader = SubtitleLoader;
        let doc = loader.load(&file).unwrap();

        assert!(doc.content.contains("VTT cue text"));
        assert_eq!(doc.metadata["format"], serde_json::json!("vtt"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_subtitle_loader_metadata_duration() {
        let dir = std::env::temp_dir().join("trueno_rag_test_sub_duration");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("timed.srt");
        std::fs::write(
            &file,
            "\
1
00:01:00,000 --> 00:02:30,000
One minute in.
",
        )
        .unwrap();

        let loader = SubtitleLoader;
        let doc = loader.load(&file).unwrap();

        let duration = doc.metadata["duration_secs"].as_f64().unwrap();
        assert!((duration - 150.0).abs() < 0.1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_subtitle_loader_missing_file() {
        let loader = SubtitleLoader;
        let result = loader.load(Path::new("/nonexistent/file.srt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_subtitle_loader_cues_deserializable() {
        let dir = std::env::temp_dir().join("trueno_rag_test_sub_cues_deser");
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("test.srt");
        std::fs::write(
            &file,
            "1\n00:00:01,000 --> 00:00:04,500\nHello.\n\n2\n00:00:05,000 --> 00:00:09,000\nWorld.\n",
        )
        .unwrap();

        let loader = SubtitleLoader;
        let doc = loader.load(&file).unwrap();

        // Verify cues can be deserialized back
        let cues: Vec<crate::media::SubtitleCue> =
            serde_json::from_value(doc.metadata["subtitle_cues"].clone()).unwrap();
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Hello.");
        assert_eq!(cues[1].text, "World.");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
