//! Transcript parsing for whisper-apr JSON and plain text formats

use anyhow::{Context, Result};
use std::path::Path;

use super::types::{TranscriptInput, TranscriptSegment};

/// Whisper-apr JSON transcript format
#[derive(serde::Deserialize)]
struct WhisperTranscript {
    text: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default)]
    segments: Vec<WhisperSegment>,
}

#[derive(serde::Deserialize)]
struct WhisperSegment {
    start: f64,
    end: f64,
    text: String,
    #[serde(default)]
    tokens: Vec<serde_json::Value>,
}

fn default_language() -> String {
    "en".to_string()
}

/// Parse a transcript file, auto-detecting whisper-apr JSON vs plain text.
pub fn parse_transcript(path: &Path) -> Result<TranscriptInput> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read transcript: {}", path.display()))?;

    let source_path = path.display().to_string();

    // Try JSON first
    if let Ok(whisper) = serde_json::from_str::<WhisperTranscript>(&content) {
        let segments = whisper
            .segments
            .into_iter()
            .map(|s| TranscriptSegment { start: s.start, end: s.end, text: s.text })
            .collect();

        return Ok(TranscriptInput {
            text: whisper.text,
            language: whisper.language,
            segments,
            source_path,
        });
    }

    // Fall back to plain text
    Ok(TranscriptInput {
        text: content,
        language: "en".to_string(),
        segments: Vec::new(),
        source_path,
    })
}

/// Parse all transcript files in a directory.
pub fn parse_transcript_dir(dir: &Path) -> Result<Vec<TranscriptInput>> {
    let mut transcripts = Vec::new();

    let entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .filter_map(|e| e.ok())
        .collect();

    for entry in entries {
        let path = entry.path();
        if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext, "json" | "txt" | "md") {
                match parse_transcript(&path) {
                    Ok(t) => transcripts.push(t),
                    Err(e) => {
                        eprintln!("Warning: skipping {}: {}", path.display(), e);
                    }
                }
            }
        }
    }

    transcripts.sort_by(|a, b| a.source_path.cmp(&b.source_path));
    Ok(transcripts)
}

/// Format a timestamp in seconds as MM:SS
pub fn format_timestamp(seconds: f64) -> String {
    let mins = (seconds / 60.0) as u64;
    let secs = (seconds % 60.0) as u64;
    format!("{mins}:{secs:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_whisper_json() {
        let json = r#"{
            "text": "MLOps combines ML and DevOps practices.",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 3.5, "text": "MLOps combines ML", "tokens": []},
                {"start": 3.5, "end": 6.0, "text": "and DevOps practices.", "tokens": []}
            ]
        }"#;

        let mut f = NamedTempFile::with_suffix(".json").expect("tempfile creation failed");
        write!(f, "{json}").expect("write failed");

        let transcript = parse_transcript(f.path()).expect("unexpected failure");
        assert_eq!(transcript.language, "en");
        assert_eq!(transcript.segments.len(), 2);
        assert!(transcript.text.contains("MLOps"));
        assert!((transcript.segments[0].end - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_plain_text() {
        let text = "This is a plain text transcript about machine learning.";
        let mut f = NamedTempFile::with_suffix(".txt").expect("tempfile creation failed");
        write!(f, "{text}").expect("write failed");

        let transcript = parse_transcript(f.path()).expect("unexpected failure");
        assert_eq!(transcript.language, "en");
        assert!(transcript.segments.is_empty());
        assert!(transcript.text.contains("machine learning"));
    }

    #[test]
    fn test_parse_transcript_dir() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");

        // Create two transcript files
        let json_path = dir.path().join("lesson1.json");
        std::fs::write(
            &json_path,
            r#"{"text":"Lesson one content.","language":"en","segments":[]}"#,
        )
        .expect("unexpected failure");

        let txt_path = dir.path().join("lesson2.txt");
        std::fs::write(&txt_path, "Lesson two content.").expect("fs write failed");

        // Create a non-transcript file that should be skipped
        std::fs::write(dir.path().join("notes.rs"), "fn main() {}").expect("fs write failed");

        let transcripts = parse_transcript_dir(dir.path()).expect("unexpected failure");
        assert_eq!(transcripts.len(), 2);
    }

    #[test]
    fn test_parse_nonexistent_file() {
        let result = parse_transcript(Path::new("/nonexistent/file.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_format_timestamp() {
        assert_eq!(format_timestamp(0.0), "0:00");
        assert_eq!(format_timestamp(65.0), "1:05");
        assert_eq!(format_timestamp(3661.0), "61:01");
    }

    #[test]
    fn test_parse_whisper_json_missing_language() {
        let json = r#"{"text": "Hello", "segments": []}"#;
        let mut f = NamedTempFile::with_suffix(".json").expect("tempfile creation failed");
        write!(f, "{json}").expect("write failed");

        let transcript = parse_transcript(f.path()).expect("unexpected failure");
        assert_eq!(transcript.language, "en");
    }

    #[test]
    fn test_parse_transcript_dir_nonexistent() {
        let result = parse_transcript_dir(Path::new("/nonexistent/dir"));
        assert!(result.is_err());
    }
}
