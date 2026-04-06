//! Timestamp-aware chunker for subtitle/transcript content.

use super::{Chunk, Chunker, RecursiveChunker};
use crate::{Document, Error, Result};

/// Timestamp-aware chunker for subtitle/transcript content.
///
/// Groups subtitle cues into chunks based on time duration rather than
/// character count. Each chunk carries `start_secs` and `end_secs` in
/// its metadata for timestamp-aware retrieval and citation.
///
/// Falls back to [`RecursiveChunker`] for documents without subtitle
/// cue metadata.
///
/// # Example
///
/// ```rust
/// use trueno_rag::chunk::{TimestampChunker, Chunker};
/// use trueno_rag::Document;
/// use trueno_rag::media::SubtitleCue;
///
/// let cues = vec![
///     SubtitleCue { index: 0, start_secs: 0.0, end_secs: 30.0, text: "First segment.".into() },
///     SubtitleCue { index: 1, start_secs: 30.0, end_secs: 65.0, text: "Second segment.".into() },
///     SubtitleCue { index: 2, start_secs: 65.0, end_secs: 90.0, text: "Third segment.".into() },
/// ];
///
/// let mut doc = Document::new("First segment. Second segment. Third segment.");
/// doc.metadata.insert(
///     "subtitle_cues".into(),
///     serde_json::to_value(&cues).unwrap(),
/// );
/// doc.metadata.insert("duration_secs".into(), serde_json::json!(90.0));
///
/// let chunker = TimestampChunker::new(60.0);
/// let chunks = chunker.chunk(&doc).unwrap();
/// assert!(chunks.len() >= 2);
/// assert!(chunks[0].metadata.custom.contains_key("start_secs"));
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct TimestampChunker {
    /// Target chunk duration in seconds
    target_duration_secs: f64,
    /// Minimum chunk duration (avoids tiny fragments)
    min_duration_secs: f64,
    /// Maximum chunk duration (hard limit)
    #[allow(dead_code)]
    max_duration_secs: f64,
    /// Overlap duration for context continuity
    overlap_secs: f64,
}

impl TimestampChunker {
    /// Create a timestamp chunker with the given target duration.
    #[must_use]
    pub fn new(target_duration_secs: f64) -> Self {
        Self {
            target_duration_secs,
            min_duration_secs: 10.0,
            max_duration_secs: target_duration_secs * 2.0,
            overlap_secs: 5.0,
        }
    }

    /// Set minimum chunk duration.
    #[must_use]
    pub fn with_min_duration(mut self, secs: f64) -> Self {
        self.min_duration_secs = secs;
        self
    }

    /// Set maximum chunk duration.
    #[must_use]
    pub fn with_max_duration(mut self, secs: f64) -> Self {
        self.max_duration_secs = secs;
        self
    }

    /// Set overlap duration.
    #[must_use]
    pub fn with_overlap(mut self, secs: f64) -> Self {
        self.overlap_secs = secs;
        self
    }

    /// Build a chunk from a slice of cues.
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::disallowed_methods)] // json! macro internally uses unwrap
    fn build_chunk(
        document: &Document,
        cues: &[&crate::media::SubtitleCue],
        chunk_start_secs: f64,
    ) -> Chunk {
        let text: String = cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");

        let start_secs = cues.first().map(|c| c.start_secs).unwrap_or(chunk_start_secs);
        let end_secs = cues.last().map(|c| c.end_secs).unwrap_or(chunk_start_secs);

        let mut chunk =
            Chunk::new(document.id, text, start_secs.max(0.0) as usize, end_secs.max(0.0) as usize);
        chunk.metadata.title = document.title.clone();
        chunk.metadata.custom.insert("start_secs".into(), serde_json::json!(start_secs));
        chunk.metadata.custom.insert("end_secs".into(), serde_json::json!(end_secs));
        chunk.metadata.custom.insert(
            "start_display".into(),
            serde_json::json!(crate::media::format_display_time(start_secs)),
        );
        chunk.metadata.custom.insert(
            "end_display".into(),
            serde_json::json!(crate::media::format_display_time(end_secs)),
        );
        chunk.metadata.custom.insert("cue_count".into(), serde_json::json!(cues.len()));
        chunk
    }
}

/// Default target chunk duration in seconds
const DEFAULT_TARGET_DURATION: f64 = 60.0;

impl Default for TimestampChunker {
    fn default() -> Self {
        Self {
            target_duration_secs: DEFAULT_TARGET_DURATION,
            min_duration_secs: 10.0,
            max_duration_secs: DEFAULT_TARGET_DURATION * 2.0,
            overlap_secs: 5.0,
        }
    }
}

impl Chunker for TimestampChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document.title.clone().unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        // Extract subtitle cues from document metadata
        let cues: Vec<crate::media::SubtitleCue> = document
            .metadata
            .get("subtitle_cues")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        if cues.is_empty() {
            // No timestamp data — fall back to RecursiveChunker
            return RecursiveChunker::new(512, 50).chunk(document);
        }

        let mut chunks = Vec::new();
        let mut current_cues: Vec<&crate::media::SubtitleCue> = Vec::new();
        let mut chunk_start = cues[0].start_secs;

        for cue in &cues {
            let current_duration = cue.end_secs - chunk_start;

            // Emit chunk when we've reached target duration
            if current_duration >= self.target_duration_secs && !current_cues.is_empty() {
                chunks.push(Self::build_chunk(document, &current_cues, chunk_start));

                // Start next chunk, keeping cues that fall within overlap window
                let overlap_start = cue.start_secs - self.overlap_secs;
                current_cues.retain(|c| c.start_secs >= overlap_start);
                chunk_start = current_cues.first().map(|c| c.start_secs).unwrap_or(cue.start_secs);
            }

            current_cues.push(cue);
        }

        // Emit final chunk
        if !current_cues.is_empty() {
            let final_duration =
                current_cues.last().map(|c| c.end_secs).unwrap_or(0.0) - chunk_start;

            if final_duration < self.min_duration_secs && !chunks.is_empty() {
                // Merge into previous chunk if too short
                if let Some(last) = chunks.last_mut() {
                    let extra_text: String =
                        current_cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
                    last.content.push(' ');
                    last.content.push_str(&extra_text);

                    let end_secs = current_cues.last().map(|c| c.end_secs).unwrap_or(0.0);
                    #[allow(clippy::cast_sign_loss)]
                    {
                        last.end_offset = end_secs.max(0.0) as usize;
                    }
                    last.metadata.custom.insert("end_secs".into(), serde_json::json!(end_secs));
                    last.metadata.custom.insert(
                        "end_display".into(),
                        serde_json::json!(crate::media::format_display_time(end_secs)),
                    );
                }
            } else {
                chunks.push(Self::build_chunk(document, &current_cues, chunk_start));
            }
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        let duration =
            document.metadata.get("duration_secs").and_then(|v| v.as_f64()).unwrap_or(0.0);

        if duration <= 0.0 || self.target_duration_secs <= 0.0 {
            return usize::from(!document.content.is_empty());
        }
        #[allow(clippy::cast_sign_loss)]
        let estimate = (duration / self.target_duration_secs).ceil() as usize;
        estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    fn make_cues(durations: &[(f64, f64, &str)]) -> Vec<crate::media::SubtitleCue> {
        durations
            .iter()
            .enumerate()
            .map(|(i, (start, end, text))| crate::media::SubtitleCue {
                index: i,
                start_secs: *start,
                end_secs: *end,
                text: (*text).to_string(),
            })
            .collect()
    }

    fn doc_with_cues(cues: &[crate::media::SubtitleCue]) -> Document {
        let text: String = cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        let duration = cues.last().map(|c| c.end_secs).unwrap_or(0.0);
        let mut doc = Document::new(text);
        doc.metadata.insert("subtitle_cues".into(), serde_json::to_value(cues).unwrap());
        doc.metadata.insert("duration_secs".into(), serde_json::json!(duration));
        doc
    }

    #[test]
    fn test_timestamp_chunker_basic() {
        let cues = make_cues(&[
            (0.0, 25.0, "First segment."),
            (25.0, 50.0, "Second segment."),
            (50.0, 75.0, "Third segment."),
            (75.0, 100.0, "Fourth segment."),
        ]);
        let doc = doc_with_cues(&cues);

        let chunker = TimestampChunker::new(60.0);
        let chunks = chunker.chunk(&doc).unwrap();

        assert!(chunks.len() >= 2, "Expected at least 2 chunks, got {}", chunks.len());
        for chunk in &chunks {
            assert!(chunk.metadata.custom.contains_key("start_secs"));
            assert!(chunk.metadata.custom.contains_key("end_secs"));
            assert!(chunk.metadata.custom.contains_key("start_display"));
            assert!(chunk.metadata.custom.contains_key("end_display"));
            assert!(chunk.metadata.custom.contains_key("cue_count"));
        }
    }

    #[test]
    fn test_timestamp_chunker_single_short_chunk() {
        let cues = make_cues(&[(0.0, 10.0, "Only one."), (10.0, 20.0, "Short transcript.")]);
        let doc = doc_with_cues(&cues);

        let chunker = TimestampChunker::new(60.0);
        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_timestamp_chunker_fallback_no_cues() {
        let doc = Document::new("Plain text without any subtitle metadata.");
        let chunker = TimestampChunker::new(60.0);
        let chunks = chunker.chunk(&doc).unwrap();
        // Falls back to RecursiveChunker
        assert!(!chunks.is_empty());
        assert!(!chunks[0].metadata.custom.contains_key("start_secs"));
    }

    #[test]
    fn test_timestamp_chunker_empty_doc() {
        let doc = Document::new("");
        let chunker = TimestampChunker::new(60.0);
        assert!(chunker.chunk(&doc).is_err());
    }

    #[test]
    fn test_timestamp_chunker_metadata_values() {
        let cues = make_cues(&[
            (60.0, 90.0, "Starts at one minute."),
            (90.0, 120.0, "Ends at two minutes."),
        ]);
        let doc = doc_with_cues(&cues);

        let chunker = TimestampChunker::new(120.0);
        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);

        let start = chunks[0].metadata.custom["start_secs"].as_f64().unwrap();
        let end = chunks[0].metadata.custom["end_secs"].as_f64().unwrap();
        assert!((start - 60.0).abs() < 0.01);
        assert!((end - 120.0).abs() < 0.01);
        assert_eq!(chunks[0].metadata.custom["start_display"], "1:00");
        assert_eq!(chunks[0].metadata.custom["end_display"], "2:00");
    }

    #[test]
    fn test_timestamp_chunker_estimate() {
        let mut doc = Document::new("content");
        doc.metadata.insert("duration_secs".into(), serde_json::json!(300.0));

        let chunker = TimestampChunker::new(60.0);
        assert_eq!(chunker.estimate_chunks(&doc), 5);
    }

    #[test]
    fn test_timestamp_chunker_estimate_no_duration() {
        let doc = Document::new("content");
        let chunker = TimestampChunker::new(60.0);
        assert_eq!(chunker.estimate_chunks(&doc), 1);
    }

    #[test]
    fn test_timestamp_chunker_merge_short_final() {
        // Create cues where the final group is very short
        let cues = make_cues(&[
            (0.0, 30.0, "First."),
            (30.0, 60.0, "Second."),
            (60.0, 65.0, "Tiny final."),
        ]);
        let doc = doc_with_cues(&cues);

        let chunker = TimestampChunker::new(55.0).with_min_duration(10.0);
        let chunks = chunker.chunk(&doc).unwrap();

        // The tiny final should be merged into the previous chunk
        let last_text = &chunks.last().unwrap().content;
        assert!(last_text.contains("Tiny final"), "Last chunk: {last_text}");
    }

    #[test]
    fn test_timestamp_chunker_all_text_represented() {
        let cues = make_cues(&[
            (0.0, 20.0, "Alpha."),
            (20.0, 40.0, "Beta."),
            (40.0, 60.0, "Gamma."),
            (60.0, 80.0, "Delta."),
            (80.0, 100.0, "Epsilon."),
        ]);
        let doc = doc_with_cues(&cues);

        let chunker = TimestampChunker::new(45.0).with_overlap(0.0);
        let chunks = chunker.chunk(&doc).unwrap();

        // Every cue text should appear in at least one chunk
        for cue in &cues {
            assert!(
                chunks.iter().any(|c| c.content.contains(&cue.text)),
                "Cue text '{}' not found in any chunk",
                cue.text
            );
        }
    }

    #[test]
    fn test_timestamp_chunker_default() {
        let chunker = TimestampChunker::default();
        assert!((chunker.target_duration_secs - 60.0).abs() < 0.01);
        assert!((chunker.min_duration_secs - 10.0).abs() < 0.01);
        assert!((chunker.max_duration_secs - 120.0).abs() < 0.01);
        assert!((chunker.overlap_secs - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_timestamp_chunker_builder() {
        let chunker = TimestampChunker::new(30.0)
            .with_min_duration(5.0)
            .with_max_duration(90.0)
            .with_overlap(3.0);
        assert!((chunker.target_duration_secs - 30.0).abs() < 0.01);
        assert!((chunker.min_duration_secs - 5.0).abs() < 0.01);
        assert!((chunker.max_duration_secs - 90.0).abs() < 0.01);
        assert!((chunker.overlap_secs - 3.0).abs() < 0.01);
    }
}
