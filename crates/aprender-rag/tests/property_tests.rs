#![allow(clippy::disallowed_methods)]
//! Property-based tests for trueno-rag

use proptest::prelude::*;
use trueno_rag::{
    chunk::{Chunker, FixedSizeChunker, ParagraphChunker, RecursiveChunker, TimestampChunker},
    embed::{cosine_similarity, Embedder, MockEmbedder},
    media::{parse_subtitles, SubtitleCue, SubtitleFormat, SubtitleTrack},
    Document,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_recursive_chunker_produces_valid_chunks(
        content in "[a-zA-Z ]{100,1000}",
        chunk_size in 50usize..200,
        overlap in 0usize..50
    ) {
        let overlap = overlap.min(chunk_size / 2);
        let chunker = RecursiveChunker::new(chunk_size, overlap);
        let doc = Document::new(&content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            // All chunks should be non-empty
            for chunk in &chunks {
                prop_assert!(!chunk.content.is_empty());
            }

            // Chunk IDs should be unique
            let ids: std::collections::HashSet<_> = chunks.iter().map(|c| c.id).collect();
            prop_assert_eq!(ids.len(), chunks.len());
        }
    }

    #[test]
    fn prop_fixed_size_chunker_respects_size(
        content in "[a-zA-Z ]{200,500}",
        chunk_size in 50usize..150,
        overlap in 0usize..30
    ) {
        let overlap = overlap.min(chunk_size / 2);
        let chunker = FixedSizeChunker::new(chunk_size, overlap);
        let doc = Document::new(&content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            for chunk in &chunks {
                // Chunks should not exceed size by too much
                prop_assert!(chunk.content.len() <= chunk_size + 10);
            }
        }
    }

    #[test]
    fn prop_paragraph_chunker_groups_correctly(
        para_count in 2usize..8,
        max_paras in 1usize..4
    ) {
        // Create document with known paragraph count
        let content: String = (0..para_count)
            .map(|i| format!("Paragraph {} content here.", i))
            .collect::<Vec<_>>()
            .join("\n\n");

        let chunker = ParagraphChunker::new(max_paras);
        let doc = Document::new(&content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            // Should have roughly ceil(para_count / max_paras) chunks
            let expected_min = (para_count + max_paras - 1) / max_paras;
            prop_assert!(chunks.len() >= expected_min.saturating_sub(1));
        }
    }

    #[test]
    fn prop_embedder_produces_consistent_dimension(
        text in "[a-zA-Z ]{10,100}",
        dimension in 32usize..512
    ) {
        let embedder = MockEmbedder::new(dimension);

        if let Ok(embedding) = embedder.embed(&text) {
            prop_assert_eq!(embedding.len(), dimension);
        }
    }

    #[test]
    fn prop_cosine_similarity_bounded(
        v1 in prop::collection::vec(-1.0f32..1.0, 10..50),
        v2 in prop::collection::vec(-1.0f32..1.0, 10..50)
    ) {
        if v1.len() == v2.len() {
            let sim = cosine_similarity(&v1, &v2);
            // Cosine similarity should be in [-1, 1] range
            prop_assert!(sim >= -1.1 && sim <= 1.1);
        }
    }

    #[test]
    fn prop_document_preserves_content(content in "[a-zA-Z0-9 ]{1,500}") {
        let doc = Document::new(&content);
        prop_assert_eq!(doc.content, content);
    }

    #[test]
    fn prop_document_with_metadata(
        content in "[a-zA-Z ]{10,100}",
        title in "[a-zA-Z ]{5,30}",
        source in "[a-zA-Z/:.]{10,50}"
    ) {
        let doc = Document::new(&content)
            .with_title(&title)
            .with_source(&source);

        prop_assert_eq!(doc.content, content);
        prop_assert_eq!(doc.title, Some(title));
        prop_assert_eq!(doc.source, Some(source));
    }
}

/// Strategy to generate valid SRT cue text (no empty, no double newlines).
fn srt_text_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z ]{5,60}".prop_map(|s| s.trim().to_string()).prop_filter("non-empty", |s| !s.is_empty())
}

/// Strategy to generate a vector of subtitle cues with monotonically increasing timestamps.
fn subtitle_cues_strategy(
    count: std::ops::Range<usize>,
) -> impl Strategy<Value = Vec<SubtitleCue>> {
    count.prop_flat_map(|n| {
        proptest::collection::vec((1.0f64..10.0, srt_text_strategy()), n).prop_map(|pairs| {
            let mut time = 0.0;
            pairs
                .into_iter()
                .enumerate()
                .map(|(i, (duration, text))| {
                    let start = time;
                    let end = start + duration;
                    time = end + 0.1; // small gap between cues
                    SubtitleCue { index: i, start_secs: start, end_secs: end, text }
                })
                .collect()
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// SRT roundtrip: generate random cues → to_srt_string → parse_srt → verify cue count and content.
    #[test]
    fn prop_srt_roundtrip(cues in subtitle_cues_strategy(1..20)) {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: cues.clone(),
        };

        let srt_output = track.to_srt_string();
        let reparsed = parse_subtitles(&srt_output).unwrap();

        // Cue count must be preserved
        prop_assert_eq!(reparsed.cues.len(), cues.len());

        // Each cue text must match
        for (original, parsed) in cues.iter().zip(reparsed.cues.iter()) {
            prop_assert_eq!(&original.text, &parsed.text);
            // Timestamps preserved within 1ms (SRT has millisecond precision)
            prop_assert!((original.start_secs - parsed.start_secs).abs() < 0.002);
            prop_assert!((original.end_secs - parsed.end_secs).abs() < 0.002);
        }
    }

    /// Timestamp chunk coverage: every cue's text appears in at least one chunk.
    #[test]
    fn prop_timestamp_chunk_coverage(
        cues in subtitle_cues_strategy(2..30),
        target_secs in 10.0f64..120.0,
    ) {
        // Build a document with subtitle_cues metadata (how SubtitleLoader provides them)
        let plain_text: String = cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        let mut doc = Document::new(&plain_text);
        doc.metadata.insert(
            "subtitle_cues".into(),
            serde_json::to_value(&cues).unwrap(),
        );

        let chunker = TimestampChunker::new(target_secs).with_min_duration(0.0);
        let chunks = chunker.chunk(&doc).unwrap();

        // Must produce at least one chunk
        prop_assert!(!chunks.is_empty());

        // Every cue text must appear in at least one chunk
        for cue in &cues {
            let found = chunks.iter().any(|chunk| chunk.content.contains(&cue.text));
            prop_assert!(found, "Cue text {:?} not found in any chunk", cue.text);
        }

        // All chunks should be non-empty
        for chunk in &chunks {
            prop_assert!(!chunk.content.is_empty());
        }

        // Chunk IDs should be unique
        let ids: std::collections::HashSet<_> = chunks.iter().map(|c| c.id).collect();
        prop_assert_eq!(ids.len(), chunks.len());
    }

    /// SubtitleTrack methods: cues_in_range always returns subset, plain_text contains all cue text.
    #[test]
    fn prop_subtitle_track_invariants(cues in subtitle_cues_strategy(1..15)) {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: cues.clone(),
        };

        // Plain text contains every cue's text
        let plain = track.to_plain_text();
        for cue in &cues {
            prop_assert!(plain.contains(&cue.text));
        }

        // Duration is the end time of the last cue
        if let Some(last) = cues.last() {
            prop_assert!((track.duration_secs() - last.end_secs).abs() < 0.001);
        }

        // cues_in_range with full range returns all cues
        let all = track.cues_in_range(0.0, track.duration_secs() + 1.0);
        prop_assert_eq!(all.len(), cues.len());

        // cues_in_range with empty range before start returns nothing
        let none = track.cues_in_range(-10.0, -1.0);
        prop_assert_eq!(none.len(), 0);
    }
}

#[test]
fn test_chunker_estimate_accuracy() {
    let chunker = RecursiveChunker::new(100, 10);
    let doc = Document::new("Test content. ".repeat(50));

    let estimate = chunker.estimate_chunks(&doc);
    let actual = chunker.chunk(&doc).unwrap().len();

    // Estimate should be within 50% of actual
    let tolerance = (actual as f32 * 0.5).ceil() as usize;
    assert!(
        (estimate as i32 - actual as i32).unsigned_abs() as usize <= tolerance.max(2),
        "Estimate {} too far from actual {}",
        estimate,
        actual
    );
}
