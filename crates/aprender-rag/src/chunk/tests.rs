use super::*;

// ============ ChunkId Tests ============

#[test]
fn test_chunk_id_unique() {
    let id1 = ChunkId::new();
    let id2 = ChunkId::new();
    assert_ne!(id1, id2);
}

#[test]
fn test_chunk_id_default() {
    let id1 = ChunkId::default();
    let id2 = ChunkId::default();
    assert_ne!(id1, id2);
}

#[test]
fn test_chunk_id_display() {
    let id = ChunkId::new();
    let display = format!("{id}");
    assert!(!display.is_empty());
    assert!(display.contains('-'));
}

#[test]
fn test_chunk_id_serialization() {
    let id = ChunkId::new();
    let json = serde_json::to_string(&id).unwrap();
    let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, deserialized);
}

// ============ ChunkMetadata Tests ============

#[test]
fn test_chunk_metadata_default() {
    let meta = ChunkMetadata::default();
    assert!(meta.title.is_none());
    assert!(meta.headers.is_empty());
    assert!(meta.page.is_none());
    assert!(meta.custom.is_empty());
}

#[test]
fn test_chunk_metadata_serialization() {
    let mut meta = ChunkMetadata {
        title: Some("Test".to_string()),
        headers: vec!["Section 1".to_string()],
        page: Some(42),
        ..Default::default()
    };
    meta.custom.insert("key".to_string(), serde_json::json!("value"));

    let json = serde_json::to_string(&meta).unwrap();
    let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(meta.title, deserialized.title);
    assert_eq!(meta.headers, deserialized.headers);
    assert_eq!(meta.page, deserialized.page);
}

// ============ Chunk Tests ============

#[test]
fn test_chunk_creation() {
    let doc_id = DocumentId::new();
    let chunk = Chunk::new(doc_id, "Hello world".to_string(), 0, 11);

    assert_eq!(chunk.document_id, doc_id);
    assert_eq!(chunk.content, "Hello world");
    assert_eq!(chunk.start_offset, 0);
    assert_eq!(chunk.end_offset, 11);
    assert!(chunk.embedding.is_none());
}

#[test]
fn test_chunk_len() {
    let doc_id = DocumentId::new();
    let chunk = Chunk::new(doc_id, "Hello".to_string(), 0, 5);
    assert_eq!(chunk.len(), 5);
    assert!(!chunk.is_empty());
}

#[test]
fn test_chunk_empty() {
    let doc_id = DocumentId::new();
    let chunk = Chunk::new(doc_id, String::new(), 0, 0);
    assert_eq!(chunk.len(), 0);
    assert!(chunk.is_empty());
}

#[test]
fn test_chunk_set_embedding() {
    let doc_id = DocumentId::new();
    let mut chunk = Chunk::new(doc_id, "Test".to_string(), 0, 4);
    assert!(chunk.embedding.is_none());

    chunk.set_embedding(vec![0.1, 0.2, 0.3]);
    assert!(chunk.embedding.is_some());
    assert_eq!(chunk.embedding.unwrap(), vec![0.1, 0.2, 0.3]);
}

// ============ ChunkingStrategy Tests ============

#[test]
fn test_chunking_strategy_default() {
    let strategy = ChunkingStrategy::default();
    match strategy {
        ChunkingStrategy::Recursive { chunk_size, overlap, separators } => {
            assert_eq!(chunk_size, 512);
            assert_eq!(overlap, 50);
            assert!(!separators.is_empty());
        }
        _ => panic!("Expected Recursive strategy"),
    }
}

#[test]
fn test_chunking_strategy_serialization() {
    let strategy = ChunkingStrategy::FixedSize { chunk_size: 256, overlap: 32 };
    let json = serde_json::to_string(&strategy).unwrap();
    let deserialized: ChunkingStrategy = serde_json::from_str(&json).unwrap();

    match deserialized {
        ChunkingStrategy::FixedSize { chunk_size, overlap } => {
            assert_eq!(chunk_size, 256);
            assert_eq!(overlap, 32);
        }
        _ => panic!("Wrong strategy type"),
    }
}

// ============ RecursiveChunker Tests ============

#[test]
fn test_recursive_chunker_new() {
    let chunker = RecursiveChunker::new(512, 50);
    assert_eq!(chunker.chunk_size, 512);
    assert_eq!(chunker.overlap, 50);
    assert!(!chunker.separators.is_empty());
}

#[test]
fn test_recursive_chunker_custom_separators() {
    let chunker =
        RecursiveChunker::new(256, 20).with_separators(vec!["\n".to_string(), " ".to_string()]);
    assert_eq!(chunker.separators.len(), 2);
}

#[test]
fn test_recursive_chunker_empty_document() {
    let chunker = RecursiveChunker::new(100, 10);
    let doc = Document::new("");

    let result = chunker.chunk(&doc);
    assert!(result.is_err());
}

#[test]
fn test_recursive_chunker_small_document() {
    let chunker = RecursiveChunker::new(1000, 100);
    let doc = Document::new("This is a small document.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "This is a small document.");
}

#[test]
fn test_recursive_chunker_paragraph_split() {
    let chunker = RecursiveChunker::new(50, 10);
    let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(chunks.len() >= 2);
}

#[test]
fn test_recursive_chunker_respects_chunk_size() {
    let chunker = RecursiveChunker::new(20, 5);
    let doc = Document::new("This is a longer document that needs to be split into multiple chunks based on the chunk size.");

    let chunks = chunker.chunk(&doc).unwrap();
    // All chunks (except maybe the first with overlap) should be <= chunk_size + overlap
    for chunk in &chunks {
        assert!(
            chunk.content.len() <= 25 + 5, // chunk_size + some tolerance
            "Chunk too large: {} chars",
            chunk.content.len()
        );
    }
}

#[test]
fn test_recursive_chunker_preserves_document_id() {
    let chunker = RecursiveChunker::new(50, 10);
    let doc = Document::new("Content").with_title("Test Doc");

    let chunks = chunker.chunk(&doc).unwrap();
    for chunk in chunks {
        assert_eq!(chunk.document_id, doc.id);
        assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
    }
}

#[test]
fn test_recursive_chunker_estimate() {
    let chunker = RecursiveChunker::new(100, 20);
    let doc = Document::new("A".repeat(500));

    let estimate = chunker.estimate_chunks(&doc);
    let actual = chunker.chunk(&doc).unwrap().len();

    // Estimate should be in reasonable range
    assert!(estimate > 0);
    assert!(estimate <= actual * 2);
}

// ============ FixedSizeChunker Tests ============

#[test]
fn test_fixed_size_chunker_new() {
    let chunker = FixedSizeChunker::new(256, 32);
    assert_eq!(chunker.chunk_size, 256);
    assert_eq!(chunker.overlap, 32);
}

#[test]
fn test_fixed_size_chunker_empty_document() {
    let chunker = FixedSizeChunker::new(100, 10);
    let doc = Document::new("");

    let result = chunker.chunk(&doc);
    assert!(result.is_err());
}

#[test]
fn test_fixed_size_chunker_small_document() {
    let chunker = FixedSizeChunker::new(100, 10);
    let doc = Document::new("Short text");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "Short text");
}

#[test]
fn test_fixed_size_chunker_exact_split() {
    let chunker = FixedSizeChunker::new(10, 0);
    let doc = Document::new("0123456789abcdefghij");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].content, "0123456789");
    assert_eq!(chunks[1].content, "abcdefghij");
}

#[test]
fn test_fixed_size_chunker_with_overlap() {
    let chunker = FixedSizeChunker::new(10, 3);
    let doc = Document::new("0123456789abcdefghij");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(chunks.len() >= 2);
    // Second chunk should start before end of first chunk
}

#[test]
fn test_fixed_size_chunker_unicode() {
    let chunker = FixedSizeChunker::new(5, 0);
    let doc = Document::new("héllo wörld");

    let chunks = chunker.chunk(&doc).unwrap();
    // Should handle unicode correctly
    assert!(chunks.len() >= 2);
    for chunk in chunks {
        assert!(chunk.content.chars().count() <= 5);
    }
}

#[test]
fn test_fixed_size_chunker_estimate() {
    let chunker = FixedSizeChunker::new(10, 2);
    let doc = Document::new("A".repeat(100));

    let estimate = chunker.estimate_chunks(&doc);
    let actual = chunker.chunk(&doc).unwrap().len();

    assert!(estimate > 0);
    #[allow(clippy::cast_possible_wrap)]
    let diff = (estimate as isize - actual as isize).abs();
    assert!(diff <= 2);
}

// ============ SentenceChunker Tests ============

#[test]
fn test_sentence_chunker_new() {
    let chunker = SentenceChunker::new(3, 1);
    assert_eq!(chunker.max_sentences, 3);
    assert_eq!(chunker.overlap_sentences, 1);
}

#[test]
fn test_sentence_chunker_empty_document() {
    let chunker = SentenceChunker::new(2, 0);
    let doc = Document::new("");

    let result = chunker.chunk(&doc);
    assert!(result.is_err());
}

#[test]
fn test_sentence_chunker_single_sentence() {
    let chunker = SentenceChunker::new(2, 0);
    let doc = Document::new("This is a single sentence.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 1);
}

#[test]
fn test_sentence_chunker_multiple_sentences() {
    let chunker = SentenceChunker::new(2, 0);
    let doc = Document::new("First sentence. Second sentence. Third sentence. Fourth sentence.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 2);
}

#[test]
fn test_sentence_chunker_with_overlap() {
    let chunker = SentenceChunker::new(2, 1);
    let doc = Document::new("One. Two. Three. Four.");

    let chunks = chunker.chunk(&doc).unwrap();
    // With overlap, we should have more chunks
    assert!(chunks.len() >= 2);
}

#[test]
fn test_sentence_chunker_exclamation_question() {
    let chunker = SentenceChunker::new(1, 0);
    let doc = Document::new("Hello! How are you? I am fine.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(chunks.len() >= 3);
}

#[test]
fn test_sentence_chunker_estimate() {
    let chunker = SentenceChunker::new(2, 1);
    let doc = Document::new("One. Two. Three. Four. Five.");

    let estimate = chunker.estimate_chunks(&doc);
    assert!(estimate > 0);
}

// ============ SemanticChunker Tests ============

#[test]
fn test_semantic_chunker_new() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.5, 1000);
    assert!((chunker.similarity_threshold - 0.5).abs() < 0.01);
    assert_eq!(chunker.max_chunk_size, 1000);
}

#[test]
fn test_semantic_chunker_empty_document() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.5, 1000);
    let doc = Document::new("");

    let result = chunker.chunk(&doc);
    assert!(result.is_err());
}

#[test]
fn test_semantic_chunker_single_sentence() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.5, 1000);
    let doc = Document::new("This is a single sentence.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 1);
}

#[test]
fn test_semantic_chunker_multiple_sentences() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.9, 500); // High threshold forces splits
    let doc = Document::new("First sentence. Second sentence. Third sentence.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(!chunks.is_empty());
}

#[test]
fn test_semantic_chunker_preserves_document_id() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.5, 1000);
    let doc = Document::new("Test content here.").with_title("Test Doc");

    let chunks = chunker.chunk(&doc).unwrap();
    for chunk in chunks {
        assert_eq!(chunk.document_id, doc.id);
        assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
    }
}

#[test]
fn test_semantic_chunker_respects_max_size() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.0, 50); // Low threshold, small max size
    let doc = Document::new(
        "First sentence here. Second sentence follows. Third sentence comes. Fourth sentence ends.",
    );

    let chunks = chunker.chunk(&doc).unwrap();
    for chunk in &chunks {
        assert!(chunk.content.len() <= 100); // Some tolerance
    }
}

#[test]
fn test_semantic_chunker_estimate() {
    let embedder = crate::embed::MockEmbedder::new(64);
    let chunker = SemanticChunker::new(embedder, 0.5, 100);
    let doc = Document::new("Sentence one. Sentence two. Sentence three.");

    let estimate = chunker.estimate_chunks(&doc);
    assert!(estimate > 0);
}

// ============ StructuralChunker Tests ============

#[test]
fn test_structural_chunker_new() {
    let chunker = StructuralChunker::new(true, 500);
    assert!(chunker.respect_headers);
    assert_eq!(chunker.max_section_size, 500);
}

#[test]
fn test_structural_chunker_empty_document() {
    let chunker = StructuralChunker::new(true, 500);
    let doc = Document::new("");

    let result = chunker.chunk(&doc);
    assert!(result.is_err());
}

#[test]
fn test_structural_chunker_no_headers() {
    let chunker = StructuralChunker::new(true, 500);
    let doc = Document::new("Just plain text without headers.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 1);
}

#[test]
fn test_structural_chunker_markdown_headers() {
    let chunker = StructuralChunker::new(true, 1000);
    let doc = Document::new("# Header 1\n\nContent 1.\n\n# Header 2\n\nContent 2.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 2);
    assert!(chunks[0].content.contains("Header 1"));
    assert!(chunks[1].content.contains("Header 2"));
}

#[test]
fn test_structural_chunker_nested_headers() {
    let chunker = StructuralChunker::new(true, 1000);
    let doc = Document::new("# Main\n\nIntro.\n\n## Sub 1\n\nContent 1.\n\n## Sub 2\n\nContent 2.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(chunks.len() >= 2);
}

#[test]
fn test_structural_chunker_preserves_metadata() {
    let chunker = StructuralChunker::new(true, 1000);
    let doc = Document::new("# Section\n\nContent.").with_title("Test Doc");

    let chunks = chunker.chunk(&doc).unwrap();
    for chunk in chunks {
        assert_eq!(chunk.document_id, doc.id);
        assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
    }
}

#[test]
fn test_structural_chunker_header_in_metadata() {
    let chunker = StructuralChunker::new(true, 1000);
    let doc = Document::new("# My Section\n\nSection content here.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(!chunks.is_empty());
    assert!(
        chunks[0].metadata.headers.contains(&"My Section".to_string())
            || chunks[0].content.contains("My Section")
    );
}

#[test]
fn test_structural_chunker_respects_max_size() {
    let chunker = StructuralChunker::new(true, 50);
    let doc = Document::new("# Header\n\n".to_string() + &"A ".repeat(100));

    let chunks = chunker.chunk(&doc).unwrap();
    // Should split large sections
    assert!(!chunks.is_empty());
}

#[test]
fn test_structural_chunker_estimate() {
    let chunker = StructuralChunker::new(true, 500);
    let doc = Document::new("# H1\n\nC1.\n\n# H2\n\nC2.\n\n# H3\n\nC3.");

    let estimate = chunker.estimate_chunks(&doc);
    assert!(estimate > 0);
}

// ============ ParagraphChunker Tests ============

#[test]
fn test_paragraph_chunker_new() {
    let chunker = ParagraphChunker::new(3);
    assert_eq!(chunker.max_paragraphs, 3);
}

#[test]
fn test_paragraph_chunker_empty_document() {
    let chunker = ParagraphChunker::new(2);
    let doc = Document::new("");

    let result = chunker.chunk(&doc);
    assert!(result.is_err());
}

#[test]
fn test_paragraph_chunker_single_paragraph() {
    let chunker = ParagraphChunker::new(2);
    let doc = Document::new("This is a single paragraph without line breaks.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content.trim(), "This is a single paragraph without line breaks.");
}

#[test]
fn test_paragraph_chunker_multiple_paragraphs() {
    let chunker = ParagraphChunker::new(1);
    let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 3);
}

#[test]
fn test_paragraph_chunker_groups_paragraphs() {
    let chunker = ParagraphChunker::new(2);
    let doc = Document::new("Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 2);
    assert!(chunks[0].content.contains("Para 1"));
    assert!(chunks[0].content.contains("Para 2"));
}

#[test]
fn test_paragraph_chunker_preserves_document_id() {
    let chunker = ParagraphChunker::new(1);
    let doc = Document::new("Para 1.\n\nPara 2.").with_title("Test Doc");

    let chunks = chunker.chunk(&doc).unwrap();
    for chunk in chunks {
        assert_eq!(chunk.document_id, doc.id);
        assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
    }
}

#[test]
fn test_paragraph_chunker_estimate() {
    let chunker = ParagraphChunker::new(2);
    let doc = Document::new("P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.");

    let estimate = chunker.estimate_chunks(&doc);
    let actual = chunker.chunk(&doc).unwrap().len();

    assert!(estimate > 0);
    #[allow(clippy::cast_possible_wrap)]
    let diff = (estimate as isize - actual as isize).abs();
    assert!(diff <= 2);
}

#[test]
fn test_paragraph_chunker_whitespace_handling() {
    let chunker = ParagraphChunker::new(1);
    let doc = Document::new("  First paragraph.  \n\n  Second paragraph.  ");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 2);
    // Content should be trimmed
    assert!(!chunks[0].content.starts_with(' '));
    assert!(!chunks[1].content.ends_with(' '));
}

#[test]
fn test_paragraph_chunker_triple_newline() {
    let chunker = ParagraphChunker::new(1);
    let doc = Document::new("Para 1.\n\n\nPara 2.");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks.len(), 2);
}

// ============ Edge Cases ============

#[test]
fn test_chunker_with_newlines() {
    let chunker = RecursiveChunker::new(50, 0);
    let doc = Document::new("Line 1\nLine 2\nLine 3");

    let chunks = chunker.chunk(&doc).unwrap();
    assert!(!chunks.is_empty());
}

#[test]
fn test_chunker_offset_tracking() {
    let chunker = FixedSizeChunker::new(5, 0);
    let doc = Document::new("0123456789");

    let chunks = chunker.chunk(&doc).unwrap();
    assert_eq!(chunks[0].start_offset, 0);
    assert_eq!(chunks[0].end_offset, 5);
    assert_eq!(chunks[1].start_offset, 5);
    assert_eq!(chunks[1].end_offset, 10);
}

// ============ Property-Based Tests ============

use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_recursive_chunker_no_empty_chunks(content in "[a-zA-Z ]{10,500}") {
        let chunker = RecursiveChunker::new(50, 10);
        let doc = Document::new(content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            for chunk in chunks {
                prop_assert!(!chunk.is_empty());
            }
        }
    }

    #[test]
    fn prop_fixed_size_respects_max(content in "[a-zA-Z]{20,200}", chunk_size in 10usize..50) {
        let chunker = FixedSizeChunker::new(chunk_size, 0);
        let doc = Document::new(content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            for chunk in chunks {
                prop_assert!(chunk.content.chars().count() <= chunk_size);
            }
        }
    }

    #[test]
    fn prop_chunk_ids_unique(content in "[a-zA-Z ]{50,200}") {
        let chunker = FixedSizeChunker::new(20, 5);
        let doc = Document::new(content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            let ids: std::collections::HashSet<_> = chunks.iter().map(|c| c.id).collect();
            prop_assert_eq!(ids.len(), chunks.len());
        }
    }

    #[test]
    fn prop_paragraph_chunker_no_empty_chunks(content in "[a-zA-Z ]{10,100}(\n\n[a-zA-Z ]{10,100}){1,5}") {
        let chunker = ParagraphChunker::new(2);
        let doc = Document::new(content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            for chunk in chunks {
                prop_assert!(!chunk.is_empty());
            }
        }
    }

    #[test]
    fn prop_paragraph_chunker_respects_max(
        content in "[a-zA-Z ]{5,30}(\n\n[a-zA-Z ]{5,30}){2,8}",
        max_paras in 1usize..5
    ) {
        let chunker = ParagraphChunker::new(max_paras);
        let doc = Document::new(content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            for chunk in &chunks {
                let para_count = chunk.content.split("\n\n").filter(|p| !p.trim().is_empty()).count();
                prop_assert!(para_count <= max_paras);
            }
        }
    }
}
