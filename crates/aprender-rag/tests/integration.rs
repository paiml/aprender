#![allow(clippy::disallowed_methods)]
//! Integration tests for trueno-rag

use trueno_rag::{
    chunk::{
        Chunker, ParagraphChunker, RecursiveChunker, SentenceChunker, StructuralChunker,
        TimestampChunker,
    },
    embed::MockEmbedder,
    fusion::FusionStrategy,
    loader::{DocumentLoader, LoaderRegistry, SubtitleLoader},
    media::SubtitleCue,
    pipeline::RagPipelineBuilder,
    rerank::{LexicalReranker, NoOpReranker},
    Document,
};

#[test]
fn test_end_to_end_rag_pipeline() {
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(200, 20))
        .embedder(MockEmbedder::new(128))
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(2000)
        .build()
        .expect("Failed to build pipeline");

    // Index multiple documents
    let docs = vec![
        Document::new(
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It achieves memory safety without garbage collection.",
        )
        .with_title("Rust Overview"),
        Document::new(
            "Python is a high-level programming language known for its readability and \
             extensive standard library. It's popular for data science and web development.",
        )
        .with_title("Python Overview"),
        Document::new(
            "Machine learning is a subset of artificial intelligence that enables systems \
             to learn and improve from experience without being explicitly programmed.",
        )
        .with_title("ML Introduction"),
    ];

    let chunk_count = pipeline.index_documents(&docs).expect("Failed to index");
    assert!(chunk_count >= 3);
    assert_eq!(pipeline.document_count(), 3);

    // Query for Rust-related content
    let (results, context) = pipeline
        .query_with_context("memory safety in systems programming", 5)
        .expect("Query failed");

    assert!(!results.is_empty());
    assert!(!context.is_empty());

    // The top result should be about Rust
    let top_content = &results[0].chunk.content.to_lowercase();
    assert!(
        top_content.contains("rust") || top_content.contains("memory"),
        "Expected Rust-related content in top result"
    );
}

#[test]
fn test_different_chunking_strategies() {
    let doc = Document::new(
        "First paragraph about topic A.\n\n\
         Second paragraph about topic B.\n\n\
         Third paragraph about topic C.",
    );

    // Test ParagraphChunker
    let para_chunker = ParagraphChunker::new(1);
    let para_chunks = para_chunker.chunk(&doc).expect("ParagraphChunker failed");
    assert_eq!(para_chunks.len(), 3);

    // Test SentenceChunker
    let sent_chunker = SentenceChunker::new(2, 0);
    let sent_chunks = sent_chunker.chunk(&doc).expect("SentenceChunker failed");
    assert!(!sent_chunks.is_empty());

    // Test StructuralChunker with markdown
    let md_doc = Document::new("# Header 1\n\nContent 1.\n\n# Header 2\n\nContent 2.");
    let struct_chunker = StructuralChunker::new(true, 500);
    let struct_chunks = struct_chunker.chunk(&md_doc).expect("StructuralChunker failed");
    assert_eq!(struct_chunks.len(), 2);
}

#[test]
fn test_fusion_strategies_produce_results() {
    let strategies = vec![
        FusionStrategy::RRF { k: 60.0 },
        FusionStrategy::Linear { dense_weight: 0.7 },
        FusionStrategy::DBSF,
        FusionStrategy::Union,
    ];

    for strategy in strategies {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .fusion(strategy.clone())
            .build()
            .expect("Failed to build pipeline");

        pipeline
            .index_document(&Document::new("Test document content here."))
            .expect("Failed to index");

        let results = pipeline.query("test", 5).expect("Query failed");
        assert!(results.len() <= 5, "Strategy {:?} returned too many results", strategy);
    }
}

#[test]
fn test_context_assembly_with_citations() {
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(64))
        .reranker(NoOpReranker::new())
        .max_context_tokens(1000)
        .build()
        .expect("Failed to build pipeline");

    let doc = Document::new("Important content for citation.").with_title("Test Document");
    pipeline.index_document(&doc).expect("Failed to index");

    let (_, context) = pipeline.query_with_context("important content", 5).expect("Query failed");

    let formatted = context.format_with_citations();
    assert!(formatted.contains("[1]"), "Expected citation marker");

    let citation_list = context.citation_list();
    assert!(citation_list.contains("Test Document"), "Expected document title in citations");
}

#[test]
fn test_empty_document_handling() {
    let chunker = RecursiveChunker::new(100, 10);
    let empty_doc = Document::new("");

    let result = chunker.chunk(&empty_doc);
    assert!(result.is_err(), "Empty document should produce error");
}

#[test]
fn test_large_document_chunking() {
    let large_content = "This is a test sentence. ".repeat(1000);
    let doc = Document::new(large_content);

    let chunker = RecursiveChunker::new(500, 50);
    let chunks = chunker.chunk(&doc).expect("Chunking failed");

    // Should produce multiple chunks
    assert!(chunks.len() > 1, "Large document should produce multiple chunks");

    // Each chunk should be within size limit (with some tolerance)
    for chunk in &chunks {
        assert!(
            chunk.content.len() <= 600,
            "Chunk exceeds size limit: {} chars",
            chunk.content.len()
        );
    }
}

#[test]
fn test_query_ranking_consistency() {
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(64))
        .reranker(LexicalReranker::new())
        .build()
        .expect("Failed to build pipeline");

    // Index documents with different relevance
    pipeline
        .index_document(&Document::new("exact match query terms here"))
        .expect("Failed to index");
    pipeline
        .index_document(&Document::new("completely unrelated content"))
        .expect("Failed to index");

    let results = pipeline.query("exact match query", 5).expect("Query failed");

    if results.len() >= 2 {
        // First result should have higher score than second
        assert!(
            results[0].best_score() >= results[1].best_score(),
            "Results should be sorted by score"
        );
    }
}

// ============================================================================
// MEDIA SUPPORT INTEGRATION TESTS (Spec Section 11.3)
// ============================================================================

/// Spec 11.3.1: Load a real .srt file → chunk → embed → retrieve by query.
#[test]
fn test_srt_full_pipeline_chunk_embed_retrieve() {
    // Create a temp .srt file
    let dir = std::env::temp_dir().join("trueno_rag_integ_srt_pipeline");
    let _ = std::fs::create_dir_all(&dir);
    let srt_path = dir.join("lecture.srt");
    std::fs::write(
        &srt_path,
        "\
1
00:00:01,000 --> 00:00:10,000
Machine learning enables computers to learn from data without explicit programming.

2
00:00:11,000 --> 00:00:20,000
Deep learning uses neural networks with many layers for complex pattern recognition.

3
00:00:21,000 --> 00:00:30,000
Reinforcement learning trains agents through reward signals in an environment.

4
00:00:31,000 --> 00:00:40,000
Natural language processing handles text understanding and generation tasks.

5
00:00:41,000 --> 00:00:50,000
Computer vision focuses on image and video analysis using convolutional networks.
",
    )
    .unwrap();

    // Load via SubtitleLoader
    let loader = SubtitleLoader;
    let doc = loader.load(&srt_path).unwrap();
    assert!(doc.content.contains("Machine learning"));
    assert!(doc.metadata.contains_key("subtitle_cues"));

    // Build pipeline, index, and query
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(TimestampChunker::new(20.0).with_min_duration(0.0))
        .embedder(MockEmbedder::new(64))
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .build()
        .expect("Failed to build pipeline");

    pipeline.index_document(&doc).expect("Failed to index");

    let results = pipeline.query("neural networks deep learning", 3).expect("Query failed");

    assert!(!results.is_empty(), "Should return results for SRT content");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Spec 11.3.2: Verify timestamp metadata survives the full pipeline.
#[test]
fn test_timestamp_metadata_survives_pipeline() {
    // Build document with subtitle cues in metadata
    let cues = vec![
        SubtitleCue {
            index: 0,
            start_secs: 0.0,
            end_secs: 30.0,
            text: "Introduction to distributed systems.".into(),
        },
        SubtitleCue {
            index: 1,
            start_secs: 30.0,
            end_secs: 60.0,
            text: "Consensus algorithms like Raft and Paxos.".into(),
        },
        SubtitleCue {
            index: 2,
            start_secs: 60.0,
            end_secs: 90.0,
            text: "Fault tolerance and replication strategies.".into(),
        },
    ];

    let mut doc = Document::new("Introduction to distributed systems. Consensus algorithms like Raft and Paxos. Fault tolerance and replication strategies.")
        .with_title("Distributed Systems Lecture");
    doc.metadata.insert("subtitle_cues".into(), serde_json::to_value(&cues).unwrap());
    doc.metadata.insert("duration_secs".into(), serde_json::json!(90.0));

    // Chunk with TimestampChunker
    let chunker = TimestampChunker::new(45.0).with_min_duration(0.0);
    let chunks = chunker.chunk(&doc).expect("Chunking failed");

    assert!(!chunks.is_empty());

    // Every chunk should carry start_secs and end_secs metadata
    for chunk in &chunks {
        assert!(
            chunk.metadata.custom.contains_key("start_secs"),
            "Chunk missing start_secs metadata"
        );
        assert!(chunk.metadata.custom.contains_key("end_secs"), "Chunk missing end_secs metadata");
        assert!(
            chunk.metadata.custom.contains_key("start_display"),
            "Chunk missing start_display metadata"
        );
        assert!(
            chunk.metadata.custom.contains_key("end_display"),
            "Chunk missing end_display metadata"
        );
        assert!(
            chunk.metadata.custom.contains_key("cue_count"),
            "Chunk missing cue_count metadata"
        );
    }

    // Index into pipeline and verify metadata accessible through retrieval
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(TimestampChunker::new(45.0).with_min_duration(0.0))
        .embedder(MockEmbedder::new(64))
        .reranker(LexicalReranker::new())
        .build()
        .expect("Failed to build pipeline");

    pipeline.index_document(&doc).expect("Failed to index");

    let results = pipeline.query("consensus Raft Paxos", 3).expect("Query failed");
    assert!(!results.is_empty());

    // Retrieved chunks should still have timestamp metadata
    let top = &results[0].chunk;
    assert!(
        top.metadata.custom.contains_key("start_secs"),
        "Retrieved chunk lost start_secs metadata"
    );
}

/// Spec 11.3.3: Sidecar resolution — .mp4 + .srt in temp dir, subtitle loader selected.
#[test]
fn test_sidecar_resolution_selects_subtitle_loader() {
    let dir = std::env::temp_dir().join("trueno_rag_integ_sidecar_resolution");
    let _ = std::fs::create_dir_all(&dir);

    // Create a fake .mp4 and a real .srt sidecar
    let mp4 = dir.join("lecture.mp4");
    let srt = dir.join("lecture.srt");
    std::fs::write(&mp4, b"fake mp4 data").unwrap();
    std::fs::write(&srt, "1\n00:00:01,000 --> 00:00:05,000\nSidecar content here.\n").unwrap();

    // LoaderRegistry should find the sidecar
    let sidecar = LoaderRegistry::find_sidecar(&mp4);
    assert!(sidecar.is_some(), "Should find .srt sidecar for .mp4");
    assert_eq!(sidecar.unwrap().extension().unwrap(), "srt");

    // The .srt file itself should be loadable via registry
    let registry = LoaderRegistry::new();
    let doc = registry.load(&srt).unwrap();
    assert!(doc.content.contains("Sidecar content"));
    assert!(doc.metadata.contains_key("subtitle_cues"));

    // VTT sidecar should also work
    let mp4_2 = dir.join("talk.mp4");
    let vtt = dir.join("talk.vtt");
    std::fs::write(&mp4_2, b"fake mp4").unwrap();
    std::fs::write(&vtt, "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nVTT sidecar content.\n")
        .unwrap();

    let sidecar2 = LoaderRegistry::find_sidecar(&mp4_2);
    assert!(sidecar2.is_some());
    let doc2 = registry.load(sidecar2.as_ref().unwrap()).unwrap();
    assert!(doc2.content.contains("VTT sidecar content"));

    let _ = std::fs::remove_dir_all(&dir);
}
