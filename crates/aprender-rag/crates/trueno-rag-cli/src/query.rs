//! Query execution: dense, sparse, hybrid retrieval with optional reranking.

#[cfg(feature = "embeddings")]
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use trueno_rag::{
    chunk::RecursiveChunker,
    embed::{Embedder, TfIdfEmbedder},
    fusion::FusionStrategy,
    pipeline::RagPipelineBuilder,
    rerank::LexicalReranker,
    Chunk, Document,
};

#[cfg(feature = "embeddings")]
use trueno_rag::{EmbeddingModelType, FastEmbedder};

use crate::{PersistedChunk, PersistedIndex};

pub(crate) fn run_demo(query: &str, top_k: usize) -> Result<()> {
    println!("=== Trueno-RAG Demo ===\n");

    // Sample documents for training TF-IDF
    let sample_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Deep learning uses neural networks with many layers to learn representations of data. It has achieved breakthrough results in image and speech recognition.",
        "Natural language processing enables computers to understand, interpret, and generate human language in a valuable way.",
        "Retrieval-Augmented Generation combines retrieval systems with generative models to produce more accurate and grounded responses.",
    ];

    // Train TF-IDF embedder
    let mut embedder = TfIdfEmbedder::new(128);
    let refs: Vec<&str> = sample_texts.iter().map(AsRef::as_ref).collect();
    embedder.fit(&refs);

    // Build pipeline
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(256, 32))
        .embedder(embedder)
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(2000)
        .build()?;

    // Create documents
    let docs = vec![
        Document::new(sample_texts[0]).with_title("Machine Learning Basics"),
        Document::new(sample_texts[1]).with_title("Deep Learning Overview"),
        Document::new(sample_texts[2]).with_title("NLP Introduction"),
        Document::new(sample_texts[3]).with_title("RAG Systems"),
    ];

    // Index
    let chunk_count = pipeline.index_documents(&docs)?;
    println!("Indexed {} documents ({} chunks)\n", docs.len(), chunk_count);

    // Query
    println!("Query: \"{}\"\n", query);

    let (results, context) = pipeline.query_with_context(query, top_k)?;

    println!("Results ({}):", results.len());
    println!("{}", "-".repeat(50));

    for (i, result) in results.iter().enumerate() {
        let title = result.chunk.metadata.title.as_deref().unwrap_or("Untitled");
        println!("{}. [Score: {:.3}] {}", i + 1, result.best_score(), title);
        let preview = &result.chunk.content[..80.min(result.chunk.content.len())];
        println!("   {}...\n", preview);
    }

    println!("{}", "=".repeat(50));
    println!("Assembled Context:\n");
    println!("{}", context.format_with_citations());

    println!("\nCitations:");
    println!("{}", context.citation_list());

    Ok(())
}

pub(crate) fn run_query(
    query: &str,
    index_path: &str,
    top_k: usize,
    format: &str,
    mode: &str,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
    rerank: &str,
    hyde: bool,
) -> Result<()> {
    if !["dense", "sparse", "hybrid"].contains(&mode) {
        anyhow::bail!("Unknown mode: {mode} (expected dense, sparse, hybrid)");
    }

    let index_path = Path::new(index_path);
    let index_file = index_path.join("index.json");

    if !index_file.exists() {
        anyhow::bail!("Index not found at: {}", index_file.display());
    }

    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    // Fetch more candidates if reranking (reranker re-orders, so we need a wider pool)
    let retrieval_k = if rerank == "none" { top_k } else { top_k * 3 };

    // HyDE: expand query into hypothetical document
    let effective_query = if hyde { expand_query_hyde(query)? } else { query.to_string() };

    let scores = match mode {
        "dense" => query_dense(&effective_query, &persisted, retrieval_k)?,
        "sparse" => query_sparse(&effective_query, &persisted, retrieval_k),
        "hybrid" => {
            query_hybrid(&effective_query, &persisted, retrieval_k, fusion, fusion_k, candidates)?
        }
        _ => unreachable!(),
    };

    // Apply reranking if requested
    let scores = apply_rerank(rerank, query, &scores, &persisted.chunks, top_k)?;

    format_query_results(query, &scores, &persisted.chunks, format)
}

/// Create the correct embedder for querying based on the index's embedder_type.
///
/// If the index was built with semantic embeddings (BGE/MiniLM), returns a
/// `FastEmbedder`; otherwise returns a `TfIdfEmbedder` fit on the corpus.
pub(crate) fn create_query_embedder(persisted: &PersistedIndex) -> Result<Box<dyn Embedder>> {
    if persisted.embedder_type == "semantic" {
        #[cfg(feature = "embeddings")]
        {
            let model_type = match persisted.model_name.as_deref() {
                Some(name) if name.contains("bge-base") => EmbeddingModelType::BgeBaseEnV15,
                Some(name) if name.contains("bge-small") => EmbeddingModelType::BgeSmallEnV15,
                _ => EmbeddingModelType::AllMiniLmL6V2,
            };
            // GH-16: Status message goes to stderr to avoid contaminating --format json output
            eprintln!(
                "Using semantic embedder: {} (dim={})",
                model_type.model_name(),
                model_type.dimension()
            );
            let emb =
                FastEmbedder::new(model_type).context("Failed to initialize semantic embedder")?;
            Ok(Box::new(emb))
        }
        #[cfg(not(feature = "embeddings"))]
        {
            anyhow::bail!(
                "This index uses semantic embeddings.\n\
                 Build with: cargo build --features embeddings"
            );
        }
    } else {
        let mut emb = TfIdfEmbedder::new(persisted.dimension);
        let refs: Vec<&str> = persisted.chunks.iter().map(|c| c.content.as_str()).collect();
        emb.fit(&refs);
        Ok(Box::new(emb))
    }
}

/// Expand a query using HyDE (Hypothetical Document Embeddings).
///
/// Generates a hypothetical document via Claude API that would answer the query,
/// then concatenates it with the original query for retrieval. The hypothetical
/// document uses the same vocabulary as corpus documents, improving embedding
/// similarity for vocabulary-mismatched queries.
#[cfg(feature = "eval")]
pub(crate) fn expand_query_hyde(query: &str) -> Result<String> {
    use trueno_rag::preprocess::{AnthropicHypotheticalGenerator, HypotheticalGenerator};

    let generator = AnthropicHypotheticalGenerator::from_env()
        .map_err(|e| anyhow::anyhow!("HyDE requires ANTHROPIC_API_KEY: {e}"))?;

    eprintln!("[HyDE] Generating hypothetical document for: {}", &query[..query.len().min(60)]);
    let hypothetical =
        generator.generate(query).map_err(|e| anyhow::anyhow!("HyDE generation failed: {e}"))?;
    eprintln!("[HyDE] Generated: {}...", &hypothetical[..hypothetical.len().min(80)]);

    // Concatenate original query + hypothetical for embedding.
    // The original query preserves keyword signal for BM25,
    // the hypothetical bridges vocabulary gap for dense retrieval.
    Ok(format!("{query} {hypothetical}"))
}

#[cfg(not(feature = "eval"))]
pub(crate) fn expand_query_hyde(_query: &str) -> Result<String> {
    anyhow::bail!("HyDE requires --features eval (for Anthropic API client)")
}

/// Apply reranking to scored results.
///
/// Takes `(chunk_index, score)` pairs and reranks using the specified strategy.
/// Returns `(chunk_index, rerank_score)` pairs truncated to `top_k`.
pub(crate) fn apply_rerank(
    rerank: &str,
    query: &str,
    scores: &[(usize, f32)],
    chunks: &[PersistedChunk],
    top_k: usize,
) -> Result<Vec<(usize, f32)>> {
    use trueno_rag::rerank::Reranker;
    use trueno_rag::retrieve::RetrievalResult;
    use trueno_rag::DocumentId;

    match rerank {
        "none" => Ok(scores.iter().take(top_k).copied().collect()),
        "lexical" => {
            // Convert (index, score) pairs into RetrievalResult for the Reranker trait
            let candidates: Vec<RetrievalResult> = scores
                .iter()
                .map(|(idx, score)| {
                    let pc = &chunks[*idx];
                    let mut chunk =
                        Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
                    // Store original index in metadata for round-tripping
                    chunk.metadata.custom.insert(
                        "_idx".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(*idx)),
                    );
                    RetrievalResult {
                        chunk,
                        dense_score: Some(*score),
                        sparse_score: None,
                        fused_score: None,
                        rerank_score: None,
                    }
                })
                .collect();

            let reranker = LexicalReranker::new();
            let reranked = reranker.rerank(query, &candidates, top_k)?;

            Ok(reranked
                .into_iter()
                .map(|rr| {
                    let idx =
                        rr.chunk.metadata.custom.get("_idx").and_then(|v| v.as_u64()).unwrap_or(0)
                            as usize;
                    let score = rr.rerank_score.unwrap_or(rr.best_score());
                    (idx, score)
                })
                .collect())
        }
        _ => anyhow::bail!("Unknown rerank strategy: {rerank} (expected none, lexical)"),
    }
}

/// Rerank `RetrievedChunk` results (used in eval retrieve path).
#[cfg(feature = "eval")]
pub(crate) fn rerank_retrieved_chunks(
    rerank: &str,
    query: &str,
    mut results: Vec<trueno_rag::eval::types::RetrievedChunk>,
    top_k: usize,
) -> Result<Vec<trueno_rag::eval::types::RetrievedChunk>> {
    use trueno_rag::rerank::Reranker;
    use trueno_rag::retrieve::RetrievalResult;
    use trueno_rag::DocumentId;

    match rerank {
        "none" => {
            results.truncate(top_k);
            Ok(results)
        }
        "lexical" => {
            let candidates: Vec<RetrievalResult> = results
                .iter()
                .enumerate()
                .map(|(i, rc)| {
                    let mut chunk =
                        Chunk::new(DocumentId::new(), rc.content.clone(), 0, rc.content.len());
                    chunk.metadata.custom.insert(
                        "_idx".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(i)),
                    );
                    RetrievalResult {
                        chunk,
                        dense_score: Some(rc.score),
                        sparse_score: None,
                        fused_score: None,
                        rerank_score: None,
                    }
                })
                .collect();

            let reranker = LexicalReranker::new();
            let reranked = reranker.rerank(query, &candidates, top_k)?;

            Ok(reranked
                .into_iter()
                .map(|rr| {
                    let idx =
                        rr.chunk.metadata.custom.get("_idx").and_then(|v| v.as_u64()).unwrap_or(0)
                            as usize;
                    let mut rc = results[idx].clone();
                    rc.score = rr.rerank_score.unwrap_or(rr.best_score());
                    rc
                })
                .collect())
        }
        _ => anyhow::bail!("Unknown rerank strategy: {rerank} (expected none, lexical)"),
    }
}

/// Dense retrieval: TF-IDF or semantic cosine similarity.
pub(crate) fn query_dense(
    query: &str,
    persisted: &PersistedIndex,
    top_k: usize,
) -> Result<Vec<(usize, f32)>> {
    let embedder = create_query_embedder(persisted)?;
    let query_embedding = embedder.embed(query)?;

    let mut scores: Vec<(usize, f32)> = persisted
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    Ok(scores)
}

/// Sparse retrieval: BM25 keyword matching.
pub(crate) fn query_sparse(
    query: &str,
    persisted: &PersistedIndex,
    top_k: usize,
) -> Vec<(usize, f32)> {
    use trueno_rag::index::SparseIndex;
    use trueno_rag::{BM25Index, DocumentId};

    let mut bm25 = BM25Index::new();
    let mut chunk_map: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
    for (i, pc) in persisted.chunks.iter().enumerate() {
        let chunk = Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
        chunk_map.insert(chunk.id, i);
        bm25.add(&chunk);
    }

    let bm25_results = bm25.search(query, top_k);
    bm25_results.iter().map(|(chunk_id, score)| (chunk_map[chunk_id], *score)).collect()
}

/// Hybrid retrieval: BM25 + dense (TF-IDF or semantic) with fusion.
pub(crate) fn query_hybrid(
    query: &str,
    persisted: &PersistedIndex,
    top_k: usize,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
) -> Result<Vec<(usize, f32)>> {
    use trueno_rag::index::VectorStoreConfig;
    use trueno_rag::retrieve::HybridRetrieverConfig;
    use trueno_rag::{BM25Index, DocumentId, HybridRetriever, VectorStore};

    let fusion_strategy = parse_fusion_strategy(fusion, fusion_k)?;

    let embedder = create_query_embedder(persisted)?;
    let dim = embedder.dimension();

    let dense_store = VectorStore::new(VectorStoreConfig { dimension: dim, ..Default::default() });
    let bm25 = BM25Index::new();

    let config = HybridRetrieverConfig {
        candidates_per_source: candidates,
        fusion: fusion_strategy,
        use_dense: true,
        use_sparse: true,
    };

    let mut retriever = HybridRetriever::new(dense_store, bm25, embedder).with_config(config);

    let mut chunk_meta: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
    for (i, pc) in persisted.chunks.iter().enumerate() {
        let mut chunk = Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
        chunk.metadata.title = pc.title.clone();
        chunk.embedding = Some(persisted.embeddings[i].clone());
        chunk_meta.insert(chunk.id, i);
        retriever.index(chunk)?;
    }

    let results = retriever.retrieve(query, top_k)?;
    Ok(results
        .iter()
        .map(|rr| {
            let score = rr.fused_score.unwrap_or(rr.best_score());
            let idx = chunk_meta.get(&rr.chunk.id).copied().unwrap_or(0);
            (idx, score)
        })
        .collect())
}

/// Format and print query results in text or JSON format.
pub(crate) fn format_query_results(
    query: &str,
    scores: &[(usize, f32)],
    chunks: &[PersistedChunk],
    format: &str,
) -> Result<()> {
    if format == "json" {
        let results: Vec<serde_json::Value> = scores
            .iter()
            .enumerate()
            .map(|(rank, (i, score))| {
                let chunk = &chunks[*i];
                let mut result = serde_json::json!({
                    "rank": rank + 1,
                    "score": score,
                    "content": chunk.content,
                    "title": chunk.title,
                    "source": chunk.source,
                });
                if let Some(start) = chunk.start_secs {
                    result["start_secs"] = serde_json::json!(start);
                }
                if let Some(end) = chunk.end_secs {
                    result["end_secs"] = serde_json::json!(end);
                }
                result
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("Query: \"{query}\"\n");
        println!("Results ({}):", scores.len());
        println!("{}", "-".repeat(50));

        for (rank, (i, score)) in scores.iter().enumerate() {
            let chunk = &chunks[*i];
            let title = chunk.title.as_deref().unwrap_or("Untitled");
            let time_info = match (chunk.start_secs, chunk.end_secs) {
                (Some(start), Some(end)) => format!(
                    " [{}–{}]",
                    trueno_rag::media::format_display_time(start),
                    trueno_rag::media::format_display_time(end),
                ),
                _ => String::new(),
            };
            println!("{}. [Score: {:.3}] {}{}", rank + 1, score, title, time_info);
            let preview = &chunk.content[..80.min(chunk.content.len())];
            println!("   {preview}...\n");
        }
    }
    Ok(())
}

/// Compute cosine similarity between two vectors
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub(crate) fn parse_fusion_strategy(fusion: &str, fusion_k: Option<f32>) -> Result<FusionStrategy> {
    match fusion {
        "rrf" => Ok(FusionStrategy::RRF { k: fusion_k.unwrap_or(60.0) }),
        "linear" => Ok(FusionStrategy::Linear { dense_weight: fusion_k.unwrap_or(0.5) }),
        "dbsf" => Ok(FusionStrategy::DBSF),
        other => anyhow::bail!("Unknown fusion strategy: {other} (expected rrf, linear, dbsf)"),
    }
}
