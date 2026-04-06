//! Evaluation framework: ground-truth generation, retrieval evaluation,
//! relevance judging, metrics, comparison, and regression gates.

#![cfg(feature = "eval")]

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use trueno_rag::{embed::Embedder, Chunk};

use crate::query::{
    cosine_similarity, create_query_embedder, expand_query_hyde, parse_fusion_strategy,
    rerank_retrieved_chunks,
};
use crate::{EvalAction, PersistedIndex};

pub(crate) fn run_eval(action: EvalAction) -> Result<()> {
    match action {
        EvalAction::Sample { index, output, sample_size, seed } => {
            run_eval_sample(&index, &output, sample_size, seed)
        }

        EvalAction::Generate { index, output, sample_size, seed, model, dry_run } => {
            run_eval_generate(&index, &output, sample_size, seed, &model, dry_run)
        }

        EvalAction::Retrieve {
            index,
            ground_truth,
            output,
            top_k,
            mode,
            fusion,
            fusion_k,
            candidates,
            rerank,
            hyde,
        } => run_eval_retrieve(
            &index,
            &ground_truth,
            &output,
            top_k,
            &mode,
            &fusion,
            fusion_k,
            candidates,
            &rerank,
            hyde,
        ),

        EvalAction::Judge { retrieval_results, ground_truth, output, cache, top_k, model } => {
            // GH-16: Warn that --ground-truth is accepted but not yet used by the judge
            if !ground_truth.is_empty() {
                eprintln!(
                    "Warning: --ground-truth '{}' is not yet used by eval judge (LLM judges relevance without reference answers). Flag accepted for future use.",
                    ground_truth
                );
            }
            let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
            rt.block_on(run_eval_judge(&retrieval_results, &output, &cache, top_k, &model))
        }

        EvalAction::Metrics { retrieval_results, judgments, output } => {
            run_eval_metrics(&retrieval_results, &judgments, &output)
        }

        EvalAction::Compare { baseline, candidate } => run_eval_compare(&baseline, &candidate),

        EvalAction::Gate { results, min_mrr, min_hit5 } => {
            run_eval_gate(&results, min_mrr, min_hit5)
        }
    }
}

fn run_eval_sample(
    index_path: &str,
    output_path: &str,
    sample_size: usize,
    seed: u64,
) -> Result<()> {
    use trueno_rag::eval::generate::IndexChunk;
    use trueno_rag::eval::{AnthropicClient, GroundTruthGenerator};

    let index_file = Path::new(index_path).join("index.json");
    if !index_file.exists() {
        anyhow::bail!("Index not found: {}", index_file.display());
    }

    println!("Loading index from {}...", index_file.display());
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    let chunks: Vec<IndexChunk> = persisted
        .chunks
        .iter()
        .map(|c| IndexChunk {
            content: c.content.clone(),
            source: c.source.clone().unwrap_or_default(),
            title: c.title.clone(),
            start_secs: c.start_secs,
            end_secs: c.end_secs,
        })
        .collect();

    println!("Loaded {} chunks", chunks.len());

    // Use a dummy client -- sample_chunks doesn't call the API
    let client = AnthropicClient::new("sample-only");
    let gen = GroundTruthGenerator::new(client, "none", sample_size, seed);
    let sampled = gen.sample_chunks(&chunks);

    // Write sampled chunks as JSONL
    let mut file = std::io::BufWriter::new(fs::File::create(output_path)?);
    use std::io::Write;

    #[derive(serde::Serialize)]
    struct SampledChunkOutput {
        content: String,
        source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_secs: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end_secs: Option<f64>,
        course: String,
        domain: String,
    }

    for s in &sampled {
        let entry = SampledChunkOutput {
            content: s.content.clone(),
            source: s.source.clone(),
            start_secs: s.start_secs,
            end_secs: s.end_secs,
            course: s.course.clone(),
            domain: s.domain.clone(),
        };
        serde_json::to_writer(&mut file, &entry)?;
        writeln!(file)?;
    }

    println!("\nSampled {} chunks saved to: {output_path}", sampled.len());
    Ok(())
}

fn run_eval_generate(
    index_path: &str,
    output_path: &str,
    sample_size: usize,
    seed: u64,
    model: &str,
    dry_run: bool,
) -> Result<()> {
    use trueno_rag::eval::{generate::IndexChunk, AnthropicClient, GroundTruthGenerator};

    let index_file = Path::new(index_path).join("index.json");
    if !index_file.exists() {
        anyhow::bail!("Index not found: {}", index_file.display());
    }

    println!("Loading index from {}...", index_file.display());
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    let chunks: Vec<IndexChunk> = persisted
        .chunks
        .iter()
        .map(|c| IndexChunk {
            content: c.content.clone(),
            source: c.source.clone().unwrap_or_default(),
            title: c.title.clone(),
            start_secs: c.start_secs,
            end_secs: c.end_secs,
        })
        .collect();

    println!("Loaded {} chunks", chunks.len());

    if dry_run {
        let client = AnthropicClient::new("dry-run");
        let gen = GroundTruthGenerator::new(client, model, sample_size, seed);
        let sampled = gen.sample_chunks(&chunks);
        println!("\nDry run: would generate {} questions", sampled.len());
        for s in sampled.iter().take(10) {
            println!("  [{}] {}: {}...", s.domain, s.course, &s.content[..s.content.len().min(80)]);
        }
        return Ok(());
    }

    let client = AnthropicClient::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let gen = GroundTruthGenerator::new(client, model, sample_size, seed);

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let results = rt.block_on(gen.generate(&chunks)).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Write JSONL output
    let mut file = std::io::BufWriter::new(fs::File::create(output_path)?);
    use std::io::Write;
    for entry in &results {
        serde_json::to_writer(&mut file, entry)?;
        writeln!(file)?;
    }

    println!("\nGround truth saved to: {output_path} ({} entries)", results.len());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_eval_retrieve(
    index_path: &str,
    ground_truth_path: &str,
    output_path: &str,
    top_k: usize,
    mode: &str,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
    rerank: &str,
    hyde: bool,
) -> Result<()> {
    use trueno_rag::eval::types::GroundTruthEntry;
    use trueno_rag::DocumentId;

    // Validate mode
    if !["dense", "sparse", "hybrid"].contains(&mode) {
        anyhow::bail!("Unknown mode: {mode} (expected dense, sparse, hybrid)");
    }

    let index_file = Path::new(index_path).join("index.json");
    if !index_file.exists() {
        anyhow::bail!("Index not found: {}", index_file.display());
    }

    // Load ground truth
    let gt_text = fs::read_to_string(ground_truth_path)
        .with_context(|| format!("Failed to read {ground_truth_path}"))?;
    let queries: Vec<GroundTruthEntry> = gt_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse ground truth JSONL")?;

    println!("Loaded {} queries from {}", queries.len(), ground_truth_path);
    let hyde_label = if hyde { " | HyDE: on" } else { "" };
    println!("Mode: {mode} | Fusion: {fusion} | Candidates: {candidates} | Top-k: {top_k} | Rerank: {rerank}{hyde_label}");

    // Fetch more candidates when reranking to give the reranker a wider pool
    let retrieval_k = if rerank == "none" { top_k } else { top_k * 3 };

    // Load index
    println!("Loading index from {}...", index_file.display());
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;
    println!("Index: {} chunks, dim={}", persisted.chunks.len(), persisted.dimension);

    // Build embedder (auto-detects semantic vs TF-IDF from index metadata)
    let embedder = create_query_embedder(&persisted)?;

    // Convert PersistedChunks to Chunks with embeddings for hybrid/sparse modes
    let build_chunks = |with_embeddings: bool| -> Vec<Chunk> {
        persisted
            .chunks
            .iter()
            .enumerate()
            .map(|(i, pc)| {
                let mut chunk =
                    Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
                chunk.metadata.title = pc.title.clone();
                chunk.metadata.custom.insert(
                    "source".to_string(),
                    serde_json::Value::String(pc.source.clone().unwrap_or_default()),
                );
                if with_embeddings {
                    chunk.embedding = Some(persisted.embeddings[i].clone());
                }
                chunk
            })
            .collect()
    };

    // HyDE: pre-expand all queries if enabled (batches API calls before retrieval loop)
    let expanded_queries: Option<Vec<String>> =
        if hyde { Some(expand_all_queries_hyde(&queries)?) } else { None };

    // Helper: get effective query (HyDE-expanded or original)
    let effective_query = |i: usize, original: &str| -> String {
        expanded_queries.as_ref().map_or_else(|| original.to_string(), |eq| eq[i].clone())
    };

    // Run queries per mode, writing results to output file
    let mut output_file = std::io::BufWriter::new(fs::File::create(output_path)?);

    match mode {
        "dense" => eval_retrieve_dense(
            &queries,
            &persisted,
            &*embedder,
            &effective_query,
            retrieval_k,
            top_k,
            rerank,
            &mut output_file,
        )?,
        "sparse" => eval_retrieve_sparse(
            &queries,
            &persisted,
            &build_chunks,
            &effective_query,
            retrieval_k,
            top_k,
            rerank,
            &mut output_file,
        )?,
        "hybrid" => eval_retrieve_hybrid(
            &queries,
            &persisted,
            embedder,
            &build_chunks,
            &effective_query,
            retrieval_k,
            top_k,
            rerank,
            fusion,
            fusion_k,
            candidates,
            &mut output_file,
        )?,
        _ => unreachable!(),
    }

    println!("\nRetrieval results saved to: {output_path}");
    Ok(())
}

/// Expand all ground-truth queries via HyDE (Hypothetical Document Embeddings).
fn expand_all_queries_hyde(
    queries: &[trueno_rag::eval::types::GroundTruthEntry],
) -> Result<Vec<String>> {
    println!("HyDE enabled — expanding {} queries via Claude API...", queries.len());
    let mut expanded = Vec::with_capacity(queries.len());
    for (i, entry) in queries.iter().enumerate() {
        eprint!("[HyDE {}/{}] ", i + 1, queries.len());
        expanded.push(expand_query_hyde(&entry.query)?);
    }
    println!("HyDE expansion complete.");
    Ok(expanded)
}

/// Dense eval retrieval: cosine similarity over embeddings.
fn eval_retrieve_dense(
    queries: &[trueno_rag::eval::types::GroundTruthEntry],
    persisted: &PersistedIndex,
    embedder: &dyn Embedder,
    effective_query: &dyn Fn(usize, &str) -> String,
    retrieval_k: usize,
    top_k: usize,
    rerank: &str,
    output: &mut impl std::io::Write,
) -> Result<()> {
    use trueno_rag::eval::types::{RetrievalResultEntry, RetrievedChunk};

    for (i, entry) in queries.iter().enumerate() {
        print!("[{}/{}] {}...", i + 1, queries.len(), &entry.query[..entry.query.len().min(60)]);

        let eq = effective_query(i, &entry.query);
        let start = std::time::Instant::now();
        let query_embedding = embedder.embed(&eq)?;

        let mut scores: Vec<(usize, f32)> = persisted
            .embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| (idx, cosine_similarity(&query_embedding, emb)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(retrieval_k);
        let latency = start.elapsed().as_secs_f64();

        let results: Vec<RetrievedChunk> = scores
            .iter()
            .map(|(idx, score)| {
                let chunk = &persisted.chunks[*idx];
                RetrievedChunk {
                    content: chunk.content.clone(),
                    source: chunk.source.clone(),
                    score: *score,
                    title: chunk.title.clone(),
                    start_secs: chunk.start_secs,
                    end_secs: chunk.end_secs,
                }
            })
            .collect();
        let results = rerank_retrieved_chunks(rerank, &entry.query, results, top_k)?;

        println!(" {} results ({:.2}s)", results.len(), latency);
        serde_json::to_writer(
            &mut *output,
            &RetrievalResultEntry {
                query: entry.query.clone(),
                domain: entry.domain.clone(),
                course: entry.course.clone(),
                results,
                latency_s: latency,
            },
        )?;
        writeln!(output)?;
    }
    Ok(())
}

/// Sparse eval retrieval: BM25 keyword matching.
fn eval_retrieve_sparse(
    queries: &[trueno_rag::eval::types::GroundTruthEntry],
    persisted: &PersistedIndex,
    build_chunks: &dyn Fn(bool) -> Vec<Chunk>,
    effective_query: &dyn Fn(usize, &str) -> String,
    retrieval_k: usize,
    top_k: usize,
    rerank: &str,
    output: &mut impl std::io::Write,
) -> Result<()> {
    use trueno_rag::eval::types::{RetrievalResultEntry, RetrievedChunk};
    use trueno_rag::index::SparseIndex;
    use trueno_rag::BM25Index;

    println!("Building BM25 index from {} chunks...", persisted.chunks.len());
    let start_build = std::time::Instant::now();
    let mut bm25 = BM25Index::new();
    let chunks = build_chunks(false);
    let mut chunk_map: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
    for (i, chunk) in chunks.iter().enumerate() {
        chunk_map.insert(chunk.id, i);
        bm25.add(chunk);
    }
    println!("BM25 index built in {:.2}s", start_build.elapsed().as_secs_f64());

    for (i, entry) in queries.iter().enumerate() {
        print!("[{}/{}] {}...", i + 1, queries.len(), &entry.query[..entry.query.len().min(60)]);

        let eq = effective_query(i, &entry.query);
        let start = std::time::Instant::now();
        let bm25_results = bm25.search(&eq, retrieval_k);
        let latency = start.elapsed().as_secs_f64();

        let results: Vec<RetrievedChunk> = bm25_results
            .iter()
            .map(|(chunk_id, score)| {
                let idx = chunk_map[chunk_id];
                let pc = &persisted.chunks[idx];
                RetrievedChunk {
                    content: pc.content.clone(),
                    source: pc.source.clone(),
                    score: *score,
                    title: pc.title.clone(),
                    start_secs: pc.start_secs,
                    end_secs: pc.end_secs,
                }
            })
            .collect();
        let results = rerank_retrieved_chunks(rerank, &entry.query, results, top_k)?;

        println!(" {} results ({:.2}s)", results.len(), latency);
        serde_json::to_writer(
            &mut *output,
            &RetrievalResultEntry {
                query: entry.query.clone(),
                domain: entry.domain.clone(),
                course: entry.course.clone(),
                results,
                latency_s: latency,
            },
        )?;
        writeln!(output)?;
    }
    Ok(())
}

/// Hybrid eval retrieval: BM25 + dense with RRF fusion.
#[allow(clippy::too_many_arguments)]
fn eval_retrieve_hybrid(
    queries: &[trueno_rag::eval::types::GroundTruthEntry],
    persisted: &PersistedIndex,
    embedder: Box<dyn Embedder>,
    build_chunks: &dyn Fn(bool) -> Vec<Chunk>,
    effective_query: &dyn Fn(usize, &str) -> String,
    retrieval_k: usize,
    top_k: usize,
    rerank: &str,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
    output: &mut impl std::io::Write,
) -> Result<()> {
    use trueno_rag::eval::types::{RetrievalResultEntry, RetrievedChunk};
    use trueno_rag::index::VectorStoreConfig;
    use trueno_rag::retrieve::HybridRetrieverConfig;
    use trueno_rag::{BM25Index, HybridRetriever, VectorStore};

    let fusion_strategy = parse_fusion_strategy(fusion, fusion_k)?;
    let dim = embedder.dimension();
    println!("Building hybrid retriever (BM25 + dense dim={dim}, fusion={fusion})...");
    let start_build = std::time::Instant::now();

    let dense_store = VectorStore::new(VectorStoreConfig { dimension: dim, ..Default::default() });
    let bm25 = BM25Index::new();
    let config = HybridRetrieverConfig {
        candidates_per_source: candidates,
        fusion: fusion_strategy,
        use_dense: true,
        use_sparse: true,
    };
    let mut retriever = HybridRetriever::new(dense_store, bm25, embedder).with_config(config);

    let chunks = build_chunks(true);
    let n_chunks = chunks.len();
    let mut chunk_meta: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
    for (i, chunk) in chunks.into_iter().enumerate() {
        chunk_meta.insert(chunk.id, i);
        retriever.index(chunk)?;
    }
    println!(
        "Hybrid retriever built: {} chunks in {:.2}s",
        n_chunks,
        start_build.elapsed().as_secs_f64()
    );

    for (i, entry) in queries.iter().enumerate() {
        print!("[{}/{}] {}...", i + 1, queries.len(), &entry.query[..entry.query.len().min(60)]);

        let eq = effective_query(i, &entry.query);
        let start = std::time::Instant::now();
        let retrieval_results = retriever.retrieve(&eq, retrieval_k)?;
        let latency = start.elapsed().as_secs_f64();

        let results: Vec<RetrievedChunk> = retrieval_results
            .iter()
            .map(|rr| {
                let score = rr.fused_score.unwrap_or(rr.best_score());
                if let Some(&idx) = chunk_meta.get(&rr.chunk.id) {
                    let pc = &persisted.chunks[idx];
                    RetrievedChunk {
                        content: pc.content.clone(),
                        source: pc.source.clone(),
                        score,
                        title: pc.title.clone(),
                        start_secs: pc.start_secs,
                        end_secs: pc.end_secs,
                    }
                } else {
                    RetrievedChunk {
                        content: rr.chunk.content.clone(),
                        source: None,
                        score,
                        title: None,
                        start_secs: None,
                        end_secs: None,
                    }
                }
            })
            .collect();
        let results = rerank_retrieved_chunks(rerank, &entry.query, results, top_k)?;

        println!(" {} results ({:.2}s)", results.len(), latency);
        serde_json::to_writer(
            &mut *output,
            &RetrievalResultEntry {
                query: entry.query.clone(),
                domain: entry.domain.clone(),
                course: entry.course.clone(),
                results,
                latency_s: latency,
            },
        )?;
        writeln!(output)?;
    }
    Ok(())
}

async fn run_eval_judge(
    retrieval_results_path: &str,
    output_path: &str,
    cache_path: &str,
    top_k: usize,
    model: &str,
) -> Result<()> {
    use trueno_rag::eval::{
        types::{JudgeCache, RetrievalResultEntry},
        AnthropicClient, RelevanceJudge,
    };

    // Load retrieval results
    let text = fs::read_to_string(retrieval_results_path)
        .with_context(|| format!("Failed to read {retrieval_results_path}"))?;
    let results: Vec<RetrievalResultEntry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse retrieval results JSONL")?;

    eprintln!("Loaded {} retrieval results", results.len());

    // Load cache
    let cache = JudgeCache::load(Path::new(cache_path));
    eprintln!("Cache: {} entries loaded from {}", cache.entries.len(), cache_path);

    let client = AnthropicClient::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut judge = RelevanceJudge::new(client, model, cache);

    let eval_output = judge.evaluate(&results, top_k).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    // Save cache
    judge.cache().save(Path::new(cache_path)).context("Failed to save judge cache")?;
    eprintln!("Cache saved: {} entries to {}", judge.cache().entries.len(), cache_path);

    // Save results
    let json = serde_json::to_string_pretty(&eval_output)?;
    fs::write(output_path, json)?;
    eprintln!("Results saved to: {output_path}");

    Ok(())
}

fn run_eval_metrics(
    retrieval_results_path: &str,
    judgments_path: &str,
    output_path: &str,
) -> Result<()> {
    use trueno_rag::eval::metrics::{compute_metrics_from_judgments, format_metrics_summary};
    use trueno_rag::eval::types::{JudgmentEntry, RetrievalResultEntry};

    // Load retrieval results
    let text = fs::read_to_string(retrieval_results_path)
        .with_context(|| format!("Failed to read {retrieval_results_path}"))?;
    let results: Vec<RetrievalResultEntry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse retrieval results JSONL")?;

    // Load judgments
    let jtext = fs::read_to_string(judgments_path)
        .with_context(|| format!("Failed to read {judgments_path}"))?;
    let judgments: Vec<JudgmentEntry> = jtext
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse judgments JSONL")?;

    println!("Loaded {} retrieval results, {} judgments", results.len(), judgments.len());

    let eval_output = compute_metrics_from_judgments(&results, &judgments);

    println!("\n{}", format_metrics_summary(&eval_output.aggregate, &eval_output.by_domain));

    let json = serde_json::to_string_pretty(&eval_output)?;
    fs::write(output_path, json)?;
    println!("Results saved to: {output_path}");

    Ok(())
}

fn run_eval_compare(baseline_path: &str, candidate_path: &str) -> Result<()> {
    use trueno_rag::eval::{judge::compare_results, types::EvalOutput};

    let baseline: EvalOutput = serde_json::from_str(
        &fs::read_to_string(baseline_path)
            .with_context(|| format!("Failed to read {baseline_path}"))?,
    )?;
    let candidate: EvalOutput = serde_json::from_str(
        &fs::read_to_string(candidate_path)
            .with_context(|| format!("Failed to read {candidate_path}"))?,
    )?;

    println!("{}", compare_results(&baseline, &candidate));
    Ok(())
}

fn run_eval_gate(results_path: &str, min_mrr: f64, min_hit5: f64) -> Result<()> {
    use trueno_rag::eval::{judge::check_gate, types::EvalOutput};

    let output: EvalOutput = serde_json::from_str(
        &fs::read_to_string(results_path)
            .with_context(|| format!("Failed to read {results_path}"))?,
    )?;

    match check_gate(&output, min_mrr, min_hit5) {
        Ok(()) => {
            println!(
                "Regression gate PASSED (MRR={:.4}, Hit@5={:.4})",
                output.aggregate.mrr, output.aggregate.hit_rate_5
            );
            Ok(())
        }
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    }
}
