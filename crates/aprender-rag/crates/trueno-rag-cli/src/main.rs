//! Trueno-RAG CLI
//!
//! Command-line interface for the Trueno-RAG pipeline.
//!
//! ## Features
//!
//! - `embeddings` - Enable production semantic embeddings via fastembed (ONNX Runtime)
//!
//! ## Usage
//!
//! ```bash
//! # Build with semantic embeddings support
//! cargo build --release --features embeddings
//!
//! # Index documents with semantic embeddings
//! trueno-rag index --path docs/ --output index/ --embedder semantic
//!
//! # Index with recursive directory walking and subtitle support
//! trueno-rag index --path /data/ --output index/ --recursive
//!
//! # Index with timestamp-aware chunking for media transcripts
//! trueno-rag index --path /data/ --output index/ --recursive --chunk-strategy timestamp
//! ```

mod discover;
mod eval_cmd;
mod extract;
mod incremental;
mod ingest;
mod query;
mod transcribe;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use trueno_rag::loader::LoaderRegistry;

/// Embedder type selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum EmbedderType {
    /// TF-IDF statistical embeddings (default, no downloads)
    #[default]
    Tfidf,
    /// Semantic embeddings via fastembed (requires `embeddings` feature)
    Semantic,
}

/// Model selection for semantic embeddings
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum SemanticModel {
    /// all-MiniLM-L6-v2: Fast, good quality (384 dims)
    #[default]
    MiniLm,
    /// BGE-small-en-v1.5: Balanced performance (384 dims)
    BgeSmall,
    /// BGE-base-en-v1.5: Higher quality (768 dims)
    BgeBase,
}

/// Chunking strategy selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum ChunkStrategy {
    /// Auto-select: TimestampChunker for media, RecursiveChunker for text
    #[default]
    Auto,
    /// Recursive character splitting (works for all content)
    Recursive,
    /// Timestamp-aware chunking (best for subtitle/transcript content)
    Timestamp,
}

/// Compute backend for transcription
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum BackendType {
    /// CPU with SIMD acceleration
    #[default]
    Cpu,
    /// GPU via wgpu (cross-platform)
    Gpu,
    /// NVIDIA CUDA (Linux/Windows)
    Cuda,
}

#[derive(Parser)]
#[command(name = "trueno-rag")]
#[command(author = "Pragmatic AI Labs")]
#[command(version)]
#[command(about = "Pure-Rust RAG pipeline CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a demo RAG query
    Demo {
        /// Query string
        #[arg(short, long, default_value = "What is machine learning?")]
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "3")]
        top_k: usize,
    },

    /// Index documents from a file or directory
    Index {
        /// Path to document(s)
        #[arg(short, long)]
        path: String,

        /// Output directory for index
        #[arg(short, long)]
        output: String,

        /// Chunk size in characters (for recursive chunker)
        #[arg(long, default_value = "512")]
        chunk_size: usize,

        /// Chunk overlap in characters (for recursive chunker)
        #[arg(long, default_value = "64")]
        chunk_overlap: usize,

        /// Embedding dimension (only for tfidf embedder)
        #[arg(long, default_value = "256")]
        dimension: usize,

        /// Embedder type (tfidf or semantic)
        #[arg(short, long, value_enum, default_value = "tfidf")]
        embedder: EmbedderType,

        /// Model for semantic embeddings (mini-lm, bge-small, bge-base)
        #[arg(short, long, value_enum, default_value = "mini-lm")]
        model: SemanticModel,

        /// Recursively scan subdirectories
        #[arg(short, long, default_value = "false")]
        recursive: bool,

        /// Chunking strategy (auto, recursive, timestamp)
        #[arg(long, value_enum, default_value = "auto")]
        chunk_strategy: ChunkStrategy,

        /// Number of parallel loading jobs
        #[arg(short, long, default_value = "1")]
        jobs: usize,

        /// Write a JSON manifest of indexed files and chunks
        #[arg(long, default_value = "false")]
        manifest: bool,

        /// Glob patterns to exclude files/directories (repeatable)
        #[arg(long)]
        exclude: Vec<String>,

        /// Deduplicate chunks with identical content (keeps first occurrence)
        #[arg(long, default_value = "false")]
        dedup: bool,

        /// Also export a SQLite+FTS5 index (requires sqlite feature)
        #[arg(long, default_value = "false")]
        sqlite: bool,

        /// Incremental mode: only re-index changed files (requires --sqlite)
        #[arg(long, default_value = "false")]
        incremental: bool,
    },

    /// Query the RAG pipeline
    Query {
        /// Query string
        query: String,

        /// Path to index directory
        #[arg(short, long)]
        index: String,

        /// Number of results
        #[arg(short, long, default_value = "5")]
        top_k: usize,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Retrieval mode: dense, sparse (BM25), hybrid (BM25 + dense RRF)
        #[arg(long, default_value = "hybrid")]
        mode: String,

        /// Fusion strategy (hybrid mode only): rrf, linear, dbsf
        #[arg(long, default_value = "rrf")]
        fusion: String,

        /// Fusion parameter: RRF k value or Linear dense_weight
        #[arg(long)]
        fusion_k: Option<f32>,

        /// Candidates per source for hybrid retrieval
        #[arg(long, default_value = "50")]
        candidates: usize,

        /// Reranking strategy: none, lexical
        #[arg(long, default_value = "none")]
        rerank: String,

        /// Enable HyDE (Hypothetical Document Embeddings) query expansion.
        /// Generates a hypothetical answer via Claude API and uses it for retrieval.
        /// Requires ANTHROPIC_API_KEY environment variable and --features eval.
        #[arg(long, default_value = "false")]
        hyde: bool,
    },

    /// Batch transcribe media files to .srt sidecars
    Transcribe {
        /// Path to directory containing media files
        #[arg(short, long)]
        path: String,

        /// Recursively scan subdirectories
        #[arg(short, long, default_value = "false")]
        recursive: bool,

        /// Skip files that already have .srt/.vtt sidecars
        #[arg(long, default_value = "true")]
        skip_existing: bool,

        /// Number of parallel transcription jobs (CPU mode)
        #[arg(short, long, default_value = "1")]
        jobs: usize,

        /// Path to Whisper .apr model file (e.g. base.apr, large-v3-turbo.apr)
        #[arg(short, long)]
        model: Option<String>,

        /// Compute backend (cpu, gpu, cuda)
        #[arg(short, long, value_enum, default_value = "cpu")]
        backend: BackendType,

        /// Only report what would be transcribed (dry run)
        #[arg(long, default_value = "false")]
        dry_run: bool,

        /// Initial prompt to condition decoder vocabulary (e.g. "AWS, Kubernetes, YAML")
        #[arg(long)]
        prompt: Option<String>,

        /// Path to file with hotwords (one per line) to boost during decoding
        #[arg(long)]
        hotwords: Option<String>,

        /// Glob patterns to exclude files/directories (repeatable)
        #[arg(long)]
        exclude: Vec<String>,
    },

    /// Extract keyframes from video files at scene changes (requires ffmpeg)
    ExtractFrames {
        /// Path to directory containing video files
        #[arg(short, long)]
        path: String,

        /// Recursively scan subdirectories
        #[arg(short, long, default_value = "false")]
        recursive: bool,

        /// Scene change detection threshold (0.0-1.0, lower = more frames)
        #[arg(long, default_value = "0.3")]
        threshold: f64,

        /// Minimum seconds between extracted frames
        #[arg(long, default_value = "5.0")]
        min_interval: f64,

        /// Number of parallel extraction jobs
        #[arg(short, long, default_value = "4")]
        jobs: usize,

        /// Skip videos that already have frames/ directory
        #[arg(long, default_value = "true")]
        skip_existing: bool,

        /// Only report what would be extracted (dry run)
        #[arg(long, default_value = "false")]
        dry_run: bool,

        /// Glob patterns to exclude files/directories (repeatable)
        #[arg(long)]
        exclude: Vec<String>,
    },

    /// Show pipeline info
    Info,

    /// Evaluation framework: generate ground truth, run retrieval, judge relevance
    #[cfg(feature = "eval")]
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },
}

/// Eval sub-subcommands
#[cfg(feature = "eval")]
#[derive(Subcommand)]
enum EvalAction {
    /// Sample chunks from index for ground truth generation (no API needed)
    Sample {
        /// Path to index directory (containing index.json)
        #[arg(short, long)]
        index: String,

        /// Output path for sampled-chunks JSONL
        #[arg(short, long)]
        output: String,

        /// Number of chunks to sample
        #[arg(long, default_value = "250")]
        sample_size: usize,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Generate synthetic ground truth from an index via Claude API (requires ANTHROPIC_API_KEY)
    Generate {
        /// Path to index directory (containing index.json)
        #[arg(short, long)]
        index: String,

        /// Output path for ground-truth JSONL
        #[arg(short, long)]
        output: String,

        /// Number of query-chunk pairs to generate
        #[arg(long, default_value = "250")]
        sample_size: usize,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Claude model for question generation
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        model: String,

        /// Sample chunks only — no API calls (dry run)
        #[arg(long, default_value = "false")]
        dry_run: bool,
    },

    /// Run retrieval queries from ground truth and dump raw results
    Retrieve {
        /// Path to index directory
        #[arg(short, long)]
        index: String,

        /// Path to ground-truth JSONL
        #[arg(short, long)]
        ground_truth: String,

        /// Output path for retrieval results JSONL
        #[arg(short, long)]
        output: String,

        /// Number of results per query
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Retrieval mode: dense (TF-IDF only), sparse (BM25 only), hybrid (fused)
        #[arg(long, default_value = "dense")]
        mode: String,

        /// Fusion strategy (hybrid mode only): rrf, linear, dbsf
        #[arg(long, default_value = "rrf")]
        fusion: String,

        /// Fusion parameter: RRF k value or Linear dense_weight
        #[arg(long)]
        fusion_k: Option<f32>,

        /// Candidates per source for hybrid retrieval
        #[arg(long, default_value = "50")]
        candidates: usize,

        /// Reranking strategy: none, lexical
        #[arg(long, default_value = "none")]
        rerank: String,

        /// Enable HyDE (Hypothetical Document Embeddings) query expansion.
        /// Generates a hypothetical answer via Claude API and uses it for retrieval.
        /// Requires ANTHROPIC_API_KEY environment variable and --features eval.
        #[arg(long, default_value = "false")]
        hyde: bool,
    },

    /// Judge retrieval results for relevance via Claude API and compute metrics
    Judge {
        /// Path to retrieval-results JSONL
        #[arg(short, long)]
        retrieval_results: String,

        /// Path to ground-truth JSONL (for metadata)
        #[arg(short, long)]
        ground_truth: String,

        /// Output path for eval results JSON
        #[arg(short, long)]
        output: String,

        /// Path to judge cache JSON (created if absent)
        #[arg(long, default_value = "judge-cache.json")]
        cache: String,

        /// Number of results to judge per query
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Claude model for judging
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        model: String,
    },

    /// Compute IR metrics from pre-judged results (no API needed)
    Metrics {
        /// Path to retrieval-results JSONL
        #[arg(short, long)]
        retrieval_results: String,

        /// Path to judgments JSONL (produced by Claude Code or external judge)
        #[arg(short, long)]
        judgments: String,

        /// Output path for eval results JSON
        #[arg(short, long)]
        output: String,
    },

    /// Compare two eval result files
    Compare {
        /// Baseline results JSON
        #[arg(long)]
        baseline: String,

        /// Candidate results JSON
        #[arg(long)]
        candidate: String,
    },

    /// Regression gate — exit non-zero if below thresholds
    Gate {
        /// Path to eval results JSON
        #[arg(long)]
        results: String,

        /// Minimum MRR threshold
        #[arg(long, default_value = "0.50")]
        min_mrr: f64,

        /// Minimum Hit@5 threshold
        #[arg(long, default_value = "0.70")]
        min_hit5: f64,
    },
}

/// Persisted index structure
#[derive(Serialize, Deserialize)]
struct PersistedIndex {
    chunks: Vec<PersistedChunk>,
    embeddings: Vec<Vec<f32>>,
    dimension: usize,
    /// Embedder type used (for query compatibility)
    #[serde(default)]
    embedder_type: String,
    /// Model name (for semantic embeddings)
    #[serde(default)]
    model_name: Option<String>,
}

/// Persisted chunk data
#[derive(Serialize, Deserialize)]
struct PersistedChunk {
    content: String,
    title: Option<String>,
    source: Option<String>,
    /// Timestamp metadata for media-derived chunks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    start_secs: Option<f64>,
    /// Timestamp metadata for media-derived chunks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    end_secs: Option<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Demo { query, top_k } => query::run_demo(&query, top_k)?,
        Commands::Index {
            path,
            output,
            chunk_size,
            chunk_overlap,
            dimension,
            embedder,
            model,
            recursive,
            chunk_strategy,
            jobs,
            manifest,
            exclude,
            dedup,
            sqlite,
            incremental,
        } => {
            if incremental {
                incremental::run_index_incremental(
                    &path,
                    &output,
                    chunk_size,
                    chunk_overlap,
                    dimension,
                    embedder,
                    model,
                    recursive,
                    chunk_strategy,
                    jobs,
                    &exclude,
                    dedup,
                )?
            } else {
                ingest::run_index(
                    &path,
                    &output,
                    chunk_size,
                    chunk_overlap,
                    dimension,
                    embedder,
                    model,
                    recursive,
                    chunk_strategy,
                    jobs,
                    manifest,
                    &exclude,
                    dedup,
                    sqlite,
                )?
            }
        }
        Commands::Query {
            query,
            index,
            top_k,
            format,
            mode,
            fusion,
            fusion_k,
            candidates,
            rerank,
            hyde,
        } => query::run_query(
            &query, &index, top_k, &format, &mode, &fusion, fusion_k, candidates, &rerank, hyde,
        )?,
        Commands::Transcribe {
            path,
            recursive,
            skip_existing,
            jobs,
            model,
            backend,
            dry_run,
            prompt,
            hotwords,
            exclude,
        } => transcribe::run_transcribe(
            &path,
            recursive,
            skip_existing,
            jobs,
            model.as_deref(),
            backend,
            dry_run,
            prompt.as_deref(),
            hotwords.as_deref(),
            &exclude,
        )?,
        Commands::ExtractFrames {
            path,
            recursive,
            threshold,
            min_interval,
            jobs,
            skip_existing,
            dry_run,
            exclude,
        } => extract::run_extract_frames(
            &path,
            recursive,
            threshold,
            min_interval,
            jobs,
            skip_existing,
            dry_run,
            &exclude,
        )?,
        Commands::Info => run_info(),
        #[cfg(feature = "eval")]
        Commands::Eval { action } => eval_cmd::run_eval(action)?,
    }

    Ok(())
}

fn run_info() {
    println!("Trueno-RAG Pipeline");
    println!("==================");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Components:");
    println!(
        "  - Chunkers: Recursive, Fixed, Sentence, Paragraph, Semantic, Structural, Timestamp"
    );
    #[cfg(feature = "embeddings")]
    println!("  - Embedders: TF-IDF, FastEmbed (semantic) ✓");
    #[cfg(not(feature = "embeddings"))]
    println!("  - Embedders: TF-IDF (trainable), Mock (testing)");
    println!("  - Fusion: RRF, Linear, DBSF, Convex, Union, Intersection");
    println!("  - Rerankers: Lexical, CrossEncoder (mock), Composite");
    println!();
    println!("Supported formats:");
    let registry = LoaderRegistry::new();
    let exts: Vec<&str> = registry.supported_extensions();
    println!("  {}", exts.join(", "));
    println!();
    #[cfg(feature = "embeddings")]
    {
        println!("Semantic Embedding Models:");
        println!("  - mini-lm: sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast)");
        println!("  - bge-small: BAAI/bge-small-en-v1.5 (384 dims, balanced)");
        println!("  - bge-base: BAAI/bge-base-en-v1.5 (768 dims, quality)");
    }
    #[cfg(not(feature = "embeddings"))]
    {
        println!("Note: Build with --features embeddings for semantic search");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use trueno_rag::fusion::FusionStrategy;
    use trueno_rag::Document;

    // Re-import functions from submodules for testing
    use crate::discover::*;
    use crate::incremental::*;
    use crate::ingest::*;
    use crate::query::*;
    use crate::transcribe::TranscribeManifest;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_discover_files_single_file() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_single");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        fs::write(&file, "hello").unwrap();

        let registry = LoaderRegistry::new();
        let files = discover_files(&file, false, &registry, &None).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_files_directory_non_recursive() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_dir");
        let sub = dir.join("sub");
        let _ = fs::create_dir_all(&sub);
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(dir.join("b.md"), "b").unwrap();
        fs::write(dir.join("c.mp4"), "c").unwrap(); // unsupported
        fs::write(sub.join("d.txt"), "d").unwrap(); // in subdir

        let registry = LoaderRegistry::new();
        let files = discover_files(&dir, false, &registry, &None).unwrap();
        // Only a.txt and b.md in top-level (not c.mp4, not sub/d.txt)
        assert_eq!(files.len(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_files_recursive() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_recursive");
        let sub = dir.join("sub");
        let deep = sub.join("deep");
        let _ = fs::create_dir_all(&deep);
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(sub.join("b.srt"), "1\n00:00:01,000 --> 00:00:02,000\nb\n").unwrap();
        fs::write(deep.join("c.md"), "c").unwrap();

        let registry = LoaderRegistry::new();
        let files = discover_files(&dir, true, &registry, &None).unwrap();
        assert_eq!(files.len(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_files_unsupported_single() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_unsup");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("video.mp4");
        fs::write(&file, "fake").unwrap();

        let registry = LoaderRegistry::new();
        let result = discover_files(&file, false, &registry, &None);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_classify_files() {
        let files = vec![
            PathBuf::from("/data/a.txt"),
            PathBuf::from("/data/b.txt"),
            PathBuf::from("/data/c.srt"),
            PathBuf::from("/data/d.md"),
        ];
        let counts = classify_files(&files);
        assert_eq!(counts["txt"], 2);
        assert_eq!(counts["srt"], 1);
        assert_eq!(counts["md"], 1);
    }

    #[test]
    fn test_chunk_strategy_default() {
        // Ensure Auto is the default
        let strategy = ChunkStrategy::default();
        assert!(matches!(strategy, ChunkStrategy::Auto));
    }

    #[test]
    fn test_build_exclude_set_empty() {
        let result = build_exclude_set(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_build_exclude_set_valid() {
        let patterns = vec!["*/RAW".to_string(), "*/RAW/*".to_string()];
        let result = build_exclude_set(&patterns).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_build_exclude_set_invalid() {
        let patterns = vec!["[invalid".to_string()];
        let result = build_exclude_set(&patterns);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_excluded_matches() {
        let patterns = vec!["*/RAW".to_string(), "*/RAW/*".to_string()];
        let exclude = build_exclude_set(&patterns).unwrap();
        assert!(is_excluded(Path::new("/data/courses/aws/RAW"), &exclude));
        assert!(is_excluded(Path::new("/data/courses/aws/RAW/video.mp4"), &exclude));
        assert!(!is_excluded(Path::new("/data/courses/aws/build/video.srt"), &exclude));
    }

    #[test]
    fn test_is_excluded_none() {
        assert!(!is_excluded(Path::new("/any/path"), &None));
    }

    #[test]
    fn test_discover_files_with_exclude() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_exclude");
        let raw = dir.join("RAW");
        let build = dir.join("build");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&raw);
        let _ = fs::create_dir_all(&build);
        fs::write(dir.join("keep.txt"), "keep").unwrap();
        fs::write(raw.join("skip.txt"), "skip").unwrap();
        fs::write(build.join("also_keep.txt"), "keep2").unwrap();

        let registry = LoaderRegistry::new();
        let exclude = build_exclude_set(&["*/RAW".to_string(), "*/RAW/*".to_string()]).unwrap();
        let files = discover_files(&dir, true, &registry, &exclude).unwrap();

        // Should have keep.txt and build/also_keep.txt but NOT RAW/skip.txt
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| !f.to_string_lossy().contains("RAW")));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "sqlite")]
    fn test_export_sqlite_creates_db() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_sqlite_export");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let persisted = PersistedIndex {
            chunks: vec![
                PersistedChunk {
                    content: "Rust is a systems language.".to_string(),
                    title: Some("Rust Basics".to_string()),
                    source: Some("docs/rust.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "The borrow checker ensures safety.".to_string(),
                    title: Some("Rust Basics".to_string()),
                    source: Some("docs/rust.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "Python is interpreted.".to_string(),
                    title: Some("Python Intro".to_string()),
                    source: Some("docs/python.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
            ],
            embeddings: vec![vec![0.0; 4]; 3],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        export_sqlite(&persisted, &dir).unwrap();

        let db_path = dir.join("index.sqlite");
        assert!(db_path.exists(), "index.sqlite should be created");

        // Verify doc and chunk counts via SqliteIndex API
        let idx = trueno_rag::SqliteIndex::open(&db_path).unwrap();
        assert_eq!(idx.document_count().unwrap(), 2, "2 unique source docs");
        assert_eq!(idx.chunk_count().unwrap(), 3, "3 chunks total");

        // Verify FTS5 search works
        let results = idx.search_fts("borrow checker", 5).unwrap();
        assert!(!results.is_empty(), "FTS5 should find 'borrow checker'");
        assert!(
            results[0].content.contains("borrow checker"),
            "Top result should contain query terms"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "sqlite")]
    fn test_export_sqlite_groups_by_source() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_sqlite_grouping");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // 4 chunks from 2 docs
        let persisted = PersistedIndex {
            chunks: vec![
                PersistedChunk {
                    content: "chunk A1".to_string(),
                    title: Some("Doc A".to_string()),
                    source: Some("a.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "chunk A2".to_string(),
                    title: Some("Doc A".to_string()),
                    source: Some("a.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "chunk B1".to_string(),
                    title: None,
                    source: Some("b.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "chunk B2".to_string(),
                    title: None,
                    source: Some("b.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
            ],
            embeddings: vec![vec![0.0; 4]; 4],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        export_sqlite(&persisted, &dir).unwrap();

        let idx = trueno_rag::SqliteIndex::open(dir.join("index.sqlite")).unwrap();
        assert_eq!(idx.document_count().unwrap(), 2);
        assert_eq!(idx.chunk_count().unwrap(), 4);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "sqlite")]
    fn test_export_sqlite_unknown_source() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_sqlite_unknown");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let persisted = PersistedIndex {
            chunks: vec![PersistedChunk {
                content: "orphan chunk".to_string(),
                title: None,
                source: None, // no source
                start_secs: None,
                end_secs: None,
            }],
            embeddings: vec![vec![0.0; 4]],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        export_sqlite(&persisted, &dir).unwrap();

        let idx = trueno_rag::SqliteIndex::open(dir.join("index.sqlite")).unwrap();
        assert_eq!(idx.document_count().unwrap(), 1, "unknown doc grouped");
        assert_eq!(idx.chunk_count().unwrap(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "sqlite")]
    fn test_export_sqlite_replaces_stale_db() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_sqlite_replace");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Create a stale DB
        fs::write(dir.join("index.sqlite"), b"stale data").unwrap();

        let persisted = PersistedIndex {
            chunks: vec![PersistedChunk {
                content: "fresh content".to_string(),
                title: None,
                source: Some("fresh.txt".to_string()),
                start_secs: None,
                end_secs: None,
            }],
            embeddings: vec![vec![0.0; 4]],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        export_sqlite(&persisted, &dir).unwrap();

        let idx = trueno_rag::SqliteIndex::open(dir.join("index.sqlite")).unwrap();
        assert_eq!(idx.document_count().unwrap(), 1);
        assert_eq!(idx.chunk_count().unwrap(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(not(feature = "sqlite"))]
    fn test_export_sqlite_stub_errors() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_sqlite_stub");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let persisted = PersistedIndex {
            chunks: vec![],
            embeddings: vec![],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        let result = export_sqlite(&persisted, &dir);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("sqlite"),
            "Error should mention sqlite feature"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "sqlite")]
    fn test_export_sqlite_empty_index() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_sqlite_empty");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let persisted = PersistedIndex {
            chunks: vec![],
            embeddings: vec![],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        export_sqlite(&persisted, &dir).unwrap();

        let idx = trueno_rag::SqliteIndex::open(dir.join("index.sqlite")).unwrap();
        assert_eq!(idx.document_count().unwrap(), 0);
        assert_eq!(idx.chunk_count().unwrap(), 0);

        let _ = fs::remove_dir_all(&dir);
    }

    // ================================================================
    // Coverage-targeted tests for uncovered pure-logic functions
    // ================================================================

    #[test]
    fn test_parse_fusion_strategy_rrf() {
        let result = parse_fusion_strategy("rrf", None).unwrap();
        assert!(matches!(result, FusionStrategy::RRF { k } if (k - 60.0).abs() < 0.001));
    }

    #[test]
    fn test_parse_fusion_strategy_rrf_custom_k() {
        let result = parse_fusion_strategy("rrf", Some(30.0)).unwrap();
        assert!(matches!(result, FusionStrategy::RRF { k } if (k - 30.0).abs() < 0.001));
    }

    #[test]
    fn test_parse_fusion_strategy_linear() {
        let result = parse_fusion_strategy("linear", Some(0.7)).unwrap();
        assert!(
            matches!(result, FusionStrategy::Linear { dense_weight } if (dense_weight - 0.7).abs() < 0.001)
        );
    }

    #[test]
    fn test_parse_fusion_strategy_dbsf() {
        let result = parse_fusion_strategy("dbsf", None).unwrap();
        assert!(matches!(result, FusionStrategy::DBSF));
    }

    #[test]
    fn test_parse_fusion_strategy_unknown() {
        let result = parse_fusion_strategy("unknown", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_finish_load_report_success() {
        let docs = vec![Document::new("test content".to_string())];
        let result = finish_load_report(docs, 0).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_finish_load_report_with_errors() {
        let docs = vec![Document::new("test content".to_string())];
        let result = finish_load_report(docs, 3).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_finish_load_report_all_failed() {
        let docs: Vec<Document> = vec![];
        let result = finish_load_report(docs, 5);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("5 errors"));
    }

    #[test]
    fn test_report_media_text_split_no_media() {
        let docs = vec![Document::new("plain text".to_string())];
        // Should not panic -- no media documents means no output
        report_media_text_split(&docs);
    }

    #[test]
    fn test_report_media_text_split_with_media() {
        let mut doc = Document::new("media content".to_string());
        doc.metadata
            .insert("subtitle_cues".to_string(), serde_json::Value::String("cue data".to_string()));
        let docs = vec![doc, Document::new("plain text".to_string())];
        // Should print "1 with timestamps, 1 plain text"
        report_media_text_split(&docs);
    }

    #[test]
    fn test_query_sparse_basic() {
        let persisted = PersistedIndex {
            chunks: vec![
                PersistedChunk {
                    content: "Rust borrow checker and ownership model".to_string(),
                    title: Some("Rust".to_string()),
                    source: Some("rust.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "Python garbage collector and reference counting".to_string(),
                    title: Some("Python".to_string()),
                    source: Some("python.txt".to_string()),
                    start_secs: None,
                    end_secs: None,
                },
            ],
            embeddings: vec![vec![0.0; 4]; 2],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        let results = query_sparse("borrow checker", &persisted, 5);
        assert!(!results.is_empty(), "BM25 should find 'borrow checker'");
        // First result should be the Rust chunk (index 0)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_query_sparse_empty_corpus() {
        let persisted = PersistedIndex {
            chunks: vec![],
            embeddings: vec![],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        let results = query_sparse("anything", &persisted, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_dense_tfidf() {
        // TF-IDF embedder doesn't require external models -- it can be created inline
        let persisted = PersistedIndex {
            chunks: vec![
                PersistedChunk {
                    content: "alpha beta gamma".to_string(),
                    title: None,
                    source: None,
                    start_secs: None,
                    end_secs: None,
                },
                PersistedChunk {
                    content: "delta epsilon zeta".to_string(),
                    title: None,
                    source: None,
                    start_secs: None,
                    end_secs: None,
                },
            ],
            embeddings: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            dimension: 4,
            embedder_type: "tfidf".to_string(),
            model_name: None,
        };

        let result = query_dense("alpha beta", &persisted, 2);
        assert!(result.is_ok());
        let scores = result.unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_format_query_results_text() {
        let chunks = vec![PersistedChunk {
            content: "Rust is a systems programming language focused on safety".to_string(),
            title: Some("Rust Intro".to_string()),
            source: Some("rust.txt".to_string()),
            start_secs: None,
            end_secs: None,
        }];
        let scores = vec![(0_usize, 0.95_f32)];
        let result = format_query_results("Rust", &scores, &chunks, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_query_results_json() {
        let chunks = vec![PersistedChunk {
            content: "Rust is a systems programming language".to_string(),
            title: Some("Rust".to_string()),
            source: Some("rust.txt".to_string()),
            start_secs: Some(10.5),
            end_secs: Some(25.0),
        }];
        let scores = vec![(0_usize, 0.8_f32)];
        let result = format_query_results("Rust", &scores, &chunks, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_query_results_json_no_timestamps() {
        let chunks = vec![PersistedChunk {
            content: "plain text content without timestamps".to_string(),
            title: None,
            source: Some("doc.txt".to_string()),
            start_secs: None,
            end_secs: None,
        }];
        let scores = vec![(0_usize, 0.5_f32)];
        let result = format_query_results("plain", &scores, &chunks, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_query_results_text_with_timestamps() {
        let chunks = vec![
            PersistedChunk {
                content: "lecture content about PDCA cycle in software engineering and continuous improvement".to_string(),
                title: Some("PDCA Lecture".to_string()),
                source: Some("lecture.srt".to_string()),
                start_secs: Some(120.0),
                end_secs: Some(180.0),
            },
        ];
        let scores = vec![(0_usize, 0.9_f32)];
        let result = format_query_results("PDCA", &scores, &chunks, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_rerank_none() {
        let chunks = vec![PersistedChunk {
            content: "alpha".to_string(),
            title: None,
            source: None,
            start_secs: None,
            end_secs: None,
        }];
        let scores = vec![(0_usize, 1.0_f32)];
        let result = apply_rerank("none", "test", &scores, &chunks, 5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_apply_rerank_lexical() {
        let chunks = vec![
            PersistedChunk {
                content: "Rust borrow checker ensures memory safety through ownership".to_string(),
                title: Some("Rust Safety".to_string()),
                source: Some("rust.txt".to_string()),
                start_secs: None,
                end_secs: None,
            },
            PersistedChunk {
                content:
                    "Python garbage collector manages memory automatically with reference counting"
                        .to_string(),
                title: Some("Python GC".to_string()),
                source: Some("python.txt".to_string()),
                start_secs: None,
                end_secs: None,
            },
        ];
        let scores = vec![(0_usize, 0.9_f32), (1_usize, 0.5_f32)];
        let result = apply_rerank("lexical", "borrow checker memory", &scores, &chunks, 5).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_apply_rerank_unknown() {
        let chunks = vec![];
        let scores = vec![];
        let result = apply_rerank("invalid", "test", &scores, &chunks, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_manifest_save_load() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_manifest");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let manifest = TranscribeManifest {
            completed: vec!["a.mp4".to_string(), "b.mp4".to_string()],
            failed: vec!["c.mp4".to_string()],
        };
        manifest.save(&dir).unwrap();

        let loaded = TranscribeManifest::load(&dir);
        assert_eq!(loaded.completed.len(), 2);
        assert_eq!(loaded.failed.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_chunk_and_embed_timestamp_strategy() {
        use trueno_rag::chunk::RecursiveChunker;
        use trueno_rag::chunk::TimestampChunker;
        use trueno_rag::embed::MockEmbedder;

        let embedder = MockEmbedder::new(4);
        let recursive = RecursiveChunker::new(512, 64);
        let timestamp = TimestampChunker::new(30.0);

        let doc = Document::new(
            "This is a lecture about Rust programming and memory safety concepts".to_string(),
        );
        // TimestampChunker with no cues falls back to RecursiveChunker
        let docs = vec![doc];

        let result = chunk_and_embed(
            &docs,
            &embedder,
            &recursive,
            &timestamp,
            ChunkStrategy::Timestamp,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_discover_and_load_empty_dir() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_load_empty");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Only create an unsupported file
        fs::write(dir.join("video.mp4"), "not a real file").unwrap();

        let result = discover_and_load(&dir, false, 1, &None);
        assert!(result.is_err(), "Should fail with no supported files");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_documents_sequential_progress() {
        // Create 101 files to trigger the progress reporting line
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_seq_progress");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut files = Vec::new();
        for i in 0..101 {
            let file = dir.join(format!("file_{i:03}.txt"));
            fs::write(&file, format!("Content of file {i}")).unwrap();
            files.push(file);
        }

        let registry = LoaderRegistry::new();
        let result = load_documents_sequential(&files, &registry);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 101);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_media_files_recursive_subdirs() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_media_recursive");
        let sub = dir.join("sub");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.join("a.mp4"), "fake").unwrap();
        fs::write(sub.join("b.mp4"), "fake").unwrap();

        let files = discover_media_files(&dir, true, &None).unwrap();
        assert_eq!(files.len(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_documents_sequential_with_error() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_seq_load_err");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Create a valid text file and a binary that will fail to load as text
        fs::write(dir.join("good.txt"), "valid text content").unwrap();
        fs::write(dir.join("bad.bin"), &[0xFF, 0xFE, 0x00, 0x01]).unwrap();

        let registry = LoaderRegistry::new();
        let files = vec![dir.join("good.txt"), dir.join("bad.bin")];
        let result = load_documents_sequential(&files, &registry);
        // Should succeed with at least the good file loaded (bad.bin may or may not error)
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_chunk_and_embed_empty_document() {
        use trueno_rag::chunk::RecursiveChunker;
        use trueno_rag::chunk::TimestampChunker;
        use trueno_rag::embed::MockEmbedder;

        let embedder = MockEmbedder::new(4);
        let recursive = RecursiveChunker::new(512, 64);
        let timestamp = TimestampChunker::new(30.0);

        // One empty doc + one valid doc
        let docs = vec![
            Document::new(String::new()), // empty -- should be skipped
            Document::new("Some real content that has actual words".to_string()),
        ];

        let result = chunk_and_embed(
            &docs,
            &embedder,
            &recursive,
            &timestamp,
            ChunkStrategy::Recursive,
            false,
        );
        assert!(result.is_ok());
        let (chunks, embeddings) = result.unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks.len(), embeddings.len());
    }

    #[test]
    fn test_chunk_and_embed_with_dedup() {
        use trueno_rag::chunk::RecursiveChunker;
        use trueno_rag::chunk::TimestampChunker;
        use trueno_rag::embed::MockEmbedder;

        let embedder = MockEmbedder::new(4);
        let recursive = RecursiveChunker::new(512, 64);
        let timestamp = TimestampChunker::new(30.0);

        // Two identical documents -- dedup should remove duplicates
        let docs = vec![
            Document::new("Duplicate content for dedup testing.".to_string()),
            Document::new("Duplicate content for dedup testing.".to_string()),
        ];

        let result = chunk_and_embed(
            &docs,
            &embedder,
            &recursive,
            &timestamp,
            ChunkStrategy::Recursive,
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_discover_media_files_single_file() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_media_single");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("test.mp4");
        fs::write(&file, "fake media").unwrap();

        let files = discover_media_files(&file, false, &None).unwrap();
        assert_eq!(files.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_media_files_single_non_media() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_media_nonmedia");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("test.txt");
        fs::write(&file, "not media").unwrap();

        let result = discover_media_files(&file, false, &None);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_media_files_with_exclude() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_media_exclude");
        let raw = dir.join("RAW");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&raw);
        fs::write(dir.join("keep.mp4"), "fake media").unwrap();
        fs::write(raw.join("skip.mp4"), "fake media").unwrap();

        let exclude = build_exclude_set(&["*/RAW".to_string(), "*/RAW/*".to_string()]).unwrap();
        let files = discover_media_files(&dir, true, &exclude).unwrap();

        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().contains("keep.mp4"));

        let _ = fs::remove_dir_all(&dir);
    }

    // --- Incremental indexing tests ---

    #[test]
    fn test_compute_file_hashes() {
        let dir = std::env::temp_dir().join("trueno_rag_test_hashes");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        fs::write(dir.join("a.txt"), "hello").unwrap();
        fs::write(dir.join("b.txt"), "world").unwrap();

        let files = vec![dir.join("a.txt"), dir.join("b.txt")];
        let hashes = compute_file_hashes(&files).unwrap();
        assert_eq!(hashes.len(), 2);

        // Same content should produce same hash
        let hash_a = hashes[0].1;
        let hashes2 = compute_file_hashes(&[dir.join("a.txt")]).unwrap();
        assert_eq!(hash_a, hashes2[0].1);

        // Different content should produce different hash
        assert_ne!(hashes[0].1, hashes[1].1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_diff_fingerprints_detects_new() {
        let current =
            vec![(PathBuf::from("/a.md"), [1u8; 32]), (PathBuf::from("/b.md"), [2u8; 32])];
        let stored: HashMap<String, Vec<u8>> = HashMap::new();

        let (changed, deleted) = diff_fingerprints(&current, &stored);
        assert_eq!(changed.len(), 2);
        assert!(deleted.is_empty());
    }

    #[test]
    fn test_diff_fingerprints_detects_changed() {
        let current = vec![(PathBuf::from("/a.md"), [2u8; 32])];
        let mut stored: HashMap<String, Vec<u8>> = HashMap::new();
        stored.insert("/a.md".to_string(), vec![1u8; 32]);

        let (changed, deleted) = diff_fingerprints(&current, &stored);
        assert_eq!(changed.len(), 1);
        assert!(deleted.is_empty());
    }

    #[test]
    fn test_diff_fingerprints_detects_unchanged() {
        let current = vec![(PathBuf::from("/a.md"), [1u8; 32])];
        let mut stored: HashMap<String, Vec<u8>> = HashMap::new();
        stored.insert("/a.md".to_string(), vec![1u8; 32]);

        let (changed, deleted) = diff_fingerprints(&current, &stored);
        assert!(changed.is_empty());
        assert!(deleted.is_empty());
    }

    #[test]
    fn test_diff_fingerprints_detects_deleted() {
        let current: Vec<(PathBuf, [u8; 32])> = vec![];
        let mut stored: HashMap<String, Vec<u8>> = HashMap::new();
        stored.insert("/a.md".to_string(), vec![1u8; 32]);

        let (changed, deleted) = diff_fingerprints(&current, &stored);
        assert!(changed.is_empty());
        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0], "/a.md");
    }
}
