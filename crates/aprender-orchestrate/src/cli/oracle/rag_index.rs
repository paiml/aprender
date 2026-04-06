//! RAG indexing commands
//!
//! This module contains the RAG indexing logic, separated from query commands
//! to keep file sizes manageable.
//!
//! When the `rag` feature is enabled, indexes into SQLite+FTS5 via
//! `trueno_rag::sqlite::SqliteIndex`. Otherwise falls back to
//! `HybridRetriever` + JSON persistence.

use crate::ansi_colors::Colorize;
use crate::oracle;

#[cfg(feature = "rag")]
use crate::cli::oracle_indexing::ChunkIndexer;
use crate::cli::oracle_indexing::{
    check_dir_for_changes, doc_fingerprint_changed, index_dir_group,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Print a phase progress indicator to stderr.
fn eprint_phase(phase: &str) {
    eprintln!("  {} {}", "[   index]".dimmed(), phase);
}

/// Print a labeled statistic with the label in bright yellow.
fn print_stat(label: &str, value: impl std::fmt::Display) {
    println!("{}: {}", label.bright_yellow(), value);
}

/// Clean up stale JSON files from pre-SQLite era.
///
/// Two-stage cleanup per spec Section 4.2.4:
/// 1. First run: renames `.json` → `.json.bak` (recoverable)
/// 2. Second run: deletes `.json.bak` files (permanent)
///
/// This gives one full reindex cycle for manual recovery before
/// the ~600MB of stale JSON is permanently removed.
#[cfg(feature = "rag")]
fn cleanup_stale_json(persistence: &oracle::rag::persistence::RagPersistence) {
    let cache = persistence.cache_path();
    for name in &["index.json", "documents.json", "manifest.json"] {
        let path = cache.join(name);
        let bak = cache.join(format!("{name}.bak"));

        // Stage 2: delete .bak files from a previous cleanup
        if bak.exists() {
            match std::fs::remove_file(&bak) {
                Ok(()) => eprintln!("  {} Deleted {name}.bak", "[   clean]".dimmed()),
                Err(e) => eprintln!("  {} Failed to delete {name}.bak: {e}", "[   clean]".dimmed()),
            }
        }

        // Stage 1: rename live .json → .bak
        if path.exists() {
            match std::fs::rename(&path, &bak) {
                Ok(()) => eprintln!("  {} Renamed {name} → {name}.bak", "[   clean]".dimmed()),
                Err(e) => eprintln!("  {} Failed to rename {name}: {e}", "[   clean]".dimmed()),
            }
        }
    }
}

/// Default SQLite database path for the RAG index (delegates to rag module).
#[cfg(feature = "rag")]
fn sqlite_index_path() -> std::path::PathBuf {
    super::rag::sqlite_index_path()
}

/// Check if the existing RAG index is current (no files changed).
/// Returns true if index is up to date and no rebuild is needed.
fn is_index_current(
    persistence: &oracle::rag::persistence::RagPersistence,
    rust_stack_dirs: &[String],
    rust_corpus_dirs: &[String],
    python_corpus_dirs: &[String],
    rust_config: &oracle::rag::ChunkerConfig,
    python_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
) -> bool {
    let Ok(Some(fingerprints)) = persistence.load_fingerprints_only() else {
        return false;
    };
    if fingerprints.is_empty() {
        return false;
    }

    println!("{}", "Checking for changes against stored fingerprints...".dimmed());

    let changed = detect_dir_changes(
        rust_stack_dirs,
        rust_corpus_dirs,
        python_corpus_dirs,
        rust_config,
        python_config,
        model_hash,
        &fingerprints,
    );

    if changed == 0 {
        println!(
            "{}",
            "Index is current (no files changed since last index)".bright_green().bold()
        );
        println!();
        return true;
    }

    println!("{} files changed, rebuilding index...", changed.to_string().bright_yellow());
    println!();
    false
}

/// Detect changes across all directory groups by checking doc fingerprints and source dirs.
/// Returns the number of changed sources (0 = no changes).
fn detect_dir_changes(
    rust_stack_dirs: &[String],
    rust_corpus_dirs: &[String],
    python_corpus_dirs: &[String],
    rust_config: &oracle::rag::ChunkerConfig,
    python_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> usize {
    use std::path::Path;

    let rust_dirs = rust_stack_dirs
        .iter()
        .chain(rust_corpus_dirs.iter())
        .map(|d| (d.as_str(), rust_config, "rs"));
    let python_dirs = python_corpus_dirs.iter().map(|d| (d.as_str(), python_config, "py"));

    for (dir, config, ext) in rust_dirs.chain(python_dirs) {
        let path = Path::new(dir);
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let component = canonical.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        if path.exists()
            && check_component_changed(path, component, config, model_hash, existing, ext)
        {
            return 1;
        }
    }
    0
}

/// Check if a single file within a component directory has changed fingerprint.
fn check_component_file_changed(
    base_path: &std::path::Path,
    filename: &str,
    component: &str,
    config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> bool {
    let path = base_path.join(filename);
    if path.exists() {
        let doc_id = format!("{}/{}", component, filename);
        return doc_fingerprint_changed(&path, &doc_id, config, model_hash, existing);
    }
    false
}

/// Check if any file in a single component directory has changed.
fn check_component_changed(
    path: &std::path::Path,
    component: &str,
    config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    if any_root_md_changed(path, component, config, model_hash, existing) {
        return true;
    }

    // Check src/ directory, falling back to root for Python flat-layout packages
    let (scan_dir, base) = resolve_scan_dir(path, extension);
    scan_dir.exists()
        && check_dir_for_changes(
            &scan_dir, &base, component, config, model_hash, existing, extension,
        )
}

/// Check if any root-level .md file in the component directory has changed.
fn any_root_md_changed(
    path: &std::path::Path,
    component: &str,
    config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> bool {
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|ext| ext == "md") {
            let Some(file_name) = p.file_name() else { continue };
            let fname = file_name.to_string_lossy().to_string();
            if check_component_file_changed(path, &fname, component, config, model_hash, existing) {
                return true;
            }
        }
    }
    false
}

/// Resolve the directory to scan for source files and its base path.
fn resolve_scan_dir(
    path: &std::path::Path,
    extension: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let src_dir = path.join("src");
    if src_dir.exists() {
        (src_dir.clone(), src_dir.parent().unwrap_or(&src_dir).to_path_buf())
    } else if extension == "py" {
        (path.to_path_buf(), path.to_path_buf())
    } else {
        (src_dir, path.to_path_buf())
    }
}

// ============================================================================
// SQLite Indexing Backend (rag feature)
// ============================================================================

/// Chunk indexer that collects chunks per document and flushes them into
/// `SqliteIndex` as batch transactions.
#[cfg(feature = "rag")]
pub(crate) struct SqliteChunkIndexer {
    index: trueno_rag::sqlite::SqliteIndex,
    /// Pending chunks grouped by document: doc_id → Vec<(chunk_id, content)>
    pending: std::collections::HashMap<String, Vec<(String, String)>>,
}

#[cfg(feature = "rag")]
impl SqliteChunkIndexer {
    fn new(index: trueno_rag::sqlite::SqliteIndex) -> Self {
        Self { index, pending: std::collections::HashMap::new() }
    }

    /// Flush all pending chunks into SQLite.
    fn flush(
        &self,
        fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    ) -> anyhow::Result<()> {
        for (doc_id, chunks) in &self.pending {
            let fp = fingerprints.get(doc_id).map(|fp| (doc_id.as_str(), &fp.content_hash));
            self.index
                .insert_document(doc_id, None, Some(doc_id), "", chunks, fp)
                .map_err(|e| anyhow::anyhow!("SQLite insert failed for {doc_id}: {e}"))?;
        }
        Ok(())
    }

    /// Get reference to underlying SqliteIndex.
    fn index(&self) -> &trueno_rag::sqlite::SqliteIndex {
        &self.index
    }
}

#[cfg(feature = "rag")]
impl ChunkIndexer for SqliteChunkIndexer {
    fn index_chunk(&mut self, chunk_id: &str, content: &str) {
        // chunk_id format: "component/file.rs#line"
        let doc_id = chunk_id.split('#').next().unwrap_or(chunk_id);
        self.pending
            .entry(doc_id.to_string())
            .or_default()
            .push((chunk_id.to_string(), content.to_string()));
    }
}

/// Save the RAG index to SQLite, print stats, and optimize.
#[cfg(feature = "rag")]
fn save_rag_index_sqlite(
    sqlite_indexer: &SqliteChunkIndexer,
    reindexer: &oracle::rag::HeijunkaReindexer,
    indexed_count: usize,
    total_chunks: usize,
    fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "{}: {} documents, {} chunks indexed",
        "Complete".bright_green().bold(),
        indexed_count,
        total_chunks
    );
    println!();

    let reindex_stats = reindexer.stats();
    print_stat("Reindexer", format!("{} documents tracked", reindex_stats.tracked_documents));
    println!();

    println!("{}", "Flushing to SQLite...".dimmed());
    {
        let _flush_span = span("sqlite_flush");
        sqlite_indexer.flush(fingerprints)?;
    }

    // Store metadata
    let index = sqlite_indexer.index();
    index
        .set_metadata("batuta_version", env!("CARGO_PKG_VERSION"))
        .map_err(|e| anyhow::anyhow!("Failed to set metadata: {e}"))?;
    index
        .set_metadata(
            "indexed_at",
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis().to_string())
                .unwrap_or_default(),
        )
        .map_err(|e| anyhow::anyhow!("Failed to set metadata: {e}"))?;

    // Rebuild index segments for query performance
    {
        let _opt_span = span("sqlite_optimize");
        index.optimize().map_err(|e| anyhow::anyhow!("Optimize failed: {e}"))?;
    }

    let doc_count = index.document_count().map_err(|e| anyhow::anyhow!("Count failed: {e}"))?;
    let chunk_count = index.chunk_count().map_err(|e| anyhow::anyhow!("Count failed: {e}"))?;

    let db_path = sqlite_index_path();
    let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    println!(
        "{}: {} documents, {} chunks in SQLite ({:.1} MB)",
        "Saved".bright_green().bold(),
        doc_count,
        chunk_count,
        db_size as f64 / 1_048_576.0,
    );
    println!("  {}: {:?}", "Path".dimmed(), db_path);
    println!();

    Ok(())
}

// ============================================================================
// JSON Persistence Backend (fallback without rag feature)
// ============================================================================

/// Print summary stats and save the RAG index to disk as JSON.
#[cfg(not(feature = "rag"))]
#[allow(clippy::too_many_arguments)]
fn save_rag_index_json(
    persistence: &oracle::rag::persistence::RagPersistence,
    retriever: oracle::rag::HybridRetriever,
    reindexer: oracle::rag::HeijunkaReindexer,
    indexed_count: usize,
    total_chunks: usize,
    fingerprints: std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: std::collections::HashMap<String, String>,
) -> anyhow::Result<()> {
    use oracle::rag::persistence::{CorpusSource, PersistedDocuments};

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!(
        "{}: {} documents, {} chunks indexed",
        "Complete".bright_green().bold(),
        indexed_count,
        total_chunks
    );
    println!();

    let stats = retriever.stats();
    print_stat("Vocabulary", format!("{} unique terms", stats.total_terms));
    print_stat("Avg doc length", format!("{:.1} tokens", stats.avg_doc_length));
    println!();

    let reindex_stats = reindexer.stats();
    print_stat("Reindexer", format!("{} documents tracked", reindex_stats.tracked_documents));
    println!();

    let corpus_sources = vec![CorpusSource {
        id: "sovereign-ai-stack".to_string(),
        commit: None,
        doc_count: indexed_count,
        chunk_count: total_chunks,
    }];

    println!("{}", "Saving index to disk...".dimmed());

    let persisted_index = retriever.to_persisted();
    let persisted_docs = PersistedDocuments {
        documents: std::collections::HashMap::new(),
        fingerprints,
        total_chunks,
        chunk_contents,
    };

    match persistence.save(&persisted_index, &persisted_docs, corpus_sources) {
        Ok(()) => {
            println!(
                "{}: Index saved to {:?}",
                "Saved".bright_green().bold(),
                persistence.cache_path()
            );
        }
        Err(e) => {
            println!("{}: Failed to save index: {}", "Warning".bright_yellow(), e);
        }
    }
    println!();

    Ok(())
}

// ============================================================================
// Public Commands
// ============================================================================

/// Common chunker configs, directory lists, and setup for RAG indexing.
struct IndexConfig {
    rust_chunker_config: oracle::rag::ChunkerConfig,
    python_chunker_config: oracle::rag::ChunkerConfig,
    rust_stack_dirs: Vec<String>,
    rust_corpus_dirs: Vec<String>,
    python_corpus_dirs: Vec<String>,
    model_hash: [u8; 32],
    /// Number of private directories merged from `.batuta-private.toml`.
    private_dir_count: usize,
}

impl IndexConfig {
    fn new() -> Self {
        let mut config = Self {
            rust_chunker_config: oracle::rag::ChunkerConfig::new(
                512,
                64,
                &["\n## ", "\n### ", "\n#### ", "\nfn ", "\npub fn ", "\nimpl "],
            ),
            python_chunker_config: oracle::rag::ChunkerConfig::new(
                512,
                64,
                &["\n## ", "\n### ", "\n#### ", "\ndef ", "\nclass ", "\n    def ", "\nasync def "],
            ),
            rust_stack_dirs: vec![
                // Core compute
                "../trueno",
                "../trueno-db",
                "../trueno-graph",
                "../trueno-rag",
                "../trueno-viz",
                "../trueno-zram",
                "../trueno-ublk",
                // ML/Training/Inference
                "../aprender",
                "../entrenar",
                "../realizar",
                "../whisper.apr",
                "../alimentar",
                // Distribution/Registry
                "../repartir",
                "../pacha",
                // Simulation/Games/Education
                "../jugar",
                "../simular",
                "../profesor",
                // Transpilers
                "../depyler",
                "../bashrs",
                "../decy",
                "../rascal",
                "../ruchy",
                "../ruchyruchy",
                // Quality/Tooling/Tracing
                "../provable-contracts",
                "../forjar",
                "../apr-qa",
                "../renacer",
                "../paiml-mcp-agent-toolkit",
                "../certeza",
                "../verificar",
                "../probar",
                "../presentar",
                "../cohete",
                "../duende",
                "../pepita",
                "../manzana",
                "../copia",
                "../pforge",
                "../rust-mcp-sdk",
                "../organizational-intelligence-plugin",
                // CLI/Utilities
                "../reaper",
                "../pzsh",
                "../mp4convertor",
                "../wos",
                "../single-shot-eval",
                "../ubuntu-config-scripts",
                // Benchmarking/Courses (Rust)
                "../deterministic-llm-coding",
                "../compiled-rust-benchmarking",
                "../HF-Production-ML",
                "../HF-Hub-Ecosystem",
                // Ruchy ecosystem
                "../rosetta-ruchy",
                "../ruchy-docker",
                "../ruchy-lambda",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            rust_corpus_dirs: vec![
                // Ground truth corpora
                "../batuta-ground-truth-mlops-corpus",
                "../apr-model-qa-playbook",
                "../tgi-ground-truth-corpus",
                "../mixed-python-rust-ground-truth",
                "../mixed-rust-lean-ground-truth",
                "../lean-ground-truth",
                "../safe-lua-groundtruth",
                "../reprorusted-c-cli",
                // Books/Cookbooks
                "../batuta-cookbook",
                "../apr-cookbook",
                "../sovereign-ai-book",
                "../sovereign-ai-stack-book",
                "../pmat-book",
                "../ruchy-book",
                "../ruchy-cli-tools-book",
                "../ruchy-cookbook",
                "../ruchy-repl-demos",
                "../ald-cookbook",
                "../prs-cookbook",
                "../rust-data-engineering",
                // Courses (content/docs)
                "../advanced-prompting-with-github-copilot",
                "../agentic-ai",
                "../ghcp-for-systems-level-development",
                "../GitHub-Copilot-Mastery-Capstone",
                "../responsible-ai-dev",
                "../windsurf",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            python_corpus_dirs: vec![
                "../hf-ground-truth-corpus",
                "../jax-ground-truth-corpus",
                "../vllm-ground-truth-corpus",
                "../algorithm-competition-corpus",
                "../databricks-ground-truth-corpus",
                "../ludwig-ground-truth-corpus",
                "../tiny-model-ground-truth",
                // Transpilation corpora
                "../reprorusted-python-cli",
                "../reprorusted-std-only",
                "../fully-typed-reprorusted-python-cli",
                // HuggingFace courses
                "../huggingface-fine-tuning",
                "../HF-Advanced-Fine-Tuning",
                "../llms-with-huggingface",
                // Databricks courses
                "../databricks-data-engineering",
                "../DB-mlops-genai",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            model_hash: [0u8; 32], // BM25 has no model weights
            private_dir_count: 0,
        };
        config.merge_private();
        config
    }

    /// Merge directories from `.batuta-private.toml` if it exists.
    fn merge_private(&mut self) {
        match crate::config::PrivateConfig::load_optional() {
            Ok(Some(private)) => {
                self.apply_private(&private);
            }
            Ok(None) => {} // No private config file — normal
            Err(e) => {
                eprintln!(
                    "  {} Failed to parse {}: {}",
                    "[warning]".bright_yellow(),
                    crate::config::PRIVATE_CONFIG_FILENAME,
                    e
                );
            }
        }
    }

    /// Pure merge: append private directories to existing lists.
    fn apply_private(&mut self, private: &crate::config::PrivateConfig) {
        let count = private.dir_count();
        if count == 0 {
            return;
        }
        self.rust_stack_dirs.extend(private.private.rust_stack_dirs.iter().cloned());
        self.rust_corpus_dirs.extend(private.private.rust_corpus_dirs.iter().cloned());
        self.python_corpus_dirs.extend(private.private.python_corpus_dirs.iter().cloned());
        self.private_dir_count = count;
    }
}

/// Run the indexing pipeline, dispatching to SQLite or JSON based on feature.
fn run_indexing(config: &IndexConfig, force: bool) -> anyhow::Result<()> {
    use oracle::rag::{
        fingerprint::DocumentFingerprint, persistence::RagPersistence, HeijunkaReindexer,
        SemanticChunker,
    };

    let persistence = RagPersistence::new();

    if force {
        println!("{}", "Force rebuild requested (old cache retained until save)...".dimmed());
    }

    eprint_phase("Checking fingerprints...");
    if !force
        && is_index_current(
            &persistence,
            &config.rust_stack_dirs,
            &config.rust_corpus_dirs,
            &config.python_corpus_dirs,
            &config.rust_chunker_config,
            &config.python_chunker_config,
            config.model_hash,
        )
    {
        return Ok(());
    }

    let rust_chunker = SemanticChunker::from_config(&config.rust_chunker_config);
    let python_chunker = SemanticChunker::from_config(&config.python_chunker_config);
    let mut reindexer = HeijunkaReindexer::new();
    let mut fingerprints: std::collections::HashMap<String, DocumentFingerprint> =
        std::collections::HashMap::new();
    let mut indexed_count = 0;
    let mut total_chunks = 0;
    let mut chunk_contents: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Dispatch to SQLite or JSON indexer
    #[cfg(feature = "rag")]
    {
        let db_path = sqlite_index_path();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if force {
            // Only delete DB on explicit force rebuild
            let _ = std::fs::remove_file(&db_path);
        }
        let sqlite_index = trueno_rag::sqlite::SqliteIndex::open(&db_path)
            .map_err(|e| anyhow::anyhow!("Failed to open SQLite index: {e}"))?;
        let mut sqlite_indexer = SqliteChunkIndexer::new(sqlite_index);

        run_index_phases(
            config,
            &rust_chunker,
            &python_chunker,
            &mut reindexer,
            &mut sqlite_indexer,
            &mut indexed_count,
            &mut total_chunks,
            &mut fingerprints,
            &mut chunk_contents,
        );

        eprint_phase("Saving to SQLite...");
        save_rag_index_sqlite(
            &sqlite_indexer,
            &reindexer,
            indexed_count,
            total_chunks,
            &fingerprints,
        )?;

        // Save fingerprints.json for fast is_index_current() checks
        eprint_phase("Saving fingerprints...");
        let _ = persistence.save_fingerprints_only(&fingerprints);

        // Clean up stale JSON files from pre-SQLite era
        cleanup_stale_json(&persistence);
    }

    #[cfg(not(feature = "rag"))]
    {
        let mut retriever = oracle::rag::HybridRetriever::new();

        run_index_phases(
            config,
            &rust_chunker,
            &python_chunker,
            &mut reindexer,
            &mut retriever,
            &mut indexed_count,
            &mut total_chunks,
            &mut fingerprints,
            &mut chunk_contents,
        );

        eprint_phase("Saving index...");
        save_rag_index_json(
            &persistence,
            retriever,
            reindexer,
            indexed_count,
            total_chunks,
            fingerprints,
            chunk_contents,
        )?;
    }

    Ok(())
}

/// Run all three indexing phases (Rust stack, Python corpora, Rust corpora).
#[allow(clippy::too_many_arguments)]
fn run_index_phases(
    config: &IndexConfig,
    rust_chunker: &oracle::rag::SemanticChunker,
    python_chunker: &oracle::rag::SemanticChunker,
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn crate::cli::oracle_indexing::ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    // Index Rust stack components
    eprint_phase("Indexing Rust stack...");
    println!("{}", "Scanning Rust stack repositories...".dimmed());
    println!();

    index_dir_group(
        &config.rust_stack_dirs,
        false,
        rust_chunker,
        &config.rust_chunker_config,
        config.model_hash,
        "rs",
        true,
        true,
        reindexer,
        indexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    );

    // Index Python ground truth corpora
    eprint_phase("Indexing Python corpora...");
    println!();
    println!("{}", "Scanning Python ground truth corpora...".dimmed());
    println!();

    index_dir_group(
        &config.python_corpus_dirs,
        true,
        python_chunker,
        &config.python_chunker_config,
        config.model_hash,
        "py",
        true,
        true,
        reindexer,
        indexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    );

    // Index Rust ground truth corpora
    eprint_phase("Indexing Rust corpora...");
    println!();
    println!("{}", "Scanning Rust ground truth corpora...".dimmed());
    println!();

    index_dir_group(
        &config.rust_corpus_dirs,
        true,
        rust_chunker,
        &config.rust_chunker_config,
        config.model_hash,
        "rs",
        true,
        true,
        reindexer,
        indexer,
        indexed_count,
        total_chunks,
        fingerprints,
        chunk_contents,
    );
}

/// Index stack documentation for RAG
pub fn cmd_oracle_rag_index(force: bool) -> anyhow::Result<()> {
    println!("{}", "RAG Indexer (Heijunka Mode)".bright_cyan().bold());
    #[cfg(feature = "rag")]
    println!("{}", "(SQLite+FTS5 backend)".dimmed());
    #[cfg(not(feature = "rag"))]
    println!("{}", "(JSON fallback backend)".dimmed());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    let config = IndexConfig::new();

    if config.private_dir_count > 0 {
        println!(
            "{}: {} private directories merged from {}",
            "Private".bright_cyan(),
            config.private_dir_count,
            crate::config::PRIVATE_CONFIG_FILENAME
        );
        println!();
    }

    run_indexing(&config, force)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_config_apply_private_adds_dirs() {
        let mut config = IndexConfig::new();
        let original_rust_stack = config.rust_stack_dirs.len();
        let original_rust_corpus = config.rust_corpus_dirs.len();
        let original_python = config.python_corpus_dirs.len();

        let private = crate::config::PrivateConfig {
            private: crate::config::PrivateExtensions {
                rust_stack_dirs: vec!["../rmedia".to_string(), "../infra".to_string()],
                rust_corpus_dirs: vec!["../internal-cookbook".to_string()],
                python_corpus_dirs: vec!["../private-notebooks".to_string()],
                endpoints: vec![],
            },
        };

        config.apply_private(&private);

        assert_eq!(config.rust_stack_dirs.len(), original_rust_stack + 2);
        assert_eq!(config.rust_corpus_dirs.len(), original_rust_corpus + 1);
        assert_eq!(config.python_corpus_dirs.len(), original_python + 1);
        assert_eq!(config.private_dir_count, 4);
        assert!(config.rust_stack_dirs.contains(&"../rmedia".to_string()));
        assert!(config.rust_stack_dirs.contains(&"../infra".to_string()));
    }

    #[test]
    fn test_index_config_apply_private_empty_is_noop() {
        let mut config = IndexConfig::new();
        let original_rust_stack = config.rust_stack_dirs.len();
        let original_rust_corpus = config.rust_corpus_dirs.len();
        let original_python = config.python_corpus_dirs.len();
        let original_private_count = config.private_dir_count;

        let private = crate::config::PrivateConfig::default();
        config.apply_private(&private);

        assert_eq!(config.rust_stack_dirs.len(), original_rust_stack);
        assert_eq!(config.rust_corpus_dirs.len(), original_rust_corpus);
        assert_eq!(config.python_corpus_dirs.len(), original_python);
        assert_eq!(config.private_dir_count, original_private_count);
    }
}
