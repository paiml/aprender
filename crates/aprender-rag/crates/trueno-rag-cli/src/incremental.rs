//! Incremental indexing: only re-process changed/new files, remove deleted files.
//!
//! Uses SQLite fingerprints table (blake3 hash per source file) to detect
//! changed, new, and deleted files. Only the delta is re-indexed, making
//! subsequent runs much faster than a full re-index.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use crate::{ChunkStrategy, EmbedderType, SemanticModel};

#[cfg(feature = "sqlite")]
use {
    crate::discover::{build_exclude_set, discover_files},
    crate::ingest::{chunk_and_embed, create_embedder, finish_load_report, load_documents},
    crate::PersistedChunk,
    std::path::Path,
    trueno_rag::{
        chunk::{RecursiveChunker, TimestampChunker},
        loader::LoaderRegistry,
    },
};

/// Entry point for incremental indexing. Delegates to the feature-gated inner implementation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_index_incremental(
    path: &str,
    output: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    dimension: usize,
    embedder_type: EmbedderType,
    model: SemanticModel,
    recursive: bool,
    chunk_strategy: ChunkStrategy,
    jobs: usize,
    exclude_patterns: &[String],
    dedup: bool,
) -> Result<()> {
    run_index_incremental_inner(
        path,
        output,
        chunk_size,
        chunk_overlap,
        dimension,
        embedder_type,
        model,
        recursive,
        chunk_strategy,
        jobs,
        exclude_patterns,
        dedup,
    )
}

/// Compute blake3 hashes for a list of files.
#[allow(dead_code)] // Used by sqlite-gated code and tests
pub(crate) fn compute_file_hashes(files: &[PathBuf]) -> Result<Vec<(PathBuf, [u8; 32])>> {
    let mut results = Vec::with_capacity(files.len());
    for file in files {
        let data = fs::read(file).with_context(|| format!("Failed to read {}", file.display()))?;
        let hash: [u8; 32] = *blake3::hash(&data).as_bytes();
        results.push((file.clone(), hash));
    }
    Ok(results)
}

/// Diff current file hashes against stored fingerprints.
///
/// Returns `(changed_or_new, deleted_paths)`. A file is "changed" if its blake3
/// hash differs from the stored fingerprint. A file is "new" if it has no stored
/// fingerprint. A file is "deleted" if its stored fingerprint has no corresponding
/// file on disk.
#[allow(dead_code)] // Used by sqlite-gated code and tests
pub(crate) fn diff_fingerprints(
    current: &[(PathBuf, [u8; 32])],
    stored: &HashMap<String, Vec<u8>>,
) -> (Vec<(PathBuf, [u8; 32])>, Vec<String>) {
    let mut changed = Vec::new();
    let mut current_paths: HashSet<String> = HashSet::new();

    for (path, hash) in current {
        let path_str = path.to_string_lossy().to_string();
        current_paths.insert(path_str.clone());

        match stored.get(&path_str) {
            Some(stored_hash) if stored_hash.as_slice() == hash.as_slice() => {
                // Unchanged -- skip
            }
            _ => {
                // New or changed
                changed.push((path.clone(), *hash));
            }
        }
    }

    let deleted: Vec<String> =
        stored.keys().filter(|k| !current_paths.contains(k.as_str())).cloned().collect();

    (changed, deleted)
}

// ── SQLite-gated implementation ────────────────────────────────────────

/// Full incremental indexing pipeline (requires `sqlite` feature).
///
/// Steps:
/// 1. Open existing SQLite index
/// 2. Discover files on disk, compute blake3 hashes
/// 3. Diff against stored fingerprints to find changed/new/deleted
/// 4. Remove deleted sources from index
/// 5. Load, chunk, and embed changed files
/// 6. Insert new chunks with updated fingerprints
#[cfg(feature = "sqlite")]
#[allow(clippy::too_many_arguments)]
fn run_index_incremental_inner(
    path: &str,
    output: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    dimension: usize,
    embedder_type: EmbedderType,
    #[allow(unused_variables)] model: SemanticModel,
    recursive: bool,
    chunk_strategy: ChunkStrategy,
    jobs: usize,
    exclude_patterns: &[String],
    dedup: bool,
) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        anyhow::bail!("Path not found: {}", path.display());
    }

    let sqlite_index = open_existing_index(output)?;

    // Discover and diff
    let exclude = build_exclude_set(exclude_patterns)?;
    let registry = LoaderRegistry::new();
    let files = discover_files(path, recursive, &registry, &exclude)?;
    println!("Found {} files on disk", files.len());

    let file_hashes = compute_file_hashes(&files)?;
    let (changed, deleted) = diff_against_stored(&sqlite_index, &file_hashes)?;

    if changed.is_empty() && deleted.is_empty() {
        println!("No changes detected — index is up to date.");
        return Ok(());
    }

    println!("{} files changed/new, {} files deleted", changed.len(), deleted.len());

    // Remove deleted files
    remove_deleted_sources(&sqlite_index, &deleted)?;

    if changed.is_empty() {
        println!("Only deletions — no re-indexing needed.");
        sqlite_index.optimize().map_err(|e| anyhow::anyhow!("Failed to optimize: {e}"))?;
        return Ok(());
    }

    // Load, chunk, embed changed files
    let changed_paths: Vec<PathBuf> = changed.iter().map(|(p, _)| p.clone()).collect();
    let documents = load_documents(&changed_paths, &registry, jobs)?;
    let documents = finish_load_report(documents, 0)?;

    let (embedder_box, _dim, _name, _model_name) =
        create_embedder(embedder_type, dimension, model, &documents)?;

    let recursive_chunker = RecursiveChunker::new(chunk_size, chunk_overlap);
    let timestamp_chunker = TimestampChunker::new(60.0);
    let (all_chunks, _embeddings) = chunk_and_embed(
        &documents,
        &*embedder_box,
        &recursive_chunker,
        &timestamp_chunker,
        chunk_strategy,
        dedup,
    )?;

    // Insert changed documents into SQLite
    incremental_insert(&sqlite_index, &all_chunks, &changed)?;
    optimize_and_report(&sqlite_index)?;

    Ok(())
}

/// Open an existing SQLite index, failing if it does not exist.
#[cfg(feature = "sqlite")]
fn open_existing_index(output: &str) -> Result<trueno_rag::SqliteIndex> {
    let db_path = Path::new(output).join("index.sqlite");
    if !db_path.exists() {
        anyhow::bail!(
            "No existing SQLite index at {}. Run a full index first (without --incremental).",
            db_path.display()
        );
    }
    trueno_rag::SqliteIndex::open(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to open SQLite index: {e}"))
}

/// Compare file hashes against stored fingerprints in the SQLite index.
#[cfg(feature = "sqlite")]
fn diff_against_stored(
    sqlite_index: &trueno_rag::SqliteIndex,
    file_hashes: &[(PathBuf, [u8; 32])],
) -> Result<(Vec<(PathBuf, [u8; 32])>, Vec<String>)> {
    let stored_fps = sqlite_index
        .list_fingerprints()
        .map_err(|e| anyhow::anyhow!("Failed to list fingerprints: {e}"))?;
    let stored_map: HashMap<String, Vec<u8>> = stored_fps.into_iter().collect();
    Ok(diff_fingerprints(file_hashes, &stored_map))
}

/// Remove deleted sources from the SQLite index, printing each removal.
#[cfg(feature = "sqlite")]
fn remove_deleted_sources(
    sqlite_index: &trueno_rag::SqliteIndex,
    deleted: &[String],
) -> Result<()> {
    for del_path in deleted {
        let removed = sqlite_index
            .remove_by_source(del_path)
            .map_err(|e| anyhow::anyhow!("Failed to remove {del_path}: {e}"))?;
        if removed > 0 {
            println!("  Removed: {} ({} docs)", del_path, removed);
        }
    }
    Ok(())
}

/// Optimize the SQLite index and print final document/chunk counts.
#[cfg(feature = "sqlite")]
fn optimize_and_report(sqlite_index: &trueno_rag::SqliteIndex) -> Result<()> {
    sqlite_index.optimize().map_err(|e| anyhow::anyhow!("Failed to optimize: {e}"))?;
    let stats =
        sqlite_index.document_count().map_err(|e| anyhow::anyhow!("Failed to count docs: {e}"))?;
    let chunk_count =
        sqlite_index.chunk_count().map_err(|e| anyhow::anyhow!("Failed to count chunks: {e}"))?;
    println!("Incremental update complete: {} docs, {} chunks total", stats, chunk_count);
    Ok(())
}

/// Insert changed documents into SQLite with fingerprints.
#[cfg(feature = "sqlite")]
fn incremental_insert(
    sqlite_index: &trueno_rag::SqliteIndex,
    chunks: &[PersistedChunk],
    changed: &[(PathBuf, [u8; 32])],
) -> Result<()> {
    use std::collections::BTreeMap;

    // Build a hash lookup for changed files
    let hash_map: HashMap<String, [u8; 32]> =
        changed.iter().map(|(p, h)| (p.to_string_lossy().to_string(), *h)).collect();

    // Group chunks by source
    let mut doc_chunks: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut doc_titles: HashMap<String, Option<String>> = HashMap::new();

    for (i, pc) in chunks.iter().enumerate() {
        let doc_id = pc.source.as_deref().unwrap_or("unknown").to_string();
        let chunk_id = format!("{}#{}", doc_id, i);
        doc_chunks.entry(doc_id.clone()).or_default().push((chunk_id, pc.content.clone()));
        doc_titles.entry(doc_id).or_insert_with(|| pc.title.clone());
    }

    for (doc_id, chunk_pairs) in &doc_chunks {
        let title = doc_titles.get(doc_id).and_then(|t| t.as_deref());
        let content: String =
            chunk_pairs.iter().map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n");

        let fingerprint = hash_map.get(doc_id).map(|h| (doc_id.as_str(), h));

        sqlite_index
            .insert_document(
                doc_id,
                title,
                Some(doc_id.as_str()),
                &content,
                chunk_pairs,
                fingerprint,
            )
            .map_err(|e| anyhow::anyhow!("Failed to insert document {doc_id}: {e}"))?;

        println!("  Updated: {} ({} chunks)", doc_id, chunk_pairs.len());
    }

    Ok(())
}

/// Stub when sqlite feature is not enabled.
#[cfg(not(feature = "sqlite"))]
#[allow(clippy::too_many_arguments)]
fn run_index_incremental_inner(
    _path: &str,
    _output: &str,
    _chunk_size: usize,
    _chunk_overlap: usize,
    _dimension: usize,
    _embedder_type: EmbedderType,
    _model: SemanticModel,
    _recursive: bool,
    _chunk_strategy: ChunkStrategy,
    _jobs: usize,
    _exclude_patterns: &[String],
    _dedup: bool,
) -> Result<()> {
    anyhow::bail!(
        "Incremental indexing requires the 'sqlite' feature.\n\
         Build with: cargo build --features sqlite"
    );
}
