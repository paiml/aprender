//! Document loading, chunking, embedding, and index persistence.
//!
//! Covers the full indexing pipeline: discover files, load documents,
//! chunk content, compute embeddings, save to JSON and/or SQLite.

use anyhow::{Context, Result};
use globset::GlobSet;
use rayon::prelude::*;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use trueno_rag::{
    chunk::{RecursiveChunker, TimestampChunker},
    embed::{Embedder, TfIdfEmbedder},
    loader::LoaderRegistry,
    Chunk, Chunker, Document,
};

#[cfg(feature = "embeddings")]
use trueno_rag::{EmbeddingModelType, FastEmbedder};

use crate::discover::{build_exclude_set, classify_files, discover_files};
use crate::{ChunkStrategy, EmbedderType, PersistedChunk, PersistedIndex, SemanticModel};

/// Load documents from discovered files, reporting progress and errors.
/// Uses rayon parallel loading when `jobs` > 1.
pub(crate) fn load_documents(
    files: &[PathBuf],
    registry: &LoaderRegistry,
    jobs: usize,
) -> Result<Vec<Document>> {
    if jobs > 1 {
        load_documents_parallel(files, registry, jobs)
    } else {
        load_documents_sequential(files, registry)
    }
}

pub(crate) fn load_documents_sequential(
    files: &[PathBuf],
    registry: &LoaderRegistry,
) -> Result<Vec<Document>> {
    let mut documents = Vec::new();
    let mut load_errors = 0usize;

    for (i, file) in files.iter().enumerate() {
        match registry.load(file) {
            Ok(doc) => documents.push(doc),
            Err(e) => {
                eprintln!("  Warning: failed to load {}: {}", file.display(), e);
                load_errors += 1;
            }
        }
        if (i + 1) % 100 == 0 {
            println!("  Loaded {}/{} files...", i + 1, files.len());
        }
    }

    finish_load_report(documents, load_errors)
}

fn load_documents_parallel(
    files: &[PathBuf],
    registry: &LoaderRegistry,
    jobs: usize,
) -> Result<Vec<Document>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .context("Failed to create thread pool")?;

    let documents = Mutex::new(Vec::new());
    let load_errors = Mutex::new(0usize);

    pool.install(|| {
        files.par_iter().for_each(|file| match registry.load(file) {
            Ok(doc) => {
                documents.lock().expect("documents mutex poisoned").push(doc);
            }
            Err(e) => {
                eprintln!("  Warning: failed to load {}: {}", file.display(), e);
                *load_errors.lock().expect("load_errors mutex poisoned") += 1;
            }
        });
    });

    let documents = documents.into_inner().expect("documents mutex poisoned");
    let load_errors = load_errors.into_inner().expect("load_errors mutex poisoned");
    finish_load_report(documents, load_errors)
}

pub(crate) fn finish_load_report(
    documents: Vec<Document>,
    load_errors: usize,
) -> Result<Vec<Document>> {
    if documents.is_empty() {
        anyhow::bail!("All files failed to load ({} errors)", load_errors);
    }

    if load_errors > 0 {
        println!("Loaded {} documents ({} failed)", documents.len(), load_errors);
    } else {
        println!("Loaded {} documents", documents.len());
    }

    Ok(documents)
}

/// Select the appropriate chunker for a document based on strategy.
fn chunk_document(
    doc: &Document,
    strategy: ChunkStrategy,
    recursive_chunker: &RecursiveChunker,
    timestamp_chunker: &TimestampChunker,
) -> Result<Vec<Chunk>> {
    let use_timestamps = match strategy {
        ChunkStrategy::Timestamp => true,
        ChunkStrategy::Recursive => false,
        ChunkStrategy::Auto => doc.metadata.contains_key("subtitle_cues"),
    };
    if use_timestamps {
        Ok(timestamp_chunker.chunk(doc)?)
    } else {
        Ok(recursive_chunker.chunk(doc)?)
    }
}

/// Convert a raw Chunk into a PersistedChunk, carrying forward document source.
fn to_persisted_chunk(chunk: &Chunk, doc: &Document) -> PersistedChunk {
    PersistedChunk {
        content: chunk.content.clone(),
        title: chunk.metadata.title.clone(),
        source: doc.source.clone(),
        start_secs: chunk.metadata.custom.get("start_secs").and_then(serde_json::Value::as_f64),
        end_secs: chunk.metadata.custom.get("end_secs").and_then(serde_json::Value::as_f64),
    }
}

/// Check if a chunk is a duplicate based on content hash. Returns true if novel.
fn is_novel_chunk(seen: &mut HashSet<u64>, content: &str) -> bool {
    let mut hasher = std::hash::DefaultHasher::new();
    content.hash(&mut hasher);
    seen.insert(hasher.finish())
}

/// Embed and collect chunks from a single document, deduplicating if enabled.
fn embed_doc_chunks(
    doc: &Document,
    chunks: Vec<Chunk>,
    embedder: &dyn Embedder,
    dedup: bool,
    seen: &mut HashSet<u64>,
    all_chunks: &mut Vec<PersistedChunk>,
    all_embeddings: &mut Vec<Vec<f32>>,
) -> Result<usize> {
    let mut dedup_count = 0usize;
    for chunk in chunks {
        if dedup && !is_novel_chunk(seen, &chunk.content) {
            dedup_count += 1;
            continue;
        }
        all_embeddings.push(embedder.embed(&chunk.content)?);
        all_chunks.push(to_persisted_chunk(&chunk, doc));
    }
    Ok(dedup_count)
}

/// Chunk documents and compute embeddings, returning parallel vectors.
pub(crate) fn chunk_and_embed(
    documents: &[Document],
    embedder: &dyn Embedder,
    recursive_chunker: &RecursiveChunker,
    timestamp_chunker: &TimestampChunker,
    strategy: ChunkStrategy,
    dedup: bool,
) -> Result<(Vec<PersistedChunk>, Vec<Vec<f32>>)> {
    let mut all_chunks = Vec::new();
    let mut all_embeddings = Vec::new();

    let mut seen: HashSet<u64> = HashSet::new();
    let mut dedup_count = 0usize;
    let mut skipped_empty = 0usize;
    for doc in documents {
        if doc.content.is_empty() {
            skipped_empty += 1;
            continue;
        }
        let chunks = chunk_document(doc, strategy, recursive_chunker, timestamp_chunker)?;
        dedup_count += embed_doc_chunks(
            doc,
            chunks,
            embedder,
            dedup,
            &mut seen,
            &mut all_chunks,
            &mut all_embeddings,
        )?;
    }

    if skipped_empty > 0 {
        println!("Skipped {} empty documents", skipped_empty);
    }
    if dedup && dedup_count > 0 {
        println!("Deduplicated: removed {} duplicate chunks", dedup_count);
    }

    Ok((all_chunks, all_embeddings))
}

/// Discover files and load documents, reporting progress.
pub(crate) fn discover_and_load(
    path: &Path,
    recursive: bool,
    jobs: usize,
    exclude: &Option<GlobSet>,
) -> Result<(Vec<PathBuf>, HashMap<String, usize>, Vec<Document>)> {
    let registry = LoaderRegistry::new();
    let files = discover_files(path, recursive, &registry, exclude)?;

    if files.is_empty() {
        let exts = registry.supported_extensions().join(", ");
        anyhow::bail!("No supported files found at: {} (supported: {})", path.display(), exts);
    }

    let classification = classify_files(&files);
    println!(
        "Scanning {}{}... found {} files",
        path.display(),
        if recursive { " (recursive)" } else { "" },
        files.len()
    );
    for (ext, count) in &classification {
        println!("  {} .{} files", count, ext);
    }

    if jobs > 1 {
        println!("Loading with {} parallel jobs", jobs);
    }
    let documents = load_documents(&files, &registry, jobs)?;
    report_media_text_split(&documents);
    Ok((files, classification, documents))
}

/// Print how many documents have timestamp metadata vs plain text.
pub(crate) fn report_media_text_split(documents: &[Document]) {
    let media_count = documents.iter().filter(|d| d.metadata.contains_key("subtitle_cues")).count();
    if media_count > 0 {
        let text_count = documents.len() - media_count;
        println!("  {} with timestamps, {} plain text", media_count, text_count);
    }
}

/// Create an embedder based on the selected type and return it with metadata.
pub(crate) fn create_embedder(
    embedder_type: EmbedderType,
    dimension: usize,
    #[allow(unused_variables)] model: SemanticModel,
    documents: &[Document],
) -> Result<(Box<dyn Embedder>, usize, String, Option<String>)> {
    match embedder_type {
        EmbedderType::Tfidf => {
            let mut embedder = TfIdfEmbedder::new(dimension);
            let doc_texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
            embedder.fit(&doc_texts);
            println!("Using TF-IDF embedder (dimension: {})", dimension);
            Ok((Box::new(embedder), dimension, "tfidf".to_string(), None))
        }
        EmbedderType::Semantic => {
            #[cfg(feature = "embeddings")]
            {
                let model_type = match model {
                    SemanticModel::MiniLm => EmbeddingModelType::AllMiniLmL6V2,
                    SemanticModel::BgeSmall => EmbeddingModelType::BgeSmallEnV15,
                    SemanticModel::BgeBase => EmbeddingModelType::BgeBaseEnV15,
                };
                println!(
                    "Loading semantic model: {} (dimension: {})",
                    model_type.model_name(),
                    model_type.dimension()
                );
                let embedder = FastEmbedder::new(model_type)
                    .context("Failed to initialize semantic embedder")?;
                let dim = embedder.dimension();
                let name = model_type.model_name().to_string();
                Ok((Box::new(embedder), dim, "semantic".to_string(), Some(name)))
            }
            #[cfg(not(feature = "embeddings"))]
            {
                anyhow::bail!(
                    "Semantic embeddings require the 'embeddings' feature.\n\
                     Build with: cargo build --features embeddings"
                );
            }
        }
    }
}

/// Save a persisted index to disk, optionally writing a manifest and/or SQLite export.
pub(crate) fn save_index(
    persisted: &PersistedIndex,
    output: &str,
    manifest: bool,
    files: &[PathBuf],
    classification: &HashMap<String, usize>,
    sqlite: bool,
) -> Result<()> {
    let output_path = Path::new(output);
    fs::create_dir_all(output_path)?;

    let index_file = output_path.join("index.json");
    let json = serde_json::to_string_pretty(persisted)?;
    fs::write(&index_file, json)?;
    println!("Index saved to: {}", index_file.display());

    if manifest {
        let manifest_data = build_index_manifest(files, classification, &persisted.chunks);
        let manifest_file = output_path.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest_data)?;
        fs::write(&manifest_file, manifest_json)?;
        println!("Manifest saved to: {}", manifest_file.display());
    }

    if sqlite {
        export_sqlite(persisted, output_path)?;
    }

    Ok(())
}

/// Export a `PersistedIndex` to a `SqliteIndex` for FTS5 BM25 search.
#[cfg(feature = "sqlite")]
pub(crate) fn export_sqlite(persisted: &PersistedIndex, output_path: &Path) -> Result<()> {
    use std::collections::BTreeMap;

    let db_path = output_path.join("index.sqlite");
    // Remove stale DB so we get a clean export
    if db_path.exists() {
        fs::remove_file(&db_path)?;
    }
    let sqlite_index = trueno_rag::SqliteIndex::open(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to create SQLite index: {e}"))?;

    // Group chunks by source document
    let mut doc_chunks: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut doc_titles: HashMap<String, Option<String>> = HashMap::new();
    for (i, pc) in persisted.chunks.iter().enumerate() {
        let doc_id = pc.source.as_deref().unwrap_or("unknown").to_string();
        let chunk_id = format!("{}#{}", doc_id, i);
        doc_chunks.entry(doc_id.clone()).or_default().push((chunk_id, pc.content.clone()));
        doc_titles.entry(doc_id).or_insert_with(|| pc.title.clone());
    }

    let doc_count = doc_chunks.len();
    let chunk_count = persisted.chunks.len();

    for (doc_id, chunks) in &doc_chunks {
        let title = doc_titles.get(doc_id).and_then(|t| t.as_deref());
        // Concatenate all chunk content as the document-level content
        let content: String = chunks.iter().map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n");
        sqlite_index
            .insert_document(doc_id, title, Some(doc_id.as_str()), &content, chunks, None)
            .map_err(|e| anyhow::anyhow!("Failed to insert document {doc_id}: {e}"))?;
    }

    sqlite_index.optimize().map_err(|e| anyhow::anyhow!("Failed to optimize SQLite index: {e}"))?;

    println!(
        "SQLite index saved to: {} ({} docs, {} chunks)",
        db_path.display(),
        doc_count,
        chunk_count,
    );
    Ok(())
}

/// Stub when sqlite feature is not enabled.
#[cfg(not(feature = "sqlite"))]
pub(crate) fn export_sqlite(_persisted: &PersistedIndex, _output_path: &Path) -> Result<()> {
    anyhow::bail!(
        "SQLite export requires the 'sqlite' feature.\n\
         Build with: cargo build --features sqlite"
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_index(
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
    manifest: bool,
    exclude_patterns: &[String],
    dedup: bool,
    sqlite: bool,
) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        anyhow::bail!("Path not found: {}", path.display());
    }

    let exclude = build_exclude_set(exclude_patterns)?;
    let (files, classification, documents) = discover_and_load(path, recursive, jobs, &exclude)?;
    let (embedder_box, actual_dimension, embedder_name, model_name) =
        create_embedder(embedder_type, dimension, model, &documents)?;

    let recursive_chunker = RecursiveChunker::new(chunk_size, chunk_overlap);
    let timestamp_chunker = TimestampChunker::new(60.0);
    let (all_chunks, all_embeddings) = chunk_and_embed(
        &documents,
        &*embedder_box,
        &recursive_chunker,
        &timestamp_chunker,
        chunk_strategy,
        dedup,
    )?;

    println!("Indexed {} documents ({} chunks)", documents.len(), all_chunks.len());

    let persisted = PersistedIndex {
        chunks: all_chunks,
        embeddings: all_embeddings,
        dimension: actual_dimension,
        embedder_type: embedder_name,
        model_name,
    };

    save_index(&persisted, output, manifest, &files, &classification, sqlite)
}

/// Build a JSON manifest of indexed files and chunks.
pub(crate) fn build_index_manifest(
    files: &[PathBuf],
    classification: &HashMap<String, usize>,
    chunks: &[PersistedChunk],
) -> serde_json::Value {
    let file_entries: Vec<serde_json::Value> = files
        .iter()
        .map(|f| {
            let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
            serde_json::json!({
                "path": f.to_string_lossy(),
                "extension": ext,
            })
        })
        .collect();

    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "file_count": files.len(),
        "chunk_count": chunks.len(),
        "format_counts": classification,
        "files": file_entries,
    })
}
