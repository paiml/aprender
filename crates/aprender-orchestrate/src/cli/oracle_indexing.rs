//! RAG Indexing Functions
//!
//! This module contains file indexing helpers for the RAG system.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::oracle;

/// Trait for indexing backends that accept individual chunks.
///
/// Abstracts over `HybridRetriever` (in-memory BM25+TF-IDF) and
/// SQLite-backed indexing (FTS5 BM25).
pub(crate) trait ChunkIndexer {
    /// Index a single chunk identified by `chunk_id` with `content`.
    fn index_chunk(&mut self, chunk_id: &str, content: &str);
}

impl ChunkIndexer for oracle::rag::HybridRetriever {
    fn index_chunk(&mut self, chunk_id: &str, content: &str) {
        self.index_document(chunk_id, content);
    }
}

/// Check if a directory name should be skipped during indexing (all languages)
fn should_skip_directory(name: &str) -> bool {
    name.starts_with('.') || name == "target" || name == "__pycache__"
}

/// Check if a directory should be skipped for Python indexing
fn should_skip_python_dir(name: &str) -> bool {
    name.starts_with('.') || name == "__pycache__"
}

/// Check if a directory should be skipped for Rust indexing
fn should_skip_rust_dir(name: &str) -> bool {
    name.starts_with('.') || name == "target"
}

/// Check if a directory should be skipped for markdown (hidden only)
fn should_skip_hidden_dir(name: &str) -> bool {
    name.starts_with('.')
}

/// Check if a directory should be recursed into for indexing
fn should_recurse_dir(path: &std::path::Path, skip_fn: fn(&str) -> bool) -> bool {
    path.file_name().and_then(|n| n.to_str()).is_some_and(|name| !skip_fn(name))
}

/// Index chunks from a file and update indexer/counters, returns chunk count.
///
/// Stores full chunk content in `chunk_contents` for the JSON fallback path.
/// When the `rag` feature is enabled (SQLite backend), chunk content is stored
/// in the database directly, so `chunk_contents` population is skipped to avoid
/// cloning 389K strings (~200MB) into a HashMap that is never read.
#[allow(clippy::too_many_arguments)]
fn index_file_chunks(
    content: &str,
    doc_id: &str,
    chunker: &oracle::rag::SemanticChunker,
    indexer: &mut dyn ChunkIndexer,
    total_chunks: &mut usize,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) -> usize {
    let chunks = chunker.split(content);
    let chunk_count = chunks.len();
    for chunk in &chunks {
        let chunk_id = format!("{}#{}", doc_id, chunk.start_line);
        // Only populate chunk_contents for JSON fallback path; SQLite stores
        // content in the DB directly, so this clone is unnecessary overhead.
        #[cfg(not(feature = "rag"))]
        chunk_contents.insert(chunk_id.clone(), chunk.content.clone());
        #[cfg(feature = "rag")]
        let _ = &chunk_contents;
        indexer.index_chunk(&chunk_id, &chunk.content);
        *total_chunks += 1;
    }
    chunk_count
}

/// Print progress for indexed file (only if > 1 chunk)
fn print_file_indexed(relative_path: &std::path::Path, chunks_len: usize) {
    use oracle::rag::tui::inline;

    if chunks_len > 1 {
        let bar = inline::bar(chunks_len as f64, 20.0, 15);
        println!(
            "  {} {:40} {} ({} chunks)",
            "✓".bright_green(),
            relative_path.display().to_string().cyan(),
            bar,
            chunks_len
        );
    }
}

/// Print progress for indexed markdown doc (always shows)
fn print_markdown_indexed(display_path: &str, chunks_len: usize) {
    use oracle::rag::tui::inline;

    let bar = inline::bar(chunks_len as f64, 20.0, 15);
    println!("  {} {:40} {} ({} chunks)", "✓".bright_green(), display_path.cyan(), bar, chunks_len);
}

/// Check if file content is trivial (empty or too short)
fn is_trivial_content(content: &str) -> bool {
    content.trim().is_empty() || content.lines().count() < 5
}

/// Index a single named document (e.g., CLAUDE.md, README.md) with fingerprint + print.
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_doc_file(
    file_path: &std::path::Path,
    doc_id: &str,
    display_name: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return;
    };

    fingerprints.insert(
        doc_id.to_string(),
        oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash),
    );
    reindexer.enqueue(doc_id, file_path.to_path_buf(), 0);

    let chunk_count =
        index_file_chunks(&content, doc_id, chunker, indexer, total_chunks, chunk_contents);
    *indexed_count += 1;

    print_markdown_indexed(display_name, chunk_count);
}

/// Check if a single document's fingerprint has changed compared to stored fingerprints.
///
/// Uses mtime pre-filter: if file mtime < stored indexed_at, skip the expensive
/// content read + hash (O(stat) instead of O(file_read)).
pub(crate) fn doc_fingerprint_changed(
    file_path: &std::path::Path,
    doc_id: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> bool {
    // mtime pre-filter: skip content read if file hasn't been modified since last index
    if let Some(stored_fp) = existing_fingerprints.get(doc_id) {
        if let Ok(meta) = std::fs::metadata(file_path) {
            if let Ok(mtime) = meta.modified() {
                let mtime_ms = mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                if mtime_ms < stored_fp.indexed_at {
                    return false;
                }
            }
        }
    }

    let Ok(content) = std::fs::read_to_string(file_path) else {
        return false;
    };
    let current_fp =
        oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash);
    match existing_fingerprints.get(doc_id) {
        Some(stored_fp) if !stored_fp.needs_reindex(&current_fp) => false,
        Some(_) => true,
        None => true, // New file
    }
}

/// Index all files for a single component directory (CLAUDE.md, README.md, src/, specs/, book/).
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_component(
    path: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    extension: &str,
    include_specs: bool,
    include_book: bool,
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    // Index all root-level .md files (CLAUDE.md, README.md, CONTRIBUTING.md, etc.)
    if let Ok(entries) = std::fs::read_dir(path) {
        let mut md_files: Vec<_> = entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md") && e.path().is_file())
            .collect();
        // Sort for deterministic indexing order (CLAUDE.md first if present)
        md_files.sort_by_key(|e| e.file_name());
        for entry in md_files {
            let md_path = entry.path();
            let Some(fname) = md_path.file_name() else { continue };
            let file_name = fname.to_string_lossy();
            let doc_id = format!("{}/{}", component, file_name);
            index_doc_file(
                &md_path,
                &doc_id,
                &doc_id,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }

    // Index source files (P2)
    let src_dir = path.join("src");
    let (scan_dir, base) = if src_dir.exists() {
        (src_dir.clone(), src_dir.parent().unwrap_or(&src_dir).to_path_buf())
    } else if extension == "py" {
        // Python packages often use flat layouts (e.g. databricks/, mypackage/)
        // instead of src/. Fall back to scanning the root directory.
        (path.to_path_buf(), path.to_path_buf())
    } else {
        // Rust always uses src/
        (src_dir.clone(), path.to_path_buf())
    };
    if scan_dir.exists() {
        match extension {
            "rs" => index_rust_files(
                &scan_dir,
                &base,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            ),
            "py" => index_python_files(
                &scan_dir,
                &base,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            ),
            _ => {}
        }
    }

    // Index specification docs (P1)
    if include_specs {
        let specs_dir = path.join("docs/specifications");
        if specs_dir.exists() {
            index_markdown_files(
                &specs_dir,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }

    // Index mdBook documentation (P1)
    if include_book {
        let book_dir = path.join("book/src");
        if book_dir.exists() {
            index_markdown_files_recursive(
                &book_dir,
                book_dir.parent().unwrap_or(&book_dir),
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }
}

/// Index all components in a list of directories, optionally printing "not found" for missing dirs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_dir_group(
    dirs: &[String],
    show_not_found: bool,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    extension: &str,
    include_specs: bool,
    include_book: bool,
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    for dir in dirs {
        let path = std::path::Path::new(dir);
        if !path.exists() {
            if show_not_found {
                println!("  {} {} (not found)", "⊘".dimmed(), dir.dimmed());
            }
            continue;
        }
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let component = canonical.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        index_component(
            path,
            component,
            chunker,
            chunker_config,
            model_hash,
            extension,
            include_specs,
            include_book,
            reindexer,
            indexer,
            indexed_count,
            total_chunks,
            fingerprints,
            chunk_contents,
        );
    }
}

/// Process a single source file for RAG indexing: read, fingerprint, chunk, and print.
#[allow(clippy::too_many_arguments)]
fn process_and_index_file(
    path: &std::path::Path,
    base_dir: &std::path::Path,
    doc_id_prefix: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    if is_trivial_content(&content) {
        return;
    }

    let relative_path = path.strip_prefix(base_dir).unwrap_or(path);
    let doc_id = format!("{}/{}", doc_id_prefix, relative_path.display());

    fingerprints.insert(
        doc_id.clone(),
        oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash),
    );

    reindexer.enqueue(&doc_id, path.to_path_buf(), 0);

    let chunk_count =
        index_file_chunks(&content, &doc_id, chunker, indexer, total_chunks, chunk_contents);
    *indexed_count += 1;

    print_file_indexed(relative_path, chunk_count);
}

/// Check if a single file has changed compared to existing fingerprints.
///
/// Uses mtime pre-filter: if file mtime < stored indexed_at, skip the expensive
/// content read + hash (O(stat) instead of O(file_read)).
fn check_file_changed(
    path: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
) -> Option<bool> {
    use oracle::rag::fingerprint::DocumentFingerprint;

    // mtime pre-filter: skip content read if file hasn't been modified since last index
    let relative_path = path.strip_prefix(base_dir).unwrap_or(path);
    let doc_id = format!("{}/{}", component, relative_path.display());
    if let Some(stored_fp) = existing_fingerprints.get(&doc_id) {
        if let Ok(meta) = std::fs::metadata(path) {
            if let Ok(mtime) = meta.modified() {
                let mtime_ms = mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                if mtime_ms < stored_fp.indexed_at {
                    return Some(false);
                }
            }
        }
    }

    let content = std::fs::read_to_string(path).ok()?;
    if is_trivial_content(&content) {
        return Some(false);
    }

    let current_fp = DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash);

    match existing_fingerprints.get(&doc_id) {
        Some(stored_fp) if !stored_fp.needs_reindex(&current_fp) => Some(false),
        _ => Some(true), // Changed or new file
    }
}

/// Check if a single entry (file or directory) has changes.
fn entry_has_changes(
    path: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    if path.is_dir() && should_recurse_dir(path, should_skip_directory) {
        return check_dir_for_changes(
            path,
            base_dir,
            component,
            chunker_config,
            model_hash,
            existing_fingerprints,
            extension,
        );
    }
    !path.is_dir()
        && path.extension().is_some_and(|ext| ext == extension)
        && check_file_changed(
            path,
            base_dir,
            component,
            chunker_config,
            model_hash,
            existing_fingerprints,
        ) == Some(true)
}

pub(crate) fn check_dir_for_changes(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    existing_fingerprints: &std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    extension: &str,
) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };

    entries.flatten().any(|entry| {
        entry_has_changes(
            &entry.path(),
            base_dir,
            component,
            chunker_config,
            model_hash,
            existing_fingerprints,
            extension,
        )
    })
}

/// Recursively index Python files in a directory
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_python_files(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() && should_recurse_dir(&path, should_skip_python_dir) {
            index_python_files(
                &path,
                base_dir,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        } else if path.extension().is_some_and(|ext| ext == "py") {
            process_and_index_file(
                &path,
                base_dir,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }
}

/// Recursively index Rust source files in a directory
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_rust_files(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() && should_recurse_dir(&path, should_skip_rust_dir) {
            index_rust_files(
                &path,
                base_dir,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            process_and_index_file(
                &path,
                base_dir,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }
}

/// Index markdown files in a directory (non-recursive)
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_markdown_files(
    dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.extension().is_none_or(|ext| ext != "md") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }

        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file.md");
        let doc_id = format!("{}/docs/{}", component, file_name);

        fingerprints.insert(
            doc_id.clone(),
            oracle::rag::DocumentFingerprint::new(content.as_bytes(), chunker_config, model_hash),
        );

        reindexer.enqueue(&doc_id, path.clone(), 0);

        let chunk_count =
            index_file_chunks(&content, &doc_id, chunker, indexer, total_chunks, chunk_contents);
        *indexed_count += 1;

        let display_path = format!("{}/docs/{}", component, file_name);
        print_markdown_indexed(&display_path, chunk_count);
    }
}

/// Recursively index markdown files in a directory (for mdBook)
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_markdown_files_recursive(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    component: &str,
    chunker: &oracle::rag::SemanticChunker,
    chunker_config: &oracle::rag::ChunkerConfig,
    model_hash: [u8; 32],
    reindexer: &mut oracle::rag::HeijunkaReindexer,
    indexer: &mut dyn ChunkIndexer,
    indexed_count: &mut usize,
    total_chunks: &mut usize,
    fingerprints: &mut std::collections::HashMap<String, oracle::rag::DocumentFingerprint>,
    chunk_contents: &mut std::collections::HashMap<String, String>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() && should_recurse_dir(&path, should_skip_hidden_dir) {
            index_markdown_files_recursive(
                &path,
                base_dir,
                component,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        } else if path.extension().is_some_and(|ext| ext == "md") {
            let book_prefix = format!("{}/book", component);
            process_and_index_file(
                &path,
                base_dir,
                &book_prefix,
                chunker,
                chunker_config,
                model_hash,
                reindexer,
                indexer,
                indexed_count,
                total_chunks,
                fingerprints,
                chunk_contents,
            );
        }
    }
}
