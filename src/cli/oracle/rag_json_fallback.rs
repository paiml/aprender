//! JSON fallback query backend for RAG oracle (when `rag` feature is disabled).

use crate::ansi_colors::Colorize;
use crate::oracle;

use std::sync::{Arc, LazyLock, RwLock};

use super::rag_helpers::format_timestamp;

/// Session-scoped cache for loaded RAG index data (JSON fallback only).
/// Uses Arc to allow cheap cloning without copying the index.
/// When `rag` feature is enabled, SQLite handles caching natively.
#[cfg(not(feature = "rag"))]
pub(super) static RAG_INDEX_CACHE: LazyLock<RwLock<Option<Arc<RagIndexData>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Loaded RAG index data (JSON backend only; SQLite path doesn't use this).
#[cfg(not(feature = "rag"))]
pub(in crate::cli::oracle) struct RagIndexData {
    pub(in crate::cli::oracle) retriever: oracle::rag::HybridRetriever,
    pub(in crate::cli::oracle) doc_count: usize,
    pub(in crate::cli::oracle) chunk_count: usize,
    pub(in crate::cli::oracle) chunk_contents: std::collections::HashMap<String, String>,
}

/// Try to load RAG index from JSON, returns None if not found.
/// Uses session-scoped cache to avoid re-loading from disk on every query.
#[cfg(not(feature = "rag"))]
pub(in crate::cli::oracle) fn rag_load_index() -> anyhow::Result<Option<Arc<RagIndexData>>> {
    use oracle::rag::persistence::RagPersistence;

    // Check session cache first
    {
        let cache =
            RAG_INDEX_CACHE.read().map_err(|e| anyhow::anyhow!("RAG cache lock poisoned: {e}"))?;
        if let Some(ref data) = *cache {
            eprintln!("  {} hit -- using session-cached index", "[   cache]".dimmed());
            println!(
                "{}: {} documents, {} chunks (session cache)",
                "Index".bright_green(),
                data.doc_count,
                data.chunk_count
            );
            println!();
            return Ok(Some(Arc::clone(data)));
        }
    }

    let persistence = RagPersistence::new();
    match persistence.load() {
        Ok(Some((persisted_index, persisted_docs, manifest))) => {
            let retriever = oracle::rag::HybridRetriever::from_persisted(persisted_index.clone());
            let doc_count = {
                let unique_docs: std::collections::HashSet<&str> = persisted_index
                    .doc_lengths
                    .keys()
                    .map(|k| k.split('#').next().unwrap_or(k.as_str()))
                    .collect();
                unique_docs.len()
            };
            let chunk_count = persisted_docs.total_chunks;
            let chunk_contents = persisted_docs.chunk_contents.clone();

            println!(
                "{}: Loaded from disk (indexed {})",
                "Index".bright_green(),
                format_timestamp(manifest.indexed_at)
            );
            println!(
                "  {} documents, {} chunks, {} sources",
                doc_count,
                chunk_count,
                manifest.sources.len()
            );
            println!();

            let data = Arc::new(RagIndexData { retriever, doc_count, chunk_count, chunk_contents });

            // Store in session cache
            {
                let mut cache = RAG_INDEX_CACHE
                    .write()
                    .map_err(|e| anyhow::anyhow!("RAG cache lock poisoned: {e}"))?;
                *cache = Some(Arc::clone(&data));
            }

            Ok(Some(data))
        }
        Ok(None) => {
            println!(
                "{}",
                "No cached index found. Run 'batuta oracle --rag-index' first."
                    .bright_yellow()
                    .bold()
            );
            println!();
            Ok(None)
        }
        Err(e) => {
            println!("{}: {} - rebuilding index recommended", "Warning".bright_yellow(), e);
            println!();
            Ok(None)
        }
    }
}
