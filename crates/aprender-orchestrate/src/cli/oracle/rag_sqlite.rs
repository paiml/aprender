//! SQLite+FTS5 query backend for RAG oracle.

use crate::ansi_colors::Colorize;

/// A search result from the SQLite FTS5 backend.
#[cfg(feature = "rag")]
pub(in crate::cli::oracle) struct SqliteSearchResult {
    /// Chunk identifier (e.g., "trueno/CLAUDE.md#5")
    pub(in crate::cli::oracle) chunk_id: String,
    /// Document identifier (e.g., "trueno/CLAUDE.md")
    pub(in crate::cli::oracle) doc_id: String,
    /// Chunk text content
    pub(in crate::cli::oracle) content: String,
    /// BM25 relevance score (higher = more relevant)
    pub(in crate::cli::oracle) score: f64,
}

/// Default SQLite database path for the RAG index.
#[cfg(feature = "rag")]
pub(in crate::cli::oracle) fn sqlite_index_path() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
        .join("batuta/rag/index.sqlite")
}

/// Load RAG index from SQLite. Returns None if the database doesn't exist.
#[cfg(feature = "rag")]
pub(in crate::cli::oracle) fn rag_load_sqlite(
) -> anyhow::Result<Option<trueno_rag::sqlite::SqliteIndex>> {
    let db_path = sqlite_index_path();
    if !db_path.exists() {
        return Ok(None);
    }

    let index = trueno_rag::sqlite::SqliteIndex::open(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to open SQLite index: {e}"))?;

    Ok(Some(index))
}

/// Search the SQLite FTS5 index.
#[cfg(feature = "rag")]
pub(in crate::cli::oracle) fn rag_search_sqlite(
    index: &trueno_rag::sqlite::SqliteIndex,
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<SqliteSearchResult>> {
    let fts_results =
        index.search_fts(query, k).map_err(|e| anyhow::anyhow!("FTS5 search failed: {e}"))?;

    Ok(fts_results
        .into_iter()
        .map(|r| SqliteSearchResult {
            chunk_id: r.chunk_id,
            doc_id: r.doc_id,
            content: r.content,
            score: r.score,
        })
        .collect())
}

/// Extract component name from doc_id (e.g., "trueno/CLAUDE.md" -> "trueno").
#[cfg(feature = "rag")]
pub(in crate::cli::oracle) fn extract_component(doc_id: &str) -> String {
    doc_id.split('/').next().unwrap_or("unknown").to_string()
}

/// Load all configured SQLite indices (main oracle + private endpoints).
#[cfg(feature = "rag")]
pub(super) fn rag_load_all_indices(
) -> anyhow::Result<Vec<(String, trueno_rag::sqlite::SqliteIndex)>> {
    let mut indices = Vec::new();

    let main_path = sqlite_index_path();
    if main_path.exists() {
        let idx = trueno_rag::sqlite::SqliteIndex::open(&main_path)
            .map_err(|e| anyhow::anyhow!("Failed to open main index: {e}"))?;
        indices.push(("oracle".to_string(), idx));
    }

    match crate::config::PrivateConfig::load_optional() {
        Err(e) => {
            eprintln!("Warning: failed to load private config: {e}");
        }
        Ok(None) => {}
        Ok(Some(config)) => {
            for ep in &config.private.endpoints {
                if ep.endpoint_type == "local" {
                    let path = std::path::Path::new(&ep.index_path);
                    if path.exists() {
                        match trueno_rag::sqlite::SqliteIndex::open(path) {
                            Ok(idx) => indices.push((ep.name.clone(), idx)),
                            Err(e) => {
                                eprintln!(
                                    "  {} Failed to open endpoint {}: {}",
                                    "[warning]".bright_yellow(),
                                    ep.name,
                                    e
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(indices)
}

/// Search all indices and fuse results via Reciprocal Rank Fusion.
#[cfg(feature = "rag")]
pub(super) fn rag_search_multi(
    indices: &[(String, trueno_rag::sqlite::SqliteIndex)],
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<SqliteSearchResult>> {
    use std::collections::HashMap;

    let rrf_k = 60.0_f64;
    let mut score_map: HashMap<String, (f64, SqliteSearchResult)> = HashMap::new();

    for (source_name, index) in indices {
        let results = index
            .search_fts(query, k)
            .map_err(|e| anyhow::anyhow!("FTS5 search on {source_name} failed: {e}"))?;

        for (rank, r) in results.into_iter().enumerate() {
            let rrf_score = 1.0 / (rrf_k + rank as f64 + 1.0);
            let key = format!("{}:{}", source_name, r.chunk_id);
            let entry = score_map.entry(key).or_insert_with(|| {
                (
                    0.0,
                    SqliteSearchResult {
                        chunk_id: r.chunk_id,
                        doc_id: r.doc_id,
                        content: r.content,
                        score: 0.0,
                    },
                )
            });
            entry.0 += rrf_score;
        }
    }

    let mut results: Vec<_> = score_map.into_values().collect();
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    Ok(results
        .into_iter()
        .map(|(score, mut r)| {
            r.score = score;
            r
        })
        .collect())
}

/// Dispatch search to single-index (direct) or multi-index (RRF fusion).
#[cfg(feature = "rag")]
pub(super) fn rag_dispatch_search(
    indices: &[(String, trueno_rag::sqlite::SqliteIndex)],
    query: &str,
    k: usize,
) -> anyhow::Result<Vec<SqliteSearchResult>> {
    if indices.len() == 1 {
        rag_search_sqlite(&indices[0].1, query, k)
    } else {
        rag_search_multi(indices, query, k)
    }
}
