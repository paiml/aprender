//! PMAT Query integration for Oracle mode
//!
//! Provides function-level quality-annotated code search via `pmat query`,
//! optionally combined with RAG document-level retrieval for a hybrid view.
//!
//! ## v2.0 Enhancements
//!
//! 1. RRF-fused ranking in combined mode (Cormack et al. 2009)
//! 2. Cross-project search via `--pmat-all-local`
//! 3. Result caching with mtime-based invalidation
//! 4. Quality distribution summary
//! 5. Documentation backlinks from RAG index

#[path = "pmat_query_cache.rs"]
mod pmat_query_cache;
#[path = "pmat_query_display.rs"]
mod pmat_query_display;
#[path = "pmat_query_fusion.rs"]
mod pmat_query_fusion;
#[path = "pmat_query_types.rs"]
mod pmat_query_types;

#[cfg(test)]
#[path = "pmat_query_tests.rs"]
mod pmat_query_tests;

// Re-export all public types so callers are unaffected
pub use pmat_query_types::{FusedResult, PmatQueryOptions, PmatQueryResult, QualitySummary};

use super::types::OracleOutputFormat;
use crate::ansi_colors::Colorize;

/// Check that pmat is available, printing install instructions if not.
fn check_pmat_available() -> bool {
    use crate::tools;
    let registry = tools::ToolRegistry::detect();
    if registry.pmat.is_none() {
        eprintln!("{}: pmat is not installed or not in PATH.", "Error".bright_red().bold());
        eprintln!();
        eprintln!("Install pmat:");
        eprintln!("  cargo install pmat");
        eprintln!();
        eprintln!("Or check: https://github.com/paiml/pmat");
        return false;
    }
    true
}

/// Execute PMAT query, optionally combined with RAG retrieval.
#[allow(clippy::too_many_arguments)]
pub fn cmd_oracle_pmat_query(
    query: Option<String>,
    project_path: Option<String>,
    limit: usize,
    min_grade: Option<String>,
    max_complexity: Option<u32>,
    include_source: bool,
    also_rag: bool,
    all_local: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use pmat_query_cache::{attach_rag_backlinks, run_cross_project_query};
    use pmat_query_display::{
        pmat_display_combined, pmat_display_results, run_pmat_query, show_pmat_query_usage,
    };

    if !check_pmat_available() {
        return Ok(());
    }

    let query_text = match query {
        Some(q) if !q.is_empty() => q,
        _ => {
            show_pmat_query_usage();
            return Ok(());
        }
    };

    // GH-47: Status headers go to stderr to avoid contaminating --format json output
    eprintln!("{}", "PMAT Query Mode".bright_cyan().bold());
    eprintln!("{}", "\u{2500}".repeat(50).dimmed());
    eprintln!();

    let opts = PmatQueryOptions {
        query: query_text.clone(),
        project_path,
        limit,
        min_grade,
        max_complexity,
        include_source,
    };

    let mut pmat_results = if all_local {
        eprintln!("{}", "Cross-project search (all local PAIML projects)".dimmed());
        eprintln!();
        run_cross_project_query(&opts)?
    } else {
        run_pmat_query(&opts)?
    };

    if pmat_results.is_empty() {
        eprintln!("{}", "No functions matched the query. Try broadening your search.".dimmed());
        return Ok(());
    }

    display_pmat_with_optional_rag(&query_text, &mut pmat_results, also_rag, format)
}

/// Display PMAT results, optionally fused with RAG retrieval.
fn display_pmat_with_optional_rag(
    query_text: &str,
    pmat_results: &mut [PmatQueryResult],
    also_rag: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use pmat_query_cache::attach_rag_backlinks;
    use pmat_query_display::{pmat_display_combined, pmat_display_results};

    if !also_rag {
        return pmat_display_results(query_text, pmat_results, format);
    }

    let rag_results = match load_rag_results(query_text)? {
        Some((results, chunk_contents)) => {
            attach_rag_backlinks(pmat_results, &chunk_contents);
            results
        }
        None => Vec::new(),
    };
    pmat_display_combined(query_text, pmat_results, &rag_results, format)
}

/// Load RAG results and chunk contents for combined display.
/// Returns None (with warning) if index unavailable.
///
/// When the `rag` feature is enabled, uses SQLite+FTS5 backend directly.
/// Otherwise falls back to JSON-based `HybridRetriever`.
#[cfg(feature = "rag")]
fn load_rag_results(
    query_text: &str,
) -> anyhow::Result<
    Option<(Vec<crate::oracle::rag::RetrievalResult>, std::collections::HashMap<String, String>)>,
> {
    let index = match super::rag::rag_load_sqlite()? {
        Some(idx) => idx,
        None => {
            eprintln!(
                "{}: RAG index not available. Showing PMAT results only.",
                "Warning".bright_yellow()
            );
            eprintln!("  Run 'batuta oracle --rag-index' to enable combined search.");
            eprintln!();
            return Ok(None);
        }
    };

    let sqlite_results = super::rag::rag_search_sqlite(&index, query_text, 10)?;

    let results: Vec<crate::oracle::rag::RetrievalResult> = sqlite_results
        .iter()
        .map(|r| {
            let component = super::rag::extract_component(&r.doc_id);
            let max_score = sqlite_results.first().map(|first| first.score).unwrap_or(1.0).max(1.0);
            crate::oracle::rag::RetrievalResult {
                id: r.chunk_id.clone(),
                component,
                source: r.doc_id.clone(),
                content: r.content.chars().take(200).collect(),
                score: r.score / max_score,
                start_line: 1,
                end_line: 1,
                score_breakdown: crate::oracle::rag::ScoreBreakdown {
                    bm25_score: r.score,
                    dense_score: 0.0,
                    rrf_score: 0.0,
                    rerank_score: None,
                },
            }
        })
        .collect();

    // SQLite stores content in the DB, so chunk_contents HashMap is empty.
    // Backlinks will be a no-op (acceptable tradeoff vs cloning 389K strings).
    Ok(Some((results, std::collections::HashMap::new())))
}

/// Load RAG results and chunk contents for combined display (JSON fallback).
/// Returns None (with warning) if index unavailable.
#[cfg(not(feature = "rag"))]
fn load_rag_results(
    query_text: &str,
) -> anyhow::Result<
    Option<(Vec<crate::oracle::rag::RetrievalResult>, std::collections::HashMap<String, String>)>,
> {
    use crate::oracle::rag::DocumentIndex;

    let index_data = match super::rag::rag_load_index()? {
        Some(data) => data,
        None => {
            eprintln!(
                "{}: RAG index not available. Showing PMAT results only.",
                "Warning".bright_yellow()
            );
            eprintln!("  Run 'batuta oracle --rag-index' to enable combined search.");
            eprintln!();
            return Ok(None);
        }
    };

    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(query_text, &empty_index, 10);

    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content.clone_from(snippet);
        }
    }

    Ok(Some((results, index_data.chunk_contents.clone())))
}
