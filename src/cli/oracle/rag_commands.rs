//! Public RAG oracle command implementations.

use crate::ansi_colors::Colorize;
use crate::oracle;

use crate::cli::oracle::types::OracleOutputFormat;

#[cfg(feature = "rag")]
use super::rag_sqlite::{
    extract_component, rag_dispatch_search, rag_load_all_indices, SqliteSearchResult,
};

use super::rag_display::{rag_display_results, rag_print_profiling_summary, rag_show_usage};

/// RAG query using SQLite+FTS5 backend with multi-index support.
#[cfg(feature = "rag")]
pub(super) fn cmd_oracle_rag_sqlite(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;
    use std::time::Instant;

    let total_start = Instant::now();
    let _total_span = trace.then(|| span("total_query"));

    // GH-47: Status headers go to stderr to avoid contaminating --format json output
    eprintln!("{}", "RAG Oracle Mode".bright_cyan().bold());
    eprintln!("{}", "(SQLite+FTS5 backend)".dimmed());
    if profile || trace {
        eprintln!("{}", "(profiling enabled)".dimmed());
    }
    eprintln!("{}", "---".repeat(17).dimmed());
    eprintln!();

    let load_start = Instant::now();
    let _load_span = trace.then(|| span("index_load"));
    let indices = rag_load_all_indices()?;
    drop(_load_span);
    let load_ms = load_start.elapsed().as_millis();

    if indices.is_empty() {
        eprintln!(
            "{}",
            "No SQLite index found. Run 'batuta oracle --rag-index' first.".bright_yellow().bold()
        );
        eprintln!();
        return Ok(());
    }

    let mut total_docs = 0usize;
    let mut total_chunks = 0usize;
    for (name, idx) in &indices {
        let doc_count = idx.document_count().map_err(|e| anyhow::anyhow!("{e}"))?;
        let chunk_count = idx.chunk_count().map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!(
            "  {} {}: {} docs, {} chunks",
            "Index".bright_yellow(),
            name.cyan(),
            doc_count,
            chunk_count,
        );
        total_docs += doc_count;
        total_chunks += chunk_count;
    }
    if indices.len() > 1 {
        eprintln!(
            "  {}: {} docs, {} chunks across {} indices",
            "Total".bright_yellow(),
            total_docs,
            total_chunks,
            indices.len(),
        );
    }
    eprintln!();

    let query_text = match query {
        Some(q) => q,
        None => {
            rag_show_usage();
            return Ok(());
        }
    };

    let retrieve_start = Instant::now();
    let _retrieve_span = trace.then(|| span("fts5_search"));
    let sqlite_results = rag_dispatch_search(&indices, &query_text, 10)?;
    drop(_retrieve_span);
    let retrieve_ms = retrieve_start.elapsed().as_millis();

    if sqlite_results.is_empty() {
        eprintln!("{}", "No results found. Try running --rag-index first.".dimmed());
        return Ok(());
    }

    let results: Vec<oracle::rag::RetrievalResult> = sqlite_results
        .iter()
        .map(|r| {
            let component = extract_component(&r.doc_id);
            let max_score = sqlite_results
                .first()
                .map(|first| first.score)
                .unwrap_or(1.0)
                .max(f64::MIN_POSITIVE);
            oracle::rag::RetrievalResult {
                id: r.chunk_id.clone(),
                component,
                source: r.doc_id.clone(),
                content: r.content.chars().take(200).collect(),
                score: r.score / max_score,
                start_line: 1,
                end_line: 1,
                score_breakdown: oracle::rag::ScoreBreakdown {
                    bm25_score: r.score,
                    dense_score: 0.0,
                    rrf_score: 0.0,
                    rerank_score: None,
                },
            }
        })
        .collect();

    rag_display_results(&query_text, &results, format)?;

    let total_ms = total_start.elapsed().as_millis();
    eprintln!(
        "{}",
        format!(
            "load={}ms  search={}ms  total={}ms  indices={}",
            load_ms,
            retrieve_ms,
            total_ms,
            indices.len()
        )
        .dimmed()
    );

    if profile || trace {
        rag_print_profiling_summary();
    }

    Ok(())
}

/// RAG-based query using indexed documentation (without profiling).
#[allow(dead_code, clippy::needless_return)]
pub fn cmd_oracle_rag(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
    #[cfg(feature = "rag")]
    {
        return cmd_oracle_rag_sqlite(query, format, false, false);
    }

    #[cfg(not(feature = "rag"))]
    {
        cmd_oracle_rag_json(query, format)
    }
}

/// JSON fallback RAG query (when rag feature is not enabled).
#[cfg(not(feature = "rag"))]
fn cmd_oracle_rag_json(query: Option<String>, format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::DocumentIndex;

    use super::rag_json_fallback::rag_load_index;

    // GH-47: Status headers go to stderr to avoid contaminating --format json output
    eprintln!("{}", "RAG Oracle Mode".bright_cyan().bold());
    eprintln!("{}", "---".repeat(17).dimmed());
    eprintln!();

    let index_data = match rag_load_index()? {
        Some(data) => data,
        None => return Ok(()),
    };

    eprintln!(
        "{}: {} documents, {} chunks",
        "Index".bright_yellow(),
        index_data.doc_count,
        index_data.chunk_count
    );
    eprintln!();

    let query_text = match query {
        Some(q) => q,
        None => {
            rag_show_usage();
            return Ok(());
        }
    };

    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(&query_text, &empty_index, 10);

    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content = snippet.chars().take(200).collect();
        }
    }

    if results.is_empty() {
        eprintln!("{}", "No results found. Try running --rag-index first.".dimmed());
        return Ok(());
    }

    rag_display_results(&query_text, &results, format)?;
    Ok(())
}

/// RAG-based query with profiling support.
#[allow(clippy::needless_return)]
pub fn cmd_oracle_rag_with_profile(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    #[cfg(feature = "rag")]
    {
        return cmd_oracle_rag_sqlite(query, format, profile, trace);
    }

    #[cfg(not(feature = "rag"))]
    {
        cmd_oracle_rag_json_with_profile(query, format, profile, trace)
    }
}

/// RAG answer generation: retrieve chunks, feed as context to Claude, generate answer.
#[cfg(feature = "rag")]
pub fn cmd_oracle_rag_answer(
    query: Option<String>,
    model: &str,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let indices = rag_load_all_indices()?;
    if indices.is_empty() {
        anyhow::bail!("No SQLite index found. Run 'batuta oracle --rag-index' first.");
    }

    let query_text = match query {
        Some(q) => q,
        None => {
            anyhow::bail!(
                "--answer requires a query. Usage: batuta oracle --answer \"your question\""
            );
        }
    };

    let results = rag_dispatch_search(&indices, &query_text, 10)?;
    if results.is_empty() {
        println!("{}", "No relevant context found.".bright_yellow());
        return Ok(());
    }

    let context = build_answer_context(&results);

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY not set. Required for --answer mode."))?;

    let answer = call_anthropic_answer(&api_key, model, &query_text, &context)?;

    display_answer(&query_text, &answer, &results, format);

    Ok(())
}

/// Build a context string from retrieved chunks for the LLM prompt.
#[cfg(feature = "rag")]
fn build_answer_context(results: &[SqliteSearchResult]) -> String {
    let mut context = String::new();
    for (i, r) in results.iter().enumerate() {
        context.push_str(&format!("[Source {}] {}\n", i + 1, r.doc_id));
        context.push_str(&r.content);
        context.push_str("\n\n");
    }
    context
}

/// Call the Anthropic API to generate an answer from retrieved context.
#[cfg(feature = "rag")]
fn call_anthropic_answer(
    api_key: &str,
    model: &str,
    query: &str,
    context: &str,
) -> anyhow::Result<String> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 2048,
            "system": "You are a knowledgeable technical assistant. Answer the user's question using ONLY the provided context from the sovereign AI stack documentation and video lectures. Cite sources by number (e.g., [Source 1]). If the context doesn't contain enough information to answer, say so explicitly.",
            "messages": [{
                "role": "user",
                "content": format!(
                    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer using only the context above, citing sources by number."
                )
            }]
        });

        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("API request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {}: {}", status, text);
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse API response: {e}"))?;

        let text = json["content"][0]["text"]
            .as_str()
            .unwrap_or("No response generated.")
            .to_string();

        Ok(text)
    })
}

/// Display the generated answer with source citations.
#[cfg(feature = "rag")]
fn display_answer(
    query: &str,
    answer: &str,
    sources: &[SqliteSearchResult],
    format: OracleOutputFormat,
) {
    match format {
        OracleOutputFormat::Json => {
            let source_list: Vec<serde_json::Value> = sources
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    serde_json::json!({
                        "index": i + 1,
                        "doc_id": s.doc_id,
                        "chunk_id": s.chunk_id,
                        "score": s.score,
                    })
                })
                .collect();
            let output = serde_json::json!({
                "query": query,
                "answer": answer,
                "sources": source_list,
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
        }
        _ => {
            println!("{}", "Oracle Answer".bright_cyan().bold());
            println!("{}", "---".repeat(17).dimmed());
            println!();
            println!("{}: {}", "Q".bright_yellow().bold(), query);
            println!();
            println!("{}", answer);
            println!();
            println!("{}", "Sources".bright_yellow().bold());
            println!("{}", "---".repeat(10).dimmed());
            for (i, s) in sources.iter().take(5).enumerate() {
                println!("  [{}] {} (score: {:.3})", i + 1, s.doc_id.cyan(), s.score);
            }
        }
    }
}

/// Stub for non-rag builds.
#[cfg(not(feature = "rag"))]
pub fn cmd_oracle_rag_answer(
    _query: Option<String>,
    _model: &str,
    _format: OracleOutputFormat,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "--answer requires the 'rag' feature. Build with: cargo build --features native,rag"
    );
}

/// JSON fallback RAG query with profiling (when rag feature is not enabled).
#[cfg(not(feature = "rag"))]
fn cmd_oracle_rag_json_with_profile(
    query: Option<String>,
    format: OracleOutputFormat,
    profile: bool,
    trace: bool,
) -> anyhow::Result<()> {
    use oracle::rag::profiling::span;
    use oracle::rag::DocumentIndex;
    use std::time::Instant;

    use super::rag_json_fallback::rag_load_index;

    let total_start = Instant::now();
    let _total_span = trace.then(|| span("total_query"));

    // GH-47: Status headers go to stderr to avoid contaminating --format json output
    eprintln!("{}", "RAG Oracle Mode".bright_cyan().bold());
    if profile || trace {
        eprintln!("{}", "(profiling enabled)".dimmed());
    }
    eprintln!("{}", "---".repeat(17).dimmed());
    eprintln!();

    let load_start = Instant::now();
    let _load_span = trace.then(|| span("index_load"));
    let index_data = match rag_load_index()? {
        Some(data) => data,
        None => return Ok(()),
    };
    drop(_load_span);
    let load_ms = load_start.elapsed().as_millis();

    eprintln!(
        "{}: {} documents, {} chunks",
        "Index".bright_yellow(),
        index_data.doc_count,
        index_data.chunk_count
    );
    eprintln!();

    let query_text = match query {
        Some(q) => q,
        None => {
            rag_show_usage();
            return Ok(());
        }
    };

    let retrieve_start = Instant::now();
    let _retrieve_span = trace.then(|| span("retrieve"));
    let empty_index = DocumentIndex::default();
    let mut results = index_data.retriever.retrieve(&query_text, &empty_index, 10);
    drop(_retrieve_span);
    let retrieve_ms = retrieve_start.elapsed().as_millis();

    let enrich_start = Instant::now();
    let _enrich_span = trace.then(|| span("enrich_results"));
    for result in &mut results {
        if let Some(snippet) = index_data.chunk_contents.get(&result.id) {
            result.content = snippet.chars().take(200).collect();
        }
    }
    drop(_enrich_span);
    let enrich_ms = enrich_start.elapsed().as_millis();

    if results.is_empty() {
        eprintln!("{}", "No results found. Try running --rag-index first.".dimmed());
        return Ok(());
    }

    rag_display_results(&query_text, &results, format)?;

    let total_ms = total_start.elapsed().as_millis();
    eprintln!(
        "{}",
        format!(
            "load={}ms  retrieve={}ms  enrich={}ms  total={}ms",
            load_ms, retrieve_ms, enrich_ms, total_ms
        )
        .dimmed()
    );

    if profile || trace {
        rag_print_profiling_summary();
    }

    Ok(())
}
