//! Display and formatting helpers for RAG oracle results.

use crate::ansi_colors::Colorize;
use crate::oracle;

use crate::cli::oracle::types::OracleOutputFormat;

/// Format results as JSON.
pub(super) fn rag_format_results_json(
    query_text: &str,
    results: &[oracle::rag::RetrievalResult],
) -> anyhow::Result<()> {
    let json = serde_json::json!({
        "query": query_text,
        "results": results.iter().map(|r| {
            serde_json::json!({
                "component": r.component,
                "source": r.source,
                "score": r.score,
                "content": r.content,
            })
        }).collect::<Vec<_>>()
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

/// Format results as Markdown.
pub(super) fn rag_format_results_markdown(
    query_text: &str,
    results: &[oracle::rag::RetrievalResult],
) {
    println!("## RAG Query Results\n");
    println!("**Query:** {}\n", query_text);
    for (i, result) in results.iter().enumerate() {
        println!("### {}. {} ({})\n", i + 1, result.component, result.source);
        println!("**Score:** {:.3}\n", result.score);
        if !result.content.is_empty() {
            println!("```\n{}\n```\n", result.content);
        }
    }
}

/// Format results as colored text.
pub(super) fn rag_format_results_text(query_text: &str, results: &[oracle::rag::RetrievalResult]) {
    use oracle::rag::tui::inline;

    println!("{}: {}", "Query".bright_cyan(), query_text);
    println!();

    for (i, result) in results.iter().enumerate() {
        let score_bar = inline::score_bar(result.score, 10);
        println!(
            "{}. [{}] {} {}",
            i + 1,
            result.component.bright_yellow(),
            result.source.dimmed(),
            score_bar
        );
        if !result.content.is_empty() {
            let preview: String = result.content.chars().take(200).collect();
            println!("   {}", preview.dimmed());
        }
        println!();
    }
}

/// Show usage instructions for RAG queries.
pub(super) fn rag_show_usage() {
    println!("{}", "Usage: batuta oracle --rag \"your query here\"".dimmed());
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!("  {} {}", "batuta oracle --rag".cyan(), "\"How do I train a model?\"".dimmed());
    println!("  {} {}", "batuta oracle --rag".cyan(), "\"SIMD tensor operations\"".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --rag-index".cyan(),
        "# Index stack documentation first".dimmed()
    );
}

/// Print profiling summary for RAG queries.
pub(super) fn rag_print_profiling_summary() {
    use oracle::rag::profiling::GLOBAL_METRICS;

    // GH-47: Profiling output goes to stderr to avoid contaminating --format json
    eprintln!();
    eprintln!("{}", "---".repeat(17).dimmed());
    eprintln!("{}", "Profiling Summary".bright_cyan().bold());

    let summary = GLOBAL_METRICS.summary();
    for (name, stats) in &summary.spans {
        eprintln!(
            "  {}: {:.2}ms (count: {})",
            name.bright_yellow(),
            stats.total_us as f64 / 1000.0,
            stats.count
        );
    }

    eprintln!(
        "  {}: {:.1}%",
        "Cache hit rate".bright_yellow(),
        GLOBAL_METRICS.cache_hit_rate() * 100.0
    );
}

/// Format and display RAG results based on output format.
pub(super) fn rag_display_results(
    query_text: &str,
    results: &[oracle::rag::RetrievalResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => rag_format_results_json(query_text, results)?,
        OracleOutputFormat::Markdown => rag_format_results_markdown(query_text, results),
        OracleOutputFormat::Text => rag_format_results_text(query_text, results),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            eprintln!("No code available for RAG results (try --format text)");
            std::process::exit(1);
        }
    }
    Ok(())
}
