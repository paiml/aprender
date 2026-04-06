//! Display and formatting functions for PMAT Query results.

use crate::ansi_colors::Colorize;

use super::pmat_query_fusion::{compute_quality_summary, format_summary_line};
use super::pmat_query_types::{FusedResult, PmatQueryResult};
use crate::cli::oracle::types::OracleOutputFormat;

/// Parse `pmat query` JSON output into structured results.
pub(super) fn parse_pmat_query_output(json: &str) -> anyhow::Result<Vec<PmatQueryResult>> {
    let results: Vec<PmatQueryResult> = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("Failed to parse pmat query output: {e}"))?;
    Ok(results)
}

/// Invoke `pmat query` and return parsed results.
pub(super) fn run_pmat_query(
    opts: &super::pmat_query_types::PmatQueryOptions,
) -> anyhow::Result<Vec<PmatQueryResult>> {
    use super::pmat_query_cache::{load_cached_results, save_cache};

    // Check cache first
    if let Some(cached) = load_cached_results(&opts.query, opts.project_path.as_deref()) {
        eprintln!("  {} hit \u{2014} using cached results", "[   cache]".dimmed());
        return Ok(cached);
    }

    let limit_str = opts.limit.to_string();
    let mut args: Vec<&str> = vec!["query", &opts.query, "--format", "json", "--limit", &limit_str];

    let grade_val;
    if let Some(ref grade) = opts.min_grade {
        grade_val = grade.clone();
        args.push("--min-grade");
        args.push(&grade_val);
    }

    let complexity_str;
    if let Some(max) = opts.max_complexity {
        complexity_str = max.to_string();
        args.push("--max-complexity");
        args.push(&complexity_str);
    }

    if opts.include_source {
        args.push("--include-source");
    }

    let working_dir = opts.project_path.as_ref().map(std::path::Path::new);
    let output = crate::tools::run_tool("pmat", &args, working_dir)?;
    let results = parse_pmat_query_output(&output)?;

    // Save to cache
    save_cache(&opts.query, opts.project_path.as_deref(), &results);

    Ok(results)
}

/// Grade badge with color.
pub(super) fn grade_badge(grade: &str) -> String {
    match grade {
        "A" => format!("[{}]", "A".bright_green().bold()),
        "B" => format!("[{}]", "B".green()),
        "C" => format!("[{}]", "C".bright_yellow()),
        "D" => format!("[{}]", "D".yellow()),
        "F" => format!("[{}]", "F".bright_red().bold()),
        other => format!("[{}]", other.dimmed()),
    }
}

/// Score bar for TDG score (0-100 scale).
pub(super) fn tdg_score_bar(score: f64, width: usize) -> String {
    let filled = ((score / 100.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{} {:.1}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty), score)
}

/// Print quality summary line in text mode.
fn print_quality_summary(results: &[PmatQueryResult]) {
    if results.is_empty() {
        return;
    }
    let summary = compute_quality_summary(results);
    println!("{}: {}", "Summary".bright_yellow().bold(), format_summary_line(&summary));
    println!();
}

/// Format results as colored text.
fn pmat_format_results_text(query_text: &str, results: &[PmatQueryResult]) {
    println!("{}: {}", "PMAT Query".bright_cyan(), query_text);
    println!("{}", "\u{2500}".repeat(50).dimmed());
    println!();

    for (i, r) in results.iter().enumerate() {
        let badge = grade_badge(&r.tdg_grade);
        let score_bar = tdg_score_bar(r.tdg_score, 10);
        let project_prefix =
            r.project.as_ref().map(|p| format!("[{}] ", p.bright_blue())).unwrap_or_default();
        println!(
            "{}. {} {}{}:{}  {}          {}",
            i + 1,
            badge,
            project_prefix,
            r.file_path.cyan(),
            r.start_line,
            r.function_name.bright_yellow(),
            score_bar
        );
        if !r.signature.is_empty() {
            println!("   {}", r.signature.dimmed());
        }
        println!("   Complexity: {} | Big-O: {} | SATD: {}", r.complexity, r.big_o, r.satd_count);
        if let Some(ref doc) = r.doc_comment {
            let preview: String = doc.chars().take(120).collect();
            println!("   {}", preview.dimmed());
        }
        if !r.rag_backlinks.is_empty() {
            println!("   {} {}", "See also:".bright_green(), r.rag_backlinks.join(", ").dimmed());
        }
        if let Some(ref src) = r.source {
            println!("   {}", "\u{2500}".repeat(40).dimmed());
            for line in src.lines().take(10) {
                #[cfg(feature = "syntect")]
                crate::cli::syntax::print_highlighted_line(
                    line,
                    crate::cli::syntax::Language::Rust,
                    "   ",
                );
                #[cfg(not(feature = "syntect"))]
                println!("   {}", line);
            }
            if src.lines().count() > 10 {
                println!("   {}", "...".dimmed());
            }
        }
        println!();
    }

    print_quality_summary(results);
}

/// Format results as JSON with query metadata envelope.
fn pmat_format_results_json(query_text: &str, results: &[PmatQueryResult]) -> anyhow::Result<()> {
    let summary = compute_quality_summary(results);
    let json = serde_json::json!({
        "query": query_text,
        "source": "pmat",
        "result_count": results.len(),
        "summary": {
            "grades": summary.grades,
            "avg_complexity": summary.avg_complexity,
            "total_satd": summary.total_satd,
            "complexity_range": [summary.complexity_range.0, summary.complexity_range.1],
        },
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

/// Format results as a markdown table.
fn pmat_format_results_markdown(query_text: &str, results: &[PmatQueryResult]) {
    println!("## PMAT Query Results\n");
    println!("**Query:** {}\n", query_text);
    println!("| # | Grade | File | Function | TDG | Complexity | Big-O |");
    println!("|---|-------|------|----------|-----|------------|-------|");
    for (i, r) in results.iter().enumerate() {
        println!(
            "| {} | {} | {}:{} | `{}` | {:.1} | {} | {} |",
            i + 1,
            r.tdg_grade,
            r.file_path,
            r.start_line,
            r.function_name,
            r.tdg_score,
            r.complexity,
            r.big_o
        );
    }
    let summary = compute_quality_summary(results);
    println!("\n**Summary:** {}", format_summary_line(&summary));
}

/// Display results in the requested format.
pub(super) fn pmat_display_results(
    query_text: &str,
    results: &[PmatQueryResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => pmat_format_results_json(query_text, results)?,
        OracleOutputFormat::Markdown => pmat_format_results_markdown(query_text, results),
        OracleOutputFormat::Text => pmat_format_results_text(query_text, results),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            for r in results {
                if let Some(ref src) = r.source {
                    println!("// {}:{} - {}", r.file_path, r.start_line, r.function_name);
                    println!("{}", src);
                    println!();
                }
            }
            if results.iter().all(|r| r.source.is_none()) {
                eprintln!("No source code in results (try --pmat-include-source)");
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

/// Display RRF-fused combined PMAT + RAG results.
pub(super) fn pmat_display_combined(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    rag_results: &[crate::oracle::rag::RetrievalResult],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use super::pmat_query_fusion::rrf_fuse_results;

    let fused = rrf_fuse_results(pmat_results, rag_results, 20);

    match format {
        OracleOutputFormat::Json => display_combined_json(query_text, pmat_results, &fused)?,
        OracleOutputFormat::Markdown => {
            display_combined_markdown(query_text, pmat_results, &fused);
        }
        OracleOutputFormat::Text => display_combined_text(pmat_results, &fused),
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => display_combined_code(&fused),
    }
    Ok(())
}

fn display_combined_json(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    fused: &[(FusedResult, f64)],
) -> anyhow::Result<()> {
    let summary = compute_quality_summary(pmat_results);
    let json = serde_json::json!({
        "query": query_text,
        "mode": "rrf_fused",
        "k": 60,
        "result_count": fused.len(),
        "summary": {
            "grades": summary.grades,
            "avg_complexity": summary.avg_complexity,
            "total_satd": summary.total_satd,
            "complexity_range": [summary.complexity_range.0, summary.complexity_range.1],
        },
        "results": fused.iter().map(|(item, score)| {
            let mut v = serde_json::to_value(item).unwrap_or_default();
            if let Some(obj) = v.as_object_mut() {
                obj.insert("rrf_score".to_string(), serde_json::json!(score));
            }
            v
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn display_combined_markdown(
    query_text: &str,
    pmat_results: &[PmatQueryResult],
    fused: &[(FusedResult, f64)],
) {
    println!("## Combined PMAT + RAG Results (RRF-fused)\n");
    println!("**Query:** {}\n", query_text);
    println!("| # | Type | Source | Score |");
    println!("|---|------|--------|-------|");
    for (i, (item, score)) in fused.iter().enumerate() {
        match item {
            FusedResult::Function(r) => {
                println!(
                    "| {} | fn | {}:{} `{}` [{}] | {:.3} |",
                    i + 1,
                    r.file_path,
                    r.start_line,
                    r.function_name,
                    r.tdg_grade,
                    score
                );
            }
            FusedResult::Document { component, source, .. } => {
                println!("| {} | doc | [{}] {} | {:.3} |", i + 1, component, source, score);
            }
        }
    }
    let summary = compute_quality_summary(pmat_results);
    println!("\n**Summary (functions):** {}", format_summary_line(&summary));
}

fn display_combined_text(pmat_results: &[PmatQueryResult], fused: &[(FusedResult, f64)]) {
    println!("{} (RRF k=60)", "Combined Search".bright_cyan().bold());
    println!("{}", "\u{2500}".repeat(50).dimmed());
    println!();

    for (i, (item, score)) in fused.iter().enumerate() {
        display_combined_text_item(i, item, *score);
    }

    print_quality_summary(pmat_results);
}

fn display_combined_text_item(i: usize, item: &FusedResult, score: f64) {
    let score_pct = (score * 100.0) as usize;
    let bar_filled = (score * 10.0).round() as usize;
    let bar_empty = 10_usize.saturating_sub(bar_filled);
    let bar = format!(
        "{}{} {:3}%",
        "\u{2588}".repeat(bar_filled),
        "\u{2591}".repeat(bar_empty),
        score_pct,
    );

    match item {
        FusedResult::Function(r) => {
            let badge = grade_badge(&r.tdg_grade);
            let project_prefix =
                r.project.as_ref().map(|p| format!("[{}] ", p.bright_blue())).unwrap_or_default();
            println!(
                "{}. {} {} {}{}:{}  {}  {}",
                i + 1,
                "[fn]".bright_cyan(),
                badge,
                project_prefix,
                r.file_path.cyan(),
                r.start_line,
                r.function_name.bright_yellow(),
                bar,
            );
            println!(
                "   Complexity: {} | Big-O: {} | SATD: {}",
                r.complexity, r.big_o, r.satd_count
            );
            if !r.rag_backlinks.is_empty() {
                println!(
                    "   {} {}",
                    "See also:".bright_green(),
                    r.rag_backlinks.join(", ").dimmed()
                );
            }
        }
        FusedResult::Document { component, source, content, .. } => {
            println!(
                "{}. {} [{}] {} {}",
                i + 1,
                "[doc]".bright_green(),
                component.bright_yellow(),
                source.dimmed(),
                bar,
            );
            if !content.is_empty() {
                let preview: String = content.chars().take(200).collect();
                println!("   {}", preview.dimmed());
            }
        }
    }
    println!();
}

fn display_combined_code(fused: &[(FusedResult, f64)]) {
    for (item, _) in fused {
        if let FusedResult::Function(r) = item {
            if let Some(ref src) = r.source {
                println!("// {}:{} - {}", r.file_path, r.start_line, r.function_name);
                println!("{}", src);
                println!();
            }
        }
    }
    let has_source = fused
        .iter()
        .any(|(item, _)| matches!(item, FusedResult::Function(r) if r.source.is_some()));
    if !has_source {
        eprintln!("No source code in results (try --pmat-include-source)");
        std::process::exit(1);
    }
}

/// Show usage hint when no query is provided.
pub(super) fn show_pmat_query_usage() {
    println!("{}", "Usage: batuta oracle --pmat-query \"your query here\"".dimmed());
    println!();
    println!("{}", "Examples:".bright_yellow());
    println!("  {} {}", "batuta oracle --pmat-query".cyan(), "\"error handling\"".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"serialize\" --pmat-min-grade A".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"allocator\" --pmat-include-source --format json".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"cache\" --rag  # combined RRF-fused search".dimmed()
    );
    println!(
        "  {} {}",
        "batuta oracle --pmat-query".cyan(),
        "\"tokenizer\" --pmat-all-local  # search all projects".dimmed()
    );
}
