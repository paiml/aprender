//! RAG index statistics and dashboard commands.

use crate::ansi_colors::Colorize;
use crate::oracle;

use crate::cli::oracle::types::OracleOutputFormat;

#[cfg(feature = "rag")]
use super::rag_sqlite::sqlite_index_path;

use super::rag_helpers::{format_timestamp, print_stat};

/// Format multi-index stats for display.
#[cfg(feature = "rag")]
pub(super) fn rag_format_multi_index_stats(
    indices: &[(String, trueno_rag::sqlite::SqliteIndex)],
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    match format {
        OracleOutputFormat::Json => {
            let index_stats: Vec<serde_json::Value> = indices
                .iter()
                .map(|(name, idx)| {
                    let path = if name == "oracle" {
                        sqlite_index_path().display().to_string()
                    } else {
                        String::new()
                    };
                    serde_json::json!({
                        "name": name,
                        "backend": "sqlite",
                        "documents": idx.document_count().unwrap_or(0),
                        "chunks": idx.chunk_count().unwrap_or(0),
                        "path": path,
                    })
                })
                .collect();
            let json = serde_json::json!({ "indices": index_stats });
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OracleOutputFormat::Markdown => {
            println!("## RAG Index Statistics (SQLite)\n");
            println!("| Index | Documents | Chunks |");
            println!("|-------|-----------|--------|");
            for (name, idx) in indices {
                println!(
                    "| {} | {} | {} |",
                    name,
                    idx.document_count().unwrap_or(0),
                    idx.chunk_count().unwrap_or(0),
                );
            }
        }
        OracleOutputFormat::Text => {
            let mut total_docs = 0usize;
            let mut total_chunks = 0usize;
            for (name, idx) in indices {
                let doc_count = idx.document_count().unwrap_or(0);
                let chunk_count = idx.chunk_count().unwrap_or(0);
                total_docs += doc_count;
                total_chunks += chunk_count;
                print_stat(
                    &format!("Index ({name})"),
                    format!("{} docs, {} chunks", doc_count, chunk_count).cyan(),
                );
            }
            if indices.len() > 1 {
                print_stat(
                    "Total",
                    format!(
                        "{} docs, {} chunks across {} indices",
                        total_docs,
                        total_chunks,
                        indices.len()
                    )
                    .cyan(),
                );
            }
            print_stat("Backend", "SQLite+FTS5".cyan());
            print_stat("Path (oracle)", format!("{:?}", sqlite_index_path()));
        }
        OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
            eprintln!("No code available for RAG stats (try --format text)");
            std::process::exit(1);
        }
    }
    Ok(())
}

/// Show RAG index statistics.
pub fn cmd_oracle_rag_stats(format: OracleOutputFormat) -> anyhow::Result<()> {
    println!("{}", "RAG Index Statistics".bright_cyan().bold());
    println!("{}", "---".repeat(17).dimmed());
    println!();

    #[cfg(feature = "rag")]
    {
        use super::rag_sqlite::rag_load_all_indices;

        let indices = rag_load_all_indices()?;
        if !indices.is_empty() {
            rag_format_multi_index_stats(&indices, format)?;
            println!();
            return Ok(());
        }
    }

    cmd_oracle_rag_stats_json(format)
}

/// Show RAG stats from JSON persistence (fallback).
pub(super) fn cmd_oracle_rag_stats_json(format: OracleOutputFormat) -> anyhow::Result<()> {
    use oracle::rag::persistence::RagPersistence;

    let persistence = RagPersistence::new();

    match persistence.stats()? {
        Some(manifest) => match format {
            OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
                eprintln!("No code available for RAG stats (try --format text)");
                std::process::exit(1);
            }
            OracleOutputFormat::Json => {
                let json = serde_json::json!({
                    "backend": "json",
                    "version": manifest.version,
                    "indexed_at": manifest.indexed_at,
                    "batuta_version": manifest.batuta_version,
                    "sources": manifest.sources.iter().map(|s| {
                        serde_json::json!({
                            "id": s.id,
                            "commit": s.commit,
                            "doc_count": s.doc_count,
                            "chunk_count": s.chunk_count,
                        })
                    }).collect::<Vec<_>>()
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
            OracleOutputFormat::Markdown => {
                println!("## RAG Index Statistics (JSON)\n");
                println!("| Property | Value |");
                println!("|----------|-------|");
                println!("| Backend | JSON |");
                println!("| Version | {} |", manifest.version);
                println!("| Batuta Version | {} |", manifest.batuta_version);
                println!("| Indexed At | {} |", format_timestamp(manifest.indexed_at));
                println!();
                println!("### Sources\n");
                for source in &manifest.sources {
                    println!(
                        "- **{}**: {} docs, {} chunks",
                        source.id, source.doc_count, source.chunk_count
                    );
                }
            }
            OracleOutputFormat::Text => {
                print_stat("Backend", "JSON".cyan());
                print_stat("Index version", manifest.version.cyan());
                print_stat("Batuta version", manifest.batuta_version.cyan());
                print_stat("Indexed", format_timestamp(manifest.indexed_at).cyan());
                print_stat("Cache path", format!("{:?}", persistence.cache_path()));
                println!();

                let total_docs: usize = manifest.sources.iter().map(|s| s.doc_count).sum();
                let total_chunks: usize = manifest.sources.iter().map(|s| s.chunk_count).sum();

                print_stat("Total", format!("{} documents", total_docs));
                print_stat("Total", format!("{} chunks", total_chunks));
                println!();

                if !manifest.sources.is_empty() {
                    println!("{}", "Sources:".bright_yellow());
                    for source in &manifest.sources {
                        println!(
                            "  {} {}: {} docs, {} chunks",
                            "*".bright_blue(),
                            source.id.cyan(),
                            source.doc_count,
                            source.chunk_count
                        );
                        if let Some(commit) = &source.commit {
                            println!("    {} commit: {}", "".dimmed(), commit.dimmed());
                        }
                    }
                }
            }
        },
        None => {
            println!(
                "{}",
                "No cached index found. Run 'batuta oracle --rag-index' to create one."
                    .bright_yellow()
            );
        }
    }

    println!();
    Ok(())
}

/// Launch the TUI dashboard for RAG oracle.
pub fn cmd_oracle_rag_dashboard() -> anyhow::Result<()> {
    #[cfg(feature = "presentar-terminal")]
    {
        use oracle::rag::tui::OracleDashboard;
        let mut dashboard = OracleDashboard::new();
        dashboard.run()
    }
    #[cfg(not(feature = "presentar-terminal"))]
    {
        anyhow::bail!(
            "TUI dashboard requires the 'tui' feature. Install with: cargo install batuta --features tui"
        )
    }
}
