//! Coursera asset CLI command handler

use std::path::{Path, PathBuf};

use crate::ansi_colors::Colorize;
use crate::oracle::coursera::{banner, key_concepts, reflection, transcript, vocabulary};

use super::types::OracleOutputFormat;

/// Coursera asset type (matches `--asset` flag)
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum CourseraAssetType {
    /// 1200x400 banner PNG with title and concept bubbles
    Banner,
    /// Reflection reading with Bloom's taxonomy questions and arXiv citations
    Reflection,
    /// Key concepts extraction with definitions and code examples
    #[value(name = "key-concepts")]
    KeyConcepts,
    /// Course vocabulary organized by category
    Vocabulary,
}

/// Handle `batuta oracle --asset <type> --transcript <path>` dispatch.
pub fn cmd_oracle_asset(
    asset_type: CourseraAssetType,
    transcript_path: PathBuf,
    output: Option<PathBuf>,
    topic: Option<String>,
    course_title: Option<String>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    println!("{} {}", "Oracle".bright_cyan().bold(), "Coursera Asset Generation".dimmed());

    match asset_type {
        CourseraAssetType::Banner => handle_banner(transcript_path, output, course_title),
        CourseraAssetType::Reflection => {
            handle_reflection(transcript_path, topic.as_deref(), format)
        }
        CourseraAssetType::KeyConcepts => handle_key_concepts(transcript_path, format),
        CourseraAssetType::Vocabulary => handle_vocabulary(transcript_path, format),
    }
}

fn handle_banner(
    transcript_path: PathBuf,
    output: Option<PathBuf>,
    course_title: Option<String>,
) -> anyhow::Result<()> {
    let t = transcript::parse_transcript(&transcript_path)?;
    let title = course_title.unwrap_or_else(|| "Course".to_string());
    let config = banner::banner_config_from_transcript(&t, &title);
    let svg = banner::generate_banner_svg(&config);

    let output_path = output.unwrap_or_else(|| PathBuf::from("banner.png"));

    if output_path.extension().is_some_and(|e| e == "svg") {
        std::fs::write(&output_path, &svg)?;
        println!("{} SVG written to {}", "OK".bright_green().bold(), output_path.display());
    } else {
        // Try PNG rasterization
        match banner::svg_to_png(&svg, config.width, config.height) {
            Ok(png_bytes) => {
                std::fs::write(&output_path, &png_bytes)?;
                println!(
                    "{} PNG written to {} ({} bytes)",
                    "OK".bright_green().bold(),
                    output_path.display(),
                    png_bytes.len()
                );
            }
            Err(e) => {
                // Fall back to SVG
                let svg_path = output_path.with_extension("svg");
                std::fs::write(&svg_path, &svg)?;
                println!("{} {}", "Note".bright_yellow().bold(), e);
                println!(
                    "{} SVG fallback written to {}",
                    "OK".bright_green().bold(),
                    svg_path.display()
                );
            }
        }
    }

    Ok(())
}

fn handle_reflection(
    transcript_path: PathBuf,
    topic: Option<&str>,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    let t = transcript::parse_transcript(&transcript_path)?;
    let reading = reflection::generate_reflection(&t, topic);

    match format {
        OracleOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&reading)?);
        }
        OracleOutputFormat::Markdown => {
            print!("{}", reflection::render_reflection_markdown(&reading));
        }
        _ => {
            // Text format: colorized output
            print_reflection_text(&reading);
        }
    }

    Ok(())
}

fn handle_key_concepts(transcript_path: PathBuf, format: OracleOutputFormat) -> anyhow::Result<()> {
    let t = transcript::parse_transcript(&transcript_path)?;
    let reading = key_concepts::generate_key_concepts(&t);

    match format {
        OracleOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&reading)?);
        }
        OracleOutputFormat::Markdown => {
            print!("{}", key_concepts::render_key_concepts_markdown(&reading));
        }
        _ => {
            print_key_concepts_text(&reading);
        }
    }

    Ok(())
}

fn handle_vocabulary(transcript_path: PathBuf, format: OracleOutputFormat) -> anyhow::Result<()> {
    let path = Path::new(&transcript_path);

    let transcripts = if path.is_dir() {
        transcript::parse_transcript_dir(path)?
    } else {
        vec![transcript::parse_transcript(path)?]
    };

    let entries = vocabulary::extract_vocabulary(&transcripts);

    match format {
        OracleOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&entries)?);
        }
        OracleOutputFormat::Markdown => {
            print!("{}", vocabulary::render_vocabulary_markdown(&entries));
        }
        _ => {
            print_vocabulary_text(&entries);
        }
    }

    Ok(())
}

// ============================================================================
// Text output helpers
// ============================================================================

fn print_reflection_text(reading: &crate::oracle::coursera::ReflectionReading) {
    println!("\n{}", "Key Themes".bright_yellow().bold());
    for theme in &reading.themes {
        println!("  - {theme}");
    }

    println!("\n{}", "Reflection Questions".bright_yellow().bold());
    for (i, q) in reading.questions.iter().enumerate() {
        println!("  {}. [{}] {}", i + 1, format!("{}", q.thinking_level).bright_cyan(), q.question);
    }

    println!("\n{}", "Further Reading".bright_yellow().bold());
    for cite in &reading.citations {
        println!("  {} ({}) — {}", cite.authors.dimmed(), cite.year, cite.title.bright_white());
        println!("    {}", cite.url.dimmed());
    }
    println!();
}

fn print_key_concepts_text(reading: &crate::oracle::coursera::KeyConceptsReading) {
    println!("\n{}", "Key Concepts".bright_yellow().bold());
    for concept in &reading.concepts {
        println!(
            "  {} [{}]: {}",
            concept.term.bright_white().bold(),
            format!("{}", concept.category).dimmed(),
            concept.definition
        );
    }

    if !reading.code_examples.is_empty() {
        println!("\n{}", "Code Examples".bright_yellow().bold());
        for ex in &reading.code_examples {
            println!(
                "  {} ({}): {}",
                ex.related_concept.bright_cyan(),
                ex.language,
                ex.code.lines().next().unwrap_or("")
            );
        }
    }
    println!();
}

fn print_vocabulary_text(entries: &[crate::oracle::coursera::VocabularyEntry]) {
    println!("\n{} ({} terms)", "Vocabulary".bright_yellow().bold(), entries.len());
    for entry in entries {
        println!(
            "  {} [{}] (x{}) — {}",
            entry.term.bright_white().bold(),
            format!("{}", entry.category).dimmed(),
            entry.frequency,
            entry.definition
        );
    }
    println!();
}
