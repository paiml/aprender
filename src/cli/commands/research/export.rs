//! Research export subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{ExportArgs, ExportFormat};
use crate::research::{AnonymizationConfig, LiterateDocument, NotebookExporter, ResearchArtifact};
use std::path::Path;

pub fn run_research_export(args: ExportArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Exporting: {} -> {}", args.input.display(), args.output.display()),
    );

    match args.format {
        ExportFormat::Notebook => export_notebook(&args.input, &args.output, level),
        ExportFormat::Html => export_html(&args.input, &args.output, level),
        ExportFormat::AnonymizedJson => export_anonymized_json(&args, level),
        ExportFormat::RoCrate => {
            Err("Use 'entrenar research bundle' for RO-Crate export".to_string())
        }
    }
}

/// Export a literate document as a Jupyter notebook
fn export_notebook(input: &Path, output: &Path, level: LogLevel) -> Result<(), String> {
    let content =
        std::fs::read_to_string(input).map_err(|e| format!("Failed to read input: {e}"))?;

    let doc = LiterateDocument::parse_markdown(&content);
    let notebook = NotebookExporter::from_literate(&doc);

    let ipynb = notebook.to_ipynb();
    std::fs::write(output, &ipynb).map_err(|e| format!("Failed to write notebook: {e}"))?;

    log(level, LogLevel::Normal, &format!("Notebook exported: {} cells", notebook.cell_count()));
    Ok(())
}

/// Export a literate document as HTML
fn export_html(input: &Path, output: &Path, level: LogLevel) -> Result<(), String> {
    let content =
        std::fs::read_to_string(input).map_err(|e| format!("Failed to read input: {e}"))?;

    let doc = LiterateDocument::parse_markdown(&content);
    let html = doc.to_html();

    std::fs::write(output, &html).map_err(|e| format!("Failed to write HTML: {e}"))?;

    log(level, LogLevel::Normal, "HTML exported successfully");
    Ok(())
}

/// Export a research artifact as anonymized JSON
fn export_anonymized_json(args: &ExportArgs, level: LogLevel) -> Result<(), String> {
    if !args.anonymize {
        return Err("--anonymize flag required for anonymized export".to_string());
    }

    let salt = args.anon_salt.as_ref().ok_or("--anon-salt required for anonymization")?;

    let yaml = std::fs::read_to_string(&args.input)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    let config = AnonymizationConfig::new(salt);
    let anon = config.anonymize(&artifact);

    let json = serde_json::to_string_pretty(&anon).map_err(|e| format!("JSON error: {e}"))?;

    std::fs::write(&args.output, &json).map_err(|e| format!("Failed to write JSON: {e}"))?;

    log(level, LogLevel::Normal, &format!("Anonymized artifact: {}", anon.anonymous_id));
    Ok(())
}
