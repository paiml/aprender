//! Research cite subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{CitationFormat, CiteArgs};
use crate::research::{CitationMetadata, ResearchArtifact};

pub fn run_research_cite(args: CiteArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Generating citation from: {}", args.artifact.display()));

    // Load artifact
    let yaml = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    // Create citation
    let mut citation = CitationMetadata::new(artifact, args.year);

    if let Some(journal) = &args.journal {
        citation = citation.with_journal(journal);
    }
    if let Some(volume) = &args.volume {
        citation = citation.with_volume(volume);
    }
    if let Some(pages) = &args.pages {
        citation = citation.with_pages(pages);
    }
    if let Some(url) = &args.url {
        citation = citation.with_url(url);
    }

    // Generate output
    let output = match args.format {
        CitationFormat::Bibtex => citation.to_bibtex(),
        CitationFormat::Cff => citation.to_cff(),
        CitationFormat::Json => {
            serde_json::to_string_pretty(&citation).map_err(|e| format!("JSON error: {e}"))?
        }
    };

    // Write or print
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, &output).map_err(|e| format!("Failed to write file: {e}"))?;
        log(level, LogLevel::Normal, &format!("Citation saved to: {}", output_path.display()));
    } else {
        println!("{output}");
    }

    Ok(())
}
