//! Research bundle subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::BundleArgs;
use crate::research::{ResearchArtifact, RoCrate};

pub fn run_research_bundle(args: BundleArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Bundling RO-Crate: {}", args.output.display()));

    // Load artifact
    let yaml = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    let mut crate_pkg = RoCrate::from_artifact(&artifact, &args.output);

    // Add files
    for file_path in &args.file {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read {}: {e}", file_path.display()))?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("Invalid file name: {}", file_path.display()))?;

        crate_pkg.add_text_file(file_name, &content);
    }

    if args.zip {
        let zip_path = args.output.with_extension("zip");
        let zip_data = crate_pkg.to_zip().map_err(|e| format!("ZIP error: {e}"))?;
        std::fs::write(&zip_path, &zip_data).map_err(|e| format!("Failed to write ZIP: {e}"))?;

        log(
            level,
            LogLevel::Normal,
            &format!("RO-Crate ZIP created: {} ({} bytes)", zip_path.display(), zip_data.len()),
        );
    } else {
        crate_pkg.to_directory().map_err(|e| format!("Failed to create directory: {e}"))?;

        log(
            level,
            LogLevel::Normal,
            &format!(
                "RO-Crate directory created: {} ({} entities)",
                args.output.display(),
                crate_pkg.entity_count()
            ),
        );
    }

    Ok(())
}
