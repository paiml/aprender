//! Research deposit subcommand

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{ArchiveProviderArg, DepositArgs};
use crate::research::{ArchiveDeposit, ArchiveProvider, ResearchArtifact};

pub fn run_research_deposit(args: DepositArgs, level: LogLevel) -> Result<(), String> {
    let provider = match args.provider {
        ArchiveProviderArg::Zenodo => ArchiveProvider::Zenodo,
        ArchiveProviderArg::Figshare => ArchiveProvider::Figshare,
        ArchiveProviderArg::Dryad => ArchiveProvider::Dryad,
        ArchiveProviderArg::Dataverse => ArchiveProvider::Dataverse,
    };

    log(level, LogLevel::Normal, &format!("Preparing deposit to: {provider}"));

    // Load artifact
    let yaml = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("Failed to read artifact: {e}"))?;

    let artifact: ResearchArtifact =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse artifact: {e}"))?;

    let mut deposit = ArchiveDeposit::new(provider, artifact);

    // Add files
    for file_path in &args.file {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read {}: {e}", file_path.display()))?;

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| format!("Invalid file name: {}", file_path.display()))?;

        deposit = deposit.with_text_file(file_name, &content);
    }

    if args.dry_run {
        log(level, LogLevel::Normal, "Dry run - deposit validated:");
        log(level, LogLevel::Normal, &format!("  Provider: {}", deposit.provider));
        log(level, LogLevel::Normal, &format!("  Title: {}", deposit.metadata.title));
        log(level, LogLevel::Normal, &format!("  Files: {}", deposit.files.len()));
        log(level, LogLevel::Verbose, &format!("  Base URL: {}", provider.base_url()));
    } else {
        // Note: Actual deposit would require async HTTP client
        // For now, we just validate the deposit structure
        log(
            level,
            LogLevel::Normal,
            "Deposit prepared (actual upload requires API token and network access)",
        );
        log(
            level,
            LogLevel::Normal,
            &format!("  Provider: {} ({})", deposit.provider, provider.base_url()),
        );
        log(level, LogLevel::Normal, &format!("  Files ready: {}", deposit.files.len()));
    }

    Ok(())
}
