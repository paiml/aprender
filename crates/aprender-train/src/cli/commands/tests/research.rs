//! Research command tests

use super::super::*;
use super::base::{create_safetensors_file, create_test_config};
use crate::cli::LogLevel;
use crate::config::*;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper to run research init command via the public API
fn run_research_init_helper(
    id: &str,
    title: &str,
    artifact_type: crate::config::ArtifactTypeArg,
    license: crate::config::LicenseArg,
    output: PathBuf,
    author: Option<String>,
    orcid: Option<String>,
    affiliation: Option<String>,
    description: Option<String>,
    keywords: Option<String>,
    doi: Option<String>,
) -> Result<(), String> {
    use crate::config::{ResearchArgs, ResearchCommand, ResearchInitArgs};

    let args = ResearchArgs {
        command: ResearchCommand::Init(ResearchInitArgs {
            id: id.to_string(),
            title: title.to_string(),
            artifact_type,
            license,
            output,
            author,
            orcid,
            affiliation,
            description,
            keywords,
            doi,
        }),
    };

    research::run_research(args, LogLevel::Quiet)
}

#[test]
fn test_research_init_basic() {
    use crate::config::{ArtifactTypeArg, LicenseArg};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("artifact.yaml");

    let result = run_research_init_helper(
        "test-artifact-001",
        "Test Research Artifact",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        output.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    );

    assert!(result.is_ok(), "Research init failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_init_with_author() {
    use crate::config::{ArtifactTypeArg, LicenseArg};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("artifact.yaml");

    let result = run_research_init_helper(
        "test-artifact-002",
        "Test Research Artifact",
        ArtifactTypeArg::Paper,
        LicenseArg::Mit,
        output.clone(),
        Some("John Doe".to_string()),
        Some("0000-0002-1825-0097".to_string()),
        Some("University of Testing".to_string()),
        Some("A test artifact for unit tests".to_string()),
        Some("test, research, artifact".to_string()),
        Some("10.1000/test".to_string()),
    );

    assert!(result.is_ok(), "Research init with author failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_init_all_artifact_types() {
    use crate::config::{ArtifactTypeArg, LicenseArg};

    let dir = TempDir::new().expect("temp file creation should succeed");

    // Test each artifact type
    let artifact_types = [
        ArtifactTypeArg::Dataset,
        ArtifactTypeArg::Paper,
        ArtifactTypeArg::Model,
        ArtifactTypeArg::Code,
        ArtifactTypeArg::Notebook,
        ArtifactTypeArg::Workflow,
    ];

    for (i, artifact_type) in artifact_types.iter().enumerate() {
        let output = dir.path().join(format!("artifact_{i}.yaml"));

        let result = run_research_init_helper(
            &format!("test-{i}"),
            &format!("Test {i}"),
            *artifact_type,
            LicenseArg::CcBy4,
            output.clone(),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert!(result.is_ok(), "Research init failed for {artifact_type:?}: {result:?}");
    }
}

#[test]
fn test_research_init_all_licenses() {
    use crate::config::{ArtifactTypeArg, LicenseArg};

    let dir = TempDir::new().expect("temp file creation should succeed");

    // Test each license type
    let licenses = [
        LicenseArg::CcBy4,
        LicenseArg::CcBySa4,
        LicenseArg::Cc0,
        LicenseArg::Mit,
        LicenseArg::Apache2,
        LicenseArg::Gpl3,
        LicenseArg::Bsd3,
    ];

    for (i, license) in licenses.iter().enumerate() {
        let output = dir.path().join(format!("license_{i}.yaml"));

        let result = run_research_init_helper(
            &format!("license-test-{i}"),
            &format!("License Test {i}"),
            ArtifactTypeArg::Dataset,
            *license,
            output.clone(),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert!(result.is_ok(), "Research init failed for {license:?}: {result:?}");
    }
}

/// Create a test artifact YAML file for citation tests
fn create_test_artifact(dir: &TempDir) -> PathBuf {
    use crate::config::{ArtifactTypeArg, LicenseArg};

    let output = dir.path().join("test_artifact.yaml");

    run_research_init_helper(
        "cite-test-artifact",
        "Test Article for Citation",
        ArtifactTypeArg::Paper,
        LicenseArg::CcBy4,
        output.clone(),
        Some("Jane Doe".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    output
}

#[test]
fn test_research_cite_bibtex() {
    use crate::config::{CitationFormat, CiteArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = create_test_artifact(&dir);
    let output = dir.path().join("citation.bib");

    let args = ResearchArgs {
        command: ResearchCommand::Cite(CiteArgs {
            artifact: artifact_path,
            year: 2024,
            format: CitationFormat::Bibtex,
            output: Some(output.clone()),
            journal: None,
            volume: None,
            pages: None,
            url: None,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Bibtex citation failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_cite_cff() {
    use crate::config::{CitationFormat, CiteArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = create_test_artifact(&dir);
    let output = dir.path().join("CITATION.cff");

    let args = ResearchArgs {
        command: ResearchCommand::Cite(CiteArgs {
            artifact: artifact_path,
            year: 2024,
            format: CitationFormat::Cff,
            output: Some(output.clone()),
            journal: Some("Journal of Testing".to_string()),
            volume: Some("42".to_string()),
            pages: Some("1-10".to_string()),
            url: Some("https://example.com/paper".to_string()),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "CFF citation failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_cite_json() {
    use crate::config::{CitationFormat, CiteArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = create_test_artifact(&dir);
    let output = dir.path().join("citation.json");

    let args = ResearchArgs {
        command: ResearchCommand::Cite(CiteArgs {
            artifact: artifact_path,
            year: 2024,
            format: CitationFormat::Json,
            output: Some(output.clone()),
            journal: None,
            volume: None,
            pages: None,
            url: None,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "JSON citation failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_cite_missing_artifact() {
    use crate::config::{CitationFormat, CiteArgs, ResearchArgs, ResearchCommand};

    let args = ResearchArgs {
        command: ResearchCommand::Cite(CiteArgs {
            artifact: PathBuf::from("/nonexistent/artifact.yaml"),
            year: 2024,
            format: CitationFormat::Bibtex,
            output: None,
            journal: None,
            volume: None,
            pages: None,
            url: None,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

#[test]
fn test_research_verify_invalid_file() {
    use crate::config::{ResearchArgs, ResearchCommand, VerifyArgs};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let file_path = dir.path().join("invalid.yaml");
    std::fs::write(&file_path, "not valid yaml: [").expect("file write should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Verify(VerifyArgs {
            file: file_path,
            public_key: None,
            original: None,
            verify_git: false,
        }),
    };

    // This should handle the case where the file isn't a valid signed pre-registration
    let result = research::run_research(args, LogLevel::Quiet);
    // The verify command doesn't fail for non-signed files, it just reports
    assert!(result.is_ok());
}

#[test]
fn test_research_verify_missing_file() {
    use crate::config::{ResearchArgs, ResearchCommand, VerifyArgs};

    let args = ResearchArgs {
        command: ResearchCommand::Verify(VerifyArgs {
            file: PathBuf::from("/nonexistent/file.yaml"),
            public_key: None,
            original: None,
            verify_git: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

#[test]
fn test_run_command_research_init() {
    use crate::config::{
        ArtifactTypeArg, LicenseArg, ResearchArgs, ResearchCommand, ResearchInitArgs,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("artifact.yaml");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Research(ResearchArgs {
            command: ResearchCommand::Init(ResearchInitArgs {
                id: "test-run-cmd".to_string(),
                title: "Test Run Command".to_string(),
                artifact_type: ArtifactTypeArg::Dataset,
                license: LicenseArg::CcBy4,
                output: output.clone(),
                author: None,
                orcid: None,
                affiliation: None,
                description: None,
                keywords: None,
                doi: None,
            }),
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_research_preregister_basic() {
    use crate::config::{PreregisterArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("preregistration.yaml");

    let args = ResearchArgs {
        command: ResearchCommand::Preregister(PreregisterArgs {
            title: "Test Research Question".to_string(),
            hypothesis: "Our hypothesis is X".to_string(),
            methodology: "We will use method Y".to_string(),
            analysis_plan: "We will analyze using Z".to_string(),
            notes: None,
            output: output.clone(),
            sign_key: None,
            git_timestamp: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Preregister failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_preregister_with_notes() {
    use crate::config::{PreregisterArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("preregistration.yaml");

    let args = ResearchArgs {
        command: ResearchCommand::Preregister(PreregisterArgs {
            title: "Test Research Question".to_string(),
            hypothesis: "Our hypothesis is X".to_string(),
            methodology: "We will use method Y".to_string(),
            analysis_plan: "We will analyze using Z".to_string(),
            notes: Some("Additional notes here".to_string()),
            output: output.clone(),
            sign_key: None,
            git_timestamp: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Preregister with notes failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_bundle_directory() {
    use crate::config::{ArtifactTypeArg, BundleArgs, LicenseArg, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let output = dir.path().join("ro-crate");

    // Create artifact first
    run_research_init_helper(
        "bundle-test",
        "Bundle Test Artifact",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Bundle(BundleArgs {
            artifact: artifact_path,
            output: output.clone(),
            file: vec![],
            zip: false,
            include_citations: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Bundle directory failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_bundle_zip() {
    use crate::config::{ArtifactTypeArg, BundleArgs, LicenseArg, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let output = dir.path().join("ro-crate");

    // Create artifact first
    run_research_init_helper(
        "bundle-zip-test",
        "Bundle ZIP Test Artifact",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Bundle(BundleArgs {
            artifact: artifact_path,
            output: output.clone(),
            file: vec![],
            zip: true,
            include_citations: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Bundle ZIP failed: {result:?}");
}

#[test]
fn test_research_bundle_with_files() {
    use crate::config::{ArtifactTypeArg, BundleArgs, LicenseArg, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let output = dir.path().join("ro-crate");
    let data_file = dir.path().join("data.txt");

    // Create artifact and data file
    run_research_init_helper(
        "bundle-files-test",
        "Bundle with Files Test",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");
    std::fs::write(&data_file, "test data content").expect("file write should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Bundle(BundleArgs {
            artifact: artifact_path,
            output: output.clone(),
            file: vec![data_file],
            zip: false,
            include_citations: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Bundle with files failed: {result:?}");
}

#[test]
fn test_research_deposit_dry_run() {
    use crate::config::{
        ArchiveProviderArg, ArtifactTypeArg, DepositArgs, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");

    // Create artifact first
    run_research_init_helper(
        "deposit-test",
        "Deposit Test Artifact",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Deposit(DepositArgs {
            artifact: artifact_path,
            provider: ArchiveProviderArg::Zenodo,
            token: None,
            sandbox: false,
            community: None,
            file: vec![],
            dry_run: true,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Deposit dry run failed: {result:?}");
}

#[test]
fn test_research_deposit_all_providers() {
    use crate::config::{
        ArchiveProviderArg, ArtifactTypeArg, DepositArgs, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");

    // Create artifact first
    run_research_init_helper(
        "deposit-providers-test",
        "Deposit Providers Test",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let providers = [
        ArchiveProviderArg::Zenodo,
        ArchiveProviderArg::Figshare,
        ArchiveProviderArg::Dryad,
        ArchiveProviderArg::Dataverse,
    ];

    for provider in providers {
        let args = ResearchArgs {
            command: ResearchCommand::Deposit(DepositArgs {
                artifact: artifact_path.clone(),
                provider: provider,
                token: None,
                sandbox: false,
                community: None,
                file: vec![],
                dry_run: true,
            }),
        };

        let result = research::run_research(args, LogLevel::Quiet);
        assert!(result.is_ok(), "Deposit failed for {provider:?}: {result:?}");
    }
}

#[test]
fn test_research_export_notebook() {
    use crate::config::{ExportArgs, ExportFormat, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let input = dir.path().join("document.md");
    let output = dir.path().join("notebook.ipynb");

    // Create a literate document
    std::fs::write(&input, "# Title\n\nSome text.\n\n```python\nprint('hello')\n```\n")
        .expect("file write should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input,
            format: ExportFormat::Notebook,
            output: output.clone(),
            anonymize: false,
            anon_salt: None,
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Export notebook failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_export_html() {
    use crate::config::{ExportArgs, ExportFormat, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let input = dir.path().join("document.md");
    let output = dir.path().join("output.html");

    // Create a literate document
    std::fs::write(&input, "# Title\n\nSome **bold** text.\n").expect("file write should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input,
            format: ExportFormat::Html,
            output: output.clone(),
            anonymize: false,
            anon_salt: None,
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Export HTML failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_export_anonymized() {
    use crate::config::{
        ArtifactTypeArg, ExportArgs, ExportFormat, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let output = dir.path().join("anon.json");

    // Create artifact first
    run_research_init_helper(
        "anon-test",
        "Anonymization Test",
        ArtifactTypeArg::Paper,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        Some("John Doe".to_string()),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input: artifact_path,
            format: ExportFormat::AnonymizedJson,
            output: output.clone(),
            anonymize: true,
            anon_salt: Some("test-salt-12345".to_string()),
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok(), "Export anonymized failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_research_export_anonymized_missing_flag() {
    use crate::config::{
        ArtifactTypeArg, ExportArgs, ExportFormat, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let output = dir.path().join("anon.json");

    // Create artifact first
    run_research_init_helper(
        "anon-missing-test",
        "Anonymization Missing Flag Test",
        ArtifactTypeArg::Paper,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input: artifact_path,
            format: ExportFormat::AnonymizedJson,
            output,
            anonymize: false, // Missing flag!
            anon_salt: Some("test-salt".to_string()),
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("--anonymize flag required"));
}

#[test]
fn test_research_export_rocrate_error() {
    use crate::config::{ExportArgs, ExportFormat, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let input = dir.path().join("input.md");
    let output = dir.path().join("output");

    std::fs::write(&input, "# Test\n").expect("file write should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input,
            format: ExportFormat::RoCrate,
            output,
            anonymize: false,
            anon_salt: None,
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Use 'entrenar research bundle'"));
}

// Test info command with different formats
#[test]
fn test_info_command_json() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = InfoArgs { config: config_path, format: OutputFormat::Json };

    let result = info::run_info(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_info_command_yaml() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = InfoArgs { config: config_path, format: OutputFormat::Yaml };

    let result = info::run_info(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test deposit without dry_run
#[test]
fn test_research_deposit_no_dry_run() {
    use crate::config::{
        ArchiveProviderArg, ArtifactTypeArg, DepositArgs, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");

    run_research_init_helper(
        "deposit-nodry-test",
        "Deposit No Dry Run Test",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Deposit(DepositArgs {
            artifact: artifact_path,
            provider: ArchiveProviderArg::Zenodo,
            token: None,
            sandbox: false,
            community: None,
            file: vec![],
            dry_run: false, // Test without dry run
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test deposit with files
#[test]
fn test_research_deposit_with_files() {
    use crate::config::{
        ArchiveProviderArg, ArtifactTypeArg, DepositArgs, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let data_file = dir.path().join("data.txt");

    run_research_init_helper(
        "deposit-files-test",
        "Deposit with Files Test",
        ArtifactTypeArg::Dataset,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    std::fs::write(&data_file, "test data").expect("file write should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Deposit(DepositArgs {
            artifact: artifact_path,
            provider: ArchiveProviderArg::Figshare,
            token: None,
            sandbox: false,
            community: None,
            file: vec![data_file],
            dry_run: true,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test info command verbose
#[test]
fn test_info_command_verbose() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = InfoArgs { config: config_path, format: OutputFormat::Text };

    let result = info::run_info(args, LogLevel::Verbose);
    assert!(result.is_ok());
}

// Test validate verbose
#[test]
fn test_validate_command_verbose() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = ValidateArgs { config: config_path, detailed: true };

    let result = validate::run_validate(args, LogLevel::Verbose);
    assert!(result.is_ok());
}

// Test export missing anon_salt
#[test]
fn test_research_export_anonymized_missing_salt() {
    use crate::config::{
        ArtifactTypeArg, ExportArgs, ExportFormat, LicenseArg, ResearchArgs, ResearchCommand,
    };

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = dir.path().join("artifact.yaml");
    let output = dir.path().join("anon.json");

    run_research_init_helper(
        "anon-nosalt-test",
        "Anonymization No Salt Test",
        ArtifactTypeArg::Paper,
        LicenseArg::CcBy4,
        artifact_path.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("operation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input: artifact_path,
            format: ExportFormat::AnonymizedJson,
            output,
            anonymize: true,
            anon_salt: None, // Missing salt!
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("--anon-salt required"));
}

// Test quantize with calibration data
#[test]
fn test_quantize_command_with_calibration() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let calib_path = dir.path().join("calibration.json");
    let output = dir.path().join("quantized.json");

    std::fs::write(&calib_path, "[]").expect("file write should succeed");

    let args = QuantizeArgs {
        model: model_path,
        output: output.clone(),
        bits: 4,
        method: QuantMethod::Symmetric,
        per_channel: false,
        calibration_data: Some(calib_path),
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Verbose);
    assert!(result.is_ok());
}

// Test bundle missing artifact
#[test]
fn test_research_bundle_missing_artifact() {
    use crate::config::{BundleArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Bundle(BundleArgs {
            artifact: PathBuf::from("/nonexistent/artifact.yaml"),
            output: dir.path().join("ro-crate"),
            file: vec![],
            zip: false,
            include_citations: false,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

// Test deposit missing artifact
#[test]
fn test_research_deposit_missing_artifact() {
    use crate::config::{ArchiveProviderArg, DepositArgs, ResearchArgs, ResearchCommand};

    let args = ResearchArgs {
        command: ResearchCommand::Deposit(DepositArgs {
            artifact: PathBuf::from("/nonexistent/artifact.yaml"),
            provider: ArchiveProviderArg::Zenodo,
            token: None,
            sandbox: false,
            community: None,
            file: vec![],
            dry_run: true,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

// Test export missing input
#[test]
fn test_research_export_missing_input() {
    use crate::config::{ExportArgs, ExportFormat, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");

    let args = ResearchArgs {
        command: ResearchCommand::Export(ExportArgs {
            input: PathBuf::from("/nonexistent/input.md"),
            format: ExportFormat::Notebook,
            output: dir.path().join("output.ipynb"),
            anonymize: false,
            anon_salt: None,
            kernel: "python3".to_string(),
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

// Test merge with weights count mismatch
#[test]
fn test_merge_command_weights_count_mismatch() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2],
        output,
        method: MergeMethod::Average,
        weight: None,
        density: None,
        weights: Some("0.5,0.3,0.2".to_string()), // 3 weights for 2 models
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
}

// Test init command stdout output
#[test]
fn test_init_command_stdout() {
    let args = InitArgs {
        name: "stdout_test".to_string(),
        output: None, // stdout
        template: InitTemplate::Minimal,
        model: None,
        base: None,
        method: None,
        data: None,
    };

    let result = init::run_init(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test init with qlora template
#[test]
fn test_init_command_qlora_template() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output_path = dir.path().join("qlora_config.yaml");

    let args = InitArgs {
        name: "qlora_project".to_string(),
        output: Some(output_path.clone()),
        template: InitTemplate::Qlora,
        model: Some("/path/to/model".to_string()),
        base: None,
        method: None,
        data: Some("/path/to/data".to_string()),
    };

    let result = init::run_init(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test init with full template
#[test]
fn test_init_command_full_template() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output_path = dir.path().join("full_config.yaml");

    let args = InitArgs {
        name: "full_project".to_string(),
        output: Some(output_path.clone()),
        template: InitTemplate::Full,
        model: None,
        base: None,
        method: None,
        data: None,
    };

    let result = init::run_init(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test validate with invalid YAML
#[test]
fn test_validate_command_invalid_yaml() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = dir.path().join("invalid.yaml");
    std::fs::write(&config_path, "invalid: [yaml: content").expect("file write should succeed");

    let args = ValidateArgs { config: config_path, detailed: false };

    let result = validate::run_validate(args, LogLevel::Quiet);
    assert!(result.is_err());
}

// Test inspect with columns filter
#[test]
fn test_inspect_command_with_columns() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let csv_path = dir.path().join("data.csv");
    std::fs::write(&csv_path, b"col1,col2,col3\n1,2,3\n4,5,6").expect("file write should succeed");

    let args = InspectArgs {
        input: csv_path,
        mode: InspectMode::Summary,
        columns: Some("col1,col2".to_string()),
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

// Test missing model for merge
#[test]
fn test_merge_command_missing_model() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, PathBuf::from("/nonexistent/model.safetensors")],
        output,
        method: MergeMethod::Slerp,
        weight: Some(0.5),
        density: None,
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
}

// Test quantize missing model
#[test]
fn test_quantize_command_missing_model() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("quantized.json");

    let args = QuantizeArgs {
        model: PathBuf::from("/nonexistent/model.safetensors"),
        output,
        bits: 4,
        method: QuantMethod::Symmetric,
        per_channel: false,
        calibration_data: None,
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Quiet);
    assert!(result.is_err());
}

// Test info command missing config
#[test]
fn test_info_command_missing_config() {
    let args =
        InfoArgs { config: PathBuf::from("/nonexistent/config.yaml"), format: OutputFormat::Text };

    let result = info::run_info(args, LogLevel::Quiet);
    assert!(result.is_err());
}

// Test cite stdout output
#[test]
fn test_research_cite_stdout() {
    use crate::config::{CitationFormat, CiteArgs, ResearchArgs, ResearchCommand};

    let dir = TempDir::new().expect("temp file creation should succeed");
    let artifact_path = create_test_artifact(&dir);

    let args = ResearchArgs {
        command: ResearchCommand::Cite(CiteArgs {
            artifact: artifact_path,
            year: 2024,
            format: CitationFormat::Bibtex,
            output: None, // stdout
            journal: None,
            volume: None,
            pages: None,
            url: None,
        }),
    };

    let result = research::run_research(args, LogLevel::Quiet);
    assert!(result.is_ok());
}
