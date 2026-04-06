//! Tests for research command types

use super::*;
use crate::config::cli::parse_args;
use crate::config::cli::types::{
    ArchiveProviderArg, ArtifactTypeArg, CitationFormat, ExportFormat, LicenseArg,
};
use std::path::PathBuf;

#[test]
fn test_parse_research_init() {
    let cli = parse_args([
        "entrenar",
        "research",
        "init",
        "--id",
        "dataset-2024",
        "--title",
        "My Dataset",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Init(init_args) => {
                assert_eq!(init_args.id, "dataset-2024");
                assert_eq!(init_args.title, "My Dataset");
                assert_eq!(init_args.artifact_type, ArtifactTypeArg::Dataset);
                assert_eq!(init_args.license, LicenseArg::CcBy4);
            }
            _ => panic!("Expected Init subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_preregister() {
    let cli = parse_args([
        "entrenar",
        "research",
        "preregister",
        "--title",
        "Effect of X on Y",
        "--hypothesis",
        "X improves Y by 20%",
        "--methodology",
        "RCT, n=100",
        "--analysis-plan",
        "t-test, alpha=0.05",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Preregister(prereg_args) => {
                assert_eq!(prereg_args.title, "Effect of X on Y");
                assert_eq!(prereg_args.hypothesis, "X improves Y by 20%");
            }
            _ => panic!("Expected Preregister subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_cite() {
    let cli = parse_args(["entrenar", "research", "cite", "artifact.yaml", "--year", "2024"])
        .expect("parsing should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Cite(cite_args) => {
                assert_eq!(cite_args.artifact, PathBuf::from("artifact.yaml"));
                assert_eq!(cite_args.year, 2024);
                assert_eq!(cite_args.format, CitationFormat::Bibtex);
            }
            _ => panic!("Expected Cite subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_export() {
    let cli = parse_args([
        "entrenar",
        "research",
        "export",
        "document.md",
        "--format",
        "notebook",
        "--output",
        "analysis.ipynb",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Export(export_args) => {
                assert_eq!(export_args.input, PathBuf::from("document.md"));
                assert_eq!(export_args.format, ExportFormat::Notebook);
            }
            _ => panic!("Expected Export subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_deposit() {
    let cli = parse_args([
        "entrenar",
        "research",
        "deposit",
        "artifact.yaml",
        "--provider",
        "zenodo",
        "--sandbox",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Deposit(deposit_args) => {
                assert_eq!(deposit_args.provider, ArchiveProviderArg::Zenodo);
                assert!(deposit_args.sandbox);
            }
            _ => panic!("Expected Deposit subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_bundle() {
    let cli = parse_args([
        "entrenar",
        "research",
        "bundle",
        "artifact.yaml",
        "--output",
        "./ro-crate",
        "--zip",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Bundle(bundle_args) => {
                assert_eq!(bundle_args.output, PathBuf::from("./ro-crate"));
                assert!(bundle_args.zip);
            }
            _ => panic!("Expected Bundle subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_verify() {
    let cli =
        parse_args(["entrenar", "research", "verify", "preregistration.yaml", "--verify-git"])
            .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Verify(verify_args) => {
                assert_eq!(verify_args.file, PathBuf::from("preregistration.yaml"));
                assert!(verify_args.verify_git);
            }
            _ => panic!("Expected Verify subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

// Additional coverage tests for optional fields

#[test]
fn test_parse_research_init_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "init",
        "--id",
        "model-2024",
        "--title",
        "My Model",
        "--artifact-type",
        "model",
        "--license",
        "mit",
        "--output",
        "model.yaml",
        "--author",
        "John Doe",
        "--orcid",
        "0000-0002-1825-0097",
        "--affiliation",
        "University of Test",
        "--description",
        "A test model",
        "--keywords",
        "ml,ai,test",
        "--doi",
        "10.5281/zenodo.12345",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Init(init_args) => {
                assert_eq!(init_args.artifact_type, ArtifactTypeArg::Model);
                assert_eq!(init_args.license, LicenseArg::Mit);
                assert_eq!(init_args.author, Some("John Doe".to_string()));
                assert_eq!(init_args.orcid, Some("0000-0002-1825-0097".to_string()));
                assert_eq!(init_args.affiliation, Some("University of Test".to_string()));
                assert_eq!(init_args.description, Some("A test model".to_string()));
                assert_eq!(init_args.keywords, Some("ml,ai,test".to_string()));
                assert_eq!(init_args.doi, Some("10.5281/zenodo.12345".to_string()));
            }
            _ => panic!("Expected Init subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_preregister_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "preregister",
        "--title",
        "Test Study",
        "--hypothesis",
        "H1: X > Y",
        "--methodology",
        "Survey study",
        "--analysis-plan",
        "ANOVA",
        "--notes",
        "Additional notes here",
        "--output",
        "prereg.yaml",
        "--git-timestamp",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Preregister(prereg_args) => {
                assert_eq!(prereg_args.notes, Some("Additional notes here".to_string()));
                assert_eq!(prereg_args.output, PathBuf::from("prereg.yaml"));
                assert!(prereg_args.git_timestamp);
            }
            _ => panic!("Expected Preregister subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_cite_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "cite",
        "artifact.yaml",
        "--year",
        "2024",
        "--format",
        "cff",
        "--output",
        "citation.txt",
        "--journal",
        "Nature ML",
        "--volume",
        "12",
        "--pages",
        "100-120",
        "--url",
        "https://example.com",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Cite(cite_args) => {
                assert_eq!(cite_args.format, CitationFormat::Cff);
                assert_eq!(cite_args.output, Some(PathBuf::from("citation.txt")));
                assert_eq!(cite_args.journal, Some("Nature ML".to_string()));
                assert_eq!(cite_args.volume, Some("12".to_string()));
                assert_eq!(cite_args.pages, Some("100-120".to_string()));
                assert_eq!(cite_args.url, Some("https://example.com".to_string()));
            }
            _ => panic!("Expected Cite subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_export_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "export",
        "doc.md",
        "--format",
        "html",
        "--output",
        "doc.html",
        "--anonymize",
        "--anon-salt",
        "secret123",
        "--kernel",
        "julia",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Export(export_args) => {
                assert_eq!(export_args.format, ExportFormat::Html);
                assert!(export_args.anonymize);
                assert_eq!(export_args.anon_salt, Some("secret123".to_string()));
                assert_eq!(export_args.kernel, "julia");
            }
            _ => panic!("Expected Export subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_deposit_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "deposit",
        "artifact.yaml",
        "--provider",
        "figshare",
        "--token",
        "my-api-token",
        "--community",
        "ml-research",
        "--file",
        "data.csv",
        "--file",
        "model.bin",
        "--dry-run",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Deposit(deposit_args) => {
                assert_eq!(deposit_args.provider, ArchiveProviderArg::Figshare);
                assert_eq!(deposit_args.token, Some("my-api-token".to_string()));
                assert_eq!(deposit_args.community, Some("ml-research".to_string()));
                assert_eq!(deposit_args.file.len(), 2);
                assert!(deposit_args.dry_run);
            }
            _ => panic!("Expected Deposit subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_bundle_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "bundle",
        "artifact.yaml",
        "--output",
        "./bundle",
        "--file",
        "data.csv",
        "--include-citations",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Bundle(bundle_args) => {
                assert_eq!(bundle_args.file.len(), 1);
                assert!(bundle_args.include_citations);
                assert!(!bundle_args.zip);
            }
            _ => panic!("Expected Bundle subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_parse_research_verify_all_options() {
    let cli = parse_args([
        "entrenar",
        "research",
        "verify",
        "prereg.yaml",
        "--public-key",
        "key.pub",
        "--original",
        "original.yaml",
    ])
    .expect("operation should succeed");

    match cli.command {
        crate::config::cli::Command::Research(args) => match args.command {
            ResearchCommand::Verify(verify_args) => {
                assert_eq!(verify_args.public_key, Some(PathBuf::from("key.pub")));
                assert_eq!(verify_args.original, Some(PathBuf::from("original.yaml")));
                assert!(!verify_args.verify_git);
            }
            _ => panic!("Expected Verify subcommand"),
        },
        _ => panic!("Expected Research command"),
    }
}

#[test]
fn test_research_args_debug() {
    let args = ResearchInitArgs {
        id: "test".to_string(),
        title: "Test".to_string(),
        artifact_type: ArtifactTypeArg::Dataset,
        license: LicenseArg::CcBy4,
        output: PathBuf::from("test.yaml"),
        author: None,
        orcid: None,
        affiliation: None,
        description: None,
        keywords: None,
        doi: None,
    };
    let debug = format!("{args:?}");
    assert!(debug.contains("ResearchInitArgs"));
}

#[test]
fn test_research_command_clone() {
    let init = ResearchCommand::Init(ResearchInitArgs {
        id: "test".to_string(),
        title: "Test".to_string(),
        artifact_type: ArtifactTypeArg::Dataset,
        license: LicenseArg::CcBy4,
        output: PathBuf::from("test.yaml"),
        author: None,
        orcid: None,
        affiliation: None,
        description: None,
        keywords: None,
        doi: None,
    });
    let cloned = init.clone();
    assert_eq!(init, cloned);
}
