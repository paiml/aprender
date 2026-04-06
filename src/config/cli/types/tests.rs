//! Tests for CLI type enums.

use super::*;

#[test]
fn test_output_format_from_str() {
    assert_eq!("text".parse::<OutputFormat>().expect("parsing should succeed"), OutputFormat::Text);
    assert_eq!("json".parse::<OutputFormat>().expect("parsing should succeed"), OutputFormat::Json);
    assert_eq!("yaml".parse::<OutputFormat>().expect("parsing should succeed"), OutputFormat::Yaml);
    assert_eq!("JSON".parse::<OutputFormat>().expect("parsing should succeed"), OutputFormat::Json);
    assert!("invalid".parse::<OutputFormat>().is_err());
}

#[test]
fn test_output_format_default() {
    assert_eq!(OutputFormat::default(), OutputFormat::Text);
}

#[test]
fn test_artifact_type_from_str() {
    assert_eq!(
        "dataset".parse::<ArtifactTypeArg>().expect("parsing should succeed"),
        ArtifactTypeArg::Dataset
    );
    assert_eq!(
        "paper".parse::<ArtifactTypeArg>().expect("parsing should succeed"),
        ArtifactTypeArg::Paper
    );
    assert_eq!(
        "model".parse::<ArtifactTypeArg>().expect("parsing should succeed"),
        ArtifactTypeArg::Model
    );
    assert_eq!(
        "code".parse::<ArtifactTypeArg>().expect("parsing should succeed"),
        ArtifactTypeArg::Code
    );
    assert_eq!(
        "notebook".parse::<ArtifactTypeArg>().expect("parsing should succeed"),
        ArtifactTypeArg::Notebook
    );
    assert_eq!(
        "workflow".parse::<ArtifactTypeArg>().expect("parsing should succeed"),
        ArtifactTypeArg::Workflow
    );
    assert!("invalid".parse::<ArtifactTypeArg>().is_err());
}

#[test]
fn test_artifact_type_display() {
    assert_eq!(format!("{}", ArtifactTypeArg::Dataset), "dataset");
    assert_eq!(format!("{}", ArtifactTypeArg::Paper), "paper");
    assert_eq!(format!("{}", ArtifactTypeArg::Model), "model");
    assert_eq!(format!("{}", ArtifactTypeArg::Code), "code");
    assert_eq!(format!("{}", ArtifactTypeArg::Notebook), "notebook");
    assert_eq!(format!("{}", ArtifactTypeArg::Workflow), "workflow");
}

#[test]
fn test_artifact_type_default() {
    assert_eq!(ArtifactTypeArg::default(), ArtifactTypeArg::Dataset);
}

#[test]
fn test_license_from_str() {
    assert_eq!(
        "cc-by-4.0".parse::<LicenseArg>().expect("parsing should succeed"),
        LicenseArg::CcBy4
    );
    assert_eq!(
        "cc-by-sa-4.0".parse::<LicenseArg>().expect("parsing should succeed"),
        LicenseArg::CcBySa4
    );
    assert_eq!("cc0".parse::<LicenseArg>().expect("parsing should succeed"), LicenseArg::Cc0);
    assert_eq!("mit".parse::<LicenseArg>().expect("parsing should succeed"), LicenseArg::Mit);
    assert_eq!(
        "apache-2.0".parse::<LicenseArg>().expect("parsing should succeed"),
        LicenseArg::Apache2
    );
    assert_eq!("gpl3".parse::<LicenseArg>().expect("parsing should succeed"), LicenseArg::Gpl3);
    assert_eq!("gplv3".parse::<LicenseArg>().expect("parsing should succeed"), LicenseArg::Gpl3);
    assert_eq!("bsd3".parse::<LicenseArg>().expect("parsing should succeed"), LicenseArg::Bsd3);
    assert!("invalid".parse::<LicenseArg>().is_err());
}

#[test]
fn test_license_display() {
    assert_eq!(format!("{}", LicenseArg::CcBy4), "CC-BY-4.0");
    assert_eq!(format!("{}", LicenseArg::CcBySa4), "CC-BY-SA-4.0");
    assert_eq!(format!("{}", LicenseArg::Cc0), "CC0");
    assert_eq!(format!("{}", LicenseArg::Mit), "MIT");
    assert_eq!(format!("{}", LicenseArg::Apache2), "Apache-2.0");
    assert_eq!(format!("{}", LicenseArg::Gpl3), "GPL-3.0");
    assert_eq!(format!("{}", LicenseArg::Bsd3), "BSD-3-Clause");
}

#[test]
fn test_license_default() {
    assert_eq!(LicenseArg::default(), LicenseArg::CcBy4);
}

#[test]
fn test_citation_format_from_str() {
    assert_eq!(
        "bibtex".parse::<CitationFormat>().expect("parsing should succeed"),
        CitationFormat::Bibtex
    );
    assert_eq!(
        "bib".parse::<CitationFormat>().expect("parsing should succeed"),
        CitationFormat::Bibtex
    );
    assert_eq!(
        "cff".parse::<CitationFormat>().expect("parsing should succeed"),
        CitationFormat::Cff
    );
    assert_eq!(
        "json".parse::<CitationFormat>().expect("parsing should succeed"),
        CitationFormat::Json
    );
    assert!("invalid".parse::<CitationFormat>().is_err());
}

#[test]
fn test_citation_format_display() {
    assert_eq!(format!("{}", CitationFormat::Bibtex), "bibtex");
    assert_eq!(format!("{}", CitationFormat::Cff), "cff");
    assert_eq!(format!("{}", CitationFormat::Json), "json");
}

#[test]
fn test_citation_format_default() {
    assert_eq!(CitationFormat::default(), CitationFormat::Bibtex);
}

#[test]
fn test_export_format_from_str() {
    assert_eq!(
        "notebook".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::Notebook
    );
    assert_eq!(
        "ipynb".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::Notebook
    );
    assert_eq!(
        "jupyter".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::Notebook
    );
    assert_eq!("html".parse::<ExportFormat>().expect("parsing should succeed"), ExportFormat::Html);
    assert_eq!(
        "anonymized".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::AnonymizedJson
    );
    assert_eq!(
        "anon".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::AnonymizedJson
    );
    assert_eq!(
        "ro-crate".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::RoCrate
    );
    assert_eq!(
        "rocrate".parse::<ExportFormat>().expect("parsing should succeed"),
        ExportFormat::RoCrate
    );
    assert!("invalid".parse::<ExportFormat>().is_err());
}

#[test]
fn test_export_format_display() {
    assert_eq!(format!("{}", ExportFormat::Notebook), "notebook");
    assert_eq!(format!("{}", ExportFormat::Html), "html");
    assert_eq!(format!("{}", ExportFormat::AnonymizedJson), "anonymized-json");
    assert_eq!(format!("{}", ExportFormat::RoCrate), "ro-crate");
}

#[test]
fn test_archive_provider_from_str() {
    assert_eq!(
        "zenodo".parse::<ArchiveProviderArg>().expect("parsing should succeed"),
        ArchiveProviderArg::Zenodo
    );
    assert_eq!(
        "figshare".parse::<ArchiveProviderArg>().expect("parsing should succeed"),
        ArchiveProviderArg::Figshare
    );
    assert_eq!(
        "dryad".parse::<ArchiveProviderArg>().expect("parsing should succeed"),
        ArchiveProviderArg::Dryad
    );
    assert_eq!(
        "dataverse".parse::<ArchiveProviderArg>().expect("parsing should succeed"),
        ArchiveProviderArg::Dataverse
    );
    assert!("invalid".parse::<ArchiveProviderArg>().is_err());
}

#[test]
fn test_archive_provider_display() {
    assert_eq!(format!("{}", ArchiveProviderArg::Zenodo), "zenodo");
    assert_eq!(format!("{}", ArchiveProviderArg::Figshare), "figshare");
    assert_eq!(format!("{}", ArchiveProviderArg::Dryad), "dryad");
    assert_eq!(format!("{}", ArchiveProviderArg::Dataverse), "dataverse");
}

#[test]
fn test_archive_provider_default() {
    assert_eq!(ArchiveProviderArg::default(), ArchiveProviderArg::Zenodo);
}

#[test]
fn test_shell_type_from_str() {
    assert_eq!("bash".parse::<ShellType>().expect("parsing should succeed"), ShellType::Bash);
    assert_eq!("zsh".parse::<ShellType>().expect("parsing should succeed"), ShellType::Zsh);
    assert_eq!("fish".parse::<ShellType>().expect("parsing should succeed"), ShellType::Fish);
    assert_eq!(
        "powershell".parse::<ShellType>().expect("parsing should succeed"),
        ShellType::PowerShell
    );
    assert_eq!("ps".parse::<ShellType>().expect("parsing should succeed"), ShellType::PowerShell);
    assert!("invalid".parse::<ShellType>().is_err());
}

#[test]
fn test_shell_type_display() {
    assert_eq!(format!("{}", ShellType::Bash), "bash");
    assert_eq!(format!("{}", ShellType::Zsh), "zsh");
    assert_eq!(format!("{}", ShellType::Fish), "fish");
    assert_eq!(format!("{}", ShellType::PowerShell), "powershell");
}

#[test]
fn test_shell_type_default() {
    assert_eq!(ShellType::default(), ShellType::Bash);
}

#[test]
fn test_inspect_mode_from_str() {
    assert_eq!(
        "summary".parse::<InspectMode>().expect("parsing should succeed"),
        InspectMode::Summary
    );
    assert_eq!(
        "outliers".parse::<InspectMode>().expect("parsing should succeed"),
        InspectMode::Outliers
    );
    assert_eq!(
        "distribution".parse::<InspectMode>().expect("parsing should succeed"),
        InspectMode::Distribution
    );
    assert_eq!(
        "dist".parse::<InspectMode>().expect("parsing should succeed"),
        InspectMode::Distribution
    );
    assert_eq!(
        "schema".parse::<InspectMode>().expect("parsing should succeed"),
        InspectMode::Schema
    );
    assert!("invalid".parse::<InspectMode>().is_err());
}

#[test]
fn test_inspect_mode_display() {
    assert_eq!(format!("{}", InspectMode::Summary), "summary");
    assert_eq!(format!("{}", InspectMode::Outliers), "outliers");
    assert_eq!(format!("{}", InspectMode::Distribution), "distribution");
    assert_eq!(format!("{}", InspectMode::Schema), "schema");
}

#[test]
fn test_inspect_mode_default() {
    assert_eq!(InspectMode::default(), InspectMode::Summary);
}

#[test]
fn test_audit_type_from_str() {
    assert_eq!("bias".parse::<AuditType>().expect("parsing should succeed"), AuditType::Bias);
    assert_eq!(
        "fairness".parse::<AuditType>().expect("parsing should succeed"),
        AuditType::Fairness
    );
    assert_eq!("privacy".parse::<AuditType>().expect("parsing should succeed"), AuditType::Privacy);
    assert_eq!(
        "security".parse::<AuditType>().expect("parsing should succeed"),
        AuditType::Security
    );
    assert!("invalid".parse::<AuditType>().is_err());
}

#[test]
fn test_audit_type_display() {
    assert_eq!(format!("{}", AuditType::Bias), "bias");
    assert_eq!(format!("{}", AuditType::Fairness), "fairness");
    assert_eq!(format!("{}", AuditType::Privacy), "privacy");
    assert_eq!(format!("{}", AuditType::Security), "security");
}

#[test]
fn test_audit_type_default() {
    assert_eq!(AuditType::default(), AuditType::Bias);
}
