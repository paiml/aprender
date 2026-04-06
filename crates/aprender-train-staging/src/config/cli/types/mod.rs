//! CLI type enums for formats, licenses, artifact types, etc.

mod archive_provider;
mod artifact_type;
mod audit_type;
mod citation_format;
mod export_format;
mod inspect_mode;
mod license;
mod output_format;
mod shell_type;

#[cfg(test)]
mod tests;

pub use archive_provider::ArchiveProviderArg;
pub use artifact_type::ArtifactTypeArg;
pub use audit_type::AuditType;
pub use citation_format::CitationFormat;
pub use export_format::ExportFormat;
pub use inspect_mode::InspectMode;
pub use license::LicenseArg;
pub use output_format::OutputFormat;
pub use shell_type::ShellType;
