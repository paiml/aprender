//! Export format type for CLI commands.

/// Export format for CLI
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExportFormat {
    Notebook,
    Html,
    AnonymizedJson,
    RoCrate,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "notebook" | "ipynb" | "jupyter" => Ok(ExportFormat::Notebook),
            "html" => Ok(ExportFormat::Html),
            "anonymized" | "anon" | "anonymized-json" => Ok(ExportFormat::AnonymizedJson),
            "ro-crate" | "rocrate" => Ok(ExportFormat::RoCrate),
            _ => Err(format!(
                "Unknown export format: {s}. Valid formats: notebook, html, anonymized, ro-crate"
            )),
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Notebook => write!(f, "notebook"),
            ExportFormat::Html => write!(f, "html"),
            ExportFormat::AnonymizedJson => write!(f, "anonymized-json"),
            ExportFormat::RoCrate => write!(f, "ro-crate"),
        }
    }
}
