//! Citation format type for CLI commands.

/// Citation format for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CitationFormat {
    #[default]
    Bibtex,
    Cff,
    Json,
}

impl std::str::FromStr for CitationFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bibtex" | "bib" => Ok(CitationFormat::Bibtex),
            "cff" | "citation.cff" => Ok(CitationFormat::Cff),
            "json" => Ok(CitationFormat::Json),
            _ => Err(format!("Unknown citation format: {s}. Valid formats: bibtex, cff, json")),
        }
    }
}

impl std::fmt::Display for CitationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CitationFormat::Bibtex => write!(f, "bibtex"),
            CitationFormat::Cff => write!(f, "cff"),
            CitationFormat::Json => write!(f, "json"),
        }
    }
}
