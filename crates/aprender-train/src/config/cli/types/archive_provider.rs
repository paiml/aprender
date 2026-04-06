//! Archive provider type for CLI commands.

/// Archive provider for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArchiveProviderArg {
    #[default]
    Zenodo,
    Figshare,
    Dryad,
    Dataverse,
}

impl std::str::FromStr for ArchiveProviderArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "zenodo" => Ok(ArchiveProviderArg::Zenodo),
            "figshare" => Ok(ArchiveProviderArg::Figshare),
            "dryad" => Ok(ArchiveProviderArg::Dryad),
            "dataverse" => Ok(ArchiveProviderArg::Dataverse),
            _ => Err(format!(
                "Unknown archive provider: {s}. Valid providers: zenodo, figshare, dryad, dataverse"
            )),
        }
    }
}

impl std::fmt::Display for ArchiveProviderArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveProviderArg::Zenodo => write!(f, "zenodo"),
            ArchiveProviderArg::Figshare => write!(f, "figshare"),
            ArchiveProviderArg::Dryad => write!(f, "dryad"),
            ArchiveProviderArg::Dataverse => write!(f, "dataverse"),
        }
    }
}
