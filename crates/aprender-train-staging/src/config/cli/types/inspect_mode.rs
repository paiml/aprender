//! Inspection mode type for CLI commands.

/// Inspection mode
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InspectMode {
    #[default]
    Summary,
    Outliers,
    Distribution,
    Schema,
}

impl std::str::FromStr for InspectMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "summary" => Ok(InspectMode::Summary),
            "outliers" => Ok(InspectMode::Outliers),
            "distribution" | "dist" => Ok(InspectMode::Distribution),
            "schema" => Ok(InspectMode::Schema),
            _ => Err(format!(
                "Unknown inspect mode: {s}. Valid modes: summary, outliers, distribution, schema"
            )),
        }
    }
}

impl std::fmt::Display for InspectMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InspectMode::Summary => write!(f, "summary"),
            InspectMode::Outliers => write!(f, "outliers"),
            InspectMode::Distribution => write!(f, "distribution"),
            InspectMode::Schema => write!(f, "schema"),
        }
    }
}
