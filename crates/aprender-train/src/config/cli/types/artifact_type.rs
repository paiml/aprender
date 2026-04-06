//! Artifact type for CLI commands.

/// Artifact type for CLI
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ArtifactTypeArg {
    #[default]
    Dataset,
    Paper,
    Model,
    Code,
    Notebook,
    Workflow,
}

impl std::str::FromStr for ArtifactTypeArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dataset" => Ok(ArtifactTypeArg::Dataset),
            "paper" => Ok(ArtifactTypeArg::Paper),
            "model" => Ok(ArtifactTypeArg::Model),
            "code" => Ok(ArtifactTypeArg::Code),
            "notebook" => Ok(ArtifactTypeArg::Notebook),
            "workflow" => Ok(ArtifactTypeArg::Workflow),
            _ => Err(format!(
                "Unknown artifact type: {s}. Valid types: dataset, paper, model, code, notebook, workflow"
            )),
        }
    }
}

impl std::fmt::Display for ArtifactTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactTypeArg::Dataset => write!(f, "dataset"),
            ArtifactTypeArg::Paper => write!(f, "paper"),
            ArtifactTypeArg::Model => write!(f, "model"),
            ArtifactTypeArg::Code => write!(f, "code"),
            ArtifactTypeArg::Notebook => write!(f, "notebook"),
            ArtifactTypeArg::Workflow => write!(f, "workflow"),
        }
    }
}
