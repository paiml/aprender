//! Type of research artifact.

use serde::{Deserialize, Serialize};

/// Type of research artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Structured data collection
    Dataset,
    /// Trained model weights
    Model,
    /// Source code
    Code,
    /// Academic paper or preprint
    Paper,
    /// Jupyter or computational notebook
    Notebook,
    /// Computational workflow (e.g., Snakemake, CWL)
    Workflow,
}

impl std::fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => write!(f, "Dataset"),
            Self::Model => write!(f, "Model"),
            Self::Code => write!(f, "Code"),
            Self::Paper => write!(f, "Paper"),
            Self::Notebook => write!(f, "Notebook"),
            Self::Workflow => write!(f, "Workflow"),
        }
    }
}
