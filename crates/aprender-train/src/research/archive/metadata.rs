//! Deposit metadata types.

use crate::research::artifact::ResearchArtifact;
use crate::research::citation::CitationMetadata;
use serde::{Deserialize, Serialize};

use super::identifiers::RelatedIdentifier;

/// Resource type for archives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Dataset,
    Software,
    Publication,
    Presentation,
    Poster,
    Image,
    Video,
    Other,
}

impl ResourceType {
    /// Convert from artifact type
    pub fn from_artifact_type(artifact_type: crate::research::artifact::ArtifactType) -> Self {
        use crate::research::artifact::ArtifactType;
        match artifact_type {
            ArtifactType::Dataset => Self::Dataset,
            ArtifactType::Model | ArtifactType::Code => Self::Software,
            ArtifactType::Paper => Self::Publication,
            ArtifactType::Notebook | ArtifactType::Workflow => Self::Other,
        }
    }

    /// Get Zenodo upload type string
    pub fn zenodo_type(&self) -> &'static str {
        match self {
            Self::Dataset => "dataset",
            Self::Software => "software",
            Self::Publication => "publication",
            Self::Presentation => "presentation",
            Self::Poster => "poster",
            Self::Image => "image",
            Self::Video => "video",
            Self::Other => "other",
        }
    }
}

/// Deposit metadata for archive submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositMetadata {
    /// Title
    pub title: String,
    /// Description/abstract
    pub description: String,
    /// Authors (names)
    pub authors: Vec<String>,
    /// Keywords
    pub keywords: Vec<String>,
    /// License identifier (e.g., "cc-by-4.0")
    pub license: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Related identifiers (DOIs, URLs)
    pub related_identifiers: Vec<RelatedIdentifier>,
}

impl DepositMetadata {
    /// Create from a research artifact
    pub fn from_artifact(artifact: &ResearchArtifact) -> Self {
        Self {
            title: artifact.title.clone(),
            description: artifact
                .description
                .clone()
                .unwrap_or_else(|| format!("{} - {}", artifact.title, artifact.artifact_type)),
            authors: artifact.authors.iter().map(|a| a.name.clone()).collect(),
            keywords: artifact.keywords.clone(),
            license: artifact.license.to_string().to_lowercase(),
            resource_type: ResourceType::from_artifact_type(artifact.artifact_type),
            related_identifiers: artifact
                .doi
                .iter()
                .map(RelatedIdentifier::is_identical_to)
                .collect(),
        }
    }

    /// Create from citation metadata
    pub fn from_citation(citation: &CitationMetadata) -> Self {
        let mut metadata = Self::from_artifact(&citation.artifact);
        if let Some(url) = &citation.url {
            metadata.related_identifiers.push(RelatedIdentifier::is_supplement_to(url));
        }
        metadata
    }
}
