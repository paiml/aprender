//! Research artifact with full metadata.

use serde::{Deserialize, Serialize};

use super::{ArtifactType, Author, License};

/// Research artifact with full metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResearchArtifact {
    /// Unique identifier
    pub id: String,
    /// Artifact title
    pub title: String,
    /// Authors with affiliations and roles
    pub authors: Vec<Author>,
    /// Type of artifact
    pub artifact_type: ArtifactType,
    /// License
    pub license: License,
    /// Digital Object Identifier (optional)
    pub doi: Option<String>,
    /// Version string
    pub version: String,
    /// Abstract or description
    pub description: Option<String>,
    /// Keywords for discovery
    pub keywords: Vec<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl ResearchArtifact {
    /// Create a new research artifact
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        artifact_type: ArtifactType,
        license: License,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            authors: Vec::new(),
            artifact_type,
            license,
            doi: None,
            version: "1.0.0".to_string(),
            description: None,
            keywords: Vec::new(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Add an author
    pub fn with_author(mut self, author: Author) -> Self {
        self.authors.push(author);
        self
    }

    /// Add multiple authors
    pub fn with_authors(mut self, authors: impl IntoIterator<Item = Author>) -> Self {
        self.authors.extend(authors);
        self
    }

    /// Set the DOI
    pub fn with_doi(mut self, doi: impl Into<String>) -> Self {
        self.doi = Some(doi.into());
        self
    }

    /// Set the version
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add keywords
    pub fn with_keywords(mut self, keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.keywords.extend(keywords.into_iter().map(Into::into));
        self
    }

    /// Get the first author (for citation keys)
    pub fn first_author(&self) -> Option<&Author> {
        self.authors.first()
    }
}
