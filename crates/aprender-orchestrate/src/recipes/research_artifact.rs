//! Research artifact recipe implementation.

use crate::experiment::{
    CitationMetadata, CitationType, CreditRole, ResearchArtifact, ResearchContributor,
};
use crate::recipes::RecipeResult;

/// Academic research artifact recipe
#[derive(Debug)]
pub struct ResearchArtifactRecipe {
    artifact: ResearchArtifact,
}

impl ResearchArtifactRecipe {
    /// Create a new research artifact recipe
    pub fn new(title: impl Into<String>, abstract_text: impl Into<String>) -> Self {
        Self {
            artifact: ResearchArtifact {
                title: title.into(),
                abstract_text: abstract_text.into(),
                contributors: Vec::new(),
                keywords: Vec::new(),
                doi: None,
                arxiv_id: None,
                license: "MIT".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                datasets: Vec::new(),
                code_repositories: Vec::new(),
                pre_registration: None,
            },
        }
    }

    /// Add a contributor
    pub fn add_contributor(
        &mut self,
        name: impl Into<String>,
        affiliation: impl Into<String>,
        roles: Vec<CreditRole>,
    ) {
        self.artifact.contributors.push(ResearchContributor {
            name: name.into(),
            orcid: None,
            affiliation: affiliation.into(),
            roles,
            email: None,
        });
    }

    /// Add keywords
    pub fn add_keywords(&mut self, keywords: Vec<String>) {
        self.artifact.keywords.extend(keywords);
    }

    /// Set DOI
    pub fn set_doi(&mut self, doi: impl Into<String>) {
        self.artifact.doi = Some(doi.into());
    }

    /// Add dataset reference
    pub fn add_dataset(&mut self, dataset: impl Into<String>) {
        self.artifact.datasets.push(dataset.into());
    }

    /// Add code repository
    pub fn add_repository(&mut self, repo: impl Into<String>) {
        self.artifact.code_repositories.push(repo.into());
    }

    /// Generate citation metadata
    pub fn generate_citation(&self) -> CitationMetadata {
        let authors: Vec<String> =
            self.artifact.contributors.iter().map(|c| c.name.clone()).collect();

        let now = chrono::Utc::now();

        CitationMetadata {
            citation_type: CitationType::Software,
            title: self.artifact.title.clone(),
            authors,
            year: now.format("%Y").to_string().parse().unwrap_or(2024),
            month: Some(now.format("%m").to_string().parse().unwrap_or(1)),
            doi: self.artifact.doi.clone(),
            url: self.artifact.code_repositories.first().cloned(),
            venue: None,
            volume: None,
            pages: None,
            publisher: None,
            version: Some("1.0.0".to_string()),
        }
    }

    /// Build the research artifact
    pub fn build(&self) -> RecipeResult {
        let mut result = RecipeResult::success("research-artifact");
        result = result.with_metric("contributor_count", self.artifact.contributors.len() as f64);
        result = result.with_metric("keyword_count", self.artifact.keywords.len() as f64);
        result = result.with_metric("dataset_count", self.artifact.datasets.len() as f64);

        if self.artifact.doi.is_some() {
            result = result.with_artifact("DOI registered");
        }

        let citation = self.generate_citation();
        result = result.with_artifact(format!("BibTeX: {}", citation.to_bibtex("artifact")));

        result
    }

    /// Get the artifact
    pub fn artifact(&self) -> &ResearchArtifact {
        &self.artifact
    }
}
