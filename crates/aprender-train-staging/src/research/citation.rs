//! Citation Metadata with BibTeX/CFF export (ENT-020)
//!
//! Provides citation generation in standard academic formats.

use crate::research::artifact::{ArtifactType, Author, ResearchArtifact};
use serde::{Deserialize, Serialize};

/// Citation metadata for academic reference
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitationMetadata {
    /// The research artifact being cited
    pub artifact: ResearchArtifact,
    /// Publication year
    pub year: u16,
    /// Journal or venue name (optional)
    pub journal: Option<String>,
    /// Volume number (optional)
    pub volume: Option<String>,
    /// Page range (optional)
    pub pages: Option<String>,
    /// URL to the resource
    pub url: Option<String>,
    /// Additional keywords
    pub keywords: Vec<String>,
}

impl CitationMetadata {
    /// Create citation metadata from an artifact
    pub fn new(artifact: ResearchArtifact, year: u16) -> Self {
        Self {
            artifact,
            year,
            journal: None,
            volume: None,
            pages: None,
            url: None,
            keywords: Vec::new(),
        }
    }

    /// Set journal/venue
    pub fn with_journal(mut self, journal: impl Into<String>) -> Self {
        self.journal = Some(journal.into());
        self
    }

    /// Set volume
    pub fn with_volume(mut self, volume: impl Into<String>) -> Self {
        self.volume = Some(volume.into());
        self
    }

    /// Set page range
    pub fn with_pages(mut self, pages: impl Into<String>) -> Self {
        self.pages = Some(pages.into());
        self
    }

    /// Set URL
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Add keywords
    pub fn with_keywords(mut self, keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.keywords.extend(keywords.into_iter().map(Into::into));
        self
    }

    /// Generate a citation key (author_year_firstword pattern)
    pub fn generate_citation_key(&self) -> String {
        let author_part = self
            .artifact
            .first_author()
            .map_or_else(|| "anon".to_string(), |a| a.last_name().to_lowercase());

        let first_word = self
            .artifact
            .title
            .split_whitespace()
            .next()
            .unwrap_or("untitled")
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>();

        format!("{author_part}_{year}_{first_word}", year = self.year)
    }

    /// Export to BibTeX format
    pub fn to_bibtex(&self) -> String {
        let entry_type = match self.artifact.artifact_type {
            ArtifactType::Paper => "article",
            ArtifactType::Dataset => "misc",
            ArtifactType::Model => "misc",
            ArtifactType::Code => "software",
            ArtifactType::Notebook => "misc",
            ArtifactType::Workflow => "misc",
        };

        let key = self.generate_citation_key();
        let mut bibtex = format!("@{entry_type}{{{key},\n");

        // Authors (BibTeX format: "Last1, First1 and Last2, First2")
        let authors = format_bibtex_authors(&self.artifact.authors);
        bibtex.push_str(&format!("  author = {{{}}},\n", escape_bibtex(&authors)));

        // Title
        bibtex.push_str(&format!("  title = {{{{{}}}}},\n", escape_bibtex(&self.artifact.title)));

        // Year
        bibtex.push_str(&format!("  year = {{{}}},\n", self.year));

        // Optional fields
        if let Some(journal) = &self.journal {
            bibtex.push_str(&format!("  journal = {{{}}},\n", escape_bibtex(journal)));
        }

        if let Some(volume) = &self.volume {
            bibtex.push_str(&format!("  volume = {{{volume}}},\n"));
        }

        if let Some(pages) = &self.pages {
            bibtex.push_str(&format!("  pages = {{{pages}}},\n"));
        }

        if let Some(doi) = &self.artifact.doi {
            bibtex.push_str(&format!("  doi = {{{doi}}},\n"));
        }

        if let Some(url) = &self.url {
            bibtex.push_str(&format!("  url = {{{url}}},\n"));
        }

        if !self.keywords.is_empty() {
            let kw = self.keywords.join(", ");
            bibtex.push_str(&format!("  keywords = {{{kw}}},\n"));
        }

        bibtex.push('}');
        bibtex
    }

    /// Export to CITATION.cff format (YAML)
    pub fn to_cff(&self) -> String {
        let mut cff = String::new();

        cff.push_str("cff-version: 1.2.0\n");
        cff.push_str(&format!(
            "message: \"If you use this {}, please cite it as below.\"\n",
            self.artifact.artifact_type.to_string().to_lowercase()
        ));

        // Type mapping
        let cff_type = match self.artifact.artifact_type {
            ArtifactType::Paper => "article",
            ArtifactType::Dataset => "dataset",
            ArtifactType::Model => "software",
            ArtifactType::Code => "software",
            ArtifactType::Notebook => "software",
            ArtifactType::Workflow => "software",
        };
        cff.push_str(&format!("type: {cff_type}\n"));

        // Title
        cff.push_str(&format!("title: \"{}\"\n", escape_yaml(&self.artifact.title)));

        // Version
        cff.push_str(&format!("version: \"{}\"\n", self.artifact.version));

        // License
        cff.push_str(&format!("license: {}\n", self.artifact.license));

        // DOI
        if let Some(doi) = &self.artifact.doi {
            cff.push_str(&format!("doi: {doi}\n"));
        }

        // URL
        if let Some(url) = &self.url {
            cff.push_str(&format!("url: \"{url}\"\n"));
        }

        // Date
        cff.push_str(&format!("date-released: \"{}-01-01\"\n", self.year));

        // Authors
        cff.push_str("authors:\n");
        for author in &self.artifact.authors {
            cff.push_str(&format_cff_author(author));
        }

        // Keywords
        if !self.keywords.is_empty() || !self.artifact.keywords.is_empty() {
            cff.push_str("keywords:\n");
            for kw in &self.artifact.keywords {
                cff.push_str(&format!("  - \"{kw}\"\n"));
            }
            for kw in &self.keywords {
                cff.push_str(&format!("  - \"{kw}\"\n"));
            }
        }

        // Abstract
        if let Some(desc) = &self.artifact.description {
            cff.push_str(&format!("abstract: \"{}\"\n", escape_yaml(desc)));
        }

        cff
    }
}

/// Format authors for BibTeX (Last, First and Last2, First2)
fn format_bibtex_authors(authors: &[Author]) -> String {
    authors
        .iter()
        .map(|a| {
            let parts: Vec<&str> = a.name.split_whitespace().collect();
            if parts.len() >= 2 {
                let last = parts.last().expect("parts guaranteed non-empty by len check");
                let first = parts[..parts.len() - 1].join(" ");
                format!("{last}, {first}")
            } else {
                a.name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" and ")
}

/// Format a single author for CFF
fn format_cff_author(author: &Author) -> String {
    let mut cff = String::new();

    let parts: Vec<&str> = author.name.split_whitespace().collect();
    if parts.len() >= 2 {
        let family = parts.last().expect("parts guaranteed non-empty by len check");
        let given = parts[..parts.len() - 1].join(" ");
        cff.push_str(&format!("  - family-names: \"{family}\"\n"));
        cff.push_str(&format!("    given-names: \"{given}\"\n"));
    } else {
        cff.push_str(&format!("  - name: \"{}\"\n", author.name));
    }

    if let Some(orcid) = &author.orcid {
        cff.push_str(&format!("    orcid: \"https://orcid.org/{orcid}\"\n"));
    }

    if let Some(affiliation) = author.affiliations.first() {
        cff.push_str(&format!("    affiliation: \"{}\"\n", affiliation.name));
    }

    cff
}

/// Escape special characters for BibTeX
fn escape_bibtex(s: &str) -> String {
    s.replace('&', r"\&")
        .replace('%', r"\%")
        .replace('$', r"\$")
        .replace('#', r"\#")
        .replace('_', r"\_")
        .replace('{', r"\{")
        .replace('}', r"\}")
        .replace('~', r"\~{}")
        .replace('^', r"\^{}")
}

/// Escape special characters for YAML strings
fn escape_yaml(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::artifact::{Affiliation, ContributorRole, License};

    fn create_test_artifact() -> ResearchArtifact {
        let author = Author::new("Alice Smith")
            .with_orcid("0000-0002-1825-0097")
            .expect("operation should succeed")
            .with_role(ContributorRole::Conceptualization)
            .with_affiliation(Affiliation::new("MIT"));

        ResearchArtifact::new(
            "test-001",
            "Deep Learning for Natural Language Processing",
            ArtifactType::Paper,
            License::CcBy4,
        )
        .with_author(author)
        .with_doi("10.1234/example.2024")
        .with_description("A novel approach to NLP")
    }

    #[test]
    fn test_bibtex_generation() {
        let artifact = create_test_artifact();
        let citation = CitationMetadata::new(artifact, 2024)
            .with_journal("Nature Machine Intelligence")
            .with_volume("6")
            .with_pages("123-145");

        let bibtex = citation.to_bibtex();

        assert!(bibtex.starts_with("@article{"));
        assert!(bibtex.contains("author = {Smith, Alice}"));
        assert!(bibtex.contains("title = {{Deep Learning for Natural Language Processing}}"));
        assert!(bibtex.contains("year = {2024}"));
        assert!(bibtex.contains("journal = {Nature Machine Intelligence}"));
        assert!(bibtex.contains("volume = {6}"));
        assert!(bibtex.contains("pages = {123-145}"));
        assert!(bibtex.contains("doi = {10.1234/example.2024}"));
    }

    #[test]
    fn test_bibtex_escaping_special_chars() {
        let artifact = ResearchArtifact::new(
            "test-002",
            "Machine Learning & Data Science: A 100% Complete Guide",
            ArtifactType::Paper,
            License::Mit,
        )
        .with_author(Author::new("John O'Brien"));

        let citation = CitationMetadata::new(artifact, 2024);
        let bibtex = citation.to_bibtex();

        assert!(bibtex.contains(r"Machine Learning \& Data Science"));
        assert!(bibtex.contains(r"100\% Complete"));
    }

    #[test]
    fn test_cff_generation() {
        let artifact = create_test_artifact();
        let citation = CitationMetadata::new(artifact, 2024)
            .with_url("https://example.com/paper")
            .with_keywords(["deep learning", "NLP"]);

        let cff = citation.to_cff();

        assert!(cff.contains("cff-version: 1.2.0"));
        assert!(cff.contains("type: article"));
        assert!(cff.contains("title: \"Deep Learning for Natural Language Processing\""));
        assert!(cff.contains("license: CC-BY-4.0"));
        assert!(cff.contains("doi: 10.1234/example.2024"));
        assert!(cff.contains("url: \"https://example.com/paper\""));
        assert!(cff.contains("family-names: \"Smith\""));
        assert!(cff.contains("given-names: \"Alice\""));
        assert!(cff.contains("orcid: \"https://orcid.org/0000-0002-1825-0097\""));
        assert!(cff.contains("affiliation: \"MIT\""));
        assert!(cff.contains("- \"deep learning\""));
        assert!(cff.contains("- \"NLP\""));
    }

    #[test]
    fn test_citation_key_generation() {
        let artifact = create_test_artifact();
        let citation = CitationMetadata::new(artifact, 2024);

        let key = citation.generate_citation_key();
        assert_eq!(key, "smith_2024_deep");
    }

    #[test]
    fn test_citation_key_no_author() {
        let artifact = ResearchArtifact::new(
            "test-003",
            "Anonymous Dataset",
            ArtifactType::Dataset,
            License::Cc0,
        );

        let citation = CitationMetadata::new(artifact, 2023);
        let key = citation.generate_citation_key();

        assert_eq!(key, "anon_2023_anonymous");
    }

    #[test]
    fn test_multiple_authors_bibtex() {
        let author1 = Author::new("Alice Smith");
        let author2 = Author::new("Bob Jones");
        let author3 = Author::new("Carol Williams");

        let artifact = ResearchArtifact::new(
            "test-004",
            "Collaborative Research Paper",
            ArtifactType::Paper,
            License::CcBy4,
        )
        .with_authors([author1, author2, author3]);

        let citation = CitationMetadata::new(artifact, 2024);
        let bibtex = citation.to_bibtex();

        assert!(bibtex.contains("author = {Smith, Alice and Jones, Bob and Williams, Carol}"));
    }

    #[test]
    fn test_dataset_bibtex_type() {
        let artifact = ResearchArtifact::new(
            "dataset-001",
            "ImageNet Subset",
            ArtifactType::Dataset,
            License::CcBy4,
        );

        let citation = CitationMetadata::new(artifact, 2024);
        let bibtex = citation.to_bibtex();

        assert!(bibtex.starts_with("@misc{"));
    }

    #[test]
    fn test_software_bibtex_type() {
        let artifact = ResearchArtifact::new(
            "code-001",
            "PyTorch Lightning",
            ArtifactType::Code,
            License::Apache2,
        );

        let citation = CitationMetadata::new(artifact, 2024);
        let bibtex = citation.to_bibtex();

        assert!(bibtex.starts_with("@software{"));
    }

    #[test]
    fn test_cff_single_name_author() {
        let artifact =
            ResearchArtifact::new("test-005", "Single Name Test", ArtifactType::Code, License::Mit)
                .with_author(Author::new("Madonna"));

        let citation = CitationMetadata::new(artifact, 2024);
        let cff = citation.to_cff();

        assert!(cff.contains("- name: \"Madonna\""));
    }

    #[test]
    fn test_keywords_in_bibtex() {
        let artifact = create_test_artifact();
        let citation = CitationMetadata::new(artifact, 2024).with_keywords([
            "machine learning",
            "transformers",
            "attention",
        ]);

        let bibtex = citation.to_bibtex();

        assert!(bibtex.contains("keywords = {machine learning, transformers, attention}"));
    }
}
